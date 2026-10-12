"""Vehicle Intelligence Platform — dashboard v2 (Phase 4).
Full-run car state; signals turn red after detection; Diagnose button -> fault/location/evidence.
Run: python3 -m streamlit run app.py
"""
import pickle
import numpy as np
import pandas as pd

DATADIR = "data200"
GATE_FRAC = 0.08

B = pickle.load(open("models/models.pkl", "rb"))
TRF, LRF, SRF, CL, SC = (B["TRF"], B["LRF"], B["SRF"], B["CL"], B["SC"])
CFEATS, LOCF, LOC_COLS, FEATS = (B["CFEATS"], B["LOCF"], B["LOC_COLS"], B["FEATS"])


def make_windows(run_id, df, meta, win_s=5.0, stride_s=2.5, hz=20):
    win, stride = int(win_s * hz), int(stride_s * hz)
    rows = []
    for s in range(0, len(df) - win + 1, stride):
        w = df.iloc[s:s + win]
        t_end = float(w["t"].iloc[-1])
        fault = meta.get("fault")
        import math as _m
        if fault is None or (isinstance(fault, float) and _m.isnan(fault)):
            fault = None
        onset = float(meta.get("onset", 1e9))
        label = "NORMAL" if (fault is None or t_end < onset) else fault
        f = {"run_id": run_id, "t_start": float(w["t"].iloc[0]), "t_end": t_end, "label": label,
             "location": meta.get("location"), "severity": meta.get("severity"),
             "maneuver": meta.get("maneuver")}
        for col in ["speed", "rpm", "engine_temp", "thr", "brk"]:
            f[f"{col}_mean"] = float(w[col].mean())
            f[f"{col}_std"] = float(w[col].std())
            f[f"{col}_max"] = float(w[col].max())
        f["eng_dT"] = float(w["engine_temp"].iloc[-1] - w["engine_temp"].iloc[0]) / win_s
        long_i = max(0, s + win - 1 - int(30 * hz))
        span = max((s + win - 1 - long_i) / hz, 5.0)
        f["eng_dT_long"] = float(df["engine_temp"].iloc[s + win - 1] - df["engine_temp"].iloc[long_i]) / span
        m15_i = max(0, s + win - 1 - int(15 * hz))
        f["eng_mean_15"] = float(df["engine_temp"].iloc[m15_i:s + win].mean())
        for wname in ["FL", "FR", "RL", "RR"]:
            f[f"ws_{wname}_mean"] = float(w[f"ws_{wname}"].mean())
        f["ws_LR"] = (f["ws_FL_mean"] + f["ws_RL_mean"]) / 2 - (f["ws_FR_mean"] + f["ws_RR_mean"]) / 2
        f["ws_FR_diff"] = f["ws_FL_mean"] - f["ws_FR_mean"]
        for wname in ["FL", "FR", "RL", "RR"]:
            f[f"brake_{wname}_mean"] = float(w[f"brake_{wname}"].mean())
            f[f"brake_{wname}_max"] = float(w[f"brake_{wname}"].max())
            f[f"tyre_{wname}_mean"] = float(w[f"tyre_{wname}"].mean())
            f[f"slip_{wname}_max"] = float(w[f"slip_{wname}"].abs().max())
        f["brk_temp_spread"] = max(f[f"brake_{w}_mean"] for w in ["FL", "FR", "RL", "RR"]) - min(
            f[f"brake_{w}_mean"] for w in ["FL", "FR", "RL", "RR"])
        f["slip_max"] = max(f[f"slip_{w}_max"] for w in ["FL", "FR", "RL", "RR"])
        gear_mode = float(w["gear"].mode().iloc[0])
        ratio = [3.2, 2.1, 1.5, 1.1, 0.85, 0.7][int(gear_mode) - 1]
        rpm_exp = float(w["speed"].mean()) * ratio * 3.5 * 60 / (2 * np.pi * 0.32)
        f["rpm_resid"] = float(w["rpm"].mean()) - rpm_exp
        _b0 = max(0, s + win - 1 - 600)
        _bh = df.iloc[_b0:s + win]
        _gr = int(_bh["gear"].mode().iloc[0])
        _ra = [3.2, 2.1, 1.5, 1.1, 0.85, 0.7][_gr - 1]
        _gap = _bh["speed"].values * _ra * 3.5 * 60 / (2 * np.pi * 0.32) - _bh["rpm"].values
        f["sag30"] = float((_gap > 200).mean())
        f["def30"] = float(np.maximum(_gap, 0).mean() / 1000.0)
        f["prox30"] = float(_gap.mean() / 1000.0)
        rows.append(f)
    return pd.DataFrame(rows)


def window_flags(W):
    import numpy as _n
    return (CL.predict(SC.transform(W[FEATS])) == -1)


def diagnose_run(run_id):
    import collections as _c
    df = pd.read_parquet(f"{DATADIR}/run_{int(run_id):03d}.parquet")
    man = pd.read_csv(DATADIR + "/manifest.csv")
    meta = man[man.run_id == int(run_id)].iloc[0].to_dict()
    W = make_windows(run_id, df, meta)
    flags = window_flags(W)
    nflag = int(flags.sum())
    det_t = float(W[flags].t_end.min()) if nflag else None
    status = "ABNORMAL" if (len(W) and nflag / len(W) >= GATE_FRAC) else "NORMAL"
    if status == "NORMAL":
        det_t = None
    fl = meta.get("fault")
    on = float(meta.get("onset", 0))
    post = W[W.t_end >= on] if fl is not None and str(fl) != "nan" else W
    ab = TRF.predict(post[CFEATS])
    top, n = _c.Counter(ab).most_common(1)[0]
    pt = pd.get_dummies(pd.Series(ab), prefix="pt").reindex(columns=LOC_COLS, fill_value=0)
    pf = pd.concat([post[CFEATS].reset_index(drop=True), pt.reset_index(drop=True)], axis=1)
    loctop = _c.Counter(LRF.predict(pf[LOCF])).most_common(1)[0][0]
    sev = float(SRF.predict(post[CFEATS]).mean())
    ev = ["eng_dT_long max %.2f C/s" % W["eng_dT_long"].max(),
          "brake spread max %.1f C" % W["brk_temp_spread"].max(),
          "rpm_resid min %.0f rpm" % W["rpm_resid"].min(),
          "slip max %.3f" % W["slip_max"].max()]
    return {"run": int(run_id), "status": status,
            "fault": (top if status == "ABNORMAL" else "none"),
            "conf": round(n / len(ab), 2), "location": loctop,
            "severity_s": round(sev, 2), "evidence": ev, "truth": fl,
            "detection_t": det_t, "flags": flags,
            "abnorm_frac": round(float(flags.mean()), 3)}


def split_series(df, det_t, col):
    vals = df[col]
    mask = df["t"] > (det_t if det_t is not None else 1e18)
    out = pd.DataFrame({col + " ok": vals.where(~mask), col + " FAULT": vals.where(mask)})
    out.index = df["t"].values
    return out


def main():
    import streamlit as st
    st.set_page_config(page_title="Vehicle Intelligence Platform", layout="wide")
    st.title("Vehicle Intelligence Platform — Engineering Diagnostics")
    man = pd.read_csv(DATADIR + "/manifest.csv")
    sel = st.sidebar.selectbox("Test run", man.run_id.tolist())
    meta = man[man.run_id == sel].iloc[0].to_dict()
    if st.session_state.get("run") != sel:
        st.session_state["run"] = sel
        st.session_state["df"] = pd.read_parquet(f"{DATADIR}/run_{int(sel):03d}.parquet")
        st.session_state["rep"] = diagnose_run(int(sel))
        st.session_state["pos"] = 0.0
        st.session_state["play"] = False
    df, rep = st.session_state["df"], st.session_state["rep"]
    if "pos" not in st.session_state:
        st.session_state["pos"] = 0.0
    if rep["status"] == "ABNORMAL":
        st.error(f"FAULT DETECTED @ t={rep['detection_t']:.1f}s — {rep['fault']}")
    else:
        st.success("Vehicle state: NORMAL — no fault detected")
    st.subheader(f"Run {sel} — {meta.get('maneuver')} (injected onset {meta.get('onset')}s)")
    t_end = float(df["t"].iloc[-1])

    @st.fragment
    def player():
        if "pos" not in st.session_state:
            st.session_state["pos"] = 0.0
        cplay, cslider = st.columns([1, 6])
        if cplay.button("+10s", key="stepbtn"):
            st.session_state["pos"] = min(float(st.session_state.get("pos", 0.0)) + 10.0, t_end)
        t = cslider.slider("Playback time (s)", 0.0, t_end, step=0.5, key="pos")
        row = df[df["t"] <= t].iloc[-1]
        fl = ((pd.Series(rep["flags"]).values)
              if len(rep["flags"]) == 47 else np.zeros(47, dtype=bool))
        import numpy as _np
        _wt = _np.linspace(5.0, t_end, 47)[:len(fl)]
        frac_t = float(fl[_wt <= t].mean()) if (_wt <= t).any() else 0.0
        st.write(f"State @ t={t:.1f}s — {'RED: fault active' if rep['detection_t'] is not None and t >= rep['detection_t'] else 'green: nominal'}"
                 f" (anomaly frac so far: {frac_t:.2f})")
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Speed", f"{row['speed']:.1f} m/s")
        m2.metric("RPM", f"{row['rpm']:.0f}")
        m3.metric("Engine temp", f"{row['engine_temp']:.1f} C")
        m4.metric("Anomaly frac", f"{frac_t:.2f}")

    player()

    if st.button("Diagnose fault"):
        c1, c2, c3 = st.columns(3)
        c1.metric("Fault", f"{rep['fault']} ({rep['conf']})")
        c2.metric("Location", rep["location"])
        c3.metric("Severity s", rep["severity_s"])
        st.write("Evidence:")
        for e in rep["evidence"]:
            st.write("-", e)
        with st.expander("Ground truth (hidden from model)"):
            st.write({k: meta.get(k) for k in ("fault", "location", "severity", "onset", "maneuver")})
    st.caption("Known blind spot: mild faults in gentle cruise — see notes.md evaluation.")


if __name__ == "__main__":
    main()
