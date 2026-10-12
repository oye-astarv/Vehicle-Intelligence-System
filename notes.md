# Vehicle Intelligence Program — Journey Log

## Project
VEHICLE INTELLIGENCE PLATFORM — AI-Driven Vehicle Fault Detection, Diagnosis & Engineering Intelligence.
End-to-end: simulate healthy vehicle → inject faults → collect telemetry → ML diagnosis → explainable report.
NOT a Kaggle classifier on existing dataset.

## Phases (user-defined)
1. Sim vehicle w/ faults
2. Collect telemetry + detect anomaly + classify fault
3. Fault cause localisation
4. Explainable diagnostic report + metrics

## Constraints
- Machine: Intel i7, 16GB RAM, Mac, no MATLAB license
- No BeamNG (Windows-only), no IsaacSim (Ubuntu/Windows + RTX 4080+ only, no Mac), no Simulink (no license, Intel deprecated after R2025b), no Fusion 360 CAD (mesh != dynamics)
- MVP: custom Python sim (numpy/pandas, 20-50Hz, parquet). Strong later: Webots (Intel Mac OK) or PyChrono. PyTorch for DL (CPU only, keep models small: MLP / 1D-CNN / small LSTM)

## Final Blueprint (user-approved)
Driver (throttle/brake/steering/gear)
  -> CAR (PARAMS fixed, STATE changing, update_state())
  <- FAULT INJECTOR (physical faults tweak params)
  -> SENSORS (read car.state + noise + sensor faults)
  -> TELEMETRY (timestamp + sensor readings)
  -> ML DIAGNOSER (detection + classification)
  -> OUTPUT (fault + confidence)

Rules:
- PARAMS / STATE / TELEMETRY separated. ML sees TELEMETRY only.
- Physics deterministic, fixed dt. Noise only in sensor layer.
- Faults as config with onset_time, logged in manifest, never passed to model.
- Car class does sim only. No ML / IO / plotting inside.

## Telemetry spec (frozen for Phase 1)
20-50Hz, 1 parquet per run + manifest.csv (run_id, fault_label, location, severity, maneuver, fault_onset_time).
Channels: timestamp, speed, rpm, throttle, brake, steering, gear, 4x wheel_speed, engine_temp, 4x brake_temp, 4x tyre_temp, accel_xyz, slip_x4.

## Faults v1 (updated)
- List: None, cooling_failure, brake_failure, engine_failure, tyre_degradation
- Location: required FL/FR/RL/RR for brake/tyre, None/system for cooling/engine
- Severity: coolant_eff 1.0->0.6/0.35/0.15; brake_torque x0.7/0.4/0.15; engine_scale 0.85/0.7/0.5; tyre radius x0.99/0.97/0.95 + rr 0.020/0.026/0.035
- Injector rolls random fault/severity/location/onset per run (onset 20-40s), alters params once, state corrupts via dynamics. Must log to manifest.

## Key relationships (ML gold)
- rpm/speed per gear, throttle->rpm->speed lag
- dT/dt for overheat, single-wheel bias for tyre/brake, frozen signal vs physics for sensor fault
- Validate: cruise stable temps, hard brake temp spike, corner outer wheels faster

## Canonical recipe (owner-approved)1. Normalize data (per-channel z-score, train-only stats)
2. Remove zero-std channels (slip_* identically zero in straight-line sim)
3. Parallel classification heads (type + location + severity on shared windows)
4. 5s windows for cooling/brake/tyre + 30s lookback channel for engine
(Canon labels in cells: #1 proxy drop in typeclf1; #3 calibration in calib1; #4 engine family on sag/deficit/proxy + sag30/def30/prox30 in window1. RF canon note: normalization + slip-drop mattered for CNN; RF wins are proxy-drop + horizons.)
- 2026-10-11: calib1 verified live: base 0.789, t=0.6 -> 55%@91%, t=0.7 -> 41%@96%. 30s feats in pipeline.
- CLOSE: main_polished.ipynb 24 cells, full exec zero fails. Spot-check CORRECT (brake FL 0.8). Fixed along the way: stale windows cache + guard assert, scratch-cell API drift. Type 0.789 / loc 0.73 / sev 0.19 / gate 0.08 / abstention 48%@90%. Ship it.

## Decisions log
- 2026-09-15: Deleted poisoned quick-pixel session (encrypted_content error). Kept tidy-circuit.
- 2026-09-15: Mentorship-only mode. No code unless asked.
- 2026-09-15: Rejected OpenTelemetry (SW observability, not vehicle telemetry). Rejected F1-scale volume; kept F1 logic.
- 2026-09-24: main.py created (documented pipeline) + notebook purpose headers + run_forced/SensorSystem cells added.
- 2026-09-24: Heat tuned — cooling heat_k 0.02->0.08 (cruise ~86C), brakes 8->50. Validation gate passed: NORMAL 80.6C vs cooling 0.7 94.2C (+13.6); brake RR NORMAL max 40.4 vs fault 33.5 (dead brake cold).
- 2026-09-24: Dataset built — 50 runs x 120s @20Hz (data/run_*.parquet + manifest.csv). Faults: ~11 NORMAL/10 cooling/9 brake/8 engine/12 tyre. Maneuvers cover all 5 (accel thin at 5). Severity mild 12/mod 25/sev 13.
- 2026-09-24: Phase 2 started — windowing cell added (5s win / 2.5s stride @20Hz, pre-onset windows labeled NORMAL). Smoke: 4 runs -> 188 windows x 29 features. Dataset cell now visibly build_dataset(50,120,0.05).
- 2026-09-24: Anomaly baseline added (anomaly1 cell, IsolationForest on NORMAL windows only, 8 feats). 12-run smoke: 564 windows/120 NORMAL; abnormal rate NORMAL 0.35 (FPR high — tune contamination), brake 1.00/engine 0.90/tyre 0.98. Next: tune + fault classifier.
- 2026-10-04: Clean Run All top-to-bottom, zero errors. Noisy data live (sensor fingerprints confirmed). 50 runs balanced-ish (13 brake/10 NORMAL/10 cooling/10 tyre/7 engine — engine thin). Anomaly@auto: NORMAL 0.075, brake 0.90, cooling 0.74, engine 0.92. Windowing 50 feats (dual-scale thermal in).
- 2026-10-04: Phase 2/3 cML added (typeclf1: RF type on abnormal, group splits, proxies dropped; lochd1: location RF 0.73-0.78; sevregr1: severity MAE ~0.23, unstable per-fault at n=50). NaN-label + string-feat bugs fixed. Type accuracy ~0.4 = small-data variance (1-3 test runs/fault), not feature failure.
- 2026-10-04: Stratified bulk builder added (bulkbuild1 cell -> data200/, 5x5x8=200 runs). Smoke 25/25 balanced OK. Full 200 pending user launch (~10-15 min).
- 2026-10-04: Bulk built OK — data200/ 200 runs, exact 8/combo grid, noisy. DATADIR switch cell added (datadir1); window/anomaly/type builders now read DATADIR=data200. Head-8 check: 376 windows all NORMAL correct (first 40 bulk runs are NORMAL by grid order).
- 2026-10-04: Eval cell added (eval1, caches data200/windows.parquet, 9400 windows). LOFO novelty flags only 0.35-0.55 vs 0.74-0.98 resubstitution — honest gap, boundary overfits seen runs. Needs NORMAL diversity + maneuver-robust feats before report claims.
- 2026-10-04: Maneuver holdout added — damning: hold cruise_low detection 0.035 (mild faults in gentle regime look healthier than hot-regime NORMALs), accel/brake_stop FPR 44-49%. Absolute-temp features memorize regime. Fix: maneuver-conditioned baseline (residual vs healthy same-maneuver model) before report/dashboard.
- 2026-10-04: Tried residuals (resid1, median-centering) — HURT (resub faults 0.35-0.43 vs 0.74-0.98). Median over maneuver mixes phases; absolute level carries the signal. Tried per-maneuver detectors (eval3) — worse overall except dynamic regimes. Documented negative results, kept global boundary.
- 2026-10-04: Bulk data fixed everything: type RF 0.78, loc RF 0.73-0.78 (5 classes), severity MAE ~0.20-0.23 (brake best, engine worst). Report cell (report1) + gate 0.08 calibrated (NORMAL frac ~0.06). Models exported to models/models.pkl (45MB). Dashboard app.py self-contained (needs pip install streamlit). Verified: run 0 NORMAL, run 120 engine conf 1.0, run 40 mild-cooling missed = known cruise_low blind spot.
- 2026-10-04: Bulk rebuilt (cell 23) -> models re-exported fresh (type 0.758, loc 0.77). Gate rechecked: NORMAL frac 0.04-0.06, cruise_low faults 0.0-0.17 (tyre visible, cooling/engine/brake-mild near-invisible at 5s). Gate stays 0.08; cruise_low-mild documented as blind spot, dynamic regimes reliable. Always re-export models after bulk rebuild.
- 2026-10-04: CNN track — 19ch raw 0.29 (chance-level, loss stuck ~1.07); true z-score 0.34; audit found slip_* std ~5e-17 (identically zero in straight-line sim) poisoning norm 1e10x. Try4 (15ch, proper z-score, 5-fold, 20ep): test 0.36, macro F1 0.36 (Engine R0.68 only bright spot; Brake R0.18). Preprocessing line closed — small CNN can't separate raw 5s sequences at this scale. RF 0.68-0.78 stands (split seeds differ 1 vs 0, gap dwarfs variance).
- 2026-10-04: Staging benchmark (external, signal-based, no clocks): one-sided sustained dev (temp level/sag/dTpos/bias/bspread, max-OR, per-maneuver t1@NORMAL-80th/t2@faulty-median). 3-way RF run-grouped acc 0.947 (established recall 94%, transition 51%, healthy-as-0 80% with 14% reaching stage 2). Detection delay median 6.6s p90 26.3s (n=158 runs). Negative iterations documented: global thresholds + mean-aggregation fail (t2<t1); abs-transient scores fail. Recipe for internal build approved.
- 2026-10-04: Staging CNN benchmark (external, 15ch, 12ep): overall acc 0.901 (in estimated band) BUT transition recall 0.00 — never predicts stage 1 (179 samples starved, collapses to 0/2). RF staging (51% transition recall) stands. Lesson: headline accuracy hides minority collapse twice now.
- 2026-10-04: Blend benchmark (external, RF-proba + CNN-logits -> logistic, run-grouped, straddlers dropped): RF 0.691, CNN 0.465, STACK 0.711 (+2pts, as predicted — correlated errors, modest gain). Verdict: keep stack iff free; real budget belongs to diverse views (thermal horizons, bias-weighted staging).
- 2026-10-04: Engine-evidence round (external, torque-proxy family: sag persistence, deficit integral, proxy mean): acc 0.702->0.711, engine recall 0.431->0.474, others flat (tyre 0.994->0.985 noise), new feats rank #3-4 importance. KEEP (principled, real, small). Engine still weak — needs horizons/data next, not more 5s feats.
- 2026-10-11: Proposition adopted (5s base for cooling/brake/tyre + 30s lookback channel for engine only): notebook window1 + app.py both carry sag30/def30/prox30. Full export fresh: type 0.789 (best yet), loc 0.733. Maneuver-conditioned expectation DROPPED (resid evidence: hurts). Regime gate stays policy-level (dynamic 0.55 vs cruise 0.39 engine recall).
- 2026-10-04: GRU staging benchmark (external, 30s windows @5Hz=150 steps, 15ch, 1x64, 8ep, run-grouped): acc 0.912 (healthy R0.93, established R0.84). Transition empty AGAIN at 30s scale (1604/396, no stage-1) — labeling wall (t1@80th still below faulty medians... here inverted: transition unpopulated), not a model wall. GRU ≈ RF-staging binary; no reason to prefer it. Recurrence line closed unless 60s+ horizons tried.
- 2026-10-04: Nested protocol (external, 4 folds x 3 seeds, run-grouped): TYPE 0.736±0.016 (fold-std ~0.04), LOC 0.777±0.031. Canonical numbers locked. Fresh-seed data regen (sim-overfit check) still open.
- 2026-10-04: Calibration benchmark (external): reliability near-perfect monotone (0.5-bin 0.71 ... 0.9-bin 0.99). Abstention t=0.6: 48% coverage @ 90% acc; t=0.7: 35% @ 93%. Policy: report-if-confident + abstain-to-manual = shippable honesty.
