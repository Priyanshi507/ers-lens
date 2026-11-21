# Experiment index: every test, run and verdict in one place

A reading guide to `docs/research_log.md` (the dated record, never edited after the fact; corrections are new entries).
Each row says what was asked, the prediction codes, the verdict **as it stands now**, where it ran and where the files are.
Rule for every row: predictions and pass marks were written in the log before the run; misses are reported.
Code names are reused over time (K1-K3, G1-G3, R1-R4, P1-P3, W1-W3 each mean different things in different entries); the date says which.

## 1. How things are run

| Where | Command | Used for |
|---|---|---|
| Mac terminal, repo with `.venv` active | `git am --keep-cr --whitespace=nowarn ~/Downloads/00NN-*.patch`, then `git push` | applying each patch (authored as Priyanshi507, no co-author lines) |
| Mac terminal | `python scripts/<name>.py` | detectors, hindcast, paper numbers, figures, site export (no GPU) |
| Kaggle, GPU T4 x2, Internet on, Persistence "Files only" | Cell 1 clone + `pip install -U "jax[cuda12]==0.11.2" cupy-cuda12x`; Cell 2 `scripts/fia_fix.py ...` (the DP solver) on both GPUs; Cell 3 `FileLink('<name>.zip')` | every DP solve; guide in `scripts/kaggle_<name>.md` |
| This workspace | read the zip, run `hindcast_2026.py` etc., judge against the written bars, commit, make the patch | judging; the shell here cannot reach F1 servers |

Kaggle guides, each a copy-paste of Cells 1-3: `kaggle_gpu_benchmark`, `kaggle_cuda_benchmark`, `kaggle_sectors`, `kaggle_fia_fix`, `kaggle_fia_fix_calendar`, `kaggle_race_plan`, `kaggle_rerun_strategy`, `kaggle_fia_fix_corrected`, `kaggle_rerun_fixed`, `kaggle_rerun_harvest`, `kaggle_race_fixed`, `kaggle_race_refined`, `kaggle_rerun_certified`, `kaggle_lift_coast`, `kaggle_quali`, `kaggle_altitude`, `kaggle_cars`.
Core solver call: `ERSLENS_DP_BACKEND=cuda PYTHONPATH=src python scripts/fia_fix.py --cars original --caps 250 350 --out results/<folder> --events ...`
with options `--lift-coast`, `--quali-rules`, `--push-lap`, `--altitude`, `--refine`, `--assume-recharge-mj`.
Judging: `scripts/hindcast_2026.py [--runs DIR] [--car low|high] [--altitude]`, `fia_fix_verdicts.py`, `quali_test.py`, `lift_real.py`, `paper_numbers.py`.
Test suite: about 200 tests (12 minutes); every patch ran it first.

## 2. The record, in order

| Date | Question | Codes and verdicts now | Run and results |
|---|---|---|---|
| 10-01 to 10-03 | Can drag, engine power and electric energy be recovered from speed traces? | Identifiability result (10-02): drag and power trade off exactly, so they are not separately identifiable from speed alone. ML experiment: P1 held, P2 held, P3' not held, P4 held; robustness R1 held, R2 not held; lap-time-loss chapter closed 10-03 | terminal and CPU; `results/energy`, `results/ml`, `results/ml_robustness`, `results/laptime` |
| 10-05 to 10-10 | Lap-time-optimal energy use ("Strategy Lab") | S1-S7 on three cars (10-28, final table with the corrected rules): S1 held; S3 and S4 held on two of the three cars; S5, S6, S7 not held. Per-car numbers in the 10-28 entries. Convergence rule enforced 10-10 | CPU then Kaggle; `results/strategy*` |
| 10-15 to 10-22 | Is the car model right? | Calibration fails its acceptance rule; drag and power lie on a ridge; two cars at the ends of it (`car_2026_q_low/high.yaml`) | `results/calibration*` |
| 10-18 | The energy rules were wrong | corrected before any re-run; small effect | log entry |
| 10-30 to 10-31 | Do the findings hold on every other 2026 race? | E1 not held (2 of 8 races at 80%); E2 held as written, not informative; E3 not held | OpenF1; `results/multi_season`, `laptime_season` |
| 10-31 | GPU for the solver | G1 not held; G2 not held; L1 held | `scripts/kaggle_gpu_benchmark.md` |
| 11-01 to 11-03 | Did the FIA's Miami fix change super-clipping? | C1 held 16 of 18 (later 15 of 18), C2 not held, C3 not held (later held) | `kaggle_fia_fix*`, `results/fia_fix*` |
| 11-02 | Hand-written CUDA backward pass | K1 held, K2 not held (1.86-1.96x), K3 held | `kaggle_cuda_benchmark.md` |
| 11-02, 11-13 | Whole-race energy plans | W1 not held (2 of 4), W2 not held (3 of 4), W3 not held (3 of 4) | `kaggle_race_plan`, `kaggle_race_refined` |
| 11-03 | The FIA's real 250 kW sectors | P1, P2 held 14 of 14; P3 held 14 of 14 | `kaggle_sectors.md`, `results/sectors*` |
| 11-04 | A modelling error found and fixed; everything re-run | recorded in the log | `kaggle_rerun_fixed.md` |
| 11-13 | Harvest limit corrected; price resolution; optimality certificate | H1 not held, H2 held, H3 held; F1 not held (10 of 12); R1-R4 held (certificate: gap at most 0.003 s with ten steps) | `kaggle_rerun_harvest`, `kaggle_race_refined` |
| 11-13 to 11-14 | Does the model reproduce 2026 clipping, circuit by circuit? (hindcast) | first Y1 0.79 held, Y2 11 of 16 not held; **corrected 11-14** (first measure included braking): **Y1 not held, 0.58; Y2 held, 13 of 16** | `hindcast_2026.py`, `results/fia_fix_calendar_c` |
| 11-14 | Certified re-run of everything | K1 held (535 of 538 solves certified), K2 not held, K3 not held; G1 held, G2 not held (14-17 of 18), G3 held (144 of 144) | `kaggle_rerun_certified.md` |
| 11-14 | Lift-and-coast as a choice | LC0-LC3 held, LC4 not held (ratio 1.64, bar 1.5) | `kaggle_lift_coast.md`, `results/fia_fix_calendar_lc` |
| 11-14 | Real lift-and-coast in telemetry | RL1 not held (-0.01), RL2 not held (64%), RL4 not held (real median 0 m); RL3 see log | `scripts/lift_real.py` |
| 11-14 to 11-15 | Out of sample, protocol fixed first | O1 Mexico City held (model: no clipping; real 18% of laps, 1.0 km/h); O2 withdrawn | `results/oos_real` |
| 11-15 | 2027 forecast frozen | SHA-256 73961e97...0f310f, tag `forecast-2027-v1`, Zenodo 10.5281/zenodo.24238738, OSF fh6mx | `results/forecast_2027*` |
| 11-16 | Is the size error the session (qualifying)? | QL1 not held (0.45), QL2 not held (1.94), QL3 not held, QL4 held (6 of 7) | `kaggle_quali.md`, `results/quali_*` |
| 11-17 | Uncertainty added after the runs (description only) | C1 15 of 18 p=0.008; LC2 17 of 17; Y2 13 of 16 p=0.02; QL4 6 of 7 p=0.13; Y1 interval 0.10-0.86 | `paper_numbers.py` (patch 0076) |
| 11-18 | Is the size error air density? | D1 held, D2 held (Y1 0.579 to 0.596, still under 0.6), D3 held, D4 held (Mexico: 0.00 MJ, 0.0 km/h) | `kaggle_altitude.md`, `results/fia_fix_calendar_alt*` (patches 0077-0079) |
| 11-18 | Is the size error the car? | S1 not held (3.30), S2 not held (2.97), S3 held (0.66, 0.76), S4 not held (Y2 11 and 12 of 16); neither calibrated car is a candidate | `kaggle_cars.md`, `results/fia_fix_calendar_low/high` (patches 0080-0081) |
| 11-21 | Las Vegas out of sample (O1) | limits from the FIA PDF (race 8.0 MJ, qualifying 6.0, 50 kW/s); qualifying dry; **prediction: clip** (24.1 km/h), written 11-21 before the race; judged after it | `kaggle_lasvegas.md` Part A, `results/oos_lasvegas` |
| 11-21 | What a megajoule is worth | EP1 held (18 of 18: the price is a lap-time value); EP2 not held, reversed (lowered-limit circuits 0.060 vs 0.242 s/MJ); EP3 not held, reversed (-0.41) | `kaggle_lasvegas.md` Part B, `scripts/energy_price.py`, `results/energy_price_85/80` |
| 11-21 | Can the model predict the FIA's next limits? | rule fixed first (super-clip ≥ 4.0 MJ at 8.5 MJ → limit below 8.5); calls written 11-21 before the documents: FL1 Qatar below 8.5 (6.10 MJ), FL2 Abu Dhabi below 8.5 (4.91 MJ); a public May plan of qualifying limits suggests FL1 may fail | `kaggle_fia_limits.md`, `results/fia_limits_forward` |

- 2026-11-18: paper draft 8 (answer first, plain-language section, figures 1 and 5 from committed results via `scripts/paper_figures.py`, tests D and S in Appendix Table A1, 201 tests).

## 3. What the project now says

- Direction is right: raising the super-clip limit raises super-clipping (C1, LC3), moves energy out of lift-and-coast (LC2), and qualifying and race move the same way (QL4).
- Size and ranking are not: the model clips 1.9-7 times too much at six circuits, and Y1 does not hold. Tested and not the cause: session (QL2), lift-and-coast (LC4a), air density (D1-D3), the car along the calibration ridge (S1-S2), the speed taper and the grid.
- Untested: battery not full at the line, deployment held back by teams, slipstream.
- Out of sample: Mexico City 1 of 1. Las Vegas (22 Nov) and Qatar (30 Nov) are next; the limit comes from the FIA document, the track from qualifying, the prediction is pushed before the race is opened.

## 4. Errors found and how each was handled

First hindcast included braking (corrected 11-14, Y1 0.79 to 0.58); rules error 10-18; modelling error 11-04; harvest-limit error 11-13; Mexico City command quoting 11-18 (a fault in the guide, not the model); stale numbers in finding 2 (11-13). None was hidden: each is its own dated log entry.
