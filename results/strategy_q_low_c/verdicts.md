| Check | Result | Detail |
|---|---|---|
| Convergence rule (<= 0.05 s) | held | 138 of 138 M1 results converged, 5 after grid refinement; 0 excluded |
| S1 | held | Australia 0.12, China 0.39, Miami 0.00, Canada 0.00 |
| S2 | held | Australia 24 of 27 (89%), China 26 of 27 (96%), Miami 24 of 27 (89%), Canada 27 of 27 (100%); excluded cars counted as not clipping |
| S3 | held | 4 of 4 circuits clip without the taper: Australia 0.00, China 0.50, Miami 0.00, Canada 0.00 |
| S4 | held | Australia 21 km/h lower, China 31 km/h lower, Miami 25 km/h lower, Canada 28 km/h lower |
| S5 | NOT held | Australia optimum faster by 3.658 s, China optimum faster by 3.164 s, Miami optimum faster by 3.380 s, Canada optimum faster by 2.842 s |
| S6 | NOT held | Australia M2-M1 +1.986 s, r=0.71, China M2-M1 +0.660 s, r=0.81, Miami M2-M1 +2.164 s, r=0.65, Canada M2-M1 +1.290 s, r=0.66 |
|  | S7 Australia | optimum 790 m, observed IQR 703-752 m |
|  | S7 China | optimum 765 m, observed IQR 558-662 m |
|  | S7 Miami | optimum 835 m, observed IQR 571-677 m |
|  | S7 Canada | optimum 510 m, observed IQR 521-584 m |
| S7 (exploratory) | NOT held | 0 of 4 inside |
| S1 (corrected rules) | NOT held | Australia 0.58, China 0.00, Miami 0.00, Canada 0.00 |
| S3 (corrected rules) | held | 4 of 4 clip without the taper: Australia 0.15, China 0.07, Miami 0.00, Canada 0.00 |
| S4 (corrected rules) | held | Australia 20 km/h, China 30 km/h, Miami 25 km/h, Canada 30 km/h |
| Starting charge | sensitivity | Australia 0.185 s between 30% and 70%, Canada 0.320 s between 30% and 70%, China 0.022 s between 30% and 70%, Miami 0.013 s between 30% and 70% |

**Per-race 2026 rules** (configs/rules_2026_races.yaml as it stood when these rows were run; see the research log). Same thresholds as above.

| Variant | Circuit | Clip ratio | Clips? | Deploy speed gap | Lap vs original | Convergence |
|---|---|---|---|---|---|---|
| rules | Australia | 0.58 | no | 20 km/h | +0.316 s | 0.012 s |
| rules | China | 0.00 | yes | 30 km/h | +0.080 s | 0.020 s |
| rules | Miami | 0.00 | yes | 25 km/h | +0.000 s | 0.009 s |
| rules | Canada | 0.00 | yes | 30 km/h | +0.030 s | 0.011 s |
| rules_no_taper | Australia | 0.15 | yes | 19 km/h | +0.270 s | 0.007 s |
| rules_no_taper | China | 0.07 | yes | 28 km/h | +0.050 s | 0.023 s |
| rules_no_taper | Miami | 0.00 | yes | 24 km/h | -0.077 s | 0.009 s |
| rules_no_taper | Canada | 0.00 | yes | 30 km/h | +0.025 s | 0.012 s |
| rules_h8 | Australia | 0.58 | no | 20 km/h | +0.316 s | 0.012 s |
| rules_h8 | China | 0.00 | yes | 31 km/h | +0.245 s | 0.023 s |
| rules_h8 | Miami | 0.00 | yes | 27 km/h | +0.125 s | 0.001 s |
| rules_h8 | Canada | 0.00 | yes | 30 km/h | +0.030 s | 0.011 s |
| rules_deploy250 | Miami | 0.00 | yes | 19 km/h | +0.548 s | 0.025 s |
| rules_deploy250 | Canada | 0.37 | yes | 23 km/h | +0.603 s | 0.001 s |
