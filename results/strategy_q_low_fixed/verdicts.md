| Check | Result | Detail |
|---|---|---|
| Convergence rule (<= 0.05 s) | held | 138 of 138 M1 results converged, 4 after grid refinement; 0 excluded |
| S1 | held | Australia 0.00, China 0.00, Miami 0.00, Canada 0.00 |
| S2 | NOT held | Australia 24 of 27 (89%), China 21 of 27 (78%), Miami 22 of 27 (81%), Canada 27 of 27 (100%); excluded cars counted as not clipping |
| S3 | held | 4 of 4 circuits clip without the taper: Australia 0.00, China 0.36, Miami 0.14, Canada 0.00 |
| S4 | held | Australia 21 km/h lower, China 31 km/h lower, Miami 25 km/h lower, Canada 28 km/h lower |
| S5 | NOT held | Australia optimum faster by 3.675 s, China optimum faster by 3.133 s, Miami optimum faster by 3.295 s, Canada optimum faster by 2.822 s |
| S6 | NOT held | Australia M2-M1 +3.749 s, r=0.58, China M2-M1 +1.207 s, r=0.67, Miami M2-M1 +3.493 s, r=0.46, Canada M2-M1 +1.232 s, r=0.69 |
|  | S7 Australia | optimum 790 m, observed IQR 703-752 m |
|  | S7 China | optimum 760 m, observed IQR 558-662 m |
|  | S7 Miami | optimum 830 m, observed IQR 571-677 m |
|  | S7 Canada | optimum 510 m, observed IQR 521-584 m |
| S7 (exploratory) | NOT held | 0 of 4 inside |
| S1 (corrected rules) | held | Australia 0.07, China 0.00, Miami 0.00, Canada 0.00 |
| S3 (corrected rules) | held | 4 of 4 clip without the taper: Australia 0.17, China 0.00, Miami 0.14, Canada 0.00 |
| S4 (corrected rules) | held | Australia 20 km/h, China 30 km/h, Miami 25 km/h, Canada 30 km/h |
| Starting charge | sensitivity | Australia 0.191 s between 30% and 70%, Canada 0.320 s between 30% and 70%, China 0.020 s between 30% and 70%, Miami 0.013 s between 30% and 70% |

**Per-race 2026 rules** (configs/rules_2026_races.yaml as it stood when these rows were run; see the research log). Same thresholds as above.

| Variant | Circuit | Clip ratio | Clips? | Deploy speed gap | Lap vs original | Convergence |
|---|---|---|---|---|---|---|
| rules | Australia | 0.07 | yes | 20 km/h | +0.313 s | 0.006 s |
| rules | China | 0.00 | yes | 30 km/h | +0.055 s | 0.025 s |
| rules | Miami | 0.00 | yes | 25 km/h | +0.000 s | 0.012 s |
| rules | Canada | 0.00 | yes | 30 km/h | +0.033 s | 0.012 s |
| rules_no_taper | Australia | 0.17 | yes | 19 km/h | +0.274 s | 0.007 s |
| rules_no_taper | China | 0.00 | yes | 28 km/h | +0.028 s | 0.024 s |
| rules_no_taper | Miami | 0.14 | yes | 24 km/h | -0.074 s | 0.014 s |
| rules_no_taper | Canada | 0.00 | yes | 30 km/h | +0.027 s | 0.012 s |
| rules_h8 | Australia | 0.07 | yes | 20 km/h | +0.313 s | 0.006 s |
| rules_h8 | China | 0.37 | yes | 31 km/h | +0.230 s | 0.025 s |
| rules_h8 | Miami | 0.00 | yes | 27 km/h | +0.151 s | 0.005 s |
| rules_h8 | Canada | 0.00 | yes | 30 km/h | +0.033 s | 0.012 s |
| rules_deploy250 | Miami | 0.11 | yes | 19 km/h | +0.545 s | 0.024 s |
| rules_deploy250 | Canada | 0.37 | yes | 23 km/h | +0.603 s | 0.001 s |
