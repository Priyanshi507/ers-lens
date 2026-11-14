| Check | Result | Detail |
|---|---|---|
| Convergence rule (<= 0.05 s) | held | 138 of 138 M1 results converged, 0 after grid refinement; 0 excluded |
| S1 | held | Australia 0.17, China 0.38, Miami 0.00, Canada 0.00 |
| S2 | NOT held | Australia 18 of 27 (67%), China 25 of 27 (93%), Miami 27 of 27 (100%), Canada 26 of 27 (96%); excluded cars counted as not clipping |
| S3 | held | 4 of 4 circuits clip without the taper: Australia 0.08, China 0.00, Miami 0.07, Canada 0.00 |
| S4 | NOT held | Australia 16 km/h lower, China 24 km/h lower, Miami 19 km/h lower, Canada 23 km/h lower |
| S5 | NOT held | Australia optimum faster by 2.756 s, China optimum faster by 2.177 s, Miami optimum faster by 2.453 s, Canada optimum faster by 2.010 s |
| S6 | NOT held | Australia M2-M1 +1.334 s, r=0.73, China M2-M1 +1.403 s, r=0.40, Miami M2-M1 +0.515 s, r=0.75, Canada M2-M1 +1.039 s, r=0.84 |
|  | S7 Australia | optimum 880 m, observed IQR 703-752 m |
|  | S7 China | optimum 800 m, observed IQR 558-662 m |
|  | S7 Miami | optimum 870 m, observed IQR 571-677 m |
|  | S7 Canada | optimum 610 m, observed IQR 521-584 m |
| S7 (exploratory) | NOT held | 0 of 4 inside |
| S1 (corrected rules) | NOT held | Australia 0.70, China 0.47, Miami 0.00, Canada 0.00 |
| S3 (corrected rules) | held | 3 of 4 clip without the taper: Australia 0.74, China 0.00, Miami 0.07, Canada 0.00 |
| S4 (corrected rules) | NOT held | Australia 16 km/h, China 24 km/h, Miami 19 km/h, Canada 24 km/h |
| Starting charge | sensitivity | Australia 0.210 s between 30% and 70%, Canada 0.267 s between 30% and 70%, China 0.127 s between 30% and 70%, Miami 0.072 s between 30% and 70% |

**Per-race 2026 rules** (configs/rules_2026_races.yaml as it stood when these rows were run; see the research log). Same thresholds as above.

| Variant | Circuit | Clip ratio | Clips? | Deploy speed gap | Lap vs original | Convergence |
|---|---|---|---|---|---|---|
| rules | Australia | 0.70 | no | 16 km/h | +0.343 s | 0.016 s |
| rules | China | 0.47 | yes | 24 km/h | +0.026 s | 0.002 s |
| rules | Miami | 0.00 | yes | 19 km/h | +0.000 s | 0.002 s |
| rules | Canada | 0.00 | yes | 24 km/h | +0.062 s | 0.002 s |
| rules_no_taper | Australia | 0.74 | no | 14 km/h | +0.316 s | 0.017 s |
| rules_no_taper | China | 0.00 | yes | 23 km/h | +0.009 s | 0.002 s |
| rules_no_taper | Miami | 0.07 | yes | 17 km/h | -0.055 s | 0.000 s |
| rules_no_taper | Canada | 0.00 | yes | 24 km/h | +0.050 s | 0.002 s |
| rules_h8 | Australia | 0.70 | no | 16 km/h | +0.343 s | 0.016 s |
| rules_h8 | China | 0.00 | yes | 25 km/h | +0.225 s | 0.006 s |
| rules_h8 | Miami | 0.00 | yes | 20 km/h | +0.191 s | 0.005 s |
| rules_h8 | Canada | 0.00 | yes | 24 km/h | +0.062 s | 0.002 s |
| rules_deploy250 | Miami | 0.00 | yes | 14 km/h | +0.354 s | 0.002 s |
| rules_deploy250 | Canada | 0.50 | yes | 18 km/h | +0.473 s | 0.001 s |
