| Check | Result | Detail |
|---|---|---|
| Convergence rule (<= 0.05 s) | held | 138 of 138 M1 results converged, 0 after grid refinement; 0 excluded |
| S1 | held | Australia 0.36, China 0.38, Miami 0.10, Canada 0.00 |
| S2 | NOT held | Australia 16 of 27 (59%), China 27 of 27 (100%), Miami 24 of 27 (89%), Canada 26 of 27 (96%); excluded cars counted as not clipping |
| S3 | held | 3 of 4 circuits clip without the taper: Australia 0.20, China 0.00, Miami 0.79, Canada 0.00 |
| S4 | NOT held | Australia 16 km/h lower, China 24 km/h lower, Miami 19 km/h lower, Canada 23 km/h lower |
| S5 | NOT held | Australia optimum faster by 2.675 s, China optimum faster by 2.173 s, Miami optimum faster by 2.369 s, Canada optimum faster by 2.250 s |
| S6 | NOT held | Australia M2-M1 +1.291 s, r=0.76, China M2-M1 +0.410 s, r=0.77, Miami M2-M1 +0.507 s, r=0.73, Canada M2-M1 +0.794 s, r=0.68 |
|  | S7 Australia | optimum 880 m, observed IQR 703-752 m |
|  | S7 China | optimum 800 m, observed IQR 558-662 m |
|  | S7 Miami | optimum 860 m, observed IQR 571-677 m |
|  | S7 Canada | optimum 610 m, observed IQR 521-584 m |
| S7 (exploratory) | NOT held | 0 of 4 inside |
| S1 (corrected rules) | NOT held | Australia 0.70, China 0.38, Miami 0.10, Canada 0.00 |
| S3 (corrected rules) | NOT held | 2 of 4 clip without the taper: Australia 0.74, China 0.00, Miami 0.79, Canada 0.00 |
| S4 (corrected rules) | NOT held | Australia 16 km/h, China 24 km/h, Miami 19 km/h, Canada 24 km/h |
| Starting charge | sensitivity | Australia 0.200 s between 30% and 70%, Canada 0.267 s between 30% and 70%, China 0.122 s between 30% and 70%, Miami 0.055 s between 30% and 70% |

**Per-race 2026 rules** (configs/rules_2026_races.yaml as it stood when these rows were run; see the research log). Same thresholds as above.

| Variant | Circuit | Clip ratio | Clips? | Deploy speed gap | Lap vs original | Convergence |
|---|---|---|---|---|---|---|
| rules | Australia | 0.70 | no | 16 km/h | +0.341 s | 0.016 s |
| rules | China | 0.38 | yes | 24 km/h | +0.040 s | 0.002 s |
| rules | Miami | 0.10 | yes | 19 km/h | +0.000 s | 0.004 s |
| rules | Canada | 0.00 | yes | 24 km/h | +0.060 s | 0.001 s |
| rules_no_taper | Australia | 0.74 | no | 14 km/h | +0.321 s | 0.009 s |
| rules_no_taper | China | 0.00 | yes | 23 km/h | +0.019 s | 0.001 s |
| rules_no_taper | Miami | 0.79 | no | 17 km/h | -0.078 s | 0.000 s |
| rules_no_taper | Canada | 0.00 | yes | 24 km/h | +0.048 s | 0.002 s |
| rules_h8 | Australia | 0.70 | no | 16 km/h | +0.341 s | 0.016 s |
| rules_h8 | China | 0.00 | yes | 25 km/h | +0.237 s | 0.007 s |
| rules_h8 | Miami | 0.00 | yes | 20 km/h | +0.175 s | 0.000 s |
| rules_h8 | Canada | 0.00 | yes | 24 km/h | +0.060 s | 0.001 s |
| rules_deploy250 | Miami | 0.00 | yes | 14 km/h | +0.333 s | 0.002 s |
| rules_deploy250 | Canada | 0.50 | yes | 18 km/h | +0.473 s | 0.002 s |
