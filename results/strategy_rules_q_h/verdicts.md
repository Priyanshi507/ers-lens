| Check | Result | Detail |
|---|---|---|
| Convergence rule (<= 0.05 s) | held | 138 of 138 M1 results converged, 0 after grid refinement; 0 excluded |
| S1 | held | Australia 0.00, China 0.13, Miami 0.00, Canada 0.00 |
| S2 | held | Australia 26 of 27 (96%), China 25 of 27 (93%), Miami 27 of 27 (100%), Canada 27 of 27 (100%); excluded cars counted as not clipping |
| S3 | NOT held | 2 of 4 circuits clip without the taper: Australia 0.51, China 0.29, Miami 0.51, Canada 0.00 |
| S4 | held | Australia 25 km/h lower, China 35 km/h lower, Miami 29 km/h lower, Canada 30 km/h lower |
| S5 | NOT held | Australia optimum faster by 2.209 s, China optimum faster by 1.765 s, Miami optimum faster by 1.771 s, Canada optimum faster by 1.782 s |
| S6 | NOT held | Australia M2-M1 +0.566 s, r=0.79, China M2-M1 +0.065 s, r=0.82, Miami M2-M1 +1.007 s, r=0.71, Canada M2-M1 +1.147 s, r=0.73 |
|  | S7 Australia | optimum 1020 m, observed IQR 703-752 m |
|  | S7 China | optimum 945 m, observed IQR 558-662 m |
|  | S7 Miami | optimum 1085 m, observed IQR 571-677 m |
|  | S7 Canada | optimum 765 m, observed IQR 521-584 m |
| S7 (exploratory) | NOT held | 0 of 4 inside |
| S1 (corrected rules) | held | Australia 0.14, China 0.13, Miami 0.00, Canada 0.00 |
| S3 (corrected rules) | NOT held | 2 of 4 clip without the taper: Australia 0.51, China 0.00, Miami 0.51, Canada 0.00 |
| S4 (corrected rules) | held | Australia 26 km/h, China 35 km/h, Miami 29 km/h, Canada 32 km/h |
| Starting charge | sensitivity | Australia 0.130 s between 30% and 70%, Canada 0.194 s between 30% and 70%, China 0.046 s between 30% and 70%, Miami 0.005 s between 30% and 70% |

**Per-race 2026 rules** (configs/rules_2026_races.yaml as it stood when these rows were run; see the research log). Same thresholds as above.

| Variant | Circuit | Clip ratio | Clips? | Deploy speed gap | Lap vs original | Convergence |
|---|---|---|---|---|---|---|
| rules | Australia | 0.14 | yes | 26 km/h | +0.234 s | 0.014 s |
| rules | China | 0.13 | yes | 35 km/h | +0.019 s | 0.000 s |
| rules | Miami | 0.00 | yes | 29 km/h | +0.000 s | 0.002 s |
| rules | Canada | 0.00 | yes | 32 km/h | +0.045 s | 0.002 s |
| rules_no_taper | Australia | 0.51 | no | 23 km/h | +0.101 s | 0.006 s |
| rules_no_taper | China | 0.00 | yes | 32 km/h | -0.109 s | 0.001 s |
| rules_no_taper | Miami | 0.51 | no | 27 km/h | -0.300 s | 0.004 s |
| rules_no_taper | Canada | 0.00 | yes | 32 km/h | -0.004 s | 0.004 s |
| rules_h8 | Australia | 0.14 | yes | 26 km/h | +0.234 s | 0.014 s |
| rules_h8 | China | 0.00 | yes | 37 km/h | +0.199 s | 0.007 s |
| rules_h8 | Miami | 0.00 | yes | 31 km/h | +0.094 s | 0.001 s |
| rules_h8 | Canada | 0.00 | yes | 32 km/h | +0.045 s | 0.002 s |
| rules_deploy250 | Miami | 0.00 | yes | 22 km/h | +0.568 s | 0.001 s |
| rules_deploy250 | Canada | 0.00 | yes | 23 km/h | +0.562 s | 0.001 s |
