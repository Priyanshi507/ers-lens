| Check | Result | Detail |
|---|---|---|
| Convergence rule (<= 0.05 s) | held | 138 of 138 M1 results converged, 0 after grid refinement; 0 excluded |
| S1 | held | Australia 0.00, China 0.00, Miami 0.00, Canada 0.33 |
| S2 | held | Australia 26 of 27 (96%), China 27 of 27 (100%), Miami 27 of 27 (100%), Canada 27 of 27 (100%); excluded cars counted as not clipping |
| S3 | NOT held | 2 of 4 circuits clip without the taper: Australia 0.51, China 0.29, Miami 0.51, Canada 0.00 |
| S4 | held | Australia 26 km/h lower, China 35 km/h lower, Miami 29 km/h lower, Canada 30 km/h lower |
| S5 | NOT held | Australia optimum faster by 2.026 s, China optimum faster by 1.732 s, Miami optimum faster by 1.832 s, Canada optimum faster by 1.672 s |
| S6 | NOT held | Australia M2-M1 +0.993 s, r=0.64, China M2-M1 +0.702 s, r=0.60, Miami M2-M1 +0.454 s, r=0.68, Canada M2-M1 +0.301 s, r=0.80 |
|  | S7 Australia | optimum 980 m, observed IQR 703-752 m |
|  | S7 China | optimum 940 m, observed IQR 558-662 m |
|  | S7 Miami | optimum 1085 m, observed IQR 571-677 m |
|  | S7 Canada | optimum 715 m, observed IQR 521-584 m |
| S7 (exploratory) | NOT held | 0 of 4 inside |
| S1 (corrected rules) | held | Australia 0.41, China 0.13, Miami 0.00, Canada 0.00 |
| S3 (corrected rules) | NOT held | 2 of 4 clip without the taper: Australia 0.51, China 0.00, Miami 0.51, Canada 0.00 |
| S4 (corrected rules) | held | Australia 26 km/h, China 35 km/h, Miami 29 km/h, Canada 32 km/h |
| Starting charge | sensitivity | Australia 0.105 s between 30% and 70%, Canada 0.177 s between 30% and 70%, China 0.069 s between 30% and 70%, Miami 0.026 s between 30% and 70% |

**Per-race 2026 rules** (configs/rules_2026_races.yaml as it stood when these rows were run; see the research log). Same thresholds as above.

| Variant | Circuit | Clip ratio | Clips? | Deploy speed gap | Lap vs original | Convergence |
|---|---|---|---|---|---|---|
| rules | Australia | 0.41 | yes | 26 km/h | +0.194 s | 0.006 s |
| rules | China | 0.13 | yes | 35 km/h | -0.003 s | 0.001 s |
| rules | Miami | 0.00 | yes | 29 km/h | +0.000 s | 0.002 s |
| rules | Canada | 0.00 | yes | 32 km/h | +0.031 s | 0.002 s |
| rules_no_taper | Australia | 0.51 | no | 23 km/h | +0.038 s | 0.006 s |
| rules_no_taper | China | 0.00 | yes | 32 km/h | -0.117 s | 0.001 s |
| rules_no_taper | Miami | 0.51 | no | 27 km/h | -0.325 s | 0.004 s |
| rules_no_taper | Canada | 0.00 | yes | 32 km/h | -0.054 s | 0.004 s |
| rules_h8 | Australia | 0.41 | yes | 26 km/h | +0.194 s | 0.006 s |
| rules_h8 | China | 0.12 | yes | 36 km/h | +0.170 s | 0.006 s |
| rules_h8 | Miami | 0.00 | yes | 31 km/h | +0.076 s | 0.002 s |
| rules_h8 | Canada | 0.00 | yes | 32 km/h | +0.031 s | 0.002 s |
| rules_deploy250 | Miami | 0.00 | yes | 22 km/h | +0.564 s | 0.005 s |
| rules_deploy250 | Canada | 0.25 | yes | 24 km/h | +0.555 s | 0.001 s |
