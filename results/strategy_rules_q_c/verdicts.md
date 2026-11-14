| Check | Result | Detail |
|---|---|---|
| Convergence rule (<= 0.05 s) | held | 138 of 138 M1 results converged, 0 after grid refinement; 0 excluded |
| S1 | held | Australia 0.29, China 0.00, Miami 0.00, Canada 0.34 |
| S2 | held | Australia 26 of 27 (96%), China 27 of 27 (100%), Miami 27 of 27 (100%), Canada 27 of 27 (100%); excluded cars counted as not clipping |
| S3 | NOT held | 2 of 4 circuits clip without the taper: Australia 0.51, China 0.00, Miami 0.51, Canada 0.00 |
| S4 | held | Australia 25 km/h lower, China 35 km/h lower, Miami 29 km/h lower, Canada 30 km/h lower |
| S5 | NOT held | Australia optimum faster by 2.214 s, China optimum faster by 1.772 s, Miami optimum faster by 1.774 s, Canada optimum faster by 1.784 s |
| S6 | NOT held | Australia M2-M1 +0.571 s, r=0.79, China M2-M1 +0.072 s, r=0.82, Miami M2-M1 +1.010 s, r=0.72, Canada M2-M1 +1.150 s, r=0.73 |
|  | S7 Australia | optimum 1020 m, observed IQR 703-752 m |
|  | S7 China | optimum 940 m, observed IQR 558-662 m |
|  | S7 Miami | optimum 1090 m, observed IQR 571-677 m |
|  | S7 Canada | optimum 765 m, observed IQR 521-584 m |
| S7 (exploratory) | NOT held | 0 of 4 inside |
| S1 (corrected rules) | held | Australia 0.00, China 0.26, Miami 0.00, Canada 0.00 |
| S3 (corrected rules) | NOT held | 2 of 4 clip without the taper: Australia 0.51, China 0.00, Miami 0.51, Canada 0.00 |
| S4 (corrected rules) | held | Australia 26 km/h, China 35 km/h, Miami 29 km/h, Canada 32 km/h |
| Starting charge | sensitivity | Australia 0.130 s between 30% and 70%, Canada 0.195 s between 30% and 70%, China 0.035 s between 30% and 70%, Miami 0.006 s between 30% and 70% |

**Per-race 2026 rules** (configs/rules_2026_races.yaml as it stood when these rows were run; see the research log). Same thresholds as above.

| Variant | Circuit | Clip ratio | Clips? | Deploy speed gap | Lap vs original | Convergence |
|---|---|---|---|---|---|---|
| rules | Australia | 0.00 | yes | 26 km/h | +0.245 s | 0.008 s |
| rules | China | 0.26 | yes | 35 km/h | +0.012 s | 0.005 s |
| rules | Miami | 0.00 | yes | 29 km/h | +0.000 s | 0.001 s |
| rules | Canada | 0.00 | yes | 32 km/h | +0.047 s | 0.002 s |
| rules_no_taper | Australia | 0.51 | no | 23 km/h | +0.106 s | 0.006 s |
| rules_no_taper | China | 0.00 | yes | 32 km/h | -0.111 s | 0.006 s |
| rules_no_taper | Miami | 0.51 | no | 27 km/h | -0.308 s | 0.003 s |
| rules_no_taper | Canada | 0.00 | yes | 32 km/h | -0.009 s | 0.004 s |
| rules_h8 | Australia | 0.00 | yes | 26 km/h | +0.245 s | 0.008 s |
| rules_h8 | China | 0.00 | yes | 36 km/h | +0.171 s | 0.001 s |
| rules_h8 | Miami | 0.00 | yes | 31 km/h | +0.091 s | 0.001 s |
| rules_h8 | Canada | 0.00 | yes | 32 km/h | +0.047 s | 0.002 s |
| rules_deploy250 | Miami | 0.00 | yes | 22 km/h | +0.571 s | 0.001 s |
| rules_deploy250 | Canada | 0.00 | yes | 22 km/h | +0.565 s | 0.001 s |
