| Check | Result | Detail |
|---|---|---|
| Convergence rule (<= 0.05 s) | NOT held | 129 of 130 M1 results converged, 5 after grid refinement; 1 excluded (China sweep i2d0m0) |
| S1 | held | Australia 0.39, China 0.49, Miami 0.39, Canada 0.00 |
| S2 | held | Australia 27 of 27 (100%), China 25 of 27 (93%), Miami 22 of 27 (81%), Canada 27 of 27 (100%); excluded cars counted as not clipping |
| S3 | held | 4 of 4 circuits clip without the taper: Australia 0.00, China 0.00, Miami 0.00, Canada 0.00 |
| S4 | held | Australia 21 km/h lower, China 31 km/h lower, Miami 26 km/h lower, Canada 28 km/h lower |
| S5 | NOT held | Australia optimum faster by 3.668 s, China optimum faster by 3.167 s, Miami optimum faster by 3.281 s, Canada optimum faster by 2.822 s |
| S6 | NOT held | Australia M2-M1 +1.861 s, r=0.72, China M2-M1 +1.262 s, r=0.68, Miami M2-M1 +3.480 s, r=0.45, Canada M2-M1 +1.655 s, r=0.62 |
|  | S7 Australia | optimum 785 m, observed IQR 703-752 m |
|  | S7 China | optimum 780 m, observed IQR 558-662 m |
|  | S7 Miami | optimum 820 m, observed IQR 571-677 m |
|  | S7 Canada | optimum 510 m, observed IQR 521-584 m |
| S7 (exploratory) | NOT held | 0 of 4 inside |
| S1 (corrected rules) | held | Australia 0.34, China 0.48, Miami 0.39, Canada 0.00 |
| S3 (corrected rules) | held | 4 of 4 clip without the taper: Australia 0.44, China 0.50, Miami 0.00, Canada 0.00 |
| S4 (corrected rules) | held | Australia 21 km/h, China 30 km/h, Miami 26 km/h, Canada 30 km/h |

**Corrected 2026 race rules** (configs/rules_2026_races.yaml; logged 2026-10-07, before these runs). Same thresholds as above.

| Variant | Circuit | Clip ratio | Clips? | Deploy speed gap | Lap vs original | Convergence |
|---|---|---|---|---|---|---|
| rules | Australia | 0.34 | yes | 21 km/h | +0.309 s | 0.004 s |
| rules | China | 0.48 | yes | 30 km/h | +0.121 s | 0.016 s |
| rules | Miami | 0.39 | yes | 26 km/h | +0.000 s | 0.006 s |
| rules | Canada | 0.00 | yes | 30 km/h | +0.049 s | 0.009 s |
| rules_no_taper | Australia | 0.44 | yes | 19 km/h | +0.279 s | 0.016 s |
| rules_no_taper | China | 0.50 | yes | 28 km/h | +0.064 s | 0.018 s |
| rules_no_taper | Miami | 0.00 | yes | 24 km/h | -0.069 s | 0.008 s |
| rules_no_taper | Canada | 0.00 | yes | 30 km/h | +0.035 s | 0.011 s |
| rules_h8 | Australia | 0.34 | yes | 21 km/h | +0.309 s | 0.004 s |
| rules_h8 | China | 0.00 | yes | 32 km/h | +0.270 s | 0.014 s |
| rules_h8 | Miami | 0.00 | yes | 27 km/h | +0.144 s | 0.002 s |
| rules_h8 | Canada | 0.00 | yes | 30 km/h | +0.049 s | 0.009 s |
| rules_deploy250 | Miami | 0.23 | yes | 20 km/h | +0.539 s | 0.026 s |
| rules_deploy250 | Canada | 0.37 | yes | 23 km/h | +0.603 s | 0.001 s |
