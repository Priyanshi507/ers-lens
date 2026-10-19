| Check | Result | Detail |
|---|---|---|
| Convergence rule (<= 0.05 s) | NOT held | 128 of 130 M1 results converged, 3 after grid refinement; 2 excluded (China sweep i1d0m0, China rules_no_taper nominal) |
| S1 | held | Australia 0.29, China 0.38, Miami 0.00, Canada 0.00 |
| S2 | NOT held | Australia 18 of 27 (67%), China 21 of 27 (78%), Miami 25 of 27 (93%), Canada 26 of 27 (96%); excluded cars counted as not clipping |
| S3 | held | 3 of 4 circuits clip without the taper: Australia 0.06, China 0.00, Miami 0.72, Canada 0.00 |
| S4 | NOT held | Australia 16 km/h lower, China 24 km/h lower, Miami 20 km/h lower, Canada 23 km/h lower |
| S5 | NOT held | Australia optimum faster by 2.671 s, China optimum faster by 2.170 s, Miami optimum faster by 2.334 s, Canada optimum faster by 2.241 s |
| S6 | NOT held | Australia M2-M1 +1.083 s, r=0.78, China M2-M1 +0.323 s, r=0.80, Miami M2-M1 +0.754 s, r=0.73, Canada M2-M1 +0.552 s, r=0.80 |
|  | S7 Australia | optimum 880 m, observed IQR 703-752 m |
|  | S7 China | optimum 800 m, observed IQR 558-662 m |
|  | S7 Miami | optimum 845 m, observed IQR 571-677 m |
|  | S7 Canada | optimum 610 m, observed IQR 521-584 m |
| S7 (exploratory) | NOT held | 0 of 4 inside |
| S1 (corrected rules) | NOT held | Australia 0.39, China 0.57, Miami 0.00, Canada 0.00 |
| S3 (corrected rules) | held | 3 of 4 clip without the taper: Australia 0.40, China excluded, Miami 0.00, Canada 0.00 |
| S4 (corrected rules) | NOT held | Australia 15 km/h, China 23 km/h, Miami 18 km/h, Canada 21 km/h |

**Corrected 2026 race rules** (configs/rules_2026_races.yaml; logged 2026-10-07, before these runs). Same thresholds as above.

| Variant | Circuit | Clip ratio | Clips? | Deploy speed gap | Lap vs original | Convergence |
|---|---|---|---|---|---|---|
| rules | Australia | 0.39 | yes | 15 km/h | +0.339 s | 0.002 s |
| rules | China | 0.57 | no | 23 km/h | -0.121 s | 0.010 s |
| rules | Miami | 0.00 | yes | 18 km/h | -0.181 s | 0.001 s |
| rules | Canada | 0.00 | yes | 21 km/h | -0.027 s | 0.001 s |
| rules_no_taper | Australia | 0.40 | yes | 12 km/h | +0.290 s | 0.015 s |
| rules_no_taper | China | excluded | - | - | - | not converged |
| rules_no_taper | Miami | 0.00 | yes | 16 km/h | -0.297 s | 0.003 s (refined grid) |
| rules_no_taper | Canada | 0.00 | yes | 21 km/h | -0.041 s | 0.001 s |
| rules_h8 | Australia | 0.43 | yes | 16 km/h | +0.340 s | 0.018 s |
| rules_h8 | China | 0.65 | no | 24 km/h | +0.271 s | 0.007 s |
| rules_h8 | Miami | 0.27 | yes | 17 km/h | -0.383 s | 0.026 s |
| rules_h8 | Canada | 0.00 | yes | 25 km/h | +0.068 s | 0.001 s |
| rules_deploy250 | Miami | 0.00 | yes | 13 km/h | +0.215 s | 0.001 s |
| rules_deploy250 | Canada | 0.50 | yes | 17 km/h | +0.459 s | 0.001 s |
