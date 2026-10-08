| Check | Result | Detail |
|---|---|---|
| DP converged | held | Australia 0.013 s, China 0.003 s, Miami 0.003 s, Canada 0.002 s |
| S1 | held | Australia 0.00, China 0.13, Miami 0.00, Canada 0.00 |
| S2 | held | Australia 93% of 27, China 100% of 27, Miami 100% of 27, Canada 100% of 27 |
| S3 | held | 3 of 4 circuits clip without the taper: Australia 0.51, China 0.29, Miami 0.00, Canada 0.00 |
| S4 | held | Australia 26 km/h lower, China 36 km/h lower, Miami 29 km/h lower, Canada 30 km/h lower |
| S5 | NOT held | Australia optimum faster by 2.026 s, China optimum faster by 1.724 s, Miami optimum faster by 1.824 s, Canada optimum faster by 1.670 s |
| S6 | NOT held | Australia M2-M1 +0.963 s, r=0.68, China M2-M1 +0.455 s, r=0.70, Miami M2-M1 +0.446 s, r=0.68, Canada M2-M1 +0.344 s, r=0.79 |
|  | S7 Australia | optimum 975 m, observed IQR 703-752 m |
|  | S7 China | optimum 925 m, observed IQR 558-662 m |
|  | S7 Miami | optimum 1070 m, observed IQR 571-677 m |
|  | S7 Canada | optimum 715 m, observed IQR 521-584 m |
| S7 (exploratory) | NOT held | 0 of 4 inside |
| S1 (corrected rules) | held | Australia 0.00, China 0.00, Miami 0.00, Canada 0.00 |
| S3 (corrected rules) | NOT held | 2 of 4 clip without the taper |
| S4 (corrected rules) | held | Australia 25 km/h, China 34 km/h, Miami 28 km/h, Canada 29 km/h |
| Starting charge | sensitivity | Australia 0.103 s between 30% and 70%, Canada 0.169 s between 30% and 70%, China 0.049 s between 30% and 70%, Miami 0.034 s between 30% and 70% |

**Corrected 2026 race rules** (configs/rules_2026_races.yaml; logged 2026-10-07, before these runs). Same thresholds as above.

| Variant | Circuit | Clip ratio | Clips? | Deploy speed gap | Lap vs original | Converged |
|---|---|---|---|---|---|---|
| rules | Australia | 0.00 | yes | 25 km/h | +0.196 s | 0.003 s |
| rules | China | 0.00 | yes | 34 km/h | -0.122 s | 0.006 s |
| rules | Miami | 0.00 | yes | 28 km/h | -0.023 s | 0.004 s |
| rules | Canada | 0.00 | yes | 29 km/h | -0.003 s | 0.003 s |
| rules_no_taper | Australia | 0.51 | no | 21 km/h | +0.004 s | 0.015 s |
| rules_no_taper | China | 0.43 | yes | 31 km/h | -0.256 s | 0.005 s |
| rules_no_taper | Miami | 0.55 | no | 21 km/h | -0.819 s | 0.099 s |
| rules_no_taper | Canada | 0.00 | yes | 28 km/h | -0.152 s | 0.001 s |
| rules_h8 | Australia | 0.00 | yes | 26 km/h | +0.186 s | 0.015 s |
| rules_h8 | China | 0.12 | yes | 37 km/h | +0.184 s | 0.004 s |
| rules_h8 | Miami | 0.00 | yes | 31 km/h | +0.095 s | 0.005 s |
| rules_h8 | Canada | 0.00 | yes | 32 km/h | +0.038 s | 0.000 s |
| rules_deploy250 | Miami | 0.00 | yes | 22 km/h | +0.556 s | 0.005 s |
| rules_deploy250 | Canada | 0.25 | yes | 24 km/h | +0.553 s | 0.001 s |
