| Check | Result | Detail |
|---|---|---|
| Convergence rule (<= 0.05 s) | NOT held | 128 of 130 M1 results converged, 8 after grid refinement; 2 excluded (China sweep i0d0m0, China sweep i1d2m0) |
| S1 | held | Australia 0.46, China 0.10, Miami 0.00, Canada 0.00 |
| S2 | held | Australia 25 of 27 (93%), China 24 of 27 (89%), Miami 26 of 27 (96%), Canada 27 of 27 (100%); excluded cars counted as not clipping |
| S3 | held | 4 of 4 circuits clip without the taper: Australia 0.09, China 0.00, Miami 0.00, Canada 0.27 |
| S4 | held | Australia 21 km/h lower, China 32 km/h lower, Miami 27 km/h lower, Canada 29 km/h lower |
| S5 | NOT held | Australia optimum faster by 3.673 s, China optimum faster by 3.224 s, Miami optimum faster by 3.317 s, Canada optimum faster by 2.790 s |
| S6 | NOT held | Australia M2-M1 +1.783 s, r=0.76, China M2-M1 +1.165 s, r=0.64, Miami M2-M1 +4.482 s, r=0.42, Canada M2-M1 +1.604 s, r=0.62 |
|  | S7 Australia | optimum 795 m, observed IQR 703-752 m |
|  | S7 China | optimum 765 m, observed IQR 558-662 m |
|  | S7 Miami | optimum 820 m, observed IQR 571-677 m |
|  | S7 Canada | optimum 500 m, observed IQR 521-584 m |
| S7 (exploratory) | NOT held | 0 of 4 inside |
| S1 (corrected rules) | held | Australia 0.05, China 0.40, Miami 0.10, Canada 0.00 |
| S3 (corrected rules) | held | 4 of 4 clip without the taper: Australia 0.15, China 0.00, Miami 0.00, Canada 0.00 |
| S4 (corrected rules) | held | Australia 20 km/h, China 30 km/h, Miami 25 km/h, Canada 28 km/h |

**Corrected 2026 race rules** (configs/rules_2026_races.yaml; logged 2026-10-07, before these runs). Same thresholds as above.

| Variant | Circuit | Clip ratio | Clips? | Deploy speed gap | Lap vs original | Convergence |
|---|---|---|---|---|---|---|
| rules | Australia | 0.05 | yes | 20 km/h | +0.284 s | 0.020 s |
| rules | China | 0.40 | yes | 30 km/h | -0.034 s | 0.015 s |
| rules | Miami | 0.10 | yes | 25 km/h | -0.079 s | 0.010 s |
| rules | Canada | 0.00 | yes | 28 km/h | +0.003 s | 0.015 s |
| rules_no_taper | Australia | 0.15 | yes | 19 km/h | +0.225 s | 0.033 s |
| rules_no_taper | China | 0.00 | yes | 28 km/h | -0.068 s | 0.016 s |
| rules_no_taper | Miami | 0.00 | yes | 23 km/h | -0.193 s | 0.010 s |
| rules_no_taper | Canada | 0.00 | yes | 28 km/h | -0.003 s | 0.016 s |
| rules_h8 | Australia | 0.38 | yes | 22 km/h | +0.280 s | 0.033 s |
| rules_h8 | China | 0.47 | yes | 33 km/h | +0.268 s | 0.011 s |
| rules_h8 | Miami | 0.00 | yes | 28 km/h | +0.123 s | 0.002 s |
| rules_h8 | Canada | 0.00 | yes | 31 km/h | +0.029 s | 0.010 s |
| rules_deploy250 | Miami | 0.00 | yes | 20 km/h | +0.592 s | 0.013 s |
| rules_deploy250 | Canada | 0.00 | yes | 24 km/h | +0.624 s | 0.009 s |
