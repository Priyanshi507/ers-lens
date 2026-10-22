# Calibration candidates on the fine DP grid

Fine grid 0.125 m/s, 321 charge levels; same target and acceptance rule as the coarse grid.

| CdA | ICE wheel power | Fine squared run error | Coarse | Worst DP-replay gap |
|---|---|---|---|---|
| 0.95 | 290 kW | 1.64 | 1.72 | 0.013 s |
| 0.95 | 300 kW | 1.68 | 1.67 | 0.092 s (not converged) |
| 1.00 | 300 kW | 1.66 | 1.76 | 0.009 s |
| 1.00 | 310 kW | 1.60 | 1.66 | 0.006 s |
| 1.00 | 330 kW | 2.05 | 1.86 | 0.025 s |
| 1.05 | 320 kW | 1.61 | 1.71 | 0.021 s |
| 1.10 | 340 kW | 1.79 | 1.75 | 0.025 s |
| 1.15 | 360 kW | 2.02 | 1.86 | 0.064 s (not converged) |
| 1.25 | 380 kW | 2.01 | 1.90 | 0.013 s |
| 1.35 | 400 kW | 2.02 | 1.97 | 0.041 s |
| 1.45 | 400 kW | 2.10 | 2.34 | 0.029 s |

## Leave one circuit out, fine grid

| Held-out circuit | CdA | ICE wheel power | Held-out lap error | Within 0.5 s |
|---|---|---|---|---|
| Australia | 0.95 | 300 kW | -0.561 s | no |
| China | 1.00 | 310 kW | -0.642 s | no |
| Miami | 1.00 | 300 kW | +0.552 s | no |
| Canada | 0.95 | 290 kW | +0.717 s | no |

Best on the fine grid: CdA 1.00, 310 kW.
