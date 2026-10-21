# Car calibration against 2026 pole laps

Error = model minus real time, qualifying fuel, coarse DP grid (0.5 m/s, 81 charge levels). Fit target: sum of squared errors over full-throttle runs of at least 100 m.

## Leave one circuit out

| Held-out circuit | CdA fitted on the other three | ICE wheel power | Held-out lap error | Same circuit, current car (CdA 0.95, 400 kW) |
|---|---|---|---|---|
| Australia | 0.95 | 300 kW | -0.199 s | -2.848 s |
| China | 1.00 | 330 kW | -1.071 s | -3.219 s |
| Miami | 1.00 | 300 kW | +0.820 s | -2.110 s |
| Canada | 0.95 | 290 kW | +0.918 s | -1.668 s |

All four circuits: CdA 1.00, ICE wheel power 310 kW.

## Total squared run error over all circuits (s^2)

| CdA | 280 kW | 290 kW | 300 kW | 310 kW | 320 kW | 330 kW | 340 kW | 360 kW | 380 kW | 400 kW |
|---|---|---|---|---|---|---|---|---|---|---|
| 0.95 | 2.04 | 1.72 | 1.67 | 1.78 | 1.97 | 2.30 | 2.77 | 3.73 | 5.21 | 6.57 |
| 1.00 | 2.58 | 2.05 | 1.76 | 1.66 | 1.67 | 1.86 | 2.14 |  |  |  |
| 1.05 | 3.25 | 2.74 | 2.18 | 2.07 | 1.71 | 1.72 | 1.82 | 2.42 | 3.81 | 4.68 |
| 1.10 | 4.66 | 3.46 | 2.84 | 2.43 | 1.97 | 1.78 | 1.75 |  |  |  |
| 1.15 | 6.26 | 4.65 | 3.76 | 3.07 | 2.49 | 2.11 | 1.86 | 1.86 | 2.34 | 3.12 |
| 1.25 |  |  |  |  | 4.20 |  | 2.77 | 2.08 | 1.90 | 2.31 |
| 1.35 |  |  |  |  | 6.96 |  | 4.62 | 3.15 | 2.02 | 1.97 |
| 1.45 |  |  |  |  | 10.70 |  | 7.32 | 5.10 | 2.96 | 2.34 |

## Best ICE wheel power for each CdA

| CdA | ICE wheel power | Squared run error |
|---|---|---|
| 0.95 | 300 kW | 1.67 |
| 1.00 | 310 kW | 1.66 |
| 1.05 | 320 kW | 1.71 |
| 1.10 | 340 kW | 1.75 |
| 1.15 | 360 kW | 1.86 |
| 1.25 | 380 kW | 1.90 |
| 1.35 | 400 kW | 1.97 |
| 1.45 | 400 kW | 2.34 |
