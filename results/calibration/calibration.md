# Car calibration against 2026 pole laps

Error = model minus real time, qualifying fuel, coarse DP grid (0.5 m/s, 81 charge levels). Fit target: sum of squared errors over full-throttle runs of at least 100 m.

## Leave one circuit out

| Held-out circuit | CdA fitted on the other three | ICE wheel power | Held-out lap error | Same circuit, current car (CdA 0.95, 400 kW) |
|---|---|---|---|---|
| Australia | 0.95 | 300 kW | -0.154 s | -2.952 s |
| China | 0.95 | 310 kW | -0.729 s | -3.219 s |
| Miami | 0.95 | 290 kW | +0.609 s | -2.220 s |
| Canada | 0.95 | 290 kW | +0.339 s | -2.315 s |

All four circuits: CdA 0.95, ICE wheel power 300 kW. This is on the edge of the grid, so the grid must be widened before it is used.

## Total squared run error over all circuits (s^2)

| CdA | 280 kW | 290 kW | 300 kW | 310 kW | 320 kW | 330 kW | 340 kW | 360 kW | 380 kW | 400 kW |
|---|---|---|---|---|---|---|---|---|---|---|
| 0.95 | 2.03 | 1.81 | 1.79 | 1.94 | 2.23 | 2.64 | 3.14 | 4.35 | 5.79 | 7.27 |
| 1.00 | 2.51 | 2.08 | 1.85 | 1.82 | 1.93 | 2.21 | 2.57 |  |  |  |
| 1.05 | 3.31 | 2.63 | 2.17 | 1.93 | 1.86 | 1.96 | 2.17 | 2.94 | 4.05 | 5.28 |
| 1.10 | 4.39 | 3.39 | 2.71 | 2.20 | 1.98 | 1.90 | 1.98 |  |  |  |
| 1.15 | 5.78 | 4.45 | 3.48 | 2.83 | 2.32 | 2.08 | 1.99 | 2.18 | 2.81 | 3.75 |
| 1.25 |  |  |  |  | 3.76 |  | 2.57 | 2.15 | 2.25 | 2.71 |
| 1.35 |  |  |  |  | 6.19 |  | 4.05 | 2.95 | 2.40 | 2.33 |
| 1.45 |  |  |  |  | 9.78 |  | 6.51 | 4.48 | 3.31 | 2.65 |

## Best ICE wheel power for each CdA

| CdA | ICE wheel power | Squared run error |
|---|---|---|
| 0.95 | 300 kW | 1.79 |
| 1.00 | 310 kW | 1.82 |
| 1.05 | 320 kW | 1.86 |
| 1.10 | 330 kW | 1.90 |
| 1.15 | 340 kW | 1.99 |
| 1.25 | 360 kW | 2.15 |
| 1.35 | 400 kW | 2.33 |
| 1.45 | 400 kW | 2.65 |
