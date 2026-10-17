# Car calibration against 2026 pole laps

Error = model minus real time, qualifying fuel, coarse DP grid (0.5 m/s, 81 charge levels). Fit target: sum of squared errors over full-throttle runs of at least 100 m.

## Leave one circuit out

| Held-out circuit | CdA fitted on the other three | ICE wheel power | Held-out lap error | Same circuit, current car (CdA 0.95, 400 kW) |
|---|---|---|---|---|
| Australia | 1.05 | 320 kW | -0.097 s | -2.952 s |
| China | 1.05 | 320 kW | -0.386 s | -3.219 s |
| Miami | 1.05 | 320 kW | +0.316 s | -2.220 s |
| Canada | 1.05 | 320 kW | -0.065 s | -2.315 s |

All four circuits: CdA 1.05, ICE wheel power 320 kW. This is on the edge of the grid, so the grid must be widened before it is used.

## Total squared run error over all circuits (s^2)

| CdA | 320 kW | 340 kW | 360 kW | 380 kW | 400 kW |
|---|---|---|---|---|---|
| 0.95 | 2.23 | 3.14 | 4.35 | 5.79 | 7.27 |
| 1.05 | 1.86 | 2.17 | 2.94 | 4.05 | 5.28 |
| 1.15 | 2.32 | 1.99 | 2.18 | 2.81 | 3.75 |
| 1.25 | 3.76 | 2.57 | 2.15 | 2.25 | 2.71 |
| 1.35 | 6.19 | 4.05 | 2.95 | 2.40 | 2.33 |
| 1.45 | 9.78 | 6.51 | 4.48 | 3.31 | 2.65 |
