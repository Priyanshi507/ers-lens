# Strategy Lab diagnostics (exploratory, run after the verdicts were committed)

## S7: where the speed peak falls on the longest straight

Medians over clipping race laps. 'Peak to braking' is measured back from the end of full throttle, which the corner fixes for the reference lap and race laps alike.

| Circuit | Full-throttle run, optimum / real | Start to peak, optimum / real | Peak to braking, optimum / real | Real cars clip earlier by |
|---|---|---|---|---|
| Australia | 1265 / 1182 m | 980 / 729 m | 285 / 454 m | 169 m (n=209) |
| China | 1330 / 1149 m | 940 / 608 m | 390 / 537 m | 147 m (n=229) |
| Miami | 1270 / 1202 m | 1085 / 623 m | 185 / 569 m | 384 m (n=464) |
| Canada | 1030 / 974 m | 715 / 546 m | 315 / 428 m | 113 m (n=613) |

## S5: how bang-bang the optimum is

Share of full-throttle points with deployment at the MGU-K cap (>= 98% of it), at zero (<= 2%), or in between. The optimizer could choose any level in 25 kW steps.

| Circuit | Rules | At cap | Zero | Intermediate |
|---|---|---|---|---|
| Australia | original | 66% | 33% | 1% |
| Australia | corrected | 60% | 39% | 1% |
| China | original | 53% | 43% | 4% |
| China | corrected | 53% | 43% | 4% |
| Miami | original | 78% | 20% | 2% |
| Miami | corrected | 78% | 20% | 2% |
| Canada | original | 59% | 37% | 4% |
| Canada | corrected | 54% | 42% | 4% |
