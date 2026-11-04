# Strategy Lab diagnostics (exploratory, run after the verdicts were committed)

## S7: where the speed peak falls on the longest straight

Medians over clipping race laps. 'Peak to braking' is measured back from the end of full throttle, which the corner fixes for the reference lap and race laps alike.

| Circuit | Full-throttle run, optimum / real | Start to peak, optimum / real | Peak to braking, optimum / real | Real cars clip earlier by |
|---|---|---|---|---|
| Australia | 1265 / 1182 m | 880 / 729 m | 385 / 454 m | 69 m (n=209) |
| China | 1330 / 1149 m | 800 / 608 m | 530 / 537 m | 7 m (n=229) |
| Miami | 1270 / 1202 m | 860 / 623 m | 410 / 569 m | 159 m (n=464) |
| Canada | 1030 / 974 m | 610 / 546 m | 420 / 428 m | 8 m (n=613) |

## S5: how bang-bang the optimum is

Share of full-throttle points with deployment at the MGU-K cap (>= 98% of it), at zero (<= 2%), or in between. The optimizer could choose any level in 25 kW steps.

| Circuit | Rules | At cap | Zero | Intermediate |
|---|---|---|---|---|
| Australia | original | 54% | 44% | 2% |
| Australia | corrected | 50% | 49% | 1% |
| China | original | 45% | 50% | 4% |
| China | corrected | 45% | 50% | 5% |
| Miami | original | 56% | 42% | 3% |
| Miami | corrected | 56% | 42% | 3% |
| Canada | original | 51% | 44% | 4% |
| Canada | corrected | 48% | 48% | 4% |
