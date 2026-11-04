# Strategy Lab diagnostics (exploratory, run after the verdicts were committed)

## S7: where the speed peak falls on the longest straight

Medians over clipping race laps. 'Peak to braking' is measured back from the end of full throttle, which the corner fixes for the reference lap and race laps alike.

| Circuit | Full-throttle run, optimum / real | Start to peak, optimum / real | Peak to braking, optimum / real | Real cars clip earlier by |
|---|---|---|---|---|
| Australia | 1265 / 1182 m | 790 / 729 m | 475 / 454 m | -21 m (n=209) |
| China | 1330 / 1149 m | 760 / 608 m | 570 / 537 m | -33 m (n=229) |
| Miami | 1270 / 1202 m | 830 / 623 m | 440 / 569 m | 129 m (n=464) |
| Canada | 1030 / 974 m | 510 / 546 m | 520 / 428 m | -92 m (n=613) |

## S5: how bang-bang the optimum is

Share of full-throttle points with deployment at the MGU-K cap (>= 98% of it), at zero (<= 2%), or in between. The optimizer could choose any level in 25 kW steps.

| Circuit | Rules | At cap | Zero | Intermediate |
|---|---|---|---|---|
| Australia | original | 53% | 47% | 1% |
| Australia | corrected | 48% | 50% | 1% |
| China | original | 45% | 54% | 1% |
| China | corrected | 46% | 53% | 1% |
| Miami | original | 57% | 42% | 1% |
| Miami | corrected | 57% | 42% | 1% |
| Canada | original | 49% | 48% | 3% |
| Canada | corrected | 46% | 52% | 2% |
