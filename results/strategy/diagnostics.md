# Strategy Lab diagnostics (exploratory, run after the verdicts were committed)

## S7: where the speed peak falls on the longest straight

Medians over clipping race laps. 'Peak to braking' is measured back from the end of full throttle, which the corner fixes for the reference lap and race laps alike.

| Circuit | Full-throttle run, optimum / real | Start to peak, optimum / real | Peak to braking, optimum / real | Real cars clip earlier by |
|---|---|---|---|---|
| Australia | 1265 / 1182 m | 975 / 729 m | 290 / 454 m | 164 m (n=209) |
| China | 1330 / 1149 m | 925 / 608 m | 405 / 537 m | 132 m (n=229) |
| Miami | 1270 / 1202 m | 1070 / 623 m | 200 / 569 m | 369 m (n=464) |
| Canada | 1030 / 974 m | 715 / 546 m | 315 / 428 m | 113 m (n=613) |

## S5: how bang-bang the optimum is

Share of full-throttle points with deployment at the MGU-K cap (>= 98% of it), at zero (<= 2%), or in between. The optimizer could choose any level in 25 kW steps.

| Circuit | Rules | At cap | Zero | Intermediate |
|---|---|---|---|---|
| Australia | original | 65% | 34% | 1% |
| Australia | corrected | 61% | 38% | 1% |
| China | original | 52% | 43% | 4% |
| China | corrected | 57% | 38% | 4% |
| Miami | original | 76% | 22% | 2% |
| Miami | corrected | 82% | 15% | 2% |
| Canada | original | 58% | 38% | 4% |
| Canada | corrected | 60% | 36% | 4% |

## S5: where the optimum gains its time (corrected rules)

Optimum replayed against the best tuned simple rule on the same lap; positive = optimum faster.

| Circuit | Best simple rule | Where | Optimum gains |
|---|---|---|---|
| Australia | clip_end deploy_w=300 kW, harvest_w=150 kW, clip_frac=0.5, clip_harvest_w=150 kW | total | +2.233 s |
| Australia |  | all corners | +0.402 s |
| Australia |  | all straights | +1.831 s |
| Australia |  | straight 1265 m at 2510 m | +0.625 s |
| Australia |  | straight 640 m at 170 m | +0.541 s |
| Australia |  | straight 540 m at 4025 m | +0.166 s |
| Australia |  | straight 505 m at 1855 m | +0.205 s |
| China | clip_end deploy_w=300 kW, harvest_w=75 kW, clip_frac=0.4, clip_harvest_w=350 kW | total | +1.791 s |
| China |  | all corners | +0.141 s |
| China |  | all straights | +1.649 s |
| China |  | straight 1330 m at 3940 m | +0.738 s |
| China |  | straight 710 m at 400 m | +0.247 s |
| China |  | straight 615 m at 2270 m | +0.249 s |
| China |  | straight 450 m at 1625 m | +0.214 s |
| Miami | clip_end deploy_w=350 kW, harvest_w=150 kW, clip_frac=0.3, clip_harvest_w=350 kW | total | +1.829 s |
| Miami |  | all corners | +0.453 s |
| Miami |  | all straights | +1.376 s |
| Miami |  | straight 1270 m at 2190 m | +0.399 s |
| Miami |  | straight 1115 m at 4070 m | +0.360 s |
| Miami |  | straight 660 m at 90 m | +0.312 s |
| Miami |  | straight 475 m at 1085 m | +0.286 s |
| Canada | clip_end deploy_w=300 kW, harvest_w=75 kW, clip_frac=0.4, clip_harvest_w=350 kW | total | +1.674 s |
| Canada |  | all corners | +0.142 s |
| Canada |  | all straights | +1.532 s |
| Canada |  | straight 1030 m at 65 m | +0.440 s |
| Canada |  | straight 530 m at 1270 m | +0.377 s |
| Canada |  | straight 530 m at 3005 m | +0.305 s |
| Canada |  | straight 480 m at 3725 m | +0.240 s |
