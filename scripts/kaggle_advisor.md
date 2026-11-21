# Limit advisor: LA1-LA3 (research log, 2026-11-21)

Kaggle, GPU T4 x2, Internet on, Persistence "Files only". About 2 hours (180 qualifying-lap solves).
If the session restarts, run only Cell 2 again: it continues where it stopped.

Cell 1 (new notebook; in an open one, skip the clone and pip lines and run `!git pull -q` instead):
```
%cd /kaggle/working
!rm -rf ers-lens
!git clone -q https://github.com/Priyanshi507/ers-lens.git
%cd ers-lens
!git log --oneline -1
!pip install -q -U "jax[cuda12]==0.11.2" cupy-cuda12x
```

Cell 2:
```
%%bash
cd /kaggle/working/ers-lens
export ERSLENS_DP_BACKEND=cuda PYTHONPATH=src XLA_PYTHON_CLIENT_PREALLOCATE=false
EV='Australia China Miami Canada Japan Austria Belgium "Monaco Grand Prix" "Barcelona Grand Prix" "British Grand Prix" "Hungarian Grand Prix" "Dutch Grand Prix" "Italian Grand Prix" "Spanish Grand Prix" "Azerbaijan Grand Prix" "Bahrain Grand Prix" "Singapore Grand Prix" "United States Grand Prix" "Mexico City Grand Prix" "Las Vegas Grand Prix"'
( export CUDA_VISIBLE_DEVICES=0
  for L in 50 55 60 65 70; do
    eval python scripts/fia_fix.py --push-lap --cars original --caps 350 --recharge-mj ${L:0:1}.${L:1:1} --out results/advisor/q_$L --events $EV
  done
) >> /kaggle/working/adv0.log 2>&1 &
( export CUDA_VISIBLE_DEVICES=1
  for L in 75 80 85 90; do
    eval python scripts/fia_fix.py --push-lap --cars original --caps 350 --recharge-mj ${L:0:1}.${L:1:1} --out results/advisor/q_$L --events $EV
  done
) >> /kaggle/working/adv1.log 2>&1 &
wait
tail -n 1 /kaggle/working/adv0.log; tail -n 1 /kaggle/working/adv1.log
PYTHONPATH=src:scripts python scripts/limit_advisor.py
zip -qr /kaggle/working/advisor.zip results/advisor /kaggle/working/adv0.log /kaggle/working/adv1.log
ls -lh /kaggle/working/advisor.zip
```

Cell 3:
```
%cd /kaggle/working
from IPython.display import FileLink
FileLink('advisor.zip')
```
