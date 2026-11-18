# Calibrated cars through the 16-race hindcast (Kaggle, GPU T4 x2)

Predictions S1-S4: research log, 2026-11-18 (night). Accelerator "GPU T4 x2", Internet on,
Persistence "Files only". About 40 minutes. If the session restarts, run only Cell 2 again: it
continues where it stopped. Never re-run Cell 1 after Cell 2 has started.

Cell 1:
```
%cd /kaggle/working
!rm -rf ers-lens
!git clone -q https://github.com/Priyanshi507/ers-lens.git
%cd ers-lens
!git log --oneline -1
!pip install -q -U "jax[cuda12]==0.11.2" cupy-cuda12x
!nvidia-smi -L
```

Cell 2:
```
%%bash
cd /kaggle/working/ers-lens
export ERSLENS_DP_BACKEND=cuda PYTHONPATH=src XLA_PYTHON_CLIENT_PREALLOCATE=false
HC='China Japan Miami Canada "Monaco Grand Prix" "Barcelona Grand Prix" Austria Belgium "Hungarian Grand Prix" "Dutch Grand Prix" "Italian Grand Prix" "Spanish Grand Prix" "Azerbaijan Grand Prix" "Bahrain Grand Prix" "Singapore Grand Prix" "United States Grand Prix"'
( export CUDA_VISIBLE_DEVICES=0
  eval python scripts/fia_fix.py --cars low --caps 250 350 --out results/fia_fix_calendar_low --events $HC
) >> /kaggle/working/gpu0.log 2>&1 &
( export CUDA_VISIBLE_DEVICES=1
  eval python scripts/fia_fix.py --cars high --caps 250 350 --out results/fia_fix_calendar_high --events $HC
) >> /kaggle/working/gpu1.log 2>&1 &
wait
tail -n 2 /kaggle/working/gpu0.log; tail -n 2 /kaggle/working/gpu1.log
for c in low high; do python scripts/hindcast_2026.py --runs results/fia_fix_calendar_$c --car $c | tail -3; done
zip -qr /kaggle/working/cars.zip results/fia_fix_calendar_low results/fia_fix_calendar_high /kaggle/working/gpu0.log /kaggle/working/gpu1.log
ls -lh /kaggle/working/cars.zip
```

Cell 3, as soon as Cell 2 prints the `ls` line (click the link to download):
```
%cd /kaggle/working
from IPython.display import FileLink
FileLink('cars.zip')
```
