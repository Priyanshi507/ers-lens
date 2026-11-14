# Lift-and-coast runs (Kaggle, GPU T4 x2)

Predictions LC0-LC4: research log, 2026-11-14. Accelerator "GPU T4 x2", Internet on, Persistence
"Files only". About three to four hours on both GPUs, so start it with at least five hours of GPU
quota left. Keep the tab open. If the session restarts, run only Cell 2 again: every script continues
where it stopped. Never re-run Cell 1 after Cell 2 has started: it deletes the folder.

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
CAL='Australia China Miami Canada Japan Austria Belgium "Monaco Grand Prix" "Barcelona Grand Prix" "British Grand Prix" "Hungarian Grand Prix" "Dutch Grand Prix" "Italian Grand Prix" "Spanish Grand Prix" "Azerbaijan Grand Prix" "Bahrain Grand Prix" "Singapore Grand Prix" "United States Grand Prix"'
(
  export CUDA_VISIBLE_DEVICES=0
  python scripts/fia_fix.py --cars original --caps 350 --out results/lc0_check --events Canada
  python -c "import pandas as pd; a=pd.read_csv('results/lc0_check/runs.csv').iloc[0]; b=pd.read_csv('results/fia_fix_calendar_c/runs.csv'); b=b[(b.event=='Canada')&(b.cap_kw==350)].iloc[0]; print('LC0 lap difference', a.lap_s-b.lap_s)"
  eval python scripts/fia_fix.py --cars original --caps 250 350 --lift-coast --out results/fia_fix_calendar_lc --events $CAL
  python scripts/forecast_2027.py --lift-coast --out results/forecast_2027_lc_a --events Australia China Miami Canada Japan Austria Belgium "Monaco Grand Prix" "Barcelona Grand Prix"
) >> /kaggle/working/gpu0.log 2>&1 &
(
  export CUDA_VISIBLE_DEVICES=1
  python scripts/forecast_2027.py --lift-coast --out results/forecast_2027_lc_b --events "British Grand Prix" "Hungarian Grand Prix" "Dutch Grand Prix" "Italian Grand Prix" "Spanish Grand Prix" "Azerbaijan Grand Prix" "Bahrain Grand Prix" "Singapore Grand Prix" "United States Grand Prix"
) >> /kaggle/working/gpu1.log 2>&1 &
wait
grep "LC0" /kaggle/working/gpu0.log; tail -n 2 /kaggle/working/gpu0.log; tail -n 2 /kaggle/working/gpu1.log
zip -qr /kaggle/working/lift_coast.zip results/lc0_check results/fia_fix_calendar_lc results/forecast_2027_lc_* /kaggle/working/gpu0.log /kaggle/working/gpu1.log
ls -lh /kaggle/working/lift_coast.zip
```

Cell 3, as soon as Cell 2 prints the `ls` line (click the link to download):
```
%cd /kaggle/working
from IPython.display import FileLink
FileLink('lift_coast.zip')
```
