# Air density by circuit (Kaggle, GPU T4 x2)

Predictions D1-D4: research log, 2026-11-18. Accelerator "GPU T4 x2", Internet on, Persistence
"Files only". About two hours. If the session restarts, run only Cell 2 again: it continues where it
stopped. Never re-run Cell 1 after Cell 2 has started.

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
A='Australia China Miami Canada Japan Austria Belgium "Monaco Grand Prix" "Barcelona Grand Prix"'
B='"British Grand Prix" "Hungarian Grand Prix" "Dutch Grand Prix" "Italian Grand Prix" "Spanish Grand Prix" "Azerbaijan Grand Prix" "Bahrain Grand Prix" "Singapore Grand Prix" "United States Grand Prix"'
( export CUDA_VISIBLE_DEVICES=0
  eval python scripts/fia_fix.py --cars original --caps 250 350 --altitude --out results/fia_fix_calendar_alt --events $A
) >> /kaggle/working/gpu0.log 2>&1 &
( export CUDA_VISIBLE_DEVICES=1
  eval python scripts/fia_fix.py --cars original --caps 250 350 --altitude --out results/fia_fix_calendar_alt_b --events $B
  python scripts/fia_fix.py --cars original --caps 350 --altitude --out results/fia_fix_calendar_alt_b --events "Mexico City Grand Prix"
) >> /kaggle/working/gpu1.log 2>&1 &
wait
tail -n 2 /kaggle/working/gpu0.log; tail -n 2 /kaggle/working/gpu1.log
zip -qr /kaggle/working/altitude.zip results/fia_fix_calendar_alt results/fia_fix_calendar_alt_b /kaggle/working/gpu0.log /kaggle/working/gpu1.log
ls -lh /kaggle/working/altitude.zip
```

Cell 3, as soon as Cell 2 prints the `ls` line (click the link to download):
```
%cd /kaggle/working
from IPython.display import FileLink
FileLink('altitude.zip')
```
