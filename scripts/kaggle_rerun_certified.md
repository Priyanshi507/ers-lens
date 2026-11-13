# Every DP experiment with ten bisection steps and the optimality certificate (Kaggle, GPU T4 x2)

Predictions K1-K3: research log, 2026-11-13. Accelerator "GPU T4 x2", Internet on, Persistence
"Files only". About four to five hours on both GPUs, so start it with at least six hours of GPU
quota left. Keep the tab open. If the session restarts, run only Cell 2 again: every script
continues where it stopped. Never re-run Cell 1 after Cell 2 has started: it deletes the folder.

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
strat() { for stage in "" --rules --refine; do python scripts/strategy_run.py --tracks results/tracks --car $1 --out $2 $stage || return 1; done; }
(
  export CUDA_VISIBLE_DEVICES=0
  eval python scripts/fia_fix.py --cars original --caps 250 350 --out results/fia_fix_calendar_c --events $CAL
  python scripts/sectors_run.py --out results/sectors_c
  strat configs/car_2026.yaml results/strategy_rules_q_c
) >> /kaggle/working/gpu0.log 2>&1 &
(
  export CUDA_VISIBLE_DEVICES=1
  python scripts/fia_fix.py --out results/fia_fix_c
  strat configs/car_2026_q_low.yaml results/strategy_q_low_c
  strat configs/car_2026_q_high.yaml results/strategy_q_high_c
) >> /kaggle/working/gpu1.log 2>&1 &
wait
tail -n 3 /kaggle/working/gpu0.log; tail -n 3 /kaggle/working/gpu1.log
zip -qr /kaggle/working/rerun_certified.zip results/*_c /kaggle/working/gpu0.log /kaggle/working/gpu1.log
ls -lh /kaggle/working/rerun_certified.zip
```

Cell 3, as soon as Cell 2 prints the `ls` line (click the link to download):
```
%cd /kaggle/working
from IPython.display import FileLink
FileLink('rerun_certified.zip')
```
