# Every DP experiment with the corrected harvest limit (Kaggle, GPU T4 x2)

Accelerator "GPU T4 x2", Internet on. About two and a half hours, both GPUs at once. Every result
is saved as it finishes; running Cell 2 again continues where it stopped (never re-run Cell 1
while Cell 2 is running: it deletes the folder).

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
  eval python scripts/fia_fix.py --cars original --caps 250 350 --out results/fia_fix_calendar_h --events $CAL
  python scripts/sectors_run.py --out results/sectors_h
  strat configs/car_2026.yaml results/strategy_rules_q_h
  python scripts/race_plan.py tables --out results/race_h --events Australia China
) > gpu0.log 2>&1 &
(
  export CUDA_VISIBLE_DEVICES=1
  python scripts/fia_fix.py --out results/fia_fix_h
  strat configs/car_2026_q_low.yaml results/strategy_q_low_h
  strat configs/car_2026_q_high.yaml results/strategy_q_high_h
  python scripts/race_plan.py tables --out results/race_h --events Miami Canada
) > gpu1.log 2>&1 &
wait
tail -n 3 gpu0.log; tail -n 3 gpu1.log
zip -qr /kaggle/working/rerun_harvest.zip results/*_h gpu0.log gpu1.log
ls -lh /kaggle/working/rerun_harvest.zip
```

Download rerun_harvest.zip from the Output panel (/kaggle/working, refresh, then download).
