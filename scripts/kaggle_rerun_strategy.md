# Re-running the Strategy Lab with the fixed solver (Kaggle, GPU T4 x2), part 2

Accelerator "GPU T4 x2", Internet on. Three cars (original, low-drag, high-drag end), every
stage, then --rules and --refine: about two hours, two cars at a time. Every result is saved as
it finishes; running the same cell again continues where it stopped.

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
run() {
  for stage in "" --rules --refine; do
    CUDA_VISIBLE_DEVICES=$3 python scripts/strategy_run.py --tracks results/tracks --car $1 --out $2 $stage || return 1
  done
}
( run configs/car_2026.yaml results/strategy_rules_q_fixed 0 && run configs/car_2026_q_high.yaml results/strategy_q_high_fixed 0 ) > gpu0.log 2>&1 &
( run configs/car_2026_q_low.yaml results/strategy_q_low_fixed 1 ) > gpu1.log 2>&1 &
wait
tail -3 gpu0.log gpu1.log
zip -qr /kaggle/working/rerun_part2.zip results/strategy_*_fixed gpu0.log gpu1.log
ls -lh /kaggle/working/rerun_part2.zip
```

Download rerun_part2.zip from the Output panel (/kaggle/working, refresh, then download).
