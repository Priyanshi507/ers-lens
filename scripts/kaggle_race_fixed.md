# Whole-race lap tables with the fixed solver (Kaggle, GPU T4 x2)

Accelerator "GPU T4 x2", Internet on. Four circuits, two per GPU: about 45 minutes. Each table is
saved as soon as it is done; running the cell again keeps the tables already made.

Cell 1:
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
CUDA_VISIBLE_DEVICES=0 python scripts/race_plan.py tables --out results/race_fixed --events Australia China > gpu0.log 2>&1 &
CUDA_VISIBLE_DEVICES=1 python scripts/race_plan.py tables --out results/race_fixed --events Miami Canada > gpu1.log 2>&1 &
wait
cat gpu0.log gpu1.log
zip -qr /kaggle/working/race_fixed.zip results/race_fixed gpu0.log gpu1.log
```

Download race_fixed.zip from the Output panel (/kaggle/working, refresh, then download).
