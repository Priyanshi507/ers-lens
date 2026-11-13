# Refined lap tables and the harvest-price check (Kaggle, GPU T4 x2)

Predictions R1-R3: research log, 2026-11-13. Accelerator "GPU T4 x2", Internet on, and in Session
options set Persistence to "Files only", so the results survive if the session restarts. Run the
cells in order and keep the tab open. If the session restarts, run only Cell 2 again: it continues
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
mkdir -p results/price_resolution
(
  export CUDA_VISIBLE_DEVICES=0
  python scripts/price_resolution.py --event "Bahrain Grand Prix" --caps 250 350 --bisect 4 10 > results/price_resolution/bahrain.csv
  python scripts/race_plan.py tables --out results/race_h2 --events Australia China
) > /kaggle/working/gpu0.log 2>&1 &
(
  export CUDA_VISIBLE_DEVICES=1
  python scripts/race_plan.py tables --out results/race_h2 --events Miami Canada
) > /kaggle/working/gpu1.log 2>&1 &
wait
cat results/price_resolution/bahrain.csv
tail -n 2 /kaggle/working/gpu0.log; tail -n 2 /kaggle/working/gpu1.log
zip -qr /kaggle/working/race_refined.zip results/race_h2 results/price_resolution /kaggle/working/gpu0.log /kaggle/working/gpu1.log
ls -lh /kaggle/working/race_refined.zip
```

Cell 3, as soon as Cell 2 prints the `ls` line (click the link to download):
```
%cd /kaggle/working
from IPython.display import FileLink
FileLink('race_refined.zip')
```
