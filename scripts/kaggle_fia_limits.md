# Predicting the FIA's next limits: FL1 Qatar, FL2 Abu Dhabi (research log, 2026-11-21)

Must finish BEFORE the FIA publishes either "Power Unit Information" document (Qatar's is expected
around 25 November).

## On the Mac
```
cd ~/projects/ers-lens && source .venv/bin/activate
ERSLENS_SOURCE=openf1 PYTHONPATH=src python scripts/export_tracks.py --year 2025 --out results/tracks_2025 --events "Qatar Grand Prix" "Abu Dhabi Grand Prix"
git add results/tracks_2025
git commit -m "Qatar and Abu Dhabi tracks from 2025 qualifying, for the FIA-limit forward test"
git push
```

## Kaggle (GPU T4 x2, Internet on)

Cell 1:
```
%cd /kaggle/working
!rm -rf ers-lens
!git clone -q https://github.com/Priyanshi507/ers-lens.git
%cd ers-lens
!git log --oneline -1
!ls results/tracks_2025
!pip install -q -U "jax[cuda12]==0.11.2" cupy-cuda12x
```

Cell 2, about 5 minutes; paste everything it prints:
```
%%bash
cd /kaggle/working/ers-lens
export ERSLENS_DP_BACKEND=cuda PYTHONPATH=src XLA_PYTHON_CLIENT_PREALLOCATE=false CUDA_VISIBLE_DEVICES=0
python scripts/fia_fix.py --cars original --caps 350 --recharge-mj 8.5 --assume-recharge-mj 8.5 --tracks results/tracks_2025 --out results/fia_limits_forward --events "Qatar Grand Prix" "Abu Dhabi Grand Prix"
cat results/fia_limits_forward/runs.csv
```
