# Las Vegas out of sample (O1) and the energy price (EP1-EP3)

Research log, 2026-11-21. Part A must be finished and pushed BEFORE the race starts
(Sunday 22 November, 04:00 GMT = 09:30 IST). Part B can run any time.

## On the Mac, after Las Vegas qualifying has ended (OpenF1 needs about an hour after the session)
```
cd ~/projects/ers-lens && source .venv/bin/activate && git pull
ERSLENS_SOURCE=openf1 PYTHONPATH=src python scripts/export_tracks.py --events "Las Vegas Grand Prix"
git add "results/tracks/Las Vegas Grand Prix.npz"
git commit -m "Las Vegas track from its qualifying session only (OpenF1), for the out-of-sample solve"
git push
```
If it prints SKIPPED, the data is not up yet: wait 30 minutes and run the same line again.

## Kaggle (GPU T4 x2, Internet on, Persistence "Files only")

Cell 1:
```
%cd /kaggle/working
!rm -rf ers-lens
!git clone -q https://github.com/Priyanshi507/ers-lens.git
%cd ers-lens
!git log --oneline -1
!ls "results/tracks/Las Vegas Grand Prix.npz"
!pip install -q -U "jax[cuda12]==0.11.2" cupy-cuda12x
!nvidia-smi -L
```

Cell 2, Part A, about 5 minutes; paste everything it prints:
```
%%bash
cd /kaggle/working/ers-lens
export ERSLENS_DP_BACKEND=cuda PYTHONPATH=src:scripts XLA_PYTHON_CLIENT_PREALLOCATE=false CUDA_VISIBLE_DEVICES=0
python scripts/fia_fix.py --cars original --caps 350 --out results/oos_lasvegas --events "Las Vegas Grand Prix"
cat results/oos_lasvegas/runs.csv
python - <<'P'
from pathlib import Path
from hindcast_2026 import drop_from_profile, BRAKES
print("Las Vegas drop km/h:", drop_from_profile(Path("results/oos_lasvegas/profiles/Las Vegas Grand Prix_original_350.npz"), BRAKES))
P
```

Cell 3, Part B, about 40 minutes:
```
%%bash
cd /kaggle/working/ers-lens
export ERSLENS_DP_BACKEND=cuda PYTHONPATH=src XLA_PYTHON_CLIENT_PREALLOCATE=false
EV='Australia China Miami Canada Japan Austria Belgium "Monaco Grand Prix" "Barcelona Grand Prix" "British Grand Prix" "Hungarian Grand Prix" "Dutch Grand Prix" "Italian Grand Prix" "Spanish Grand Prix" "Azerbaijan Grand Prix" "Bahrain Grand Prix" "Singapore Grand Prix" "United States Grand Prix"'
( export CUDA_VISIBLE_DEVICES=0
  eval python scripts/fia_fix.py --cars original --caps 350 --recharge-mj 8.5 --out results/energy_price_85 --events $EV
) >> /kaggle/working/gpu0.log 2>&1 &
( export CUDA_VISIBLE_DEVICES=1
  eval python scripts/fia_fix.py --cars original --caps 350 --recharge-mj 8.0 --out results/energy_price_80 --events $EV
) >> /kaggle/working/gpu1.log 2>&1 &
wait
tail -n 1 /kaggle/working/gpu0.log; tail -n 1 /kaggle/working/gpu1.log
PYTHONPATH=src:scripts python scripts/energy_price.py
zip -qr /kaggle/working/price.zip results/oos_lasvegas results/energy_price_85 results/energy_price_80
ls -lh /kaggle/working/price.zip
```

Cell 4:
```
%cd /kaggle/working
from IPython.display import FileLink
FileLink('price.zip')
```
