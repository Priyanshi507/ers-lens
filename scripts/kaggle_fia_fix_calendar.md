# FIA-fix experiment on every circuit (Kaggle T4)

GPU T4, Internet on. About 34 solves of about a minute each.

```
%cd /kaggle/working
!rm -rf ers-lens
!git clone -q https://github.com/Priyanshi507/ers-lens.git
%cd ers-lens
!pip install -q -U "jax[cuda12]==0.11.2"
!PYTHONPATH=src python scripts/fia_fix.py --cars original --caps 250 350 --assume-recharge-mj 8.5 --out results/fia_fix_calendar --events Australia China Miami Canada Japan Austria Belgium "Monaco Grand Prix" "Barcelona Grand Prix" "British Grand Prix" "Hungarian Grand Prix" "Dutch Grand Prix" "Italian Grand Prix" "Spanish Grand Prix" "Azerbaijan Grand Prix" "Bahrain Grand Prix" "Singapore Grand Prix" "United States Grand Prix"
!PYTHONPATH=src:scripts python scripts/fia_fix_verdicts.py --calendar --runs results/fia_fix_calendar/runs.csv
!cd /kaggle/working/ers-lens && zip -qr /kaggle/working/fia_fix_calendar.zip results/fia_fix_calendar
```

Download fia_fix_calendar.zip from the Output panel (/kaggle/working, refresh, then download).
