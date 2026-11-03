# Re-running the DP experiments with the fixed solver (Kaggle T4), part 1

GPU T4, Internet on. About an hour: the 250 kW sectors (42 solves), the FIA-fix calendar
(36 solves) and the four-circuit FIA-fix runs (40 solves). Each result is saved as it finishes;
re-running the solve cell continues where it stopped.

```
%cd /kaggle/working
!rm -rf ers-lens
!git clone -q https://github.com/Priyanshi507/ers-lens.git
%cd ers-lens
!git log --oneline -1
!pip install -q -U "jax[cuda12]==0.11.2" cupy-cuda12x
```

```
!ERSLENS_DP_BACKEND=cuda PYTHONPATH=src python scripts/sectors_run.py --out results/sectors_fixed
!PYTHONPATH=src:scripts python scripts/sectors_run.py --out results/sectors_fixed --verdicts
!ERSLENS_DP_BACKEND=cuda PYTHONPATH=src python scripts/fia_fix.py --cars original --caps 250 350 --out results/fia_fix_calendar_fixed --events Australia China Miami Canada Japan Austria Belgium "Monaco Grand Prix" "Barcelona Grand Prix" "British Grand Prix" "Hungarian Grand Prix" "Dutch Grand Prix" "Italian Grand Prix" "Spanish Grand Prix" "Azerbaijan Grand Prix" "Bahrain Grand Prix" "Singapore Grand Prix" "United States Grand Prix"
!PYTHONPATH=src:scripts python scripts/fia_fix_verdicts.py --calendar --runs results/fia_fix_calendar_fixed/runs.csv
!ERSLENS_DP_BACKEND=cuda PYTHONPATH=src python scripts/fia_fix.py --out results/fia_fix_fixed
!PYTHONPATH=src:scripts python scripts/fia_fix_verdicts.py --runs results/fia_fix_fixed/runs.csv
!cd /kaggle/working/ers-lens && zip -qr /kaggle/working/rerun_part1.zip results/sectors_fixed results/fia_fix_calendar_fixed results/fia_fix_fixed
```

Download rerun_part1.zip from the Output panel (/kaggle/working, refresh, then download).
