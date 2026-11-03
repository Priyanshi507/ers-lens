# FIA-fix re-run of the three circuits whose assumed recharge limit was wrong (Kaggle T4)

GPU T4, Internet on. Six solves of about a minute each. Same settings as the calendar run
(XLA backend, original car), with each circuit's FIA limit read from configs/rules_2026_races.yaml.

```
%cd /kaggle/working
!rm -rf ers-lens
!git clone -q https://github.com/Priyanshi507/ers-lens.git
%cd ers-lens
!git log --oneline -1
!pip install -q -U "jax[cuda12]==0.11.2"
!PYTHONPATH=src python scripts/fia_fix.py --cars original --caps 250 350 --out results/fia_fix_corrected --events Austria "British Grand Prix" "Italian Grand Prix"
!cd /kaggle/working/ers-lens && zip -qr /kaggle/working/fia_fix_corrected.zip results/fia_fix_corrected
```

Download fia_fix_corrected.zip from the Output panel (/kaggle/working, refresh, then download).
