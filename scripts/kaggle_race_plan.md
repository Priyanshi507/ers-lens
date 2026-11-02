# Whole-race energy plans on Kaggle (free T4)

GPU T4, Internet on. About 1 to 1.5 hours; each circuit's table is saved as soon as it is done,
and a re-run keeps the tables already made.

```
%cd /kaggle/working
!rm -rf ers-lens
!git clone -q https://github.com/Priyanshi507/ers-lens.git
%cd ers-lens
!git log --oneline -1
!pip install -q -U "jax[cuda12]==0.11.2" cupy-cuda12x
!ERSLENS_DP_BACKEND=cuda PYTHONPATH=src python scripts/race_plan.py tables --out results/race
!cd results && zip -qr /kaggle/working/race_tables.zip race
```

Download race_tables.zip from the Output panel. On the Mac, in ~/projects/ers-lens:

```
unzip -o ~/Downloads/race_tables.zip -d results
PYTHONPATH=src python scripts/race_plan.py plans --out results/race
```
