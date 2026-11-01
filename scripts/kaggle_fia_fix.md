# Running the FIA-fix experiment on Kaggle (free T4)

New notebook or the existing one: Accelerator GPU T4, Internet On. Then:

```
%cd /kaggle/working
!rm -rf ers-lens
!git clone -q https://github.com/Priyanshi507/ers-lens.git
%cd ers-lens
!pip install -q -U "jax[cuda12]==0.11.2"
!python -c "import jax; print(jax.devices())"
!PYTHONPATH=src python scripts/fia_fix.py
!cat results/fia_fix/runs.csv
!cd /kaggle/working/ers-lens && zip -qr /kaggle/working/fia_fix.zip results/fia_fix
```

About 40 solves of about a minute each. If the session stops, run the fia_fix.py line again:
finished solves are kept only while the session lives, so download first if you must stop.
Download fia_fix.zip from the Output panel (/kaggle/working) and unzip it in the repository.
