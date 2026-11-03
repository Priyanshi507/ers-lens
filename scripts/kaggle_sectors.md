# The FIA's 250 kW sectors on Kaggle (free T4)

GPU T4, Internet on. Three solves at each of 14 circuits, about half a minute each with the CUDA kernel (about 25 minutes).

```
%cd /kaggle/working
!rm -rf ers-lens
!git clone -q https://github.com/Priyanshi507/ers-lens.git
%cd ers-lens
!git log --oneline -1
!pip install -q -U "jax[cuda12]==0.11.2" cupy-cuda12x
!ERSLENS_DP_BACKEND=cuda PYTHONPATH=src python scripts/sectors_run.py --out results/sectors
!PYTHONPATH=src:scripts python scripts/sectors_run.py --out results/sectors --verdicts
!cd /kaggle/working/ers-lens && zip -qr /kaggle/working/sectors.zip results/sectors
```

Download sectors.zip from the Output panel (/kaggle/working, refresh, then download).
