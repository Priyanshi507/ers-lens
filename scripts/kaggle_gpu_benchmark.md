# Running the GPU benchmark on Kaggle (free)

1. kaggle.com → Create → New Notebook. In the right panel: Accelerator **GPU P100**
   (better float64 than the T4), Internet **On** (needs a phone-verified account).
2. Run these cells:

```
!git clone https://github.com/Priyanshi507/ers-lens.git
%cd ers-lens
!pip install -q -U "jax[cuda12]==0.11.2"
!python -c "import jax; print(jax.__version__, jax.devices())"
```

The last line must list a CUDA device (for example `CudaDevice(id=0)`). If the install fails,
try `"jax[cuda13]==0.11.2"` instead and note which worked.

```
!PYTHONPATH=src python scripts/gpu_benchmark.py --out results/gpu_benchmark/kaggle_gpu.json
!cat results/gpu_benchmark/kaggle_gpu.json
```

3. Download `results/gpu_benchmark/kaggle_gpu.json` (Output panel, or copy the printed JSON)
   and put it in the same folder of your local repository.
