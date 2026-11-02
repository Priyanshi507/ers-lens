# CUDA kernel against XLA on Kaggle (free T4)

GPU T4, Internet on. About 15 minutes.

```
%cd /kaggle/working
!rm -rf ers-lens
!git clone -q https://github.com/Priyanshi507/ers-lens.git
%cd ers-lens
!pip install -q -U "jax[cuda12]==0.11.2" cupy-cuda12x
!python -c "import cupy, jax; print(cupy.__version__, jax.devices())"
!PYTHONPATH=src python scripts/cuda_benchmark.py --out results/cuda_benchmark/t4.json
!cat results/cuda_benchmark/t4.json
```
