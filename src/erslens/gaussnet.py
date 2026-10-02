"""MLP that predicts a Gaussian (mean and standard deviation) for one scalar target.

Trained by negative log-likelihood, so the predicted standard deviation is the model's own
uncertainty; coverage of its intervals shows whether that uncertainty is honest.
"""
from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np
import optax
from scipy.stats import norm

jax.config.update("jax_enable_x64", True)

MIN_LOG_STD = -7.0


@dataclass
class Fitted:
    params: list
    x_mean: np.ndarray
    x_std: np.ndarray
    y_mean: float
    y_std: float
    train_loss: list


def _init(key, sizes):
    params = []
    for k, (n_in, n_out) in zip(jax.random.split(key, len(sizes) - 1), zip(sizes[:-1], sizes[1:])):
        params.append((jax.random.normal(k, (n_in, n_out)) * jnp.sqrt(2.0 / n_in), jnp.zeros(n_out)))
    return params


def _forward(params, x):
    for w, b in params[:-1]:
        x = jax.nn.gelu(x @ w + b)
    w, b = params[-1]
    out = x @ w + b
    return out[:, 0], jnp.maximum(out[:, 1], MIN_LOG_STD)


def _nll(params, x, y):
    mu, log_std = _forward(params, x)
    return jnp.mean(log_std + 0.5 * ((y - mu) / jnp.exp(log_std)) ** 2)


def fit(x: np.ndarray, y: np.ndarray, seed: int = 0, hidden=(256, 256), epochs: int = 40,
        batch: int = 256, lr: float = 1e-3, val_frac: float = 0.1) -> Fitted:
    x_mean, x_std = x.mean(0), x.std(0) + 1e-6
    y_mean, y_std = float(y.mean()), float(y.std() + 1e-9)
    xn, yn = (x - x_mean) / x_std, (y - y_mean) / y_std
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(x))
    n_val = int(len(x) * val_frac)
    val, tr = idx[:n_val], idx[n_val:]

    params = _init(jax.random.PRNGKey(seed), [x.shape[1], *hidden, 2])
    opt = optax.adamw(optax.cosine_decay_schedule(lr, epochs * (len(tr) // batch)), weight_decay=1e-4)
    state = opt.init(params)

    @jax.jit
    def step(params, state, xb, yb):
        loss, grads = jax.value_and_grad(_nll)(params, xb, yb)
        updates, state = opt.update(grads, state, params)
        return optax.apply_updates(params, updates), state, loss

    val_loss = jax.jit(_nll)
    best, best_params, history = np.inf, params, []
    for _ in range(epochs):
        order = rng.permutation(tr)
        for i in range(0, len(order) - batch + 1, batch):
            b = order[i:i + batch]
            params, state, _ = step(params, state, xn[b], yn[b])
        v = float(val_loss(params, xn[val], yn[val]))
        history.append(v)
        # Keep the epoch with the best held-out likelihood rather than the last one.
        if v < best:
            best, best_params = v, params
    return Fitted(best_params, x_mean, x_std, y_mean, y_std, history)


def predict(fitted: Fitted, x: np.ndarray):
    mu, log_std = _forward(fitted.params, (x - fitted.x_mean) / fitted.x_std)
    return (np.asarray(mu) * fitted.y_std + fitted.y_mean,
            np.exp(np.asarray(log_std)) * fitted.y_std)


def evaluate(y: np.ndarray, mean: np.ndarray, std: np.ndarray, level: float = 0.90) -> dict:
    z = norm.ppf(0.5 + level / 2)
    return {"rmse_mj": float(np.sqrt(np.mean((mean - y) ** 2))),
            "mae_mj": float(np.mean(np.abs(mean - y))),
            f"coverage_{int(level * 100)}": float(np.mean(np.abs(y - mean) <= z * std)),
            "mean_std_mj": float(np.mean(std))}
