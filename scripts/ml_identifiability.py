"""Does a network that predicts battery energy from telemetry learn physics, or its simulator?

Runs E1-E4 (cars identical/varied x strategies simple/flexible) and T1 (train on E2,
test on E4), then checks the predictions recorded in docs/research_log.md before running.
"""
import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from erslens.gaussnet import evaluate, fit, predict
from erslens.params import load_car_params
from erslens.synthstraight import generate

REGIMES = {"E1": ("identical", "simple"), "E2": ("identical", "flexible"),
           "E3": ("varied", "simple"), "E4": ("varied", "flexible")}


def run(n_train: int, n_test: int, epochs: int, seed: int) -> dict:
    car = load_car_params("configs/car_2026.yaml")
    data, models, results = {}, {}, {}
    for i, (name, (cars, strategies)) in enumerate(REGIMES.items()):
        x_tr, y_tr = generate(seed + 10 * i, n_train, cars, strategies, car)
        x_te, y_te = generate(seed + 10 * i + 5, n_test, cars, strategies, car)
        data[name] = (x_te, y_te)
        models[name] = fit(x_tr, y_tr, seed=seed, epochs=epochs)
        mean, std = predict(models[name], x_te)
        results[name] = {**evaluate(y_te, mean, std), "target_sd_mj": float(y_te.std())}
        print(f"{name} ({cars} cars, {strategies} strategies): {_fmt(results[name])}")
    mean, std = predict(models["E2"], data["E4"][0])
    results["T1"] = {**evaluate(data["E4"][1], mean, std), "target_sd_mj": float(data["E4"][1].std())}
    print(f"T1 (trained on E2, tested on E4): {_fmt(results['T1'])}")
    return results


def _fmt(r: dict) -> str:
    return (f"RMSE {r['rmse_mj']:.3f} MJ, 90% coverage {r['coverage_90']:.0%}, "
            f"mean predicted sd {r['mean_std_mj']:.3f} MJ (target sd {r['target_sd_mj']:.2f})")


def check_predictions(r: dict) -> list:
    rm = {k: v["rmse_mj"] for k, v in r.items()}
    checks = [
        ("P1  E1 and E2 RMSE < 0.10 MJ", rm["E1"] < 0.10 and rm["E2"] < 0.10,
         f"E1 {rm['E1']:.3f}, E2 {rm['E2']:.3f}"),
        ("P2  E4 RMSE >= 5x E2 and >= 0.30 MJ", rm["E4"] >= 5 * rm["E2"] and rm["E4"] >= 0.30,
         f"E4 {rm['E4']:.3f} = {rm['E4'] / rm['E2']:.1f}x E2"),
        ("P3' (E4/E2) >= 2 x (E3/E1)", rm["E4"] / rm["E2"] >= 2 * rm["E3"] / rm["E1"],
         f"E4/E2 = {rm['E4'] / rm['E2']:.1f}, E3/E1 = {rm['E3'] / rm['E1']:.1f}"),
        ("P4  T1 RMSE >= 3x E2 and coverage < 70%",
         rm["T1"] >= 3 * rm["E2"] and r["T1"]["coverage_90"] < 0.70,
         f"T1 {rm['T1']:.3f} = {rm['T1'] / rm['E2']:.1f}x E2, coverage {r['T1']['coverage_90']:.0%}"),
        ("P5  E1-E4 coverage 85-95%, E4 sd >= 3x E2 sd",
         all(0.85 <= r[k]["coverage_90"] <= 0.95 for k in REGIMES)
         and r["E4"]["mean_std_mj"] >= 3 * r["E2"]["mean_std_mj"],
         ", ".join(f"{k} {r[k]['coverage_90']:.0%}" for k in REGIMES)
         + f"; sd ratio {r['E4']['mean_std_mj'] / r['E2']['mean_std_mj']:.1f}"),
    ]
    return checks


def figure(r: dict, path: Path):
    names = ["E1", "E2", "E3", "E4", "T1"]
    labels = ["E1\nidentical\nsimple", "E2\nidentical\nflexible", "E3\nvaried\nsimple",
              "E4\nvaried\nflexible", "T1\ntrained E2\ntested E4"]
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
    colors = ["tab:blue", "tab:blue", "tab:orange", "tab:orange", "tab:red"]
    ax[0].bar(labels, [r[n]["rmse_mj"] for n in names], color=colors)
    ax[0].set_ylabel("battery energy error, RMSE (MJ)")
    ax[0].set_title("A. Prediction error")
    ax[1].bar(labels, [100 * r[n]["coverage_90"] for n in names], color=colors)
    ax[1].axhline(90, color="k", ls="--", lw=1)
    ax[1].set_ylabel("90% intervals containing the truth (%)")
    ax[1].set_ylim(0, 100)
    ax[1].set_title("B. Honesty of the model's uncertainty")
    fig.tight_layout()
    fig.savefig(path, dpi=150)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n-train", type=int, default=40000)
    p.add_argument("--n-test", type=int, default=10000)
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default="results/ml")
    args = p.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    results = run(args.n_train, args.n_test, args.epochs, args.seed)
    print("\nPredictions recorded before running (docs/research_log.md):")
    for label, ok, detail in check_predictions(results):
        print(f"  {'HELD    ' if ok else 'NOT HELD'}  {label}: {detail}")
    (out / "metrics.json").write_text(json.dumps(results, indent=2))
    figure(results, out / "ml_identifiability.png")
    print(f"\nSaved {out}/metrics.json and {out}/ml_identifiability.png")


if __name__ == "__main__":
    main()
