"""Robustness (seeds) and capacity/data control for the learnability experiment.

Checks predictions R1-R4 recorded in docs/research_log.md before running:
whether the main results survive reseeding, and whether the varied-car error is limited
by information (it stays put as the model and data grow) or by the model (it shrinks).
"""
import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from erslens.gaussnet import evaluate, fit, predict
from erslens.params import load_car_params
from erslens.synthstraight import car_ambiguity_mj, generate
from ml_identifiability import check_predictions, run

BASELINE = ("40k laps, 2x256", 40000, (256, 256))
CONFIGS = [BASELINE, ("160k laps, 2x256", 160000, (256, 256)),
           ("40k laps, 3x512", 40000, (512, 512, 512)),
           ("160k laps, 3x512", 160000, (512, 512, 512))]
REGIMES = {"E2": ("identical", "flexible"), "E4": ("varied", "flexible")}
TEST_SEED = {"E2": 9001, "E4": 9002}


def seeds_part(seeds, n_train, n_test, epochs):
    per_seed, verdicts = [], []
    for seed in seeds:
        print(f"\n--- seed {seed} ---")
        r = run(n_train, n_test, epochs, seed)
        per_seed.append(r)
        verdicts.append({label.split()[0]: ok for label, ok, _ in check_predictions(r)})
    summary = {name: {"rmse_mean": float(np.mean([r[name]["rmse_mj"] for r in per_seed])),
                      "rmse_sd": float(np.std([r[name]["rmse_mj"] for r in per_seed], ddof=1)),
                      "coverage_mean": float(np.mean([r[name]["coverage_90"] for r in per_seed]))}
               for name in per_seed[0]}
    return per_seed, verdicts, summary


def capacity_part(cap_seeds, n_test, epochs, car):
    results = {}
    for name, (cars, strategies) in REGIMES.items():
        x_te, y_te = generate(TEST_SEED[name], n_test, cars, strategies, car)
        for label, n_train, hidden in CONFIGS:
            rmses = []
            for seed in cap_seeds:
                x_tr, y_tr = generate(3000 + 100 * seed + (0 if name == "E2" else 50), n_train,
                                      cars, strategies, car)
                f = fit(x_tr, y_tr, seed=seed, hidden=hidden, epochs=epochs)
                rmses.append(evaluate(y_te, *predict(f, x_te))["rmse_mj"])
            results[(name, label)] = (float(np.mean(rmses)), float(np.std(rmses, ddof=1)))
            print(f"{name} {label}: RMSE {results[(name, label)][0]:.4f} "
                  f"+/- {results[(name, label)][1]:.4f} MJ")
    ambiguity = float(car_ambiguity_mj(TEST_SEED["E4"], n_test, "flexible", car).mean())
    return results, ambiguity


def check(verdicts, summary, cap, ambiguity):
    def best(name):
        return min(cap[(name, label)][0] for label, *_ in CONFIGS)

    e2_drop = 1 - best("E2") / cap[("E2", BASELINE[0])][0]
    e4_drop = 1 - best("E4") / cap[("E4", BASELINE[0])][0]
    ratio = cap[("E4", BASELINE[0])][0] / ambiguity
    stable = all(all(v[k] == verdicts[0][k] for v in verdicts) for k in ("P1", "P2", "P4", "P5"))
    cv = summary["E4"]["rmse_sd"] / summary["E4"]["rmse_mean"]
    return [
        ("R1  P1/P2/P4/P5 verdicts identical in every seed; E4 RMSE seed sd < 10% of mean",
         stable and cv < 0.10, f"verdicts stable: {stable}; E4 seed sd = {cv:.1%} of mean"),
        ("R2  E2 RMSE falls >= 30% with more data/capacity", e2_drop >= 0.30, f"E2 fell {e2_drop:.0%}"),
        ("R3  E4 RMSE falls < 25% with more data/capacity", e4_drop < 0.25, f"E4 fell {e4_drop:.0%}"),
        ("R4  baseline E4 RMSE / mean car ambiguity in [0.3, 0.9]", 0.3 <= ratio <= 0.9,
         f"{cap[('E4', BASELINE[0])][0]:.3f} / {ambiguity:.3f} = {ratio:.2f}"),
    ]


def figure(summary, cap, ambiguity, path):
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.4))
    names = ["E1", "E2", "E3", "E4", "T1"]
    ax[0].bar(names, [summary[n]["rmse_mean"] for n in names],
              yerr=[summary[n]["rmse_sd"] for n in names], capsize=4,
              color=["tab:blue", "tab:blue", "tab:orange", "tab:orange", "tab:red"])
    ax[0].set_yscale("log")
    ax[0].set_ylabel("RMSE (MJ), mean and sd over seeds")
    ax[0].set_title("A. Main result across random seeds")
    labels = [label for label, *_ in CONFIGS]
    xpos = np.arange(len(labels))
    for name, color in (("E2", "tab:blue"), ("E4", "tab:orange")):
        ax[1].errorbar(xpos, [cap[(name, l)][0] for l in labels], yerr=[cap[(name, l)][1] for l in labels],
                       marker="o", capsize=4, color=color,
                       label="identical cars (E2)" if name == "E2" else "varied cars (E4)")
    ax[1].axhline(ambiguity, color="k", ls="--", lw=1, label="car-uncertainty ambiguity")
    ax[1].set_yscale("log")
    ax[1].set_xticks(xpos, labels, fontsize=8)
    ax[1].set_ylabel("RMSE (MJ)")
    ax[1].set_title("B. More data and capacity: model limit vs information limit")
    ax[1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seeds", type=int, default=5)
    p.add_argument("--cap-seeds", type=int, default=3)
    p.add_argument("--n-train", type=int, default=40000)
    p.add_argument("--n-test", type=int, default=10000)
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--scale", type=float, default=1.0, help="shrink data sizes for smoke tests")
    p.add_argument("--out", default="results/ml_robustness")
    args = p.parse_args()

    global CONFIGS, BASELINE
    if args.scale != 1.0:
        CONFIGS = [(l, max(500, int(n * args.scale)), h) for l, n, h in CONFIGS]
        BASELINE = CONFIGS[0]
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    car = load_car_params("configs/car_2026.yaml")
    n_train, n_test = max(500, int(args.n_train * args.scale)), max(200, int(args.n_test * args.scale))

    per_seed, verdicts, summary = seeds_part(range(args.seeds), n_train, n_test, args.epochs)
    print("\n--- capacity and data control ---")
    cap, ambiguity = capacity_part(range(args.cap_seeds), n_test, args.epochs, car)
    print(f"Mean car-uncertainty ambiguity on E4 test laps: {ambiguity:.3f} MJ")

    print("\nSummary over seeds (RMSE mean +/- sd, mean 90% coverage):")
    for name, s in summary.items():
        print(f"  {name}: {s['rmse_mean']:.3f} +/- {s['rmse_sd']:.3f} MJ, coverage {s['coverage_mean']:.0%}")
    print("\nPredictions recorded before running (docs/research_log.md):")
    for label, ok, detail in check(verdicts, summary, cap, ambiguity):
        print(f"  {'HELD    ' if ok else 'NOT HELD'}  {label}: {detail}")

    (out / "robustness.json").write_text(json.dumps({
        "per_seed": per_seed, "verdicts": verdicts, "summary": summary,
        "capacity": {f"{k[0]} | {k[1]}": v for k, v in cap.items()},
        "car_ambiguity_mj": ambiguity}, indent=2))
    figure(summary, cap, ambiguity, out / "ml_robustness.png")
    print(f"\nSaved {out}/robustness.json and {out}/ml_robustness.png")


if __name__ == "__main__":
    main()
