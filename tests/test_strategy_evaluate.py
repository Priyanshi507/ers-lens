import subprocess
import sys

import pandas as pd

from erslens.strategy import REFINED_TAG


def base_row(variant, car_id="nominal", lap=72.0, pred=72.0, clip=0.0):
    return {"event": "demo", "variant": variant, "car_id": car_id, "lap_s": lap, "dp_pred_s": pred,
            "clip_ratio": clip, "deploy_weighted_kmh": 250.0, "full_throttle_mean_kmh": 280.0,
            "corr_with_nominal": 0.95}


def evaluate(tmp_path, rows):
    path = tmp_path / "runs.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    out = subprocess.run([sys.executable, "scripts/strategy_evaluate.py", "--runs", str(path)],
                         capture_output=True, text=True, check=True).stdout
    return {line.split("|")[1].strip(): line.split("|")[2].strip()
            for line in out.splitlines() if line.startswith("| ") and line.count("|") >= 4}


def rows(sweep_clips, sweep_preds):
    r = [base_row("nominal"), base_row("no_taper"), base_row("m2", lap=72.01),
         base_row("simple_flat_out", lap=72.1, pred=float("nan"))]
    r += [base_row("sweep", f"c{i}", clip=c, pred=72.0 + p) for i, (c, p) in enumerate(zip(sweep_clips, sweep_preds))]
    return r


def test_all_converged_and_clipping(tmp_path):
    v = evaluate(tmp_path, rows([0.0] * 5, [0.0] * 5))
    assert v["Convergence rule (<= 0.05 s)"] == "held"
    assert v["S1"] == v["S2"] == v["S3"] == v["S4"] == v["S5"] == "held"


def test_unconverged_clipping_car_counts_against_s2(tmp_path):
    # 4 of 5 cars clip only if the unconverged one is counted, so S2 (80%) must fail.
    v = evaluate(tmp_path, rows([0.0, 0.0, 0.0, 0.0, 0.9], [0.0, 0.0, 0.0, 0.2, 0.0]))
    assert v["Convergence rule (<= 0.05 s)"] == "NOT held"
    assert v["S2"] == "NOT held"


def test_converged_refinement_restores_the_case(tmp_path):
    r = rows([0.0, 0.0, 0.0, 0.0, 0.9], [0.0, 0.0, 0.0, 0.2, 0.0])
    r.append(base_row("sweep", "c3" + REFINED_TAG + " dv=0.125 soc=641", clip=0.1, pred=72.01))
    v = evaluate(tmp_path, r)
    assert v["Convergence rule (<= 0.05 s)"] == "held"
    assert v["S2"] == "held"


def test_unconverged_nominal_cannot_pass_s1(tmp_path):
    r = rows([0.0] * 5, [0.0] * 5)
    r[0] = base_row("nominal", pred=72.3)
    assert evaluate(tmp_path, r)["S1"] == "NOT held"
