"""Numbers in the README that a laptop can reproduce in seconds are recomputed here, so a stale
number fails the tests (finding 2's had been stale since 2026-10-02; research log, 2026-11-13)."""
import re
import subprocess
import sys
from pathlib import Path

import pytest

README = Path("README.md").read_text()
ENV = {"PYTHONPATH": "src:scripts"}


def run(*args) -> str:
    import os

    return subprocess.run([sys.executable, *args], capture_output=True, text=True, check=True,
                          env={**os.environ, **ENV}).stdout


@pytest.mark.parametrize("ice,cda,row", [(0.10, 0.20, "| ±10% | ±20% |"), (0.05, 0.10, "| ±5% | ±10% |"),
                                         (0.02, 0.05, "| ±2% | ±5% |")])
def test_finding_1_spread(ice, cda, row, tmp_path):
    out = run("scripts/identifiability_demo.py", "--max-ice-change", str(ice), "--max-cda-change", str(cda),
              "--out", str(tmp_path))
    got = float(re.search(r"spread ([\d.]+) MJ", out).group(1))
    claimed = float(re.search(re.escape(row) + r" ([\d.]+) MJ", README).group(1))
    assert got == claimed


def test_finding_2_validation():
    sys.path.insert(0, "scripts")
    from energy_validate import run as validate, summarise

    s = summarise(validate(40, 6, 0))
    text = README[README.index("### 2."):README.index("### 3.")]
    assert f"correlation **{s['corr']:.2f}**" in text
    assert f"mean error **{s['mae_kw']:.1f} kW** on a {s['true_mean_kw']} kW mean" in text
    assert f"{s['measurable_clip_laps']} of clipping laps measurable" in text
    assert s["false_positive_laps"] == 0 and "no false detections" in text
