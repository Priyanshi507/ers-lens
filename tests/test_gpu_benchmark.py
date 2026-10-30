import json
import sys

sys.path.insert(0, "scripts")
from gpu_benchmark import compare  # noqa: E402


def result(device, backward_s, solve_s, lap_s):
    return {"device": {"platform": device, "device": device, "processor": device, "jax": "0.11.2"},
            "events": {"A": {"backward": {"median_s": backward_s},
                             "solve": {"seconds": solve_s, "lap_s": lap_s}}}}


def write(tmp_path, name, data):
    p = tmp_path / name
    p.write_text(json.dumps(data))
    return str(p)


def test_speed_up_is_reported_when_results_agree(tmp_path, capsys):
    cpu = write(tmp_path, "cpu.json", result("cpu", 2.0, 600.0, 77.5812))
    gpu = write(tmp_path, "gpu.json", result("gpu", 0.1, 60.0, 77.5815))
    assert compare(cpu, gpu) == 0
    out = capsys.readouterr().out
    assert "20.0x" in out and "10.0x" in out and "| yes |" in out


def test_different_lap_times_block_the_speed_up(tmp_path, capsys):
    cpu = write(tmp_path, "cpu.json", result("cpu", 2.0, 600.0, 77.581))
    gpu = write(tmp_path, "gpu.json", result("gpu", 0.1, 60.0, 77.590))
    assert compare(cpu, gpu) == 1
    assert "no speed-up may be reported" in capsys.readouterr().out
