# Contributing to ERS-Lens

Thanks for your interest! ERS-Lens estimates the hidden battery (ERS) state of
2026 Formula 1 cars from public telemetry. Contributions of all sizes are welcome.

## Getting set up

```bash
git clone https://github.com/Priyanshi507/ers-lens.git
cd ers-lens
python3.12 -m venv .venv && source .venv/bin/activate
pip install -r requirements-lock.txt
pip install -e . --no-deps
python -m pytest
```

Tests never touch the network, so they run anywhere.

## Ways to help

- **Run the analysis on another 2026 race** and open an issue with the summary
  table. Different circuits stress energy management differently.
- **Check the physics.** Values marked `VERIFY` in `configs/car_2026.yaml` need to
  be checked against the FIA 2026 technical regulations. Cite the article number.
- **Report odd telemetry.** If a driver's data breaks the scripts, open an issue
  with the year, event and driver code.

## Ground rules

- Every change keeps `pytest` passing; new behaviour comes with a test.
- Physical constants belong in the config file, never hard-coded.
- Model inputs never use columns starting with `label_` (those are hidden truth
  from the simulator).
- Comments explain *why*, not *what*.
- Be kind in issues and reviews.

## Pull requests

1. Fork, create a branch named after the change (`fix-empty-driver`, `add-monza`).
2. Keep each PR to one idea.
3. Describe what changed and how you checked it.
