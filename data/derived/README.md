# Data

- `../../results/derate_per_lap.csv`: per-lap measurements for every lap that passed the
  quality checks (driver, lap, lap time, top speed, flat-out speed loss, distances). This is
  the dataset behind every number in the SSAC27 abstract.
- `../../results/derate_summary.csv`, `derate_sensitivity.csv`, `derate_styles.csv`,
  `abstract_figure.csv`: per-driver summaries.
- `2026_china_track.npz`: track model built from the fastest 2026 Chinese GP qualifying lap.

## Raw telemetry

Raw car telemetry is Formula 1 timing data, freely accessible through the open-source
FastF1 library (v3.8.3, pinned in `requirements-lock.txt`). It is not redistributed here.
To download exactly the data used:

    python scripts/fetch_session.py --event China --drivers ANT RUS HAM LEC BEA GAS LAW HAD SAI COL HUL LIN BOT OCO PER VER ALO STR

Norris, Piastri, Bortoleto and Albon did not start the race, so no data exists for them.
