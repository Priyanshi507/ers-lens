import argparse
from pathlib import Path

from erslens.ingest import load_driver_race, reference_track, save_track

p = argparse.ArgumentParser(description="Download a 2026 session and build track + race tables")
p.add_argument("--year", type=int, default=2026)
p.add_argument("--event", required=True, help='e.g. "China" or a round number')
p.add_argument("--drivers", nargs="+", default=["HAM", "LEC", "RUS", "ANT"])
p.add_argument("--out", default="data/real")
args = p.parse_args()

out = Path(args.out)
out.mkdir(parents=True, exist_ok=True)
event = int(args.event) if args.event.isdigit() else args.event

track = reference_track(args.year, event)
save_track(track, out / f"{track.name}_track.npz")
print(f"track {track.name}: {track.length_m:.0f} m, {track.straight_mode.mean():.0%} straight-mode")

for drv in args.drivers:
    df = load_driver_race(args.year, event, drv)
    df.to_parquet(out / f"{track.name}_{drv}_race.parquet", index=False)
    print(f"{drv}: {df['lap'].nunique()} laps, {len(df):,} samples")
