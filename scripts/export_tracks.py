"""Save the Strategy Lab's reference tracks, so machines without the FastF1 cache can run the solver.

Writes results/tracks/<event>.npz from the cached qualifying laps (the tracks every Strategy Lab
result used). Run once on a machine that has the cache:

    python scripts/export_tracks.py
"""
import argparse
from pathlib import Path

from erslens.ingest import reference_track, save_track


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--events", nargs="+", default=["Australia", "China", "Miami", "Canada"])
    ap.add_argument("--year", type=int, default=2026)
    ap.add_argument("--out", default="results/tracks")
    args = ap.parse_args()
    Path(args.out).mkdir(parents=True, exist_ok=True)
    for e in args.events:
        track = reference_track(args.year, e)
        save_track(track, Path(args.out) / f"{e}.npz")
        print(f"{e}: {track.n} points, {track.distance_m[-1]:.0f} m")


if __name__ == "__main__":
    main()
