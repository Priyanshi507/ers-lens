import argparse

from erslens.ingest import load_track
from erslens.params import load_car_params
from erslens.synth import generate_dataset
from erslens.track import demo_circuit

p = argparse.ArgumentParser(description="Generate labelled synthetic laps")
p.add_argument("--track", help="path to a .npz from fetch_session.py (default: demo circuit)")
p.add_argument("--episodes", type=int, default=200)
p.add_argument("--laps", type=int, default=5)
p.add_argument("--seed", type=int, default=0)
p.add_argument("--out", default="data/synthetic")
args = p.parse_args()

track = load_track(args.track) if args.track else demo_circuit()
car = load_car_params("configs/car_2026.yaml")
data = generate_dataset(track, car, args.episodes, args.out, args.laps, args.seed)
print(f"{len(data):,} samples, {data['episode'].nunique()} episodes -> {args.out}/")
