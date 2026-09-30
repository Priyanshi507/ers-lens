import argparse

import matplotlib.pyplot as plt
import pandas as pd

p = argparse.ArgumentParser()
p.add_argument("parquet")
p.add_argument("--episode", type=int, default=0)
p.add_argument("--out", default="episode.png")
args = p.parse_args()

df = pd.read_parquet(args.parquet)
ep = df[df["episode"] == args.episode]
km = ep["distance_m"] / 1000

fig, ax = plt.subplots(3, 1, figsize=(12, 8), sharex=True)
ax[0].plot(km, ep["speed_kmh"], lw=0.8)
ax[0].set_ylabel("speed km/h\n(observed)")
ax[1].plot(km, ep["throttle"], lw=0.8, color="tab:green")
ax[1].fill_between(km, 0, ep["brake"] * 100, color="tab:red", alpha=0.3, step="mid")
ax[1].set_ylabel("throttle / brake\n(observed)")
ax[2].plot(km, ep["label_soc_frac"] * 100, color="tab:purple")
ax[2].set_ylabel("battery %\n(HIDDEN label)")
ax[2].set_xlabel("distance km")
for lap_start in ep.groupby("lap")["distance_m"].min() / 1000:
    for a in ax:
        a.axvline(lap_start, color="grey", lw=0.5, ls="--")
fig.suptitle(f"Episode {args.episode}: what the public sees (top) vs what we must infer (bottom)")
fig.tight_layout()
fig.savefig(args.out, dpi=130)
print("saved", args.out)
