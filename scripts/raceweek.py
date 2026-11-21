"""Race-week pipeline: a dated prediction before every race, judged after it, with no one in the loop
except for the FIA's recharge limit, which must be added with its source (configs/rules_2026_races.yaml).

    python scripts/raceweek.py auto                     # what the scheduled workflow runs
    python scripts/raceweek.py predict "Qatar Grand Prix"
    python scripts/raceweek.py judge "Qatar Grand Prix"
    python scripts/raceweek.py scoreboard

Protocol (research log, 2026-11-14 and 2026-11-21): track from the event's qualifying only, original
car, 350 kW, the race's own limit, no lift-and-coast, no altitude; verdict "clip" if the model's
full-throttle drop on the longest straight is 3 km/h or more; dropped if qualifying was wet; judged
right if real race laps clip on half or more of clean laps (clip_share >= 0.5) or, for "no clip",
on less than half. A prediction file is never rewritten.
"""
import argparse
import json
import subprocess
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

CLIP_KMH = 3.0
JUDGE_AFTER = timedelta(hours=3)
PRED_DIR = ROOT / "predictions"


def slug(event: str) -> str:
    return event.lower().replace(" grand prix", "").replace(" ", "-")


def pred_path(year: int, event: str) -> Path:
    return PRED_DIR / str(year) / f"{slug(event)}.json"


def read(path: Path) -> dict | None:
    return json.loads(path.read_text()) if path.exists() else None


def write_new(path: Path, record: dict):
    if path.exists():
        raise FileExistsError(f"{path} exists: predictions are never rewritten")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(record, indent=2) + "\n")


def add_judgement(path: Path, judgement: dict):
    record = read(path)
    if record is None or record.get("status") != "predicted":
        raise ValueError(f"{path}: no open prediction to judge")
    if "judged_utc" in record:
        raise FileExistsError(f"{path} is already judged")
    record.update(judgement)
    path.write_text(json.dumps(record, indent=2) + "\n")


def verdict(drop_kmh: float) -> str:
    return "clip" if drop_kmh >= CLIP_KMH else "no clip"


def right(call: str, clip_share: float) -> bool:
    return (call == "clip") == (clip_share >= 0.5)


def decide(sessions: list[dict], now: datetime, state) -> tuple[str, str | None]:
    """The single action for this run. sessions: OpenF1 rows with meeting_name, session_name,
    date_start, date_end (UTC). state(event) gives the prediction record or None."""
    by_meeting = {}
    for s in sessions:
        by_meeting.setdefault(s["meeting_name"], {})[s["session_name"]] = s
    done = [(m, v) for m, v in by_meeting.items()
            if "Qualifying" in v and "Race" in v and _t(v["Qualifying"]["date_end"]) <= now]
    if not done:
        return "wait", None
    event, v = max(done, key=lambda mv: _t(mv[1]["Qualifying"]["date_end"]))
    record = state(event)
    race_start, race_end = _t(v["Race"]["date_start"]), _t(v["Race"]["date_end"])
    if record is None:
        return ("predict", event) if now < race_start else ("missed", event)
    if record.get("status") == "predicted" and "judged_utc" not in record and now >= race_end + JUDGE_AFTER:
        return "judge", event
    return "wait", event


def _t(text: str) -> datetime:
    return pd.Timestamp(text).tz_convert("UTC").to_pydatetime() if pd.Timestamp(text).tzinfo \
        else pd.Timestamp(text).tz_localize("UTC").to_pydatetime()


def git_commit() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, capture_output=True,
                              text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def run(*args: str):
    print("+", " ".join(args), flush=True)
    subprocess.run([sys.executable, *args], cwd=ROOT, check=True,
                   env={**__import__("os").environ, "PYTHONPATH": "src:scripts", "ERSLENS_SOURCE": "openf1"})


def wet_qualifying(year: int, event: str) -> bool:
    from erslens import openf1
    from erslens.ingest import event_info

    key = openf1.session_key(year, event_info(year, event)[1], "Q")
    return any(int(w.get("rainfall") or 0) > 0 for w in openf1.get("weather", ("session_key", key)))


def predict(year: int, event: str, now: datetime, race_start: datetime | None = None):
    from erslens.strategy import race_rules
    from erslens.params import load_car_params

    path = pred_path(year, event)
    if path.exists():
        print(f"{path} exists; nothing to do")
        return
    if race_start is not None and now >= race_start:
        raise RuntimeError(f"{event}: the race has started, a prediction now would not be out of sample")
    try:
        race_rules(load_car_params(str(ROOT / "configs/car_2026.yaml")), event,
                   str(ROOT / "configs/rules_2026_races.yaml"))
    except KeyError:
        raise SystemExit(f"{event}: add the race recharge limit from the FIA's Power Unit Information "
                         f"document, with its source, to configs/rules_2026_races.yaml, then run again")
    base = {"event": event, "year": year, "made_utc": now.isoformat(timespec="minutes"),
            "protocol": "research log 2026-11-14 and 2026-11-21"}
    if wet_qualifying(year, event):
        write_new(path, {**base, "status": "dropped", "reason": "wet qualifying"})
        return
    out = f"results/raceweek/{year}/{slug(event)}"
    run("scripts/export_tracks.py", "--year", str(year), "--events", event)
    run("scripts/fia_fix.py", "--cars", "original", "--caps", "350", "--out", out, "--events", event)
    from hindcast_2026 import BRAKES, drop_from_profile

    row = pd.read_csv(ROOT / out / "runs.csv").iloc[-1]
    if str(row.converged) != "True":
        raise RuntimeError(f"{event}: solve did not pass the convergence rule; no prediction written")
    drop = float(drop_from_profile(ROOT / out / "profiles" / f"{event}_original_350.npz", BRAKES))
    write_new(path, {**base, "status": "predicted", "call": verdict(drop), "model_drop_kmh": round(drop, 2),
                     "superclip_mj": round(float(row.superclip_mj), 2), "lap_s": round(float(row.lap_s), 3),
                     "proven_gap_s": round(float(row.optimality_gap_s), 4),
                     "price_s_per_mj": round(float(row.price_s_per_mj), 3), "code_commit": git_commit()})


def judge(year: int, event: str, now: datetime):
    path = pred_path(year, event)
    record = read(path)
    if record is None or record.get("status") != "predicted":
        raise SystemExit(f"{event}: no open prediction")
    out = f"results/raceweek/{year}/{slug(event)}/real"
    run("scripts/multi_race.py", "--year", str(year), "--events", event, "--rebase", "--out", out)
    s = pd.read_csv(ROOT / out / "R" / "race_summary.csv").iloc[-1]
    share = float(s.clip_share)
    add_judgement(path, {"judged_utc": now.isoformat(timespec="minutes"), "real_clip_share": round(share, 3),
                         "real_drop_kmh_median": float(s.drop_kmh_median), "real_clean_laps": int(s.clean_laps),
                         "right": right(record["call"], share)})


def scoreboard(year: int) -> pd.DataFrame:
    rows = [read(p) for p in sorted((PRED_DIR / str(year)).glob("*.json"))]
    cols = ["event", "made_utc", "status", "call", "model_drop_kmh", "real_clip_share", "real_drop_kmh_median",
            "right", "code_commit"]
    table = pd.DataFrame(rows, columns=cols) if rows else pd.DataFrame(columns=cols)
    out = PRED_DIR / str(year) / "scoreboard.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(out, index=False)
    return table


def openf1_sessions(year: int) -> list[dict]:
    from erslens import openf1

    names = {m["meeting_key"]: m["meeting_name"] for m in openf1.get("meetings", ("year", year))
             if not m.get("is_cancelled")}
    return [{**s, "meeting_name": names[s["meeting_key"]]} for s in openf1.get("sessions", ("year", year))
            if s.get("meeting_key") in names and s.get("session_name") in ("Qualifying", "Race")]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["auto", "predict", "judge", "scoreboard"])
    ap.add_argument("event", nargs="?")
    ap.add_argument("--year", type=int, default=2026)
    args = ap.parse_args()
    now = datetime.now(timezone.utc)
    if args.stage == "auto":
        sessions = openf1_sessions(args.year)
        action, event = decide(sessions, now, lambda e: read(pred_path(args.year, e)))
        print(f"{now:%Y-%m-%d %H:%M} UTC: {action} {event or ''}")
        if action == "predict":
            race = next(s for s in sessions if s["meeting_name"] == event and s["session_name"] == "Race")
            predict(args.year, event, now, _t(race["date_start"]))
        elif action == "judge":
            judge(args.year, event, now)
    elif args.stage == "predict":
        predict(args.year, args.event, now)
    elif args.stage == "judge":
        judge(args.year, args.event, now)
    print(scoreboard(args.year).to_string(index=False))


if __name__ == "__main__":
    main()
