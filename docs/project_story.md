# ERS-Lens: the whole project in one place

A record to present from: what the project asks, what was built, every test and how it came out,
what failed, and how to answer the questions a jury is likely to ask. Every number here comes from
`docs/research_log.md` (the dated record) or `docs/experiment_index.md` (one row per test).
Last updated 21 November 2026.

## 1. The 30-second version

In 2026 Formula 1 cars get half their power from a battery that must be refilled every lap. One
way to refill it is "super-clipping": at the end of a straight, at full throttle, the motor turns
into a generator and the car slows down. Fans and drivers disliked it, and from the Miami Grand
Prix the FIA raised the power allowed for it from 250 to 350 kW. I asked whether that fix changed
super-clipping. I built a solver that finds the provably fastest way to use the battery on every
real 2026 circuit, wrote every prediction down before running it, and checked the solver against
what real cars did. The answer: the change made super-clipping shorter, not smaller, and moved
energy out of lift-and-coast. I then turned the solver into tools a regulator can use: the lap-time
value of each megajoule of battery allowance, and a limit advisor that, given only the FIA's stated
target, ranks the circuits almost exactly as the FIA did (rank correlation 0.86).

## 2. The problem in plain words

- The battery holds 4 MJ, and the FIA caps how much can be put back each lap (5 to 9 MJ depending on
  the circuit and session).
- Energy comes back under braking, by lifting off early (lift-and-coast), or by super-clipping.
- Super-clipping makes cars slow down on straights at full throttle: confusing to watch, and a
  safety concern because a following car closes fast.
- The FIA's tools: the super-clip power limit (raised to 350 kW from Miami) and the per-circuit
  recharge limit. Nobody had a way to say, before a race, what a given setting does at a given
  circuit. That is the gap this project fills.

## 3. What I built

| Part | What it does |
|---|---|
| Track builder (`scripts/export_tracks.py`) | turns the fastest qualifying lap from public OpenF1 data into a track: corner speed limits, straights, elevation |
| Lap solver (`src/erslens/optimal.py`, `cuda_dp.py`) | dynamic programming over position, speed and battery charge; finds the fastest lap under the rules; written in JAX with a hand-written CUDA kernel and a NumPy twin that checks it |
| Optimality certificate | every solve comes with a proof of how far it can be from the true optimum (weak duality, T ≥ J(p) − p·B); solves are kept only if the gap is tiny |
| Real-lap detectors | find super-clipping and lift-and-coast in real 2026 telemetry |
| Energy price | the solver's multiplier p: seconds of lap time per extra MJ of allowance |
| Limit advisor (`scripts/limit_advisor.py`) | the qualifying limit per circuit that meets the FIA's target |
| Instant advisor (`scripts/surrogate.py`) | neural-network ensemble trained on the certified solves: predicts super-clipping for a new circuit in seconds |
| Race-week pipeline (`scripts/raceweek.py`) | every race weekend, on GitHub's free servers: track from qualifying, prediction committed before the race, verdict after; never rewritten |
| Rule checker web page | compares any two of 14 rule sets at every circuit, next to real cars |
| Tests | 208 automated tests |

Runs: GPU solves on free Kaggle T4 GPUs (no money spent); everything else on a laptop. About 3,500
saved solves.

## 4. How I worked

- Every test has its predictions and pass marks written in the dated research log before any run.
- Failures are reported as failures. Corrections are new dated entries; nothing is edited afterwards.
- Out-of-sample: the procedure was fixed first, then applied to a race whose data had not been opened
  (Mexico City, predicted after it was run) and to races not yet run (Las Vegas onward).
- The 2027 forecast was frozen before 2027: SHA-256 73961e97…0f310f, git tag `forecast-2027-v1`,
  Zenodo DOI 10.5281/zenodo.24238738, OSF registration osf.io/fh6mx.
- One command (`scripts/paper_numbers.py`) rebuilds the paper's counts from the saved solves.

## 5. Everything that was tested, in order

| When | Question | Outcome |
|---|---|---|
| Early Oct | Can drag, engine power and battery energy be read from speed traces? | No: drag and power trade off exactly. A neural network looked accurate but was confidently wrong on cars unlike its training cars (90% intervals covered 3%) |
| Oct | Optimal energy use ("Strategy Lab") | partly held; convergence rule made strict |
| Oct | Is the car model right? | calibration failed its own rule; drag and power lie on a ridge |
| 30-31 Oct | Do findings hold on every 2026 race? | not held (E1, E3) |
| 1-3 Nov | Did the Miami fix change super-clipping? | C1 held 15 of 18 races (sign test p = 0.008) |
| 3 Nov | The FIA's real 250 kW zones | held 14 of 14 |
| 4 Nov, 13 Nov | Two modelling errors found and fixed; everything re-run | logged |
| 13-14 Nov | Does the model reproduce 2026 circuit by circuit? (hindcast) | Y2 held (clips where real cars clip, 13 of 16); Y1 not held (ranking 0.58, bar 0.6); model's speed loss about 1.9 times real |
| 14 Nov | Lift-and-coast as a choice | LC2 held 17 of 17 (energy moves out of lift-and-coast); size still too large (LC4) |
| 14 Nov | Real lift-and-coast in telemetry | rare in real laps |
| 14-15 Nov | Out of sample, Mexico City | right (no clipping, as real cars) |
| 15 Nov | 2027 forecast | frozen and registered |
| 16 Nov | Is the size error qualifying vs race? | no (QL2); directions agree at 6 of 7 races (QL4) |
| 18 Nov | Is it air density? | no (D1-D4 held, nothing changes) |
| 18 Nov | Is it the car's drag and power? | no; calibrated cars are worse (S1, S2, S4 not held) |
| 21 Nov | Las Vegas out of sample | prediction "clip" (24.1 km/h) pushed before the race |
| 21 Nov | What is a megajoule worth? | EP1 held 18 of 18: the price is a true lap-time value (0.36 s/MJ at Baku, 0 at Monaco) |
| 21 Nov | Do the FIA's lowered limits follow that value? | no, reversed: they sit where super-clipping is heaviest, consistent with the regulation's own criterion (an observation, not a test) |
| 21 Nov | Predict Qatar's and Abu Dhabi's limits before the FIA | calls "below 8.5 MJ" written before the documents; a public May plan suggests Qatar's may be wrong |
| 21 Nov | Limit advisor vs the FIA's choices | LA2 held: Spearman 0.86 over 17 circuits; LA3 not held (model too strict), as predicted |
| 21 Nov | Instant advisor: a neural network trained on the certified solves, tested on unseen circuits | 3 of 4 held: error 2.15 s vs 4.94 baseline, ranks like the FIA at 0.66 with no solver, 7 s instead of GPU hours; its error bars are too narrow (62% coverage), reported |

## 6. Main findings

1. The 350 kW change made super-clipping shorter but not smaller, and moved energy out of
   lift-and-coast (C1, LC2).
2. The model gets directions right and sizes wrong: about 1.9 times too much speed loss where real
   cars clip. Tested and ruled out as causes: session, lift-and-coast, air density, the car's drag
   and power.
3. A megajoule of allowance is worth very different amounts by circuit (0 to 0.36 s per lap), and
   the price passes a strict check at all 18 circuits.
4. The FIA lowers limits where super-clipping is heaviest, not where energy is most valuable.
5. Given only the FIA's stated 2-4 s target, the advisor orders the circuits like the FIA (0.86).

## 7. What did not work, said plainly

- Circuit ranking of race speed loss: 0.58, under the 0.6 bar (Y1).
- Size of the speed loss: about 2 times too large, cause still unknown after four tests.
- The strategy and whole-race tests in October and November mostly failed (W1-W3, E1, E3).
- The first hindcast measure included braking and had to be corrected (Y1 went from 0.79 to 0.58).
- Two modelling errors and one rules error were found and fixed, each in its own log entry.

## 8. Still open

- Las Vegas (race 22 November): judge after the race.
- Qatar race limit (document about 25 November), Qatar race (29 November), Abu Dhabi (6 December).
- Not modelled: the FIA's power-reduction rate (50 or 100 kW/s), slipstream, a battery not full at
  the start of the lap, deployment held back by teams.

## 9. Who could use this

- Regulators: check a recharge limit or power limit circuit by circuit before an event; the
  advisor proposes limits for a new circuit from its qualifying lap.
- Teams: the price says where battery allowance matters for lap time; a team's own car data would
  turn the model's directions into sizes.
- Broadcasters: explain why cars slow on a straight and what one megajoule is worth here.
- Researchers: every prediction, pass mark and miss is public and can be re-run.

## 10. Questions a jury may ask, and honest answers

- *Your model gets the size wrong. Why trust it?* I use it only for directions and orderings, and
  test both: directions held (C1, LC2, QL4); ordering by harvesting difficulty matches the FIA at
  0.86. Size is stated as a limitation with four causes ruled out.
- *Isn't 0.86 just luck after many tries?* It was one test with its bar (0.5) written first; the
  target (4 s) is the FIA's own number and the limit grid is the regulation's range. Nothing was tuned.
- *Why dynamic programming and not machine learning?* The rules are hard constraints and the answer
  must be provably optimal; DP gives a certificate. My early neural-network experiment showed how
  a learned model can look accurate and be confidently wrong.
- *Is any of this specific to 2026?* The method takes any rule set; the 2027 forecast already uses
  it, frozen in advance.
- *How do I know you didn't change predictions after seeing data?* Dated log in git, public
  repository, frozen forecast with hash, DOI and OSF registration.
- *What would you do with team data?* Measure drag and power, which public data cannot separate,
  and fix the size error.

## 11. Words

- Super-clipping: harvesting at full throttle at the end of a straight; the car slows.
- Lift-and-coast: lifting off before braking to harvest.
- Recharge limit: MJ per lap the battery may be refilled; set per circuit by the FIA.
- Certificate / proven gap: a bound on how far a solve is from the best possible lap.
- Pre-registration: writing predictions and pass marks before the data or run.
- Spearman: rank correlation, 1 is the same order.

## 12. Numbers to remember

15 of 18 races (C1) · 13 of 16 (Y2) · 0.58 (Y1) · 1.9 times (size) · 17 of 17 (LC2) · 18 of 18 (EP1)
· 0.86 (LA2) · 180 of 180 advisor solves converged · 208 tests · about 3,500 solves · frozen forecast
DOI 10.5281/zenodo.24238738.
