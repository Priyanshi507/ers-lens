# Research log

## 2026-10-01: Energy estimation (Step 1)

**Goal.** Turn full-throttle speed loss into electric power and energy, the units the
FIA regulates.

**Attempt 1: fit each car's drag and ICE power, then integrate the power shortfall.**
Least squares on full-throttle acceleration between 210 and 285 km/h (above the
traction limit, below the MGU-K taper), assuming full deployment there.

Ablation on simulated cars (errors in fitted drag area / ICE power):

| Conditions                              | Drag area | ICE power |
|-----------------------------------------|-----------|-----------|
| Ideal car, perfect data                 | +3%       | +4%       |
| + 4 Hz sampling, whole-km/h rounding    | +11-19%   | +11-12%   |
| + two active-aero drag modes            | +25-34%   | +18%      |
| + realistic battery (partial deployment)| +200%     | +43-85%   |

Integral (kinetic-energy) form instead of differentiated speed fixed noise amplification
but not the last row. Quantile ("frontier") regression did not rescue it either.

**Why it fails: identifiability.** Speed only reveals net power. High drag with full
deployment produces the same trace as low drag with partial deployment, so no fitting
method can separate them from speed alone.

**Attempt 2: matched-speed differencing (adopted).** On a clipping lap the car passes
each speed twice at full throttle, before and after the peak. At equal speed, drag,
rolling resistance and ICE power cancel, leaving only the change in electric power:
swing = dKE (1/T_up + 1/T_down). Needs only mass, which the regulations fix.

Validation, 40 simulated cars (`scripts/energy_validate.py`): correlation 0.992 with
the true swing, mean absolute error 19.5 kW on a 328 kW mean, bias -4%, no false
positives on non-clipping cars, 64% of clipping laps measurable (band of at least
4 km/h needed). A 5% mass error shifts estimates by about 5%.

**Prediction stated before running on real data:** from Miami the FIA raised peak
super-clipping power from 250 to 350 kW, so if that applies in races the swing should
rise after Miami (`scripts/energy_real.py`).

## 2026-10-01 (later): Circuit effects and elevation

**Real-race result, level-ground assumption.** Swing did not rise after Miami (race
medians 193 vs 200 kW, permutation p = 0.49). Circuit differences (126-291 kW) dwarfed
the before/after difference.

**Normalising by available MGU-K power at the band speed** (taper values from config,
still marked VERIFY): Australia 0.81, China 0.82, Miami 0.83, Canada 0.84 of available
power withdrawn. Outliers were the two hilliest circuits, Austria 1.16 and Belgium 0.69
(Japan 0.60 on only 23 laps).

**Gravity does not cancel between the two passes** when they sit on different gradients.
Added each pass's elevation change to the energy balance, using the circuit elevation
profile from FastF1 position data (Z, tenths of a metre). Test: constant electric power
on a level-then-10%-uphill straight gives a false 65 kW swing assuming level ground and
-3.8 kW with the correction.

**Prediction stated before running:** the correction moves Austria down and Belgium up
toward the ~0.82 of the flat circuits. A fraction above 1 is physically possible (deploy
cut plus harvesting), so Austria may stay above 1 if harvesting is real.

## 2026-10-01: Step 2, milestone 1 (differentiable simulator)

Ported the lap simulator to JAX (`src/erslens/jaxsim.py`) with the energy strategy as an
open-loop command per 5 m of track and car parameters as differentiable inputs.

- Matches the NumPy simulator to machine precision (speed within 1e-14 m/s, battery
  energy within 1e-9 J, identical lap times).
- Gradients match central finite differences to 6 significant figures for drag area,
  ICE power and deployment efficiency. Strategy gradients are nonzero at 343 of 904
  points: those where changing deployment can change lap time.
- 771x faster per strategy than NumPy when batching 256 strategies on CPU.
- Sensitivities on the demo circuit: +0.01 drag area costs ~0.055 s/lap; +10 kW ICE
  power gains ~0.21 s/lap.

Next: calibrate the simulator's energy behaviour against the real measurements
(~82% of available MGU-K power withdrawn, ~5-7 s of clipping per straight).

## 2026-10-01: Step 2, milestone 2 (strategy fitting)

Inferring absolute battery state from speed is not identifiable (same trade-off as drag
vs deployment). Instead, fit an interpretable 3-parameter strategy per lap on the longest
straight: deployed fraction D of MGU-K cap, clip start s_c, harvest power H after it
(`src/erslens/straightfit.py`, BFGS through a JAX model including gravity).

- Recovery on synthetic laps with 4 Hz rounding and noise: D within 0.004, s_c within
  2 m, H within 1 kW; fit error equals the noise level (0.85 km/h).
- Compiled once per circuit: 0.07 s per lap.
- Dry run: two independent swing estimates agree (Spearman 0.78); a 250 to 350 kW change
  in harvest power is detected.
- Sensitivity: +/-5% ICE power or +/-10% drag shifts swing by only 2-5 kW but total
  energy used by 180-250 kJ. Swing is trustworthy; absolute energy depends on assumptions.

Prediction stated before running on real data: if the Miami increase in peak
super-clipping power (250 to 350 kW) applies in races, fitted H rises after Miami.

## 2026-10-01 (later): Milestone 2 failed its checks on real data; fixed

First real run (7 races) failed two built-in checks:
- **Sensitivity not smooth:** both +5% and -5% ICE power lowered fitted H by 110-150 kW
  (all perturbations lowered it by 110-265 kW). The clip point makes the loss multi-modal
  (late clip + strong harvest vs early clip + gentle harvest), and BFGS found the nearest
  valley. H results from that run are not interpretable.
- **Sampling bug:** fits required full throttle from the straight's first point; race laps
  often reach it later, so China kept 9 of 255 clean laps and Spa 53.
- Spa misfit (rmse 7.9 km/h, H = 0): the 3-parameter model does not describe Kemmel.

Fixes: profile over candidate clip points (19 coarse + 9 fine), with D and H fitted by
BFGS at each one; start each fit at the lap's first full-throttle sample (zero-weighted
padding keeps one compilation per circuit). Clip points fitting within 10% of the best
give an uncertainty range. Recovery holds (D within 0.01, clip within 7 m, H within 3 kW),
sensitivity is now symmetric (ICE +/-5% moves H -24/+24 kW, swing -11/+11 kW), 0.47 s/lap.

## 2026-10-02: Regulation taper, ramp model, model comparison

- Replaced the linear 290-355 km/h taper with the reported regulation formula, in one
  shared function (`physics.mguk_cap_kmh`) used by both simulators and the fitter:
  1800 - 5v kW below 340 km/h (capped at 350), 6900 - 20v between 340 and 345, zero above.
  Previously three separate copies held the wrong formula.
- Added a ramp-limited clipping model (power reduction at most 50 kW/s, VERIFY).
- Model discrimination: on step-generated laps the step model wins 100% of the time; on
  ramp-generated laps the ramp model wins 92%.
- Swing redefined as the realised power change (pre-clip power, read 40 m before the
  switch, minus power at the end of the run). On matching synthetic data: step fit bias
  +1 kW, Spearman 0.98 with the independent estimator; ramp fit bias +3 kW, Spearman 0.89.
  A ramp model on step data gives meaningless swings: model choice matters.

Rule stated before the real run: ramp preferred if lower median fit error on >= 5 of 7
circuits.

## 2026-10-02: Identifiability result (core contribution)

On a full-throttle straight, m dv/dt = (P_ice + P_e)/v - 0.5 rho CdA v^2 - crr m g.
The substitution P_ice -> P_ice + a, CdA -> CdA + delta, P_e(t) -> P_e(t) - a +
0.5 rho delta v(t)^3 leaves the dynamics, and hence the speed trace, exactly unchanged
while changing battery energy used. Electric power is identifiable from speed only up to
a function of speed; differences at matched speeds cancel any such function, so the
matched-speed estimator recovers exactly the identifiable part.

Demonstration (`scripts/identifiability_demo.py`): speeds equal to 1e-14 m/s, 100% of
4 Hz rounded telemetry rows identical, matched-speed swing identical (264.3 kW) for every
alternative. Restricting alternatives to regulation-legal deployment at every instant
(<= speed-dependent cap, >= -350 kW), battery energy on one straight still spans:
ICE +/-10%, drag +/-20%: 2.53 MJ; +/-5%/+/-10%: 1.65 MJ; +/-2%/+/-5%: 0.76 MJ
(4 MJ battery window). Absolute battery state is not recoverable from speed telemetry
without stronger information than plausible priors on car parameters.

## 2026-10-02: Regulation check

The MGU-K taper formula used since the regulation-taper update (1800 - 5v kW below
340 km/h, 6900 - 20v kW at or above 340 km/h) is confirmed in official FIA text: 2026
Formula 1 Power Unit Technical Regulations, Issue 6 (29 March 2024), which shows the
revision from earlier values (1850 - 5v; a flat 150 kW above 340 km/h). The newest
Section C edition (Issue 12, 10 June 2025) could not be read as far as Article C5, so
later changes are not excluded. The 50 kW/s ramp limit remains secondary-source only.
