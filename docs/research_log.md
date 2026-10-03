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

## 2026-10-02: ML experiment, predictions recorded before any training run

Question: does a neural network that appears to learn battery energy from telemetry
learn physics, or a property of its training simulator?

Design: synthetic full-throttle straights (240 points, 5 m apart; speed observed with
0.8 km/h noise, rounded to whole km/h). Target: battery energy used on the straight (MJ).
Factor 1, cars: identical (nominal) vs varied (ICE +/-10%, drag area +/-20%, mass 768-838 kg).
Factor 2, strategies: simple (deploy fraction D of the cap until clip point s_c, then
harvest H) vs flexible (random smooth profiles within the regulation limits).
E1 identical/simple, E2 identical/flexible, E3 varied/simple, E4 varied/flexible;
T1 trains on E2 and tests on E4. Model: MLP with a Gaussian output (mean and standard
deviation), trained by negative log-likelihood. Metric: RMSE in MJ, plus coverage of
the 90% predictive interval.

Predictions:
P1. E1 and E2 (identical cars): RMSE below 0.10 MJ.
P2. E4 (varied cars, flexible strategies): RMSE at least 5x E2's and at least 0.30 MJ.
P3. E3 (varied cars, simple strategies): RMSE at most half of E4's; the restricted
    strategy family makes battery energy look learnable.
P4. T1 (trained on identical cars): RMSE at least 3x E2's, and 90% intervals cover the
    truth on fewer than 70% of laps (overconfident under car variation).
P5. In-distribution (E1-E4): 90% interval coverage between 85% and 95%; mean predicted
    standard deviation in E4 at least 3x that in E2.

Amendment before any training run: generated samples are filtered for realism in every
regime (|energy| <= 4 MJ, the battery window; speed >= 180 km/h). Checking target ranges
showed energy varies twice as much under flexible strategies (sd 1.80 MJ) as simple ones
(0.88 MJ), so comparing E3 and E4 RMSE directly would confound identifiability with target
variance. P3 is replaced by:
P3'. The increase in RMSE caused by car variation is at least twice as large for flexible
     strategies as for simple ones: (E4 / E2) >= 2 x (E3 / E1).

## 2026-10-02: ML experiment results

| Run | RMSE (MJ) | 90% coverage | Mean predicted sd (MJ) |
| --- | --- | --- | --- |
| E1 identical/simple | 0.008 | 88% | 0.007 |
| E2 identical/flexible | 0.016 | 90% | 0.017 |
| E3 varied/simple | 0.162 | 88% | 0.145 |
| E4 varied/flexible | 0.358 | 90% | 0.317 |
| T1 train E2, test E4 | 0.636 | 3% | 0.017 |

P1 held. P2 held (E4 = 22.6x E2). P3' NOT held (E4/E2 22.6 vs E3/E1 21.1): restricting
strategies to a 3-parameter family does not restore learnability; car variation, not
strategy flexibility, drives the ambiguity. Lesson: ratio criteria with near-zero
denominators (E1 = 0.008 MJ) are fragile; future criteria use absolute differences.
P4 held: trained on identical cars, the model is confidently wrong on varied cars
(predicted sd 0.017 MJ vs RMSE 0.636 MJ, 90% coverage 3%). P5 held: trained across car
variation, its uncertainty is calibrated (coverage 88-90%; sd 0.317 vs RMSE 0.358 in E4).

Single seed, one architecture. Next: 5 seeds and a capacity/data control for E4.

## 2026-10-03: Robustness and capacity control, predictions recorded before any run

Concern: the "flexible" strategies are themselves a restricted family (five smooth bumps
through tanh), so the network may extract car information from the family's structure.
For each test lap, the car-uncertainty ambiguity sigma_amb is computed: the standard
deviation of battery energy that would remain if speed revealed nothing about ICE power,
drag area or mass (first-order, uniform priors as in the generator).

R1. With 5 seeds, the verdicts of P1, P2, P4 and P5 are the same in every seed, and the
    seed-to-seed standard deviation of E4 RMSE is below 10% of its mean.
R2. E2 (identical cars): 4x training data and a 3x512 network reduce RMSE by at least 30%
    relative to the baseline (40k laps, 2x256).
R3. E4 (varied cars): the same changes reduce RMSE by less than 25%. If R2 and R3 both
    hold, the E4 error is information-limited, not model-limited.
R4. Baseline E4 RMSE divided by the mean sigma_amb lies between 0.3 and 0.9: the network
    resolves part, but not all, of the car-uncertainty ambiguity.

## 2026-10-03: Robustness results, a training bug, and a re-run

Results (5 seeds; capacity control 3 seeds): R1 HELD (verdicts identical in all seeds,
E4 seed sd 3.8% of mean). R2 HELD (E2 fell 43%, 0.0204 -> 0.0117 MJ). R3 HELD (E4 fell
11%, 0.3555 -> 0.3181 MJ). R4 HELD (0.355 / 0.635 = 0.56). Verdicts also hold for the
exact pre-registered configuration (160k laps, 3x512). Summary over seeds: E1 0.009,
E2 0.027, E3 0.159, E4 0.365, T1 0.633 MJ (T1 90% coverage 3-9%).

Anomaly: E2 with 40k laps and 3x512 gave 0.757 +/- 1.243 MJ; two seeds near 0.01 MJ and
one near 2.2 MJ, the error of an untrained network. Cause (diagnosed from the code, not
reproduced): if the validation loss becomes NaN, "v < best" is never true, so fit()
silently returned the randomly initialised weights; log-variance was unbounded above,
allowing exp() overflow. Fix: gradient clipping (global norm 1.0), log-variance bounded
in [-7, 4], and fit() raises an error if no finite validation loss occurs.

All ML experiments are re-run with the fixed training. Expectations recorded before the
re-run: no run diverges (E2 at 40k laps, 3x512 below 0.05 MJ in every seed), and every
verdict (P1, P2, P3', P4, P5, R1-R4) is unchanged.

## 2026-10-03: Re-run with stable training

No run diverged (E2 seed sd now about 0.0001 MJ). P1, P2, P4, P5 held; P3' not held.
R1 held (E4 seed sd 1.4% of mean). R2 NOT HELD: E2 fell 21% (0.0141 -> 0.0112 MJ), not
>= 30%. R3 held: E4 fell 10% (0.3492 -> 0.3133 MJ). R4 held: 0.349 / 0.635 = 0.55.
The expectation that every verdict would be unchanged was therefore wrong.

Lesson: the earlier R2 pass (43%) was partly an artefact of the training bug, which
inflated the E2 baseline (0.0204 vs 0.0141 MJ). With stable training, E2 is near the floor
set by whole-km/h rounding. The capacity contrast (21% vs 10%) is weaker than predicted,
so it does not by itself establish an information limit. What holds: at the largest
configuration, varied-car error remains 28x identical-car error (0.313 vs 0.011 MJ), and
the network resolves about half of the car ambiguity (ratio 0.55) at every scale.

## 2026-10-03: Lap time lost to clipping (method development)

Question for strategists: how much lap time does clipping cost? Method
(`src/erslens/laptime.py`): fit pre-clip power on full-throttle running below the speed
peak (electric power = d x regulation cap), for each ICE power (+/-10%), drag area
(+/-20%) and mass (+/-35 kg) on a grid; keep legal fits within 25% of the best residual;
simulate each admissible car without clipping from the start of the fit to the end of
full throttle; time loss = observed time minus counterfactual time.

Development on simulated laps (method choices made while looking at simulation results,
so these are development findings, not tests):
- First version started at the speed peak: biased low, because clipping starts before
  the peak. Changed to start where the fit starts.
- Apparent remaining bias (-0.07 s) was a validation error: truth was measured over the
  whole straight, the estimator over the telemetry span. Compared over the same span:
  median absolute error 0.023-0.033 s on a median loss of 0.72 s, bias -0.011 to +0.006 s,
  correlation 0.99-1.00 (2 seeds x 40 laps). 90th-percentile error 0.08-0.09 s.
- Within the d x cap model, the parameter-ambiguity range is only 2-3% of the estimate;
  noise dominates. Each lap's interval = parameter range +/- 0.09 s.
Contrast with Section 3: battery energy is ambiguous by up to 63% of the battery window,
but time lost to clipping is recoverable to about 5%, assuming pre-clip deployment
follows the shape of the regulation cap.

Predictions for real races, recorded before running:
L1. At every measurable circuit (>= 20 laps), median time lost per lap on the longest
    straight is between 0.05 and 0.50 s.
L2. Per-lap time loss correlates with the matched-speed power drop (Spearman >= 0.4).
L3. On real laps with loss > 0.05 s, the median parameter-ambiguity range is under 10%
    of the estimate.

## 2026-10-03: Lap-time loss, first real-data run and diagnosis

Validation (60 simulated laps): median abs error 0.033 s, 90th percentile 0.108 s.

Real races: plausible losses at four circuits (Australia 0.124, China 0.324, Miami 0.274,
Canada 0.251 s per lap), but two checks caught problems, so the pre-registered predictions
are recorded as not held: L1 (Austria 0.000 s, Belgium -0.127 s), L2 (Spearman 0.30 on
1,645 laps) and L3 (median ambiguity range 15.7%).

Root causes identified:
1. Austria: inconsistency with the Phase 2 detector, which bridges single-sample throttle
   noise (90-98%); the lap-time code did not, ending the full-throttle run before the speed
   peak (3.8% of laps flagged vs 82% by the detector).
2. Belgium: a negative loss is physically impossible, so the pre-clip model is invalid on
   Spa's long uphill straight, where counterfactual errors accumulate with distance.
3. Validation design: simulated laps shared the estimator's own assumptions (an "inverse
   crime"), so simulation accuracy overstated real-data accuracy. This mirrors the ML
   finding that in-model accuracy does not guarantee real-world validity.

Actions: one shared definition of full throttle across the project; laps with loss below
-0.09 s flagged as model failures and reported per circuit; validation on simulated laps
that deliberately violate the estimator's assumptions. Team comparisons are held back
until these are complete.

## 2026-10-03: Lap-time loss, first real-data run and diagnosis

Validation (60 simulated laps): median abs error 0.033 s, 90th percentile 0.108 s.

Real races: plausible losses at four circuits (Australia 0.124, China 0.324, Miami 0.274,
Canada 0.251 s per lap), but two checks caught problems, so the pre-registered predictions
are recorded as not held: L1 (Austria 0.000 s, Belgium -0.127 s), L2 (Spearman 0.30 on
1,645 laps) and L3 (median ambiguity range 15.7%).

Root causes identified:
1. Austria: the lap-time code defined full throttle differently from the Phase 2 detector
   (no single-sample noise bridging, first run rather than the run containing the peak,
   window stopping at the straight's end rather than 100 m beyond).
2. Belgium: a negative loss is physically impossible, so the pre-clip model is invalid on
   Spa's long uphill straight.
3. Validation design: simulated laps shared the estimator's own assumptions (an "inverse
   crime"), so simulation accuracy overstated real-data accuracy. This mirrors the ML
   finding that in-model accuracy does not guarantee real-world validity.

Fixes: full throttle now has one shared definition (src/erslens/throttle.py) used by the
detector and the lap-time code; laps with loss below -0.09 s are flagged as model failures,
and circuits with more than 20% failures are reported as "method not valid"; L1-L3 checks
after the fixes are printed as diagnostics, not tests.

Stress test outside the model family (scripts/laptime_stress.py, 20 laps per scenario):
in-model 0.028 s median error (interval covers 95%); constant pre-clip power 0.051 s, bias
+0.046 s (90%); gradual clipping 0.028 s (100%); 2% uphill 0.030 s (95%); all three
0.062 s, bias +0.062 s (80%). The elevation correction handles gradients; deployment shape
matters most; the method overestimates when its shape assumption is wrong. Realistic
accuracy is about +/-0.06 s per lap. Spa's failure is therefore not explained by gradient.
