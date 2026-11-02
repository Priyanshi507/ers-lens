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

## 2026-10-03: Austria explained (survivorship bias); consistency check

Diagnostic (scripts/laptime_diagnose.py): in Austria the lap-time code sees the speed drop
on 72% of clean laps (detector: 82%), but returns no result on those laps: the method needs
at least 4 s of full-throttle running above 210 km/h before the speed peak to fit pre-clip
power, and Austria's short uphill straight provides less. The real-race script dropped
these laps silently, so its 3.2% "clipping share" came from the surviving, mostly
non-clipping laps: survivorship bias, not a measurement. The throttle-definition fix was
not the cause.

Fix: unmeasurable laps are recorded, not dropped; a circuit is reported only if >= 50% of
its clean laps are measurable, >= 20 laps are measured, and <= 20% are model failures.

Consistency against the detector's clipping duration (a better check than power drop
alone, since time lost depends on both the size and the duration of clipping), Spearman on
clipping laps: Miami 0.88, China 0.87, Canada 0.77, Australia 0.54.

## 2026-10-03: Lap-time loss, final run (chapter closed)

With unmeasurable laps recorded: Australia 0.133 s (IQR 0.077-0.210), China 0.331 s
(0.199-0.460), Miami 0.277 s (0.180-0.496), Canada 0.250 s (0.165-0.338) per lap; 96-99% of
clean laps measurable, failures 0-0.5%. Not reported: Austria (30% of 959 clean laps
measurable), Japan (32% of 50), Belgium (61% model failures). Team comparisons remain
exploratory. README updated with the 5-seed ML results and this finding.

## 2026-10-05: Optimal energy deployment ("Strategy Lab"), predictions recorded before any code

Question: what is the lap-time-optimal way to spend and recover electric energy on a 2026
race lap under the regulations, and does it clip?

Hypothesis: near top speed, aerodynamic drag absorbs power roughly as v^3, so a joule deployed
there buys little speed, and a joule harvested there costs little time (approximately
proportional to 1/v^3). The optimum should therefore cut electric power, or harvest, at the
end of long straights. On this view super-clipping is a feature of optimal energy management,
not a failure of it.

Problem. One steady-state race lap (mass at mid-race fuel load), car and limits from
configs/car_2026.yaml: MGU-K power cap with the FIA speed taper, 4 MJ state-of-charge window,
per-lap harvest limit, efficiencies, braking envelope. The decision is the electric power at
every track point (deploy or harvest, braking harvest included). Constraint: state of charge
at the end of the lap is at least its value at the start, so no strategy can win by draining
the battery. Objective: lap time. Tracks: the reference laps of the four reported circuits
(Australia, China, Miami, Canada); Canada is used for development.

Methods. M1, dynamic programming over state of charge (exact on its grid; the standard
method in hybrid-vehicle energy management). M2, gradient descent through the JAX simulator,
with the end-of-lap energy constraint as a penalty. M1 is the reference; M2 is the fast method.
Clipping on the longest straight is measured as in the real-data analysis: electric power at the
speed peak relative to its mean over the full-throttle running before the peak.

Predictions:
S1. For the nominal car at each of the four circuits, the optimal strategy clips on the
    longest straight: electric power at the speed peak is at most 50% of its pre-peak mean.
S2. S1 holds for at least 80% of cars in the legal parameter sweep (ICE power +/-10%, drag
    area +/-20%, mass +/-35 kg; at least 27 cars) at every circuit.
S3. With the MGU-K speed taper removed (cap 350 kW at all speeds, everything else unchanged),
    the optimum still clips on the longest straight at at least 3 of the 4 circuits. If S3
    fails, clipping is explained by the regulation taper, not by energy economics.
S4. Electric energy is spent where it buys most time: the energy-weighted mean speed of
    deployment is at least 20 km/h below the distance-weighted mean speed of full-throttle
    running.
S5. The optimum beats the best tuned simple policy (flat-out, straights-only, and
    clip-at-end-of-straight, each tuned over its parameters) by less than 0.3 s per lap: most
    of the benefit is available to a simple rule.
S6. M1 and M2 agree: lap times within 0.02 s, and deployment profiles correlated at >= 0.9.
    If they disagree, no S1-S5 result is reported until the cause is found.
S7. Comparison with real cars (exploratory; the battery is not observable): the optimal
    speed-peak position on the longest straight, measured from the start of full throttle,
    lies within the interquartile range of observed speed-peak positions at at least 3 of the
    4 circuits. If it lies after the range, real cars clip earlier than the optimum, which is
    consistent with, but does not prove, energy-limited running.

Safeguards fixed now: optimizer correctness is tested on toy tracks with known optimal answers
before any circuit result is computed; results report the full sweep, not the best case; the
simple policies get the same tuning effort as described in S5, so the comparison is fair.

## 2026-10-07: Strategy Lab methods, status before any circuit result

M1 (dynamic programming) and M2 (gradient descent through the JAX simulator) built and tested
on toy tracks with known answers (free energy gives full deployment; no energy gives the
engine-only lap; strategies are energy-neutral and respect the harvest limit).

M2 from constant starting commands gets trapped in local optima on realistic tracks (2.6 s
behind M1 on the demo circuit). Causes found: hard battery limits have zero gradient (search now
relaxes them and penalizes leaving the window, with the real limits applied in the replay); a
pure penalty gives no reward for surplus energy (now an augmented Lagrangian with a learned
energy price); in corners a non-negative command has exactly zero gradient. M2 now starts from
the best tuned simple policy, which keeps it independent of M1.

S6 not met. Demo circuit lap times: M1 coarse grid 72.150 s, M1 fine grid 71.938 s, M1 fine
grid with finer power steps 72.063 s, M2 72.095 s. The DP does not converge monotonically under
grid refinement, so its lap times are currently uncertain by about 0.1-0.2 s, well above the
0.02 s agreement S6 requires. Per the pre-registration, no S1-S5 result is reported until the
cause is found.

## 2026-10-07: Method decision before any circuit result (deviation from the pre-registration)

DP convergence, demo circuit, harvest price fixed: refining only the speed grid stalls near
72.07 s because the charge grid then limits accuracy; refining both converges (speed 0.25 m/s
with 321 charge levels: 72.040 s; speed 0.125 m/s: 72.024 s, a change of 0.016 s), and the DP's
own predicted lap time then matches the simulator replay to within 0.002-0.017 s. The earlier
non-monotone results came from re-running the harvest-price bisection on each grid.

M2 does not reach the converged optimum: from the best tuned simple policy it reaches 72.095 s,
and longer or slower optimization makes it worse (72.130, 72.430, 73.031 s), so it is trapped
near its start, not short of iterations. S6 therefore fails, and the cause is identified: local
optima of gradient descent on a lap full of switches (deploy or harvest, accelerating or braking,
battery limits), which the DP avoids by searching every state.

Decision, made before any real-circuit result exists: all results use M1. Every M1 result
reports its own convergence check (predicted minus replayed lap time), and is not used if the
check exceeds 0.05 s. Nominal cars (S1, S3, S4, S5) use the fine grid (0.125 m/s, 321 levels);
the 27-car sweep (S2) uses 0.25 m/s with 161 levels, since S2 asks whether the optimum clips,
not for lap times to the hundredth. M2 is still run and its gap to M1 reported, so the failure
of S6 is shown on every circuit, not only the demo. Prediction thresholds are unchanged.

## 2026-10-07: Regulation check and prior work, before re-running (no corrected results exist yet)

Results committed at 73401ff stand as recorded. A check of published sources found three ways
the model's rules differ from the 2026 races:
1. Race harvest limit: the model uses 8.5 MJ per lap; the FIA confirmed 9 MJ in practice and the
   race at the opening events and Miami, with power to lower it to 7 MJ at some events. No
   source found for Canada: 9 MJ assumed, 8 MJ run as a sensitivity.
2. Superclip power (harvesting at full throttle): 250 kW before Miami, 350 kW from Miami (FIA
   statement). The model allowed up to the MGU-K cap at every race, so Australia and China
   were modelled with more harvesting power than the rules allowed.
3. From Miami, deployment is 350 kW in key acceleration zones and 250 kW elsewhere. The zones
   are not published; only the extreme case (250 kW everywhere) is run, as a bound.
Corrections live in configs/rules_2026_races.yaml with a source per value; car_2026.yaml is
unchanged so the committed results stay reproducible. The simulator gained superclip and
deployment caps whose defaults change nothing (tested). The corrected runs use the same
thresholds as S1-S7.

Context: the FIA's mid-season change explicitly targeted superclip duration (about 2-4 s per
lap). If the optimum clips under the corrected rules, the regulation is pushing cars away from
optimal energy use, which is why a rule was needed.

Prior work: time-optimal energy management of F1 hybrids is established (Ebbesen, Salazar,
Elbert, Bussi, Onder, IEEE TCST; Salazar et al., IEEE TCST 2017, analytical policy; van den
Eshof, de Vries, Salazar 2026, bang-bang optimal policy for energy-limited race cars). This work
does not claim the method. Its contributions are the 2026 rules (350 kW, 4 MJ window, speed
taper, no MGU-H), public data only with convergence checks, separating the taper's effect from
energy economics, and comparison with public telemetry. The bang-bang result suggests why S5
failed: the simple rule families deploy at a constant intermediate power, which a bang-bang
optimum does not; to be tested in the S5 diagnostic.

## 2026-10-10: Convergence rule enforced on every result; S5 and S7 diagnostics

Note on the log: the two entries titled "Lap-time loss, first real-data run and diagnosis"
(2026-10-03) are a draft and its revision; both are kept unedited.

The convergence rule (2026-10-07: an M1 result is not used if its DP prediction differs from its
simulator replay by more than 0.05 s) was enforced only for the four nominal runs. Applied to all
138 M1 results committed at 6e59c49, 6 fail: Miami rules_no_taper (0.099 s) and sweep cars
Australia i0d2m2, China i0d1m1, i1d1m2, i2d2m1, Miami i0d2m1 (0.055-0.173 s). Raising the DP
speed-grid ceiling was tested as a cause and ruled out (demo circuit, no taper: top speed
98.9 m/s, below the 101 m/s ceiling; raising it to 120 m/s changed the result by 0.006 s).

Decisions, made before any re-solve: the threshold lives in one place (erslens.strategy.
CONVERGED_S) and every verdict uses only converged results. A failing case is re-solved on
finer grids, (0.125 m/s, 321 levels), then (0.125, 641), then (0.0625, 321), stopping at the
first that converges; re-solves are appended as new rows and committed rows are never edited.
A case that never converges is excluded, listed in the verdicts, and counted against its
prediction (a sweep car as not clipping, a circuit as failing). With the 6 cases excluded on
that basis, verdicts are unchanged: S2 still holds (worst case China, 24 of 27 cars, 89%) and
S3 under the corrected rules is not held (2 of 4).

Diagnostics (exploratory, after the verdicts; scripts/strategy_diagnose.py):
- S7: part of the gap was the definition. Race laps reach full throttle later than the
  qualifying reference lap (runs 56-181 m shorter), which inflates the start-to-peak distance.
  Measured back from the braking point, which the corner fixes for both, real cars still reach
  their speed peak earlier than the optimum at every circuit: by 164 m (Australia), 132 m
  (China), 369 m (Miami) and 113 m (Canada), medians over 209-613 clipping laps per
  circuit. Real cars clip for longer than the optimum. Not causal: race laps include
  traffic, slipstream and overtake mode, and the battery is not observable.
- S5: the optimum is bang-bang. 96-99% of full-throttle points deploy at the MGU-K cap or at
  zero, although the optimizer could choose any level in 25 kW steps; this matches the
  bang-bang optimal policy of van den Eshof, de Vries and Salazar (2026). The simple rule
  families set one deployment level per lap; on the demo circuit the best was an intermediate
  275 kW. The levels chosen at the four real circuits were not stored, so they, and where on
  the lap the optimum gains its time, are computed with --tracks before any S5 explanation is
  claimed.

## 2026-10-15: The car model is too fast; calibrating it before anything is reported

Check (scripts/strategy_plausibility.py, after the final verdicts at 197f82b): at qualifying
fuel, charge-neutral, the optimum is faster than the real pole lap it was built from by 3.62 s
(Australia), 3.28 s (China), 2.22 s (Miami) and 2.58 s (Canada). Even the best simple rule
beats real pole at every circuit. A real qualifying lap may spend stored charge and the model
may not, so the model should be no faster than real pole. The car model is too generous.

Where (scripts/strategy_correlation.py): off full throttle the track is capped at the real
trace, so all of the gain is under full throttle, and 2.35-3.28 s of it is above 200 km/h
(0.11-0.35 s below). Top speeds roughly agree (model 338-340 km/h, real 327-343 km/h). The
model reaches high speed too quickly on every straight. The energy rules are respected (about
8 MJ deployed per lap, harvest at the 9 MJ limit), which leaves straight-mode drag (0.95, an
estimate) and ICE power at the wheels (400 kW, no drivetrain loss) as the untested inputs.

Consequence: every result so far used this car. The qualitative results (S1-S4, the bang-bang
optimum) may survive; the sizes of gaps (S5: 1.7-2.0 s, S7 distances) cannot be reported
until the car is calibrated.

Decisions, made before the calibration runs (scripts/calibrate_car.py):
- Fit two parameters, straight-mode CdA (0.95-1.45, step 0.10) and ICE wheel power
  (320-400 kW, step 20 kW), by grid search. Target: sum of squared time errors on every
  full-throttle run of at least 100 m of the four pole laps, optimum at qualifying fuel,
  coarse DP grid (0.5 m/s, 81 charge levels).
- Assumption, stated wherever the calibrated car is used: the pole lap is close to
  energy-optimal, because its real deployment is not observable.
- Check: leave one circuit out. The fit on three circuits is applied to the fourth; the
  calibration is accepted if every held-out lap error is within 0.5 s. If the best fit lies on
  the edge of the grid, the grid is widened before anything else.
- Then the Strategy Lab is re-run with the calibrated car, with the same predictions and
  thresholds, reported next to the original results. Nothing committed is edited.

## 2026-10-15 (later): First calibration grid; drag and power are not separately identifiable

First grid (6 CdA x 5 ICE powers, 120 coarse solves, results/calibration/grid.csv): the lowest
total squared run error, 1.86 s^2, is at CdA 1.05 and 320 kW ICE wheel power, against 7.27 s^2
for the current car (0.95, 400 kW). Leave one circuit out, the fit on three circuits lands
within 0.10 s (Australia), 0.39 s (China), 0.32 s (Miami) and 0.07 s (Canada) of the held-out
pole lap, against 2.2-3.2 s for the current car, which passes the 0.5 s criterion. But the
best point is on the edge of the grid (320 kW), so by the rule logged earlier the grid is
widened before the fit is used.

The error surface is a ridge, not a bowl. The best power for each drag value: CdA 1.05 at
320 kW (1.86 s^2), 1.15 at 340 kW (1.99), 1.25 at 360 kW (2.15), 1.35 at 400 kW (2.33). Pole
lap times fix a combination of drag and wheel power, not each one. More drag with more power
gives the same lap as less of both.

Decisions, made before the widened grid and before any Strategy Lab re-run:
- Widen the grid to CdA 0.95-1.15 in steps of 0.05 and ICE wheel power 280-340 kW in steps
  of 10 kW. The calibrated car is the lowest-error point over the combined grid, accepted if
  it is not on the grid edge and every leave-one-out lap error is within 0.5 s.
- Because the split along the ridge cannot be measured, the Strategy Lab is re-run at two
  cars: the calibrated car, and a high-drag car on the ridge, CdA 1.35 with its best power
  from the first grid (400 kW). A result is reported as robust only if its verdict is the
  same for both cars; otherwise both are reported and the dependence is stated.

## 2026-10-17: Widened calibration grid; the acceptance rule is not met, and why

Widened grid (CdA 0.95-1.15 by 0.05, ICE wheel power 280-340 kW by 10 kW; 116 more coarse
solves, results/calibration/grid.csv). The lowest total squared run error is now 1.79 s^2 at
CdA 0.95 and 300 kW, again on the grid edge. Along the ridge the error barely changes: best
power per drag value 0.95 at 300 kW (1.79), 1.00 at 310 (1.82), 1.05 at 320 (1.86), 1.10 at
330 (1.90), 1.15 at 340 (1.99), 1.25 at 360 (2.15), 1.35 at 400 (2.33).

Both parts of the acceptance rule logged on 2026-10-15 fail:
- The best point is on the grid edge (CdA 0.95).
- Leave one circuit out, two held-out laps are outside 0.5 s: China -0.73 s and Miami +0.61 s
  (Australia -0.15 s, Canada +0.34 s). On the first, coarser grid all four passed; with more
  points on the flat ridge the fit picks slightly different points each time.

Reading: the circuits disagree with each other by more than the fit can resolve. At any one
ridge point the four lap errors spread by about 0.7 s (at CdA 1.05, 320 kW: China -0.39 s,
Miami +0.32 s). A single two-parameter car cannot match all four pole laps closer than that.
Likely causes, none tested: different teams set the four poles, wind and track grip differ,
and the pole laps need not be energy-optimal. Widening the grid again would chase a minimum
that the data do not determine, so the grid is not widened further.

Decisions, made before any Strategy Lab re-run:
- The calibration is reported as failing the rule as written, with the reason above. The
  calibration uncertainty is stated as about +/-0.7 s per lap (the between-circuit spread).
- Instead of one calibrated car, the Strategy Lab is re-run at both ends of the ridge, so the
  result shows what depends on the drag/power split:
  low-drag end, CdA 0.95 and 300 kW (configs/car_2026_cal_low.yaml), the lowest-error point;
  high-drag end, CdA 1.35 and 400 kW (configs/car_2026_cal_high.yaml), as logged on
  2026-10-15. Both reproduce real pole within about 0.7 s, against 2.2-3.6 s for the
  original car.
- Same predictions and thresholds as the 2026-10-05 entry. Stages run: nominal, no taper,
  the gradient method, simple rules, the legal-car sweep, and the corrected 2026 race rules.
  The starting-charge check is skipped (it changed laps by at most 0.17 s before).
- A verdict is called robust only if it is the same for the original car and both
  calibrated cars. Results go to results/strategy_cal_low and results/strategy_cal_high;
  results/strategy is not changed.

## 2026-10-18: The energy rules used so far were wrong; correction logged before any re-run

A re-check of the FIA's per-session recharge limits (sources in configs/rules_2026_races.yaml)
found two errors in the rules used since 2026-10-07:
1. Race limits. The model used 9 MJ per lap at every race. That is the limit with Overtake
   mode active; without it the limits were 8 MJ (Australia), 8.5 MJ (China), 8.5 MJ (Miami)
   and 8 MJ (Canada). The source cited for 9 MJ described the Overtake limit.
2. Qualifying limits. The calibration, plausibility and correlation scripts fit pole laps but
   applied the race rules. Qualifying limits were 7 MJ (Australia), 9 MJ (China), 8 MJ
   (Miami) and 6 MJ (Canada). The 2026-10-15 check found the optimum harvesting at its limit,
   so the model car had 1-3 MJ per lap more energy than the real car at three of the four
   pole laps. Part of "the model car is too fast" may be this, not drag or engine power.

Everything committed stands as recorded and is not edited. What it affects:
- Calibration (2026-10-15, 2026-10-17) and both calibrated cars: fitted under the wrong
  limits, so they are superseded once the re-run below exists.
- Strategy Lab corrected-rules stage (all three cars): the "rules" rows used 9 MJ. The
  existing "rules_h8" rows (8 MJ) are the correct limit for Australia and Canada; for China
  and Miami the correct 8.5 MJ lies between "rules_h8" and "rules". No verdict from this stage
  is reported until it is re-run.
- Unaffected: findings 1-6 in the README (they use no recharge limit), and every stage run
  with configs/car_2026.yaml alone (nominal, no taper, gradient, simple rules, sweep).

Fix: race limits corrected in place (git history keeps the old values); a qualifying limit per
event added; erslens.strategy.quali_rules applies it; the three pole-lap scripts use it and
take --out so earlier results are not overwritten. Overtake mode (+0.5 MJ while active) is
still not modelled; race values are without it.

Decisions, made before any re-run:
- Check first: strategy_plausibility.py with the qualifying limits. China's limit did not
  change, so China must reproduce its 2026-10-15 value (optimum 3.28 s faster than pole); any
  other value means a code error, and nothing else runs until it is found.
- Re-run the calibration grid with the qualifying limits, same points as before (both grids),
  into results/calibration_q. Same fit target and the same acceptance rule as 2026-10-15
  (best point not on the grid edge, every leave-one-out lap within 0.5 s). If the best point
  is on an edge, the grid is widened once in that direction, and the result is reported
  whichever way it falls.
- Then the Strategy Lab is re-run at both ends of the new ridge with the corrected race rules,
  same predictions and thresholds, into new folders. A verdict is called robust only if it is
  the same for the original car and both new calibrated cars.

## 2026-10-19: Plausibility check with the qualifying limits passes; the energy error was small

Before the patch was applied, the --refine pass and strategy_evaluate.py finished for both
2026-10-17 calibrated cars (results/strategy_cal_low, results/strategy_cal_high). Those cars
were fitted under the wrong limits, so their verdicts are kept as a record and not reported.

Check (scripts/strategy_plausibility.py --out results/strategy_q): China, whose limit did not
change, reproduces its 2026-10-15 value exactly (optimum 88.780 s, -3.284 s), so the code
change did nothing else. With the qualifying limits the optimum is slower by 0.131 s
(Australia, 7 MJ), 0.075 s (Miami, 8 MJ) and 0.629 s (Canada, 6 MJ), as expected from a smaller
feasible set. It is still faster than real pole at every circuit, by 3.484 s, 3.284 s,
2.140 s and 1.946 s.

Reading: the wrong recharge limits explain at most 0.63 s of the 2-3.5 s gap, and under 0.15 s
at three circuits. The rest still points at drag and ICE wheel power, as on 2026-10-15. The
calibration is re-run as logged on 2026-10-18, into results/calibration_q, with the same grid
points and acceptance rule.

## 2026-10-20: Noise in the coarse calibration solves; a check added while the grid runs

Written while the widened part of results/calibration_q is still running. The first grid has
finished (best point CdA 1.05, 320 kW, 1.71 s^2, on the grid edge, so the widened part runs
as logged). Some coarse-grid lap errors are not monotone in ICE power, which they should be:
Canada at CdA 1.05 gives +1.170 s at 300 kW, +0.062 s at 310 kW and +0.586 s at 320 kW, and
at CdA 1.10 +1.421 s, +0.312 s and +0.857 s. Calibration solves use the coarse DP grid
(0.5 m/s, 81 charge levels) and, unlike the Strategy Lab, have no convergence check, so
errors of this size may be solver noise rather than physics. That could also explain why the
leave-one-out picks moved between grids on 2026-10-17.

Decision, made before the final calibration report: after the grid finishes, the best point
for each CdA and the leave-one-out picks are re-solved on the fine grid (0.125 m/s, 321 charge
levels). The calibration is judged on the fine-grid errors, with the same acceptance rule. The
coarse-grid report is kept next to it.

## 2026-10-20 (later): Coarse calibration under qualifying limits; fine-grid check next

Coarse grid finished (results/calibration_q, 236 points, qualifying limits). Lowest total
squared run error 1.66 s^2 at CdA 1.00 and 310 kW, not on the grid edge. The ridge is flatter
than under the race limits: best power per drag value 0.95 at 300 kW (1.67), 1.00 at 310
(1.66), 1.05 at 320 (1.71), 1.10 at 340 (1.75), 1.15 at 360 (1.86), 1.25 at 380 (1.90), 1.35 at
400 (1.97). Leave one circuit out: Australia -0.199 s, China -1.071 s, Miami +0.820 s, Canada
+0.918 s, so three of four held-out laps are outside 0.5 s and the rule fails on the coarse
grid. Not judged yet: per the entry above, the verdict uses the fine-grid re-solve
(scripts/calibrate_fine.py, candidates taken from the coarse grid only, results in
results/calibration_q/fine.csv and fine.md).

## 2026-10-22: Fine-grid calibration fails the acceptance rule; the circuits disagree

Fine-grid re-solve of the 11 coarse-grid candidates (scripts/calibrate_fine.py,
results/calibration_q/fine.md). Lowest total squared run error 1.60 s^2 at CdA 1.00 and
310 kW, not the lowest or highest candidate drag. The ridge stays flat: 1.60-2.10 s^2 from
CdA 0.95 to 1.45. Fine and coarse totals differ by up to 0.24 s^2, which moves the ranking
along the ridge but does not remove it. Two candidates miss the 0.05 s convergence rule
(0.95/300 kW at Australia, 0.092 s; 1.15/360 kW at Australia, 0.064 s); all others converge.

Leave one circuit out on the fine grid, all four held-out laps are outside 0.5 s: Australia
-0.561 s (fit 0.95/300 kW, the non-converged Australia solve, so uncertain by about 0.09 s),
China -0.642 s, Miami +0.552 s, Canada +0.717 s (all converged). The rule fails on three
converged held-out laps whatever the Australia value is, so the calibration fails as written.

Reading: the coarse-grid noise logged on 2026-10-20 was real but was not the cause. The
circuits disagree with each other. At every candidate China's model lap is faster than
real (-0.30 to -1.28 s); it is closest at 0.95/290 kW and 1.00/300 kW (-0.30 s) and among
the furthest at CdA 1.15-1.35 (-1.21 to -1.23 s). Miami and Canada go the other way. At the
low-drag candidates 0.95/290, 0.95/300, 1.00/300, 1.00/310 and 1.05/320 kW they are slower
than real (+0.13 to +0.72 s); at 1.15/360, 1.25/380 and 1.35/400 kW they are close to real
(-0.21 to +0.18 s). So the circuits prefer different points on the ridge,
not one point with scatter. Possible causes, none tested: different teams set the four poles,
wind and grip, pole laps that are not energy-optimal, and speed-dependent physics the model
lacks (the two circuits that prefer more drag are the ones with the longest full-throttle
running). This is recorded as an observation only.

Decisions, made before any Strategy Lab run with these cars:
- The calibration is reported as failing the acceptance rule, with a between-circuit
  uncertainty of about +/-0.7 s per lap, as on 2026-10-17.
- The Strategy Lab is re-run at both ends of the new ridge, with the corrected race rules
  (configs/rules_2026_races.yaml as of 2026-10-18): the lowest-error fine point, CdA 1.00 and
  310 kW (configs/car_2026_q_low.yaml), and the same high-drag point as before, CdA 1.35 and
  400 kW (configs/car_2026_q_high.yaml, fine error 2.02 s^2, converged), so the high end stays
  comparable with the 2026-10-17 runs.
- Same stages, predictions and thresholds as the 2026-10-17 entry (starting-charge check
  skipped), then --refine for results that fail the convergence rule, into
  results/strategy_q_low and results/strategy_q_high. A verdict is called robust only if it is
  the same for the original car and both new calibrated cars.
- The 2026-10-17 cars' runs (results/strategy_cal_low, results/strategy_cal_high) stay as a
  record of cars fitted under the wrong limits and are not used for any verdict.

## 2026-10-28: Strategy Lab at both ends of the new ridge; which verdicts are robust

Both runs finished as logged on 2026-10-22: results/strategy_q_low (CdA 1.00, 310 kW) and
results/strategy_q_high (CdA 1.35, 400 kW), with --refine. Convergence rule: low end 129 of
130 M1 results (5 after refinement; China sweep i2d0m0 excluded), high end 128 of 130 (2 after
refinement; China sweep i1d0m0 and China rules_no_taper excluded). No verdict below depends on
the excluded results: the high-end S2 fails on Australia (18 of 27), where nothing is
excluded, and the high-end S3 under race rules is at most 2 of 4 even if China clips.

Same stages, predictions and thresholds as 2026-10-05. Original car from results/strategy.

| Prediction | Original car | Low end | High end | Robust? |
|---|---|---|---|---|
| S1 nominal optimum clips | held | held | held | yes, held |
| S2 clips in >= 80% of sweep cars | held (93-100%) | held (81-100%) | not held (Australia 67%) | no |
| S3 still clips without the taper | held (3 of 4) | held (4 of 4) | held (3 of 4) | yes, held |
| S4 deployment >= 20 km/h below full-throttle speed | held (26-36) | held (21-31) | not held (16-24) | no |
| S5 optimum within 0.3 s of best simple rule | not held (1.67-2.03 s) | not held (2.82-3.67 s) | not held (2.17-2.67 s) | yes, not held |
| S6 M1 and M2 agree | not held | not held | not held | yes, not held (cause logged 2026-10-07) |
| S7 peak position inside observed IQR | 0 of 4 | 0 of 4 | 0 of 4 | yes, not held |

S7 detail: in all three cars the optimal speed peak lies after the observed range at
Australia, China and Miami. At Canada it lies after the range for the original and high-end
cars (715 m, 610 m) and before it for the low-end car (510 m; IQR 521-584 m).

Per-race 2026 rules (limits as corrected on 2026-10-18), new cars only:

| Prediction | Low end | High end |
|---|---|---|
| S1 | held | held |
| S3 | held (4 of 4) | not held (1 of 4, China excluded) |
| S4 | held (21-30 km/h) | not held (16-25 km/h) |

The original car's per-race rows in results/strategy were run with the old 9 MJ limits
(2026-10-18 entry), so they cannot be used here. S3 and S4 under race rules already differ
between the two ends, so they are not robust whatever the original car gives. Only S1 under
race rules still waits on the original car.

Reading: at every point tested, the optimal strategy clips on the longest straight, and with
the car-file energy limit it still clips without the MGU-K taper. How widespread clipping is
across legal cars (S2), and how far below full-throttle speed energy is spent (S4), depend on
the drag/power split: the high-drag, high-power end clips less and deploys closer to top speed.
Under the per-race limits the high-drag end clips without the taper at only one circuit, so
"clipping is not just the taper" is robust only with the car-file limit. Why the high-drag end
behaves this way is not tested. The negative results (S5, S6, S7) hold for every car.

Decisions, made before the run below:
- Re-run the original car's per-race rules stage with the 2026-10-18 limits into
  results/strategy_rules_q, starting from a copy of its other rows (unaffected by the
  correction, per 2026-10-18), then --refine and strategy_evaluate.py. The copied verdicts
  (S1-S7) must reproduce results/strategy/verdicts.md exactly; any difference is a code error
  and nothing is reported until it is found.
- strategy_evaluate.py no longer says the per-race rules were "logged 2026-10-07"; the label now
  points to this log. Output only; no numbers change.
- Nothing above is reported as a finding until that run is in.

## 2026-10-28 (later): Original car under the corrected per-race rules; final robustness verdicts

Run as logged above (results/strategy_rules_q: the original car's other rows copied from
results/strategy, then --rules with the 2026-10-18 limits, then --refine). Check passed: the
copied verdicts S1-S7 are identical to results/strategy/verdicts.md, so the copy and the code
are unchanged. All 14 new per-race results converge (largest gap 0.015 s); none needed
refinement.

Original car, per-race rules: S1 held (clip ratio 0.00 at all four circuits), S3 held (3 of 4;
Australia 0.51 does not clip, just over the 0.50 threshold), S4 held (26-35 km/h). Under the old
9 MJ rows S3 had been 2 of 4, so the correction moved this verdict.

Final verdicts under the per-race rules:

| Prediction | Original car | Low end | High end | Robust? |
|---|---|---|---|---|
| S1 | held | held | held | yes, held |
| S3 | held (3 of 4) | held (4 of 4) | not held (1 of 4) | no |
| S4 | held (26-35 km/h) | held (21-30 km/h) | not held (16-25 km/h) | no |

Summary of the Strategy Lab, all three cars (this entry and the one above):
- Robust, held: the optimal strategy clips on the longest straight (S1), with the car-file
  limit and with the per-race limits; with the car-file limit it still clips without the
  MGU-K taper (S3).
- Robust, not held: the optimum beats the best tuned simple rule by 1.7-3.7 s, not under 0.3 s
  (S5); the gradient method does not reach the DP optimum (S6); the optimal speed peak is
  never inside the observed range, and lies after it at Australia, China and Miami for every
  car (S7, exploratory).
- Depends on the drag/power split, so not reported as a finding: clipping across the legal
  sweep (S2), the deployment speed gap (S4), and clipping without the taper under the per-race
  limits (S3). In each case the high-drag, high-power end is the one that fails.
These verdicts are what the Strategy Lab reports. The calibration itself failed its acceptance
rule (2026-10-22), so lap times from these cars carry the between-circuit uncertainty of about
+/-0.7 s per lap.

## 2026-10-29: Strategy Lab page; two diagnostics re-checked on all three cars (exploratory)

Added the Strategy Lab page to the Explorer (site/strategy.html, data from
scripts/export_strategy.py). The exporter recomputes every number from the saved optimal
profiles and stops if any of the 24 profiles (3 cars, 4 circuits, 2 energy limits) fails to
reproduce its runs.csv lap time and end-of-lap charge to 0.001 s and 0.001 MJ; all do. For the
original car it also reproduces the 2026-10-10 diagnostics (speed peak to braking point 290,
405, 200, 315 m; real medians from 209, 229, 464, 613 clipping laps; bang-bang shares), checked
in tests/test_export_strategy.py.

The two 2026-10-10 diagnostics were run on the original car only. Re-checked on all three cars,
after every verdict was recorded:
- Bang-bang: on every car and both energy limits, 0.7-4.7% of full-throttle points sit between
  zero and the MGU-K cap, so 95-99% are at one or the other. Same result for every car.
- Speed peak to braking point against the middle half of real clipping laps (lined up at the
  braking point, as on 2026-10-10). Original car: shorter at all four circuits with either
  limit, as logged. Low-drag end: inside at Australia and China, shorter at Miami, longer at
  Canada (car-file limit); inside at China only with the per-race limits. High-drag end: inside
  at China and Canada (car-file limit), and also at Australia with the per-race limits; shorter
  at Miami. Only Miami is shorter for every car. "Real cars clip for longer than the optimum"
  (2026-10-10) is therefore a property of the original car, not a robust result; the README
  sentence is corrected.
- Battery at the speed peak: within 0.3 MJ of empty at Australia, Miami and Canada for every
  car, but 1.0-1.2 MJ at China, where the optimum still cuts power. At China the cut is a choice
  about where energy buys most time, not a shortage. Recorded as an observation only; not
  pre-registered and not tested.

## 2026-10-30: Out-of-sample test on every other 2026 race, predictions recorded before any data

Findings 3 and 6 used seven races (Australia, China, Japan, Miami, Canada, Austria, Belgium).
The reason for that choice was never recorded, and four other races had already happened
(Monaco, Barcelona, Britain, Hungary). A reader cannot rule out that the seven were picked
because they worked. So both findings are now tested on every 2026 race not yet analysed:
Monaco, Barcelona, Britain, Hungary, Netherlands, Italy, Spain (Madrid), Azerbaijan, Bahrain,
Singapore and the United States (11 races, all held before this entry). No data from these
races has been loaded for findings 3 or 6. Italy was used once before, for the held-out
deployment test in finding 7; that test does not involve the quantities below.

Everything is unchanged from the original runs: the detector, the quality checks, the
longest-interior-straight rule, the reporting rule for time lost (at least 20 measured laps,
at most 20% model failures, at least 50% of clean laps measurable), and the car parameters.
Results go to new folders (results/multi_season, results/laptime_season); the committed
seven-race results are not edited. Every race run is listed with its outcome, including
races the method cannot measure.

Predictions:
E1. At every new race with at least 20 clean laps, at least 80% of clean laps clip by at
    least 3 km/h (the seven-race range was 82-98%).
E2. With all races, before against after the Miami changes, the normalised clipping share
    still shows no significant difference (exact permutation test, races as the unit,
    p >= 0.05), as in finding 3.
E3. At every new race that passes the reporting rule, the median time lost per lap is
    0.05-0.50 s (the range met by the four reported circuits).

No prediction is made about how many new races pass the reporting rule. A race failing E1 or
E3 is reported as a failure of the finding, not dropped. The README and the site are updated
only after the results are logged here.

## 2026-10-30 (later): First attempt at the season test could not download the new races

The first run stopped at Monaco: every FastF1 request for the new races failed (session info,
timing, car and position data), while the seven original races loaded from the local cache.
No new-race data was analysed. multi_race.py and laptime_real.py now record a race whose data
cannot be loaded in not_loaded.csv and continue with the others, instead of stopping the run.
A race in not_loaded.csv is a download failure, not a result, and is re-run before E1-E3 are
judged. Nothing about the analysis changed.

## 2026-10-30 (later): Why the downloads failed, and a second data source checked before use

The failures were not local: since about 15 October 2026 the F1 live-timing archive that FastF1
reads (livetiming.formula1.com/static) answers HTTP 401 without an F1 TV subscription (FastF1
issue #978, open, no fix). Confirmed here: that URL returns 401 while other sites load.

OpenF1 (api.openf1.org) serves its own recording of the same timing feed, free for 2023 onward.
erslens.openf1 rebuilds the ingest tables from it (selected with ERSLENS_SOURCE=openf1), following
FastF1's conventions: lap telemetry interpolated to the exact lap start and end, distance
integrated from speed as FastF1 does, position in tenths of a metre, the reference lap the
fastest qualifying lap not deleted by race control. Nothing in the analysis changes.

Check recorded before any OpenF1 run: the seven original races are re-run from OpenF1 into
results/source_check and compared with the committed FastF1 results (scripts/compare_sources.py).
OpenF1 is used for the season test only if, at every one of the seven races:
- clean laps within 10% of FastF1's count;
- share of laps clipping within 0.03, and normalised clipping share within 0.02;
- median speed lost within 1 km/h;
- share of measurable laps within 0.05, the same reporting decision, and, at reported races,
  median time lost within 0.02 s.
If any check fails, the cause is found before OpenF1 data is used, and the failure is logged here.

## 2026-10-30 (later): First source check ran on one race with broken lap data; fixed before re-running

The first OpenF1 run of the source check analysed only Australia: the shell passed the seven
race names as one argument, which FastF1 read as Australia. Australia gave 4 clean laps against
FastF1's 227, so the check failed as it should. Cause: OpenF1's lap table for the 2026
Australian race holds 45 of the race's laps (Jolpica times 1,003). Every other 2026 race has
a complete OpenF1 lap table (Jolpica and OpenF1 counts within 1% from China to Bahrain), the
pole lap is present in every OpenF1 qualifying table, and car data is complete, including
Australia's (31,180 samples for one driver).

Fix, made before any other OpenF1 result exists: when OpenF1's lap table holds less than 90%
of the laps Jolpica times, lap starts are rebuilt from Jolpica's lap times (lights-out from
OpenF1, then each driver's cumulative lap times; pit laps from Jolpica). This applies to
Australia only. Timeouts from OpenF1 are now retried. The comparison rule recorded above is
unchanged, and the source check is re-run on all seven races.

## 2026-10-30 (later): Source check result; OpenF1 used only with its own lap table

Result (results/source_check/comparison.md): 45 of 46 checks passed. At the six races where
OpenF1's own lap table was used (China, Japan, Miami, Canada, Austria, Belgium) every check
passed, most by a wide margin: clean laps within 2% (Austria 975 against 959, the rest within 3
laps), share of laps clipping within 0.015, normalised clipping share within 0.004, median speed
lost equal or 1 km/h apart, median time lost within 0.003 s at the three reported races, and the
same reporting decision at all six. The before/after comparison gives 34.2% against 34.2%,
p = 0.571 (FastF1: 34.7% against 34.1%, p = 0.51).

The one failure is Australia, the only race whose laps were rebuilt from Jolpica lap times:
148 clean laps against FastF1's 227 (35% fewer). Its other checks passed (clipping share 0.959
against 0.974, median time lost 0.115 against 0.133 s), so the laps that survive look right, but
the rebuild loses too many laps. The cause is not investigated here.

Decision, made before any data from the 11 new races is loaded, and a deviation from the rule
recorded above ("at every one of the seven races"): the Jolpica rebuild is removed. OpenF1 is
used only with its own lap table, the method that passed every check at all six races where it
was used. A race whose OpenF1 lap table holds less than 90% of the laps Jolpica times is listed
in not_loaded.csv and reported as not loaded. The probe of 2026-10-30 found complete OpenF1 lap
tables for every new race from Monaco to Bahrain; Singapore and the United States were not
reached by the probe and are checked by this rule when they run. E1-E3 are unchanged.

## 2026-10-31: Season test results; E1 and E3 not held

Run as logged (OpenF1, own lap tables; results/multi_season, results/laptime_season). All 11
new races loaded. Australia was not loaded (incomplete OpenF1 lap table), as the rule says.

| Race | Measured straight | Clean laps | Laps clipping >= 3 km/h | Share of straight clipping | Time lost per lap (median) |
| --- | --- | --- | --- | --- | --- |
| Monaco | 510 m | 960 | 4% | 1% | 0.00 s |
| Barcelona | 415 m | 633 | 5% | 0% | 0.00 s |
| Britain | 1535 m | 18 | (fewer than 20 clean laps) | | not reported |
| Hungary | 405 m | 0 | (no clean laps) | | not run |
| Netherlands | 675 m | 4 | (fewer than 20 clean laps) | | not reported |
| Italy | 1240 m | 539 | 54% | 5% | not reported (24% model failures) |
| Spain (Madrid) | 865 m | 636 | 92% | 19% | 0.04 s |
| Azerbaijan | 680 m | 582 | 1% | 0% | 0.00 s |
| Bahrain | 700 m | 510 | 23% | 6% | 0.00 s |
| Singapore | 590 m | 271 | 0% | 0% | 0.00 s |
| United States | 975 m | 700 | 93% | 32% | 0.06 s |

- E1 (at least 80% of clean laps clip at every new race with at least 20 clean laps): NOT HELD.
  It holds at 2 of 8 races (Spain 92%, United States 93%); the other six range from 0% to 54%.
- E2 (no significant before/after difference with all races): held as written, p = 0.121
  (35.0% of the straight before, 2 races; 16.7% after, 12 races). With FastF1's Australia added
  as a sensitivity check, p = 0.062, still held. Not informative: every race with almost no
  clipping is after Miami, so the comparison is confounded by circuit and says nothing about
  the rule changes.
- E3 (median time lost 0.05-0.50 s at every new race that passes the reporting rule): NOT
  HELD. It holds at 1 of 7 (United States 0.06 s); Spain is 0.04 s and the other five 0.00 s.

So finding 3 ("clipping is near-universal") and finding 6's range do not survive outside the
original seven races. The seven were not a fair sample of the calendar.

Exploratory, after the verdicts and not tested: the share of laps clipping rises with the
length of the measured straight (Spearman 0.84 over the 14 races with at least 20 clean laps).
Every measured straight of 850 m or more clips on at least 54% of laps (6 of 7 above 90%);
below 850 m only Japan (85%) and Austria (82%) clip widely. Several circuits' measured straight
is short because the method skips any straight that touches the start/finish line (lap distance
resets there), which at Baku, Bahrain and Barcelona is the main straight. This is consistent
with clipping being a long-straight effect, and with the method being unable to see it where
the longest straight crosses the line; it is not shown. Britain, Hungary and the Netherlands
lost almost every lap to the quality checks; the reasons were not examined.

README findings 3, 6 and 7 are updated with these results.

## 2026-10-31: Strategy Lab on GPU, step 1: benchmark plan and predictions, before any timing

Aim: find out how much faster the existing Strategy Lab runs on a GPU, with no change to the
solver, before building anything on top of it (whole-race strategy needs many more solves).
scripts/gpu_benchmark.py times, for the low-drag calibrated car (configs/car_2026_q_low.yaml)
at Australia, China, Miami and Canada:
- one backward pass of the dynamic program on the fine grid (0.125 m/s, 321 charge levels),
  after compilation, median of 3 runs;
- one complete solve_refined call, compilation included, exactly as the Strategy Lab runs it.
Tracks are the cached qualifying reference laps saved by scripts/export_tracks.py, so the GPU
machine (a free Kaggle notebook) solves the same circuits.

Checks, before any speed-up is reported:
- The CPU run reproduces the committed results/strategy_q_low/runs.csv nominal lap times to
  0.001 s (Australia 77.581, China 91.876, Miami 88.244, Canada 72.528 s), so the saved tracks
  are the ones every result used.
- GPU and CPU lap times agree to 0.001 s at every circuit. If not, no speed-up is reported until
  the cause is found.
- The solver runs in float64 throughout; the GPU model is recorded, because some GPUs (Kaggle's
  T4) run float64 at a small fraction of their float32 speed. The P100 is used where available.

Predictions (CPU: Apple M4 MacBook Air, JAX CPU backend):
G1. On the GPU, the fine-grid backward pass is at least 10 times faster than on the CPU at
    every circuit.
G2. A complete solve, compilation included, is at least 3 times faster on the GPU at every
    circuit (compilation and the many small host-side steps dilute the gain).
Whatever the numbers, they are reported, and step 2 (batching with vmap) is planned from them.

### 2026-10-31: CPU run on the M4, before any GPU timing

The track check passed. All four CPU lap times reproduce results/strategy_q_low/runs.csv to 0.001 s
(Australia 77.581, China 91.876, Miami 88.244, Canada 72.528 s), so the saved tracks are the ones
every Strategy Lab result used.

The backward-pass times are not consistent with each other, though. The grid is the same at every
circuit (grid_with depends only on the car), so the cost should scale with the number of track
points. Australia, China and Miami take 0.088-0.089 s per point; Canada takes 0.028 s per point, and
Australia's first call (0.065 s per point) was faster than its later runs. The likely cause is the
machine, not the solver: the M4 MacBook Air has no fan and slows down when hot, and other work on the
laptop competes for it. So cpu_m4.json is kept as recorded, and a second CPU run is added before any
GPU number is seen: backward pass only (--skip-solve), 5 repeats, on mains power, Low Power Mode off,
nothing else open, saved as cpu_m4_quiet.json. G1 is judged against both CPU files and both results
are reported; if they disagree on whether G1 held, that is stated. G2 uses cpu_m4.json, the only run
with full solves.

### 2026-10-31: GPU benchmark results and verdicts

GPU: the free Kaggle notebook offered only a Tesla T4 (no P100), JAX 0.11.2, one of its two GPUs
used. Lap times agree with the CPU to 0.001 s at all four circuits, so the speed-ups below count.

| Circuit | Backward pass: M4 first run / M4 quiet run / T4 | Speed-up vs first / quiet | Full solve: M4 / T4 | Speed-up |
|---|---|---|---|---|
| Australia | 92.1 / 29.0 / 12.6 s | 7.3x / 2.3x | 682 / 97 s | 7.1x |
| China | 95.0 / 34.9 / 13.4 s | 7.1x / 2.6x | 706 / 101 s | 7.0x |
| Miami | 93.5 / 37.0 / 13.3 s | 7.0x / 2.8x | 558 / 100 s | 5.6x |
| Canada | 24.3 / 29.8 / 10.8 s | 2.3x / 2.8x | 182 / 81 s | 2.3x |

G1 (backward pass at least 10x faster at every circuit): not held, against both CPU runs.
G2 (full solve at least 3x faster at every circuit): not held. It held at three circuits and failed
at Canada (2.3x). The three passes are also measured against the first CPU run, which the quiet run
shows was slowed about three times at those circuits, so they overstate the gain.

What the numbers show:
- On the T4 the backward pass costs the same per track point at every circuit (0.0121-0.0125 s) and
  repeats agree to within 1%, so the cost scales with track length as the code says it should.
- The first M4 run was slowed about 3x at Australia, China and Miami; the quiet run (0.028-0.035 s per
  point) matches the first run's Canada (0.028 s). Even the quiet run slowed as it went on, as a
  fanless laptop does when it warms up. Fair summary: a free T4 is 2.3-2.8x faster than the M4 for
  this solver, unchanged and in float64.
- A full solve takes about 7.5 backward passes' time on both machines, so the backward pass is the
  cost to cut; compilation is a small part.

Not yet known, and the next step: why the gain is small. One backward pass evaluates about 4.9e9
(state, command) pairs at Australia in 12.6 s on the T4, which by a rough count of the arithmetic
per pair is far below the T4's float64 peak. That suggests the time goes to memory traffic in the
interpolation (map_coordinates gathers) or to the per-point loop (lax.scan launching small kernels
1039 times), not to float64 arithmetic. This is a hypothesis to be checked with the JAX profiler
before any change to the solver.

### 2026-10-31: Correction, and what one solver step costs

The entry above guessed that the T4 runs this solver far below its float64 peak. That guess was
wrong. XLA's own cost analysis of one backward-pass step (scripts/step_cost.py, compiled on a CPU)
counts 0.83 GFLOP and 1.66-1.73 GB of memory traffic per step: 4.68 million (state, command) pairs,
about 180 floating-point operations and 370 bytes each, in 11-12 fused kernels. At the T4's peak
rates (254.4 GFLOP/s float64, 320 GB/s) that is at least 3.3 ms of arithmetic and 5.2-5.4 ms of
memory traffic per step, against 12.1 ms measured. So the T4 is already within a small factor of
both limits, and kernel launches (about 12 per step) are too few to matter. A GPU compile can fuse
differently, so step_cost.py is run on the T4 as well before anything is concluded.

What follows for the next steps, if the T4 numbers agree:
- A hand-written kernel that fuses the transition, the interpolation and the minimum (so the
  4.68 million intermediate values never go to memory) removes most of the memory traffic but
  cannot beat the float64 arithmetic floor: at best about 3.5x on the T4.
- Beyond that the arithmetic itself has to get cheaper. The T4 does 32 times more float32 than
  float64 work. Whether float32 keeps lap times within 0.001 s of the float64 results is a separate
  question, to be pre-registered and tested before any float32 result is used.

### 2026-10-31: The T4 compiles the step differently, so the correction above does not hold there

step_cost.py on the Kaggle T4 (results/gpu_benchmark/step_cost_t4.json): the GPU compiler fuses one
step into 3 kernels, not 11-12, and counts 0.455 GFLOP and 0.318 GB per step. At the T4's peak rates
that is at least 1.79 ms of float64 arithmetic and 0.99 ms of memory traffic, against 12.10 ms
measured: 15% and 8%. So on the T4 the step is not near either limit as XLA counts it, and the
conclusion of the entry above (written from the CPU compile) is withdrawn for the GPU. The CPU
counts do not carry over to the GPU; only counts from the device itself are used from here on.

The gap is not yet explained. XLA counts a float64 division or square root as one operation, but a
GPU computes each with a sequence of instructions, and the step has several divisions per pair, so
the true arithmetic floor is above 1.79 ms by an unknown factor. The interpolation also reads its
table at scattered addresses. To find which part the time goes to, step_cost.py --parts times the
transition, the interpolation and the whole step as separate compiled functions on the T4. No change
to the solver is made until that is known.

### 2026-10-31: Where one step's time goes on the T4

step_cost.py --parts on the T4 (Australia, each part compiled alone, median of 20):

| Part | Time |
|---|---|
| Transition (the physics step for every speed x charge x command) | 8.13 ms |
| Interpolation of the next point's values (4 gathers per pair) | 2.50 ms |
| Whole step, compiled alone | 13.66 ms |
| Whole step inside the solver's scan (benchmark) | 12.10 ms |

The transition is the biggest part: about 60% of the step, and alone it moves only about 150 MB
(0.5 ms at 320 GB/s), so its time is float64 arithmetic. The whole step costs more than its two
parts together (13.7 against 10.6 ms), so fusing them makes the step slower, not faster; one
possible reason is that the fused kernel recomputes the transition for each of the interpolation's
four corners. step_cost.py --hlo counts the per-pair operations in the T4's compiled step, which
shows directly whether work is repeated, before any change to the solver is designed.

### 2026-10-31: The T4's compiled step repeats nothing; next, float64 against float32 timing

step_cost.py --hlo on the T4: per (state, command) pair the compiled step has 1 division, 1 square
root, 4 gathers, and about 70 other float64 operations (20 multiplications, 13 additions,
7 subtractions, 10 minimums, 12 selects, ...). That is the same single division and square root as
the CPU compile, so no part of the physics is repeated in the compiled program, and the guess in the
entry above (the fused kernel recomputing the transition for each interpolation corner) is not
supported at this level. Code generation below the compiled program can still repeat work; this
count cannot see that.

Before any rewrite, one more measurement decides what limits the transition: the same parts timed in
float32 (step_cost.py --parts --dtype float32). If float64 throughput is the limit, float32 should
be many times faster on the T4, whose float32 rate is 32 times its float64 rate. These float32 runs
are for timing only; no float32 result is used until a separate pre-registered test shows that
float32 keeps the solver's lap times within 0.001 s of float64.

### 2026-10-31: float32 against float64 on the T4, and a layout test

step_cost.py --parts on the T4, float32 then float64 in the same session (Australia, median of 20):

| Part | float64 | float32 | float32 speed-up |
|---|---|---|---|
| Transition | 7.71 ms | 1.26 ms | 6.1x |
| Interpolation | 3.61 ms | 5.28 ms | 0.7x (slower) |
| Whole step | 13.12 ms | 8.42 ms | 1.6x |

- The transition is limited by float64 throughput: float32 runs it six times faster.
- The interpolation is not limited by arithmetic: float32 does not speed it up. The same float64
  interpolation measured 2.50 ms in an earlier session, so this part also varies by about 40%
  between sessions; its float32 slowdown may be partly that.
- So float32 alone would make the whole step only 1.6x faster, because the interpolation then
  dominates. Precision is not the first thing to change.
- In both precisions the whole step costs more than its parts together (13.1 against 11.3 ms;
  8.4 against 6.5 ms).

A likely reason the interpolation is slow: the solver's arrays are ordered (speed, charge, command),
with the command last, so neighbouring threads on the GPU hold different commands, land on
different speeds, and read table cells far apart in memory. Ordering them (speed, command, charge)
instead puts neighbouring charges side by side, which read neighbouring cells. The result is the
same: step_cost.py checks the two orders against each other on a value table that varies in both
speed and charge, and on the demo circuit the largest difference is exactly 0. Prediction, written
before the T4 run: with charge last the interpolation is faster on the T4, and so is the whole step.

### 2026-10-31: Charge-last order: the step is faster, the interpolation alone is not

step_cost.py on the T4, both orders in the same session (Australia, float64, median of 20):

| Part | Command last (vsu) | Charge last (vus) |
|---|---|---|
| Transition | 8.19 ms | 7.66 ms |
| Interpolation, alone | 2.12 ms | 3.89 ms |
| Whole step | 12.69 ms | 8.94 ms (1.42x faster) |

The largest difference between the two orders' step results on the T4 is exactly 0.

Against the prediction in the entry above: the whole step is faster with charge last (held), but
the interpolation run alone is slower, not faster (not held). So the gain does not come the way
the entry guessed. Run alone, the interpolation reads its coordinates from memory; inside the step
the GPU compiler fuses it with the transition, and that fused kernel is what the order speeds up.
Why the fused kernel gains is not yet shown.

Because the results are identical, the solver now uses charge last (optimal._transition, minimum
over axis 1). On the demo circuit (CPU) the backward pass's value checksum, the lap time, the DP's
own prediction and the harvest price are all bit-for-bit the same as before the change, and the
solver tests pass. step_cost.py keeps the old order as vsu, to measure against.

Prediction L1, written before the run: on the T4, with both orders run in one Kaggle session (the
old code from commit a34f1f8), the solver's backward pass with charge last is at least 1.3x faster
at every one of the four circuits, and the full solve gives the same lap times to 0.001 s.

### 2026-10-31: L1 held: the solver is 1.7x faster on the T4, results bit-for-bit the same

Both orders in one Kaggle session on the T4 (old code at commit a34f1f8, new at 6b52c27; median
of 3 backward passes; result in results/gpu_benchmark/t4_charge_last.json):

| Circuit | Backward pass old / new | Speed-up | Full solve old / new | Speed-up | Lap time |
|---|---|---|---|---|---|
| Australia | 12.44 / 7.50 s | 1.7x | 96 / 58 s | 1.7x | 77.581 s both |
| China | 13.34 / 7.66 s | 1.7x | 101 / 61 s | 1.7x | 91.876 s both |
| Miami | 13.34 / 7.62 s | 1.8x | 100 / 60 s | 1.7x | 88.244 s both |
| Canada | 10.77 / 6.17 s | 1.7x | 81 / 49 s | 1.7x | 72.528 s both |

L1 (at least 1.3x at every circuit, same lap times to 0.001 s): held. More than that, at every
circuit the backward pass's value checksum, the lap time, the DP's own prediction and the harvest
price are bit-for-bit equal to the first T4 run (kaggle_gpu.json), so the change is a pure speed-up.
The old code timed in this session matches the first T4 session to within 1%.

The gain inside the solver (1.7-1.8x) is larger than in the single step timed alone (1.42x); only
the solver's figure is used from here on. Against the quiet M4 run the T4 is now 3.9-4.9x faster,
but that M4 run used the old order; the M4 is timed again with the new code before any T4-against-M4
figure is quoted.

### 2026-10-31: On the M4 the new order may be slower; an A/B run decides

The M4 timed with the new (charge-last) code, set up as for the quiet run (cpu_m4_quiet_charge_last.json):
Australia 69.9, China 64.4, Miami 64.2, Canada 51.6 s per backward pass, about 0.060-0.067 s per
track point, against 0.028-0.035 s in the quiet run with the old order. Taken at face value the new
order is about twice as slow on the M4. On the 2-core x86 cloud machine used for development the
two orders took about the same time (demo circuit: 106 s old, 102 s new).

This cannot yet be told apart from the M4 itself: its first run was slowed about 3x by other load,
at 0.088 s per point. So both orders are timed again on the M4 in one sitting, alternating old, new,
old, new (Australia and Canada, 3 repeats each). If the new order is slower in both rounds, the
solver chooses the order by device: charge last on a GPU, command last on a CPU. The results are
bit-for-bit the same either way, so this changes speed only.

### 2026-10-31: A/B on the M4: the two orders run at the same speed

Old (a34f1f8) and new code on the M4 in one sitting, alternating, backward pass median of 3:

| Round | Australia old / new | Canada old / new |
|---|---|---|
| 1 | 27.85 / 27.91 s | 24.24 / 23.31 s |
| 2 | 31.07 / 29.81 s | 26.36 / 23.41 s |

The orders differ by less than the round-to-round spread, with the new one no slower in any pair.
So the slow run above (cpu_m4_quiet_charge_last.json, about 0.06 s per point) was the M4's state
at that time, not the order, and that file is not used for comparisons. The solver keeps one order
on every device; no per-device choice is needed.

With the same code on both, from the A/B rounds and the T4 run (t4_charge_last.json): Australia
27.9-29.8 s on the M4 against 7.50 s on the T4, Canada 23.3-23.4 s against 6.17 s, so a free T4
runs the solver's backward pass about 3.7-4.0x faster than the M4 at the two circuits timed both
ways. China and Miami were not in the A/B and are not quoted.

## 2026-11-01: Did the FIA's Miami fix change super-clipping? Same circuits, both rules

Question (the SSAC27 paper's title): from the Miami Grand Prix the FIA raised the peak power a
car may harvest while accelerating at full throttle (super-clipping) from 250 to 350 kW, aimed at
shortening super-clipping (sources in configs/rules_2026_races.yaml). The real races cannot
answer this on their own: each circuit ran under one rule, and the before/after comparison of
finding 3 is confounded by circuit (E2, 2026-10-31). So the Strategy Lab solves each circuit under
both caps with everything else equal: the event's race recharge limit, the same car, the same
track. Any difference is the rule's, by construction.

Design (scripts/fia_fix.py, results/fia_fix): Australia, China, Miami and Canada; the three cars
(original, low and high calibrated ends); caps 250 and 350 kW; for the original car also 150, 200,
300 and 450 kW, a dose-response sweep. 40 solves on the fine grid, run on the Kaggle T4. The
simulator now also returns superclip, the part of harvest taken while accelerating at full
throttle (tested: never above the cap or the total harvest, zero where the car is not
accelerating); the DP and every earlier result are unchanged by it (all 116 tests pass).
Measures per lap: lap time, energy harvested by super-clipping, total and longest continuous
super-clipping time, and the share of super-clipping time spent at the cap.

Seen before writing the predictions: one smoke test on the synthetic demo circuit with a coarse
grid that did not converge (not a result): with 350 kW the longest super-clipping stretch was
shorter (0.63 against 0.94 s) but super-clipping harvested more energy (2.42 against 1.87 MJ).

Predictions. Only solves that pass the convergence rule (CONVERGED_S, 0.05 s) count; a pair with
a failed solve counts against the prediction.
F1. Raising the cap from 250 to 350 kW never makes the optimal lap slower by more than 0.02 s,
    and makes it faster by less than 0.20 s, at all 12 (circuit, car) pairs.
F2. The longest continuous super-clipping stretch is shorter under 350 kW at at least 10 of the
    12 pairs (the effect the FIA intended).
F3. Super-clipping harvests more energy per lap under 350 kW at at least 10 of the 12 pairs: when
    full-throttle harvesting is allowed faster, the optimum moves harvesting there from braking
    and part throttle.
F4. Original car, caps 150 to 450 kW: super-clipping energy does not fall as the cap rises at any
    of the four circuits (no step down larger than 0.05 MJ).
No prediction is made about total super-clipping time. How these model results are compared with
the real laps is pre-registered separately, before that comparison is run.

### 2026-11-01: FIA-fix results: more energy by super-clipping, in less time; F1 and F2 not held

Run on the Kaggle T4 (results/fia_fix/runs.csv, profiles in results/fia_fix/profiles);
verdicts computed by scripts/fia_fix_verdicts.py. 38 of 40 solves pass the convergence rule;
the two that fail are Australia, low and high cars, at 350 kW (DP against replay 0.056 and
0.085 s). By the rule written before the run, those two pairs count against every prediction.

Per (circuit, car) pair, 250 kW against 350 kW:

| Pair | Lap gain | Longest stretch | Super-clip energy | Super-clip time |
|---|---|---|---|---|
| Australia original | 0.141 s | 2.74 -> 1.93 s | 4.76 -> 5.27 MJ | 20.7 -> 18.3 s |
| Australia low (failed) | 0.385 s | 4.38 -> 4.21 s | 6.20 -> 7.16 MJ | 26.2 -> 24.3 s |
| Australia high (failed) | 0.473 s | 2.76 -> 3.06 s | 5.31 -> 6.73 MJ | 22.2 -> 21.4 s |
| China original | 0.016 s | 0.57 -> 0.49 s | 1.47 -> 1.64 MJ | 7.2 -> 6.6 s |
| China low | 0.121 s | 1.73 -> 1.30 s | 3.91 -> 4.51 MJ | 17.6 -> 17.0 s |
| China high | 0.056 s | 0.88 -> 0.69 s | 1.93 -> 2.29 MJ | 8.9 -> 8.6 s |
| Miami original | 0.037 s | 1.36 -> 1.83 s | 2.59 -> 3.31 MJ | 13.6 -> 13.5 s |
| Miami low | 0.126 s | 2.41 -> 2.07 s | 4.76 -> 5.09 MJ | 21.2 -> 20.0 s |
| Miami high | 0.042 s | 1.47 -> 1.83 s | 3.10 -> 3.53 MJ | 14.1 -> 12.7 s |
| Canada original | 0.060 s | 2.36 -> 1.96 s | 2.89 -> 3.18 MJ | 12.9 -> 11.6 s |
| Canada low | 0.141 s | 2.87 -> 2.23 s | 4.17 -> 4.49 MJ | 17.8 -> 16.2 s |
| Canada high | 0.124 s | 2.31 -> 1.60 s | 3.28 -> 3.80 MJ | 13.9 -> 12.5 s |

- F1 (never slower by more than 0.02 s, faster by less than 0.20 s, all 12): NOT HELD, 10 of 12.
  No pair is slower. The two failures are the two failed solves, whose gains (0.39 and 0.47 s)
  are also above 0.20 s; on the ten others the gain is 0.016-0.141 s per lap (median 0.09 s).
- F2 (longest stretch shorter at 10 or more): NOT HELD, 8 of 12. It is longer under 350 kW for
  Miami's original and high cars (1.36 -> 1.83 and 1.47 -> 1.83 s) and Australia's high car (a
  failed solve); Australia's low car is shorter but its solve failed.
- F3 (more super-clipping energy at 10 or more): HELD, 10 of 12; the two failed solves also show
  more. On the ten, energy rises by 7-28% (median 13%).
- F4 (original car, energy never steps down by more than 0.05 MJ from 150 to 450 kW): HELD at all
  four circuits. The largest step down is China 300 -> 350 kW, 0.03 MJ. 450 kW gives the same
  result as 350 kW everywhere: the original car's MGU-K cannot harvest more than 350 kW, so a cap
  above that does nothing. For the low car (310 kW MGU-K) the real change is 250 -> 310 kW.

Exploratory, no prediction was made: total super-clipping time falls under 350 kW at all 12
pairs (by 0.5-11.5%, median 8.7% on the ten that converged), while its energy rises, so the mean
super-clipping power rises from 191-236 kW to 246-304 kW. In the model, the fix does what the FIA
wanted with time (less of it, in total and in most longest stretches) but makes super-clipping a
larger source of energy, not a smaller one. This is the optimal strategy under a simplified car,
not a measurement of real cars; whether real laps moved the same way is the next, separately
pre-registered, comparison. The two failed solves are re-run on finer grids (fia_fix.py --refine,
into refined.csv) as a robustness check; the verdicts above stay as written.

### 2026-11-01: Where the extra super-clipping happens (exploratory), and a correction

Correction to the entry above: the low car's super-clipping stops at 310 kW under the 350 kW cap
because that car's engine makes 310 kW at the wheels (ice_power_w, fitted to pole laps), and a car
cannot harvest more than its engine supplies; it is not the MGU-K. The original and high cars
reach 350 kW, the MGU-K limit, which is also why 450 kW gives the same result as 350 kW.

Exploratory, after the verdicts, no prediction (scripts/fia_fix_figures.py,
results/fia_fix/location_split.csv; the parts add up to each solve's total in runs.csv):
- Under 350 kW, super-clipping energy taken below 200 km/h rises at all 12 pairs.
- On each circuit's longest straight it falls at 9 of 12 (Australia original 0.51 -> 0.27 MJ,
  Canada original 0.46 -> 0.38 MJ). For the original car the late super-clipping on that straight
  starts later at all four circuits: Australia from 1,000 to 1,095 m into 1,260 m, China 1,250 to
  1,260 m into 1,325 m, Miami 1,130 to 1,175 m into 1,265 m, Canada 815 to 850 m into 1,025 m.
- This fits the MGU-K speed taper in the model: the MGU-K may harvest 350 kW only up to 290 km/h,
  300 kW at 300 km/h and 250 kW from 310 km/h. At straight-end speeds the 250 kW super-clipping cap
  hardly binds, so raising it frees harvesting mainly at corner exits and lower speeds.
So in the model the fix shortens and delays super-clipping at the end of the long straights, the
part spectators see, while more energy is taken in short bursts at lower speed, where the extra
100 kW can be used. docs/fia_fix_dose.png shows energy and time against the cap (original car);
docs/fia_fix_straight.png shows Canada's longest straight under both caps.

## 2026-11-01: Whole straights: re-cutting laps so no straight crosses the line

Open question from the season test (2026-10-31): clipping tracked the measured length of the
longest straight (Spearman 0.84), but that straight was found with avoid_boundary, which skips
any straight touching the start/finish line because lap distance resets there. At circuits whose
main straight crosses the line, the method measured a shorter straight, which alone could explain
low clipping (for example Bahrain 700 m, Azerbaijan 680 m, Barcelona 415 m in the season test).

Method (multi_race.py --rebase): each lap is re-cut at the slowest corner of the reference lap
(boundary_offset), which no straight crosses. New lap k runs from that corner in lap k to the
same corner in lap k+1, with distance and time carried across the line; it counts as a pit lap if
either source lap was. The reference track is rolled to start at the same corner, and the longest
straight is found as before. Detector, quality checks and thresholds are unchanged. Tested on the
synthetic demo circuit, whose 1,100 m main straight starts at the line: without re-cutting the
method finds 895 m, with it 1,095 m, and every re-cut lap clips as simulated.

Run: the same 18 races as the season test, same data source (OpenF1), output to
results/multi_rebased; the season-test results are not edited. Verdicts by
scripts/rebase_compare.py.

Predictions, before any re-cut real data is seen:
R1. At every race whose re-cut longest straight is at least 200 m longer than the season test's,
    with at least 20 clean laps in both runs, the share of clean laps clipping by 3 km/h or more
    is higher than in the season test.
R2. Among the 11 races new in the season test, with at least 20 clean laps after re-cutting, at
    least half have 80% or more of clean laps clipping (E1 held at 2 of 8 before).
R3. Over all races with at least 20 clean laps, Spearman's correlation between straight length
    and clip share is 0.7 or more: straight length still explains clipping once straights are
    measured whole.
If R1 and R2 hold, finding 3's season verdict was partly a measurement artefact and the README is
corrected; if not, the low clipping at those circuits is real.

### 2026-11-01: Whole-straights results; R1 and R3 held, R2 not held; a defect in the cut point

Run as written (results/multi_rebased, 16 races measured; Australia not loaded, as in the
season test; Britain has 18 clean laps and is reported but not used):
- R1 (higher clip share wherever the straight is 200 m or more longer): HELD, 3 of 3. Japan
  730 -> 2,030 m, 85% -> 96% of clean laps; Barcelona 415 -> 1,005 m, 5% -> 39%; Azerbaijan
  680 -> 1,980 m, 1% -> 97%. The Netherlands also lengthened (675 -> 885 m, 50% -> 67%) but had
  4 clean laps before, so it is outside R1.
- R2 (80% or more of clean laps clip at half or more of the 11 new races): NOT HELD, 3 of 10
  (Azerbaijan 97%, Spain 92%, United States 92%).
- R3 (Spearman straight length against clip share 0.7 or more): HELD, 0.80 over 16 races.
- Before against after Miami, normalised share: 33.2% (2 races) against 20.3% (14), permutation
  p = 0.150; still confounded by circuit.
So part of the season test's low clipping was a measurement artefact (Azerbaijan most of all),
and part is real (Monaco, Singapore, Hungary).

A defect found after these results, prompted by Bahrain's re-cut straight of only 760 m: the cut
point was the first point of the slowest corner. When that corner ends the main straight, the
straight still finishes at the lap boundary and is skipped, the very problem the re-cut was meant
to remove. Shown on the demo circuit: its 900 m straight ends at the 80 km/h corner and touched
the boundary. Fix: cut at the middle of the slowest corner (boundary_offset), and record for
every race the length of any straight still touching the boundary (skipped_straight_m, which
must be 0; tested). The verdicts above stay as written. The corrected run goes to
results/multi_rebased_mid and is reported as a correction, with no new predictions.

### 2026-11-01: Corrected whole-straights run: no race changes

results/multi_rebased_mid, cut at the middle of the slowest corner: skipped_straight_m is 0 at
all 17 measured races, and every race's measured straight is the same length as in the first
whole-straights run (Bahrain stays at 760 m, so Bahrain's short straight was not this defect;
760 m is what its reference lap gives, and why was not checked). Clip shares differ by at most
0.5 percentage points (Bahrain 40.0% -> 39.5%, Canada 97.9% -> 98.1%), from laps cut at a
slightly different point. R1, R2 and R3 come out the same: 3 of 3 held, 3 of 10 not held, 0.80
held. The defect was real (shown on the demo circuit) but affected no 2026 race. README finding 3
updated with the whole-straights result.

## 2026-11-01: Does the FIA-fix result hold on every circuit? Predictions C1-C3

The same-circuit experiment used four circuits. Here it is repeated on all 17 circuits of 2026
whose qualifying data loads, original car only, caps 250 and 350 kW (scripts/fia_fix.py --caps
250 350, results/fia_fix_calendar). Reference tracks for the 13 new circuits are built from
OpenF1 (export_tracks.py, which now keeps the four existing track files untouched). Race recharge
limits are sourced for Australia, China, Miami and Canada only (configs/rules_2026_races.yaml);
for the other 13 an assumed 8.5 MJ is used, the most common sourced race value, and every such
event is listed in results/fia_fix_calendar/assumptions.csv. Check before the verdicts: the four
original circuits must reproduce results/fia_fix to 0.001 s.

Predictions, a pair counting only when both solves pass the convergence rule:
C1. Super-clipping harvests more energy under 350 kW at 80% or more of the 17 circuits.
C2. Super-clipping energy below 200 km/h is higher under 350 kW at 80% or more of them.
C3. No circuit's optimal lap is slower by more than 0.02 s under 350 kW.
No prediction for the longest straight; it is reported. Circuits with no long straight (Monaco,
Singapore) are kept: the rule applies there too.

### 2026-11-02: Calendar results: C1 held, C2 and C3 not held; the low-speed shift is not general

Run on the Kaggle T4 (results/fia_fix_calendar). Correction to the entry above: the run covers 18
circuits, not 17; Australia's track was already saved, so it is included here although its
OpenF1 race laps do not load. Check passed: the four original circuits reproduce results/fia_fix
exactly (Australia 75.156 / 75.015 s, China 89.260 / 89.244 s, Miami 85.871 / 85.834 s, Canada
70.408 / 70.348 s at 250 / 350 kW). 35 of 36 solves pass the convergence rule; the United States
at 350 kW fails and counts against every prediction. 14 circuits use the assumed 8.5 MJ limit.

- C1 (more super-clipping energy under 350 kW at 80% or more): HELD, 16 of 18. Monaco has no
  super-clipping under either cap (0.000 MJ both); the United States failed to converge, though
  it also shows more (2.72 -> 2.96 MJ).
- C2 (more super-clipping energy below 200 km/h at 80% or more): NOT HELD, 12 of 18. It falls at
  Hungary (0.70 -> 0.61 MJ), Spain (0.37 -> 0.33 MJ) and Bahrain (0.25 -> 0.24 MJ), is unchanged at
  Singapore (0.26 MJ) and Monaco (0), and the United States failed.
- C3 (no circuit slower by more than 0.02 s): NOT HELD, 17 of 18, only through the United States'
  failed solve; its lap is also faster (by 0.011 s). Every converged circuit is faster or equal
  under 350 kW, by 0.000 (Monaco) to 0.357 s (Italy).
- Exploratory: super-clipping energy on the longest straight falls at 15 of 18.

So across the calendar, in the model: the 350 kW cap raises super-clipping energy almost
everywhere and takes it off the end of the longest straight almost everywhere, but where the
extra energy is taken is not always below 200 km/h. The four-circuit statement "the extra
100 kW is used mainly at corner exits and lower speeds" does not generalise and is not used.

Refined re-runs of the two failed four-circuit solves (results/fia_fix/refined.csv): Australia
low car at 350 kW converges on the finer charge grid (641 levels): lap 77.612 s against the DP's
77.606 s; its gain over 250 kW becomes 0.278 s (from 0.385 s), still above F1's 0.20 s, with more
super-clipping energy (6.36 against 6.20 MJ) and a shorter longest stretch (3.68 against 4.38 s).
Australia high car at 350 kW fails on both finer grids (DP against replay 0.083 and 0.094 s). The
F1-F4 verdicts stay as written.

### 2026-11-02: Where the extra calendar super-clipping is taken (exploratory)

From results/fia_fix_calendar/location_split.csv (speed bands fixed on 2026-11-01, before this
run; combining the two lower bands was chosen after seeing it): under 350 kW, super-clipping
energy below 250 km/h rises at 17 of 18 circuits (all but Monaco, which has none), 16 of 18
counting only converged pairs as in C1-C3 (the United States' solve failed). Energy above
250 km/h falls at 10 of 18. Where C2 failed, the gain sits at 200-250 km/h instead: Hungary
+0.21 MJ, Spain +0.09 MJ, Bahrain +0.06 MJ. Consistent with the MGU-K taper in the model, which
allows more than 250 kW only below 310 km/h. Correction to 2026-11-01: 14 circuits, not 13, use
the assumed recharge limit (assumptions.csv lists 14). README finding 9 added.

## 2026-11-02: A hand-written CUDA kernel for the backward pass; predictions K1-K3

Why not Pallas, as planned: JAX's Pallas GPU kernels need compute capability 8.0 (Ampere) or
newer through the Triton backend, which JAX 0.11 also deprecates, or Hopper through Mosaic GPU.
The free Kaggle T4 is 7.5 (jax-ml/jax issue #26799; Pallas quickstart). A CUDA kernel compiled
at run time by CuPy (NVRTC) runs on the T4.

Design (src/erslens/cuda_dp.py): one thread per (speed, charge) state loops over the 19 power
commands, so the step's 4.7 million transitions stay in registers: the physics of jaxsim._step,
the bilinear lookup of the next point's value table (map_coordinates order 1, mode nearest), and
the minimum, in one kernel launch per track point. Divisions by values fixed per state are done
once as reciprocals, leaving one division and one square root per transition (the profiling
entries of 2026-10-31 show float64 arithmetic limits the transition on the T4). Selected with
ERSLENS_DP_BACKEND=cuda; the forward pass and replay stay in JAX, and the value tables pass
between CuPy and JAX through DLPack without copying.

Checked without a GPU: a NumPy twin of the kernel matches the XLA step to 1.4e-14 s on Canada's
fine grid for two cars at six track points, and the kernel's own source, compiled as C++ with
the thread indices stubbed, gives exactly the NumPy twin's values (tests/test_cuda_dp.py).

Predictions for the Kaggle T4, both backends in one session (scripts/cuda_benchmark.py; low car,
Australia, China, Miami and Canada):
K1. The two backends' value tables differ by at most 1e-9 s anywhere, at all four circuits.
K2. The CUDA backward pass is at least 2x faster than XLA's at all four circuits.
K3. A full solve with the CUDA backward pass gives the same lap time to 0.001 s and is at least
    1.5x faster, at all four circuits.

### 2026-11-02: CUDA kernel results: K1 and K3 held, K2 not held (1.86-1.96x)

Kaggle T4, both backends in one session (results/cuda_benchmark/t4.json, commit befaf5a):

| Circuit | Backward XLA / CUDA | Speed-up | Full solve XLA / CUDA | Speed-up | Largest value difference |
|---|---|---|---|---|---|
| Australia | 7.05 / 3.79 s | 1.86x | 58 / 31 s | 1.89x | 1.4e-12 s |
| China | 7.69 / 3.93 s | 1.96x | 60 / 32 s | 1.89x | 1.1e-12 s |
| Miami | 7.61 / 3.89 s | 1.96x | 59 / 31 s | 1.89x | 1.6e-12 s |
| Canada | 6.15 / 3.14 s | 1.96x | 49 / 26 s | 1.90x | 1.7e-12 s |

- K1 (value tables within 1e-9 s): HELD, 4 of 4; the largest difference anywhere is 1.7e-12 s, and
  the first-point checksums agree to the last digit at Australia, Miami and Canada (China differs
  in the 17th significant digit).
- K2 (backward pass at least 2x faster): NOT HELD, 0 of 4; 1.86-1.96x.
- K3 (same lap to 0.001 s and full solve at least 1.5x faster): HELD, 4 of 4. The lap times are
  bit-for-bit equal to XLA's at all four circuits (for example Australia 77.58125997731432 s).

Against the first T4 run (kaggle_gpu.json, command-last order, 2026-10-31), the solver is now
3.3-3.4x faster per backward pass (Australia 12.57 -> 3.79 s) and 3.2x per full solve (96.8 ->
30.6 s), with the same results. Compared with the M4 in the A/B runs (same order as XLA now,
2026-10-31), the CUDA pass is 7.4-7.9x faster at Australia and Canada; those runs were in
different sessions on different machines. One step now takes about 3.6 ms; why it is not faster
(float64 throughput, occupancy, or memory) has not been profiled, so no reason is claimed.

## 2026-11-02: Whole-race energy plans: which charge to carry from lap to lap; predictions W1-W3

Question: over a full race under the 2026 rules, where should the battery's charge be spent? The
Strategy Lab so far solves one lap that starts and ends at the same charge. A race can carry charge
between laps: bank energy early and spend it late, or the reverse.

Prior work: Duhr, Buccheri, Balerna, Cerofolini and Onder, "Minimum-Race-Time Energy Allocation
Strategies for the Hybrid-Electric Formula 1 Power Unit", IEEE Transactions on Vehicular
Technology 72(6), 2023, study race-level allocation for the 2014-2025 power unit with a team's
model. What is new here is the 2026 rules (350 kW MGU-K with its speed taper, the per-lap recharge
limit), the real 2026 circuits from public data, and open code on a free GPU.

Design (src/erslens/race.py, scripts/race_plan.py):
- Lap level. Nine charge levels, 0 to 4 MJ in 0.5 MJ steps. For every start level and end level,
  the fastest lap that starts at one, ends at or above the other (within 5 kJ), and harvests no
  more than the event's per-lap recharge limit. One backward pass of the DP aimed at an end level
  serves all nine start levels; a ladder of twelve harvest prices finds the fastest lap within the
  limit. Every lap kept is a simulator replay, so its time, harvest and end charge are measured.
  A candidate is dropped when the DP's predicted cost and its replay differ by more than 0.05 s,
  the Strategy Lab's convergence rule; the number dropped is saved.
- Race level. A DP over laps picks the level to carry from each lap to the next, starting at 2 MJ,
  with no rule on the finishing charge.
- Baselines. "Neutral": every lap starts and ends at 2 MJ. "Simple": neutral until the last lap,
  which spends down to whatever end level is fastest.

Assumptions: the low car; fuel and mass held at their mid-race values on every lap; no safety car,
traffic, pit-lap or Overtake mode; the race distance is the winner's laps from Jolpica. Circuits:
Australia, China, Miami, Canada (the four with recorded recharge limits and converged solves).

Seen before these predictions were written: a coarse trial on the demo circuit (0.25 km/h, 81
charge points, five levels, four prices). Its neutral lap was 74.399 s against the single-lap
solve's 74.361 s; the plan gained 0.63 s by spending 1 MJ early and the rest on the last lap; the
lap time depends on the charge carried by only about 0.005 s, so when to spend is close to a tie.
W2's range is informed by that trial.

Check first: the neutral lap must be within 0.05 s of the single-lap solve that starts at 2 MJ.
If it is not, the tables are not trusted and W1-W3 are not judged.

Predictions (Kaggle T4, ERSLENS_DP_BACKEND=cuda):
W1. At every circuit the best plan is within 0.1 s of the simple plan over the race.
W2. At every circuit the best plan is 0.1 to 1.0 s faster than the neutral plan over the race.
W3. At every circuit the best plan finishes the race at 0 MJ.
