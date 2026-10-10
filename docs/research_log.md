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
