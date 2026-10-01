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
