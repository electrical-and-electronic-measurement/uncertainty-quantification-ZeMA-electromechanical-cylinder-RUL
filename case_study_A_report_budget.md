# Case Study A — Corrected Uncertainty Budget: Execution Report

## Random/systematic split of the input uncertainty, model term from a validation set, Monte Carlo-consistent combination

**Companion document to `case_study_A_report.md` and `case_study_A_report_fullwindow.md`.** This report records four runs of `case_study_A_analysis_budget.py`, which keeps the model, training procedure and held-out test cycles of the baseline. It changes how the input uncertainty is interpreted and how the terms of the budget are combined.

---

## 1. Why the original budget is not correct

The original pipeline combines `u_c = sqrt(u_physical² + u_training²)`. The quadrature sum of two independent contributions is correct GUM practice. The problems are in what goes into `u_physical` and in what is missing:

1. **Every declared input uncertainty is treated as independent random noise, per sample and per cycle** (covariance `diag(u²)`). The data rule this out. For a random component, the observed cross-cycle spread *s* satisfies *s*² = var(true) + *u*_rand², so *u*_rand ≤ *s*. For Pneumatic pressure (*u/s* ≈ 8.8), Velocity (4.3) and Acceleration plain bearing (2.1), at least 98.7%, 94.5% and 76% of *u*² must therefore be **common to all cycles**, i.e. systematic.
2. **A systematic error common to all cycles has no effect when the same instrument is used for training and inference.** A constant offset/gain error per column, *x*′ = *a*_j + *b*_j *x*, shifts and scales the training mean and standard deviation by the same amounts. The z-score is therefore unchanged, and so is the prediction. This was verified numerically.
3. **The model was evaluated far outside its training domain.** Perturbing the inputs by ≈9 standard deviations shifts the Monte Carlo mean of the prediction by +0.46 relative to *f*(*x*), half of the [0, 1] output range. This is why LPU and MC disagreed by 36%.
4. **Combination inconsistency.** The CSV `u_c` and the coverage used the LPU value, while MC was declared primary.
5. **No model-uncertainty term.** `u_training` captures only the seed-to-seed spread on a fixed split; coverage with `u_training` alone was 89.8%.

Correlating samples within a channel, with channels independent, changes little on its own: 0.641 → 0.595 on the baseline model. The decisive point is items 1–2.

---

## 2. Method

### 2.1 Random / systematic split

Per sample, *u*² = *u*_rand² + *u*_sys². The dataset declares only *u*, so *u*_rand is estimated from the data with two models (`--random-model`). **Both give upper bounds.**

| Model | Definition | Rationale |
|---|---|---|
| `bound` | *u*_rand = min(*u*, *s*); *s* = cross-cycle std of the same column (training cycles) | Largest component that can vary from cycle to cycle |
| `noise` | *u*_rand = min(*u*, *s*, σ_k); σ_k = white-noise floor of channel *k* from second differences of adjacent samples (robust MAD, Type A) | Sample-to-sample noise; signal curvature also contributes, so it is still an upper bound |

Resulting split (share of *u*², baseline input):

| Channel | *u/s* | Random share (`bound`) | Random share (`noise`) |
|---|---|---|---|
| Pneumatic pressure | 8.80 | 1.3% | 0.0% (σ = 268 Pa) |
| Velocity | 4.28 | 6.1% | 6.1% |
| Acceleration plain bearing | 2.06 | 25.9% | 23.8% |
| Acceleration piston rod | 0.68 | 99.8% | 42.5% |
| Acceleration ball bearing | 0.88 | 98.1% | 98.1% |
| Axial force | 0.22 | 100% | 17.4% |
| Active current, motor currents, sound pressure | ≤ 0.17 | 100% | 100% |

### 2.2 Budget

- **Same-instrument use** (the case study's setting): only *u*_rand is propagated, as independent per-sample noise.
  - `u_rand_lpu`: diagonal LPU.
  - `u_rand_mc`: GUM-S1 Monte Carlo with *M* = 10⁴.
- **Transfer to another, recalibrated instrument:** *u*_sys is additionally propagated as a shift common to all samples of a channel, independent between channels. This is a block-diagonal covariance with *U*_k = *u*_sys,k *u*_sys,kᵀ.
  - `u_sys_lpu`: LPU with that covariance.
  - `u_sys_mc`: MC with one N(0, 1) draw per channel.
- **Model term:** 15% of the training cycles are held out as a validation set; the test set is unchanged. Then *u*_model² = mean(*e*_val²) − mean(*u*_training,val²) − mean(*u*_rand,val²), clipped at 0, so that the residual variance is not counted twice.
- **Combination:**
  - same instrument: *u*_c = √(*u*_rand,MC² + *u*_training² + *u*_model²);
  - transfer: *u*_c,transfer = √(*u*_c² + *u*_sys,MC²).

  Coverage is computed with these Monte Carlo-based values.
- **Reference:** the original budget is recomputed on the same model as `u_indep_mc`, with every input independent and the full declared *u*.

### 2.3 Verification

- *u*_rand² + *u*_sys² = *u*² holds exactly for both models.
- A per-column affine error leaves the z-scores unchanged.
- `propagate_sys` LPU equals the Jacobian formula Σ_k (Σ_i J_ki *u*_sys,ki)² and agrees with MC (2.887·10⁻⁴ vs 2.881·10⁻⁴ on a test network).
- A reduced end-to-end run (300 cycles) produced a CSV accepted by `make_validation_figure.py`.

---

## 3. Results (953 held-out test cycles, same as the baseline)

The model is trained on 3 241 cycles instead of 3 813, because 572 are now used for validation. This explains the slightly higher RMSE compared with the earlier runs (0.0664 vs 0.0641 and 0.0694 vs 0.0655).

| Quantity | Baseline input, `bound` | Baseline input, `noise` | Without pressure, `bound` | Without pressure, `noise` |
|---|---|---|---|---|
| Test RMSE / R² | 0.0664 / 0.948 | 0.0664 / 0.948 | 0.0694 / 0.943 | 0.0694 / 0.943 |
| mean *u*_rand MC | 0.124 | 0.095 | 0.098 | 0.092 |
| LPU vs MC median rel. diff. | **6.3%** | **4.7%** | **4.7%** | **4.2%** |
| mean \|MC mean − *f*(*x*)\| | 0.022 | 0.014 | 0.048 | 0.043 |
| mean *u*_training | 0.051 | 0.051 | 0.048 | 0.048 |
| *u*_model (validation) | 0 | 0 | 0 | 0 |
| **mean *u*_c (same instrument)** | **0.136** | **0.110** | **0.111** | **0.106** |
| Coverage k=2 / k=1 | 99.9% / 96.4% | 99.8% / 93.1% | 99.5% / 95.0% | 99.5% / 93.8% |
| mean *u*_c / mean \|error\| | 2.87 | 2.31 | 2.48 | 2.36 |
| corr(*u*_c, \|error\|) | 0.40 | 0.43 | 0.51 | 0.52 |
| mean *u*_sys MC (transfer) | 0.758 | 0.767 | 0.376 | 0.380 |
| mean *u*_c,transfer | 0.771 | 0.776 | 0.394 | 0.395 |
| Original budget, mean *u*_indep MC | 1.000 | 1.000 | 0.322 | 0.322 |

Figures: `zema_lpu_vs_mc_budget_<variant>.pdf`. Panel (a) shows the random part, LPU vs MC; panel (b) shows the ±2*u*_c band.

### 3.1 Residual consistency check

The residuals bound how much propagated random input uncertainty the data allow: √(mean(*e*²) − mean(*u*_training²)) on the test set.

| Variant | rms *u*_rand MC | Allowed by residuals |
|---|---|---|
| Baseline input, `bound` / `noise` | 0.127 / 0.098 | 0.032 |
| Without pressure, `bound` / `noise` | 0.100 / 0.094 | 0.042 |

### 3.2 Blocked split (leakage check)

The label is the cycle index (`y = k/N`), and the random split puts at least one neighbour (*k*±1) of 96% of the test cycles in the training set; 64.7% have both. Consecutive cycles are nearly identical, so the random-split test error may measure interpolation between neighbours. `--split blocked` assigns whole blocks of consecutive cycles to test / validation / training (seed 0, same fractions) and drops training cycles closer than 5 cycles to a held-out cycle. No test cycle then has a training neighbour. The test set changes, so the comparison is between splits, not on the same cycles.

| Quantity | Random | Blocked 50 | Blocked 20 | Blocked 100 | Random, no pressure | Blocked 50, no pressure |
|---|---|---|---|---|---|---|
| Train / val / test cycles | 3242 / 571 / 953 | 2996 / 600 / 950 | 2741 / 580 / 960 | 3086 / 600 / 1000 | 3242 / 571 / 953 | 2996 / 600 / 950 |
| Test RMSE / R² | 0.0664 / 0.948 | 0.0739 / 0.953 | 0.0772 / 0.937 | 0.0695 / 0.935 | 0.0694 / 0.943 | 0.0813 / 0.943 |
| MAE | 0.047 | 0.051 | 0.056 | 0.052 | 0.045 | 0.052 |
| LPU vs MC median rel. diff. | 6.3% | 6.8% | 5.1% | 6.1% | 4.7% | 4.5% |
| mean *u*_c (same instrument) | 0.136 | 0.132 | 0.152 | 0.130 | 0.111 | 0.113 |
| *u*_model (validation) | 0 | 0 | 0 | 0 | 0 | 0 |
| Coverage k=2 / k=1 | 99.9% / 96.4% | 99.6% / 95.8% | 99.9% / 96.9% | 99.8% / 95.9% | 99.5% / 95.0% | 99.3% / 94.1% |
| Coverage k=2, *u*_training only | 89.8% | 90.4% | 88.5% | 87.7% | 90.3% | 88.1% |
| mean *u*_c / MAE | 2.87 | 2.59 | 2.72 | 2.49 | 2.48 | 2.19 |
| rms *u*_rand MC / allowed by residuals | 0.127 / 0.032 | 0.121 / 0.047 | 0.143 / 0.046 | 0.119 / 0.039 | 0.100 / 0.042 | 0.102 / 0.061 |
| RMSE using MC mean instead of *f*(*x*) | 0.0668 | 0.0740 | 0.0763 | 0.0809 | 0.0836 | 0.0985 |

All columns use `--random-model bound`; with `noise` the blocked-50 runs give the same RMSE and *u*_c 0.108 (baseline input) / 0.108 (no pressure), coverage k=2 99.3% / 99.2%.

Findings:

1. **Neighbour leakage is small.** Without any adjacent training cycle, RMSE rises by 5–16% (0.066 → 0.070–0.077; without pressure 0.069 → 0.081) and R² stays at 0.94–0.95. The point accuracy is not an artefact of the random split.
2. **The budget conclusions do not change.** *u*_model is still 0, coverage k=2 is still 99.3–99.9%, and the propagated random part (0.10–0.14) still exceeds what the residuals allow (0.04–0.06), though the gap narrows from ≈4× to ≈2–3×.
3. **The MC-mean bias without pressure persists** (mean |MC mean − *f*(*x*)| 0.049): using the MC mean as the estimate raises RMSE from 0.081 to 0.099, a sign that the `bound` perturbations still leave the training domain for Velocity and the plain-bearing accelerometer.

Reproduce with `--split blocked [--block-size 50] [--gap 5]`; outputs carry the suffix `_blocked<size>`.

---

## 4. Interpretation

1. **The corrected budget is 7–9 times smaller and the propagation becomes valid.** Same-instrument *u*_c drops from ≈0.98 (original) to 0.11–0.14. LPU and MC now agree within 4–6%, so first-order propagation is acceptable. The Monte Carlo bias of the point estimate falls from 0.46 to 0.01–0.05. The interval width is now of the same order as the actual error (*u*_c/MAE 2.3–2.9, previously 21), and corr(*u*_c, \|error\|) improves to 0.40–0.52.
2. **The intervals are still about twice too wide.** Coverage at k=2 is 99.5–99.9%, while the ±1·*u*_c interval already covers 93–96%. The cause is the random part, not a missing term. *u*_model comes out 0 in every run because *u*_training² + *u*_rand² already exceeds the validation residual variance. The residual check (§3.1) quantifies this: the data allow at most 0.03–0.04 of propagated random input uncertainty, against 0.09–0.13 computed. Both random models are upper bounds, and for Velocity and the plain-bearing accelerometer they hit the *u*_rand ≤ *s* cap. The true random part of those channels must be well below their cross-cycle spread.
3. **The systematic part matters only for transfer, and there it is large.**
   - With all channels, *u*_sys ≈ 0.76, dominated by Pressure.
   - Without Pressure, *u*_sys ≈ 0.38. This is larger than the original "independent" budget (0.32), because a shift common to all samples of a channel adds linearly over the samples instead of in quadrature.
   - A model used with a different, recalibrated instrument would carry this uncertainty unless it is recalibrated (re-normalized) on that instrument.
4. **Removing Pressure** still halves the transfer uncertainty. In the same-instrument budget it matters little (0.110 → 0.106 with `noise`), because the random part of Pressure is negligible once systematic and random are separated.

---

## 5. Recommendation

- **Report the same-instrument budget** *u*_c = √(*u*_rand² + *u*_training² + *u*_model²) with Monte Carlo as the primary *u*_rand. State the random/systematic split and its data-based upper bounds explicitly.
- **Report *u*_sys separately** as the additional uncertainty for deployment on another instrument, propagated as a per-channel common shift.
- **State the remaining over-coverage as a limitation of the dataset**: it declares only a total *u*, with no repeatability/noise specification. Both estimates of the random part are upper bounds, and the residuals show that the true random part is 2–4 times smaller. A tighter estimate needs sensor noise data (datasheet noise density, repeated static measurements). Fitting it to the residuals would turn the coverage check into a calibration, and is not done here.
- Caveats: one seed-0 split; the validation set reduces the training data by 15%.

---

## 6. Files

| File | Role |
|---|---|
| `case_study_A_analysis_budget.py` | Corrected-budget pipeline (`--random-model bound\|noise`, `--exclude`, `--sampling`, `--n-points`); imports the core functions of `case_study_A_analysis.py` and the loader of `case_study_A_analysis_fullwindow.py` unmodified |
| `case_study_A_results_budget_<variant>.csv` | 953 rows. Columns `y_true, y_hat, u_physical_lpu, u_physical_mc` (random part), `u_training, u_c, covered_k2, mc_bias, u_model, u_sys_lpu, u_sys_mc, u_c_transfer, u_indep_mc` |
| `pipeline_run_budget_<variant>.log` | Captured stdout, including the per-channel random/systematic split |
| `zema_lpu_vs_mc_budget_<variant>.pdf` | Validation figure (unmodified `make_validation_figure.py`) |

Variants: `first100_bound`, `first100_noise`, `first100_no_pressure_bound`, `first100_no_pressure_noise`; blocked split (§3.2): the same four with suffix `_blocked50`, plus `first100_bound_blocked20` and `first100_bound_blocked100`.

```bash
python case_study_A_analysis_budget.py --random-model bound
python case_study_A_analysis_budget.py --random-model noise
python case_study_A_analysis_budget.py --random-model bound --exclude "Pneumatic pressure"
python case_study_A_analysis_budget.py --random-model noise --exclude "Pneumatic pressure"
for t in first100_bound first100_noise first100_no_pressure_bound first100_no_pressure_noise; do
  python make_validation_figure.py --csv case_study_A_results_budget_$t.csv --out zema_lpu_vs_mc_budget_$t.pdf
done
```
