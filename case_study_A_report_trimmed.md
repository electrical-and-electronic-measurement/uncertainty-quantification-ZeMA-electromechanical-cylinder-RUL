# Case Study A — Trimmed-Feature Variant: Execution Report & Comparison

## GUM-compliant uncertainty quantification for RUL estimation, with the start-of-cycle acquisition transient trimmed

**Companion document to `case_study_A_report.md`.** This report records the execution of Case Study A with one change to Phase 1 (feature extraction): the leading within-cycle samples of Pneumatic pressure, Velocity and Axial force are dropped before the feature vector is assembled, removing what `case_study_A_eda_outliers.ipynb` identified as a measurement/DAQ acquisition transient rather than a degradation signal. Everything else — architecture, training procedure, train/test split, propagation code — is byte-for-byte the same as the baseline run, via unmodified imports from `case_study_A_analysis.py`. It accompanies `case_study_A_analysis_trimmed.py`, `case_study_A_results_trimmed.csv`, and `zema_lpu_vs_mc_trimmed.pdf`, and is meant to be read alongside the baseline `case_study_A_report.md`.

---

## 1. Summary

| | Baseline (no trim) | Trimmed |
|---|---|---|
| Input dimension | 1100 (11 channels × 100 readings/cycle) | **1084** (Pneumatic pressure 93, Velocity 94, Axial force 97; all other channels unchanged at 100) |
| Test set | 953 held-out cycles (seed 0) | same 953 cycles (identical split, seed 0) |
| Test RMSE | 0.0641 | 0.0661 |
| $R^2$ | 0.951 | 0.948 |
| mean $u_{\mathrm{physical}}$ (LPU) | 0.641 | 0.655 |
| mean $u_{\mathrm{physical}}$ (MC, primary) | 0.984 | **0.910** |
| LPU vs. MC median rel. diff. | 36.5% | **28.3%** |
| mean $u_{\mathrm{training}}$ | 0.049 | 0.050 |
| Coverage @ $k=2$ | 100.0% | 100.0% |

**Headline result.** Trimming the acquisition transient narrows the LPU-vs-Monte Carlo disagreement (36.5% → 28.3%) and lowers the Monte Carlo $u_{\mathrm{physical}}$ estimate by ~7.5% — both consistent with removing a source of local non-linearity/noise from the network's input. It does **not** fix the root cause of the over-coverage (Pressure + Velocity still account for ~75% of first-layer sensitivity, materially unchanged from baseline), does not improve point-prediction accuracy (RMSE/R² are marginally worse, within the seed-to-seed noise already characterized in `case_study_A_feature_trimming.ipynb`), and — the one genuinely new, mildly concerning finding — **reverses the sign of the weak age-vs-uncertainty correlation** that was at least directionally sensible in the baseline (§7 below). Net assessment and recommendation in §8.

---

## 2. Methodology: the trim

Settling length $K$ per channel, computed exactly as in `case_study_A_eda_outliers.ipynb` (average $|reading-cycle\ median|$ by within-cycle position, over all 4,766 cycles; $K$ = leading run exceeding 5× the steady-state baseline, i.e. positions ≥ 20):

| Channel | $K$ | Trimmed? |
|---|---|---|
| Pneumatic pressure | 7 | ✅ |
| Velocity | 6 | ✅ |
| Axial force | 3 | ✅ |
| Active current | 1 | ❌ (at the detection threshold, left untouched — matches the original recommendation) |
| all other channels | 0 | ❌ |

`build_trimmed_dataset()` (identical to the version in `case_study_A_feature_trimming.ipynb` and `case_study_A_run_analysis.ipynb`, reproduced in `case_study_A_analysis_trimmed.py`) drops each trimmed channel's own leading $K$ within-cycle samples before concatenation, giving an input dimension of $1100 - (7+6+3) = 1084$.

---

## 3. Environment and self-test

Unchanged from the baseline run — same `requirements.txt`, same `self_test_lpu()` gate (not re-run here since `propagate_lpu`/`propagate_mc` are imported unmodified from `case_study_A_analysis.py`; see `case_study_A_report.md` §3 for that result, 0.096% rel. diff., well under the 2% threshold).

---

## 4. Data and train/test split

`load_zema(cfg)` returns the same `X`, `UX`, `y` as the baseline (full-resolution, `datapoints_per_cycle=100`). The train/test split uses the same `cfg.seed=0` → identical 953 held-out cycles as the baseline run, so every comparison in this report is apples-to-apples on the *same* test cycles, differing only in which within-cycle samples feed the model.

---

## 5. Model, training, and the ensemble

Identical architecture, training procedure and ensemble convention as the baseline (§6 of `case_study_A_report.md`): MLP (128, 64, 32), softplus, Adam lr=$10^{-3}$, 200 epochs, float64, 10-seed ensemble for $u_{\mathrm{training}}$, seed 0 retained as the reference model for LPU/MC propagation and the sensitivity ranking below. Only the input dimension changes (1084 vs. 1100).

---

## 6. Propagation, combination, and results

Output: `case_study_A_results_trimmed.csv`, 953 rows, same schema as the baseline CSV (`y_true, y_hat, u_physical_lpu, u_physical_mc, u_training, u_c, covered_k2`).

```
test RMSE                  : 0.0661
test R2                    : 0.9484
mean u_physical (LPU)      : 6.5474e-01
mean u_physical (MC)       : 9.0995e-01
LPU vs MC median rel. diff : 28.33%  (Phase 5 check)
mean u_training (ensemble) : 5.0435e-02
mean u_c (combined)        : 6.5718e-01
empirical coverage @ k=2   : 100.0%  (target ~95%)
```

As in the baseline, the CSV's `u_c` column combines the **LPU** estimate; since the LPU-vs-MC gap remains well above the 5–10% threshold (§2 of `case_study_A_report.md`), Monte Carlo is still the primary $u_{\mathrm{physical}}$ estimate here too, giving $u_c^{\mathrm{MC}} = \sqrt{0.910^2+0.050^2} \approx 0.911$.

### Sensitivity ranking (first-layer $|W^{(1)}|\cdot u_n$, per channel)

| Rank | Channel | Width (cols) | $u_n$ (mean) | Share of first-layer sensitivity | Baseline share* |
|---|---|---|---|---|---|
| 1 | Pneumatic pressure | 93 | 8.864 | **50.5%** | 51.8% |
| 2 | Velocity | 94 | 4.358 | **25.0%** | 25.2% |
| 3 | Acceleration plain bearing | 100 | 2.020 | 12.3% | 11.5%† |
| 4 | Acceleration ball bearing | 100 | 0.863 | 5.3% | 4.9%† |
| 5 | Acceleration piston rod | 100 | 0.689 | 4.2% | 3.9%† |
| 6 | Axial force | 97 | 0.221 | 1.3% | 1.3% |
| 7 | Active current | 100 | 0.168 | 1.0% | 1.0% |
| 8–10 | Motor current (×3) | 100 each | ~0.021 | 0.1% each | 0.1% each |
| 11 | Sound pressure | 100 | 0.0001 | ~0.0% | ~0.0% |

\* From `case_study_A_report.md` §7. † The baseline report used the generic axis labels "Acceleration x/y/z"; this report uses the dataset's own per-sensor HDF5 labels (plain bearing / piston rod / ball bearing, same ones used throughout `case_study_A_eda.ipynb`) — the correspondence is positional (same three channels, same ranking), not a relabeling of different data.

**Trimming barely moves the sensitivity ranking.** Pressure and Velocity together still account for **75.5%** of first-layer sensitivity (vs. 77.0% at baseline) — essentially unchanged. This confirms the mechanism diagnosed in `case_study_A_report.md` §7 (a channel's *absolute* measurement-noise floor exceeding its own *steady-state* cross-cycle signal variation, amplified by z-score normalization) is a property of the channels' **steady-state** readings, not of the few leading transient samples — trimming the transient was never expected to fix this, and it doesn't.

### MC convergence check (held-out test point #0)

```
n_mc=   10,000   u_physical_mc = 9.475e-01
n_mc=  100,000   u_physical_mc = 9.506e-01   (delta 0.33% from 10k)
n_mc=1,000,000   u_physical_mc = 9.524e-01   (delta 0.19% from 100k)
```

$M=10^4$ remains adequately converged (sub-1% sampling noise) relative to the 28.3% LPU/MC gap being measured — same conclusion as the baseline (§7 of `case_study_A_report.md`).

---

## 7. Is $u_c$ actionable? — trimmed vs. baseline

| Check | Baseline (no trim) | Trimmed | Reading |
|---|---|---|---|
| mean $u_c$ (LPU-based) vs. mean $\lvert$error$\rvert$ | 0.644 vs. 0.046 (14.0×) | 0.657 vs. 0.048 (13.6×) | Still ~14× over-wide either way — trimming does not materially change the miscalibration |
| corr($u_{\mathrm{physical}}^{\mathrm{MC}}$, $y_{\mathrm{true}}$) | **+0.61** | **−0.20** | **Sign reversal.** Baseline uncertainty grows (weakly) with age, which is at least directionally sensible; trimmed uncertainty now *shrinks* with age |
| corr($u_c$, $\lvert$error$\rvert$) | 0.26 (LPU-based) / 0.15 (MC-based) | 0.05 | Already-weak signal gets weaker — $u_c$ is even less useful for flagging individually unreliable predictions |
| Early-life vs. late-life mean $u_{\mathrm{physical}}^{\mathrm{MC}}$ (terciles) | 0.933 → 1.064 (**+14%**) | 0.940 → 0.900 (**−4%**) | Baseline trend (uncertainty rises with age) qualitatively matches growing error; trimmed trend goes the *wrong* direction |
| Early-life vs. late-life mean $\lvert$error$\rvert$ | 0.033 → 0.056 (+70%) | 0.039 → 0.057 (+45%) | Actual error still grows substantially with age in both variants (unaffected by the trim, as expected — it is a property of the task, not the uncertainty budget) |

**This is the one qualitatively new finding from the trimmed run**, not just a magnitude change. Removing the transient-spike samples narrows the LPU/MC gap and lowers the *average* $u_{\mathrm{physical}}^{\mathrm{MC}}$, but it also removes whatever weak structure was making the Monte Carlo uncertainty estimate trend upward with machine age — in the trimmed variant it trends (weakly) in the wrong direction instead. A plausible mechanism: the early-life spike samples, present identically at every life stage, contributed a roughly constant amount of propagated uncertainty; removing them leaves more of the remaining signal driven by each cycle's steady-state Pressure/Velocity readings, whose *measurement* uncertainty does not have an obvious reason to track machine age the same way wear-driven signal variation does. This is offered as a plausible explanation, not a proven mechanism — it would need a dedicated per-channel ablation to confirm.

---

## 8. Validation checks — summary and recommendation

| Check | Baseline | Trimmed | Verdict |
|---|---|---|---|
| LPU vs. MC agreement | 36.5% | 28.3% | Improved, but both are far above the 5–10% threshold for treating LPU as primary — MC remains primary in both |
| mean $u_{\mathrm{physical}}^{\mathrm{MC}}$ | 0.984 | 0.910 | Improved (↓7.5%) |
| Coverage @ $k=2$ | 100.0% | 100.0% | Unchanged — still driven by Pressure+Velocity dominance (§6) |
| RMSE / $R^2$ (seed 0) | 0.0641 / 0.951 | 0.0661 / 0.948 | Marginally worse — within the ~0.18 pooled-SD seed noise already found in `case_study_A_feature_trimming.ipynb`, not a reliable effect |
| corr($u_{\mathrm{physical}}$, age) | +0.61 (sensible) | −0.20 (wrong direction) | **Regressed** |
| corr($u_c$, $\lvert$error$\rvert$) | 0.26 / 0.15 | 0.05 | **Regressed** (already weak, now weaker) |

**Figure** (`zema_lpu_vs_mc_trimmed.pdf`, same two-panel layout as the baseline's `fig:zema_lpu_vs_mc`): panel (a) shows the LPU-vs-MC scatter visibly closer to the identity line than the baseline figure (28.3% vs. 36.5% median offset); panel (b) shows the same saturated $\pm k{=}2\,u_c$ band as the baseline, still trivially covering the full test set.

```bash
python case_study_A_analysis_trimmed.py
# writes case_study_A_results_trimmed.csv; prints the summary in §6-§7 above
python make_validation_figure.py --csv case_study_A_results_trimmed.csv --out zema_lpu_vs_mc_trimmed.pdf
```

### Recommendation

**Do not adopt this trim as a blanket replacement for the baseline pipeline on current evidence.** It improves one diagnostic (LPU-vs-MC agreement, and the average magnitude of $u_{\mathrm{physical}}^{\mathrm{MC}}$) without addressing the actual cause of the over-coverage (Pressure/Velocity sensor-noise-floor amplification through normalization, §6), without improving point-prediction accuracy, and it measurably weakens the one weak-but-directionally-useful signal the uncertainty budget had (age correlation). If the goal is specifically a smaller LPU/MC gap for its own sake (e.g. to make LPU usable as a cheaper stand-in for MC), the trim is a genuine, reproducible win on that one axis — three years of evidence here is one seed-0 run, so re-confirming the sensitivity-ranking and correlation findings with the full 10-seed ensemble (not just seed 0) would be the natural next step before treating the age-correlation reversal as a real effect rather than ensemble noise. The leverage point for actually fixing the over-coverage remains what `case_study_A_report.md` §7.1 identified: the Pressure channel's normalized uncertainty amplification, not the leading-sample transient.

---

## 9. Files produced by this variant

| File | Role |
|---|---|
| `case_study_A_analysis_trimmed.py` | This variant's pipeline: imports `case_study_A_analysis.py` unmodified, adds `build_trimmed_dataset()`, reruns Phases 2/4/5 on the trimmed inputs, and prints the extra diagnostics (sensitivity ranking, MC convergence, actionability checks) used in this report |
| `case_study_A_results_trimmed.csv` | Real output, 953 test cycles × 7 columns, same schema as the baseline `case_study_A_results.csv` |
| `zema_lpu_vs_mc_trimmed.pdf` | The same two-panel validation figure as the baseline, generated from `case_study_A_results_trimmed.csv` via the unmodified `make_validation_figure.py` |
| `pipeline_run_trimmed.log` | Captured stdout of the run behind this report |
| `case_study_A_report_trimmed.md` | This document |

Does not overwrite any baseline artifact (`case_study_A_results.csv`, `zema_lpu_vs_mc.pdf`, `case_study_A_report.md`) — both variants' results coexist in the repository for direct comparison.

---

## 10. Reproducing this report from scratch

```bash
python3.10 -m venv zema_env && source zema_env/bin/activate
pip install -r requirements.txt

python case_study_A_analysis_trimmed.py
# writes case_study_A_results_trimmed.csv; expect the printed summary in §6-§7

python make_validation_figure.py --csv case_study_A_results_trimmed.csv --out zema_lpu_vs_mc_trimmed.pdf
```

Wall-clock time observed: comparable to the baseline run (dataset already cached; 10-seed ensemble training + 953-cycle LPU/Monte Carlo propagation at `n_mc=10,000`, well under an hour on CPU).
