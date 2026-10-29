# Case Study A — Window-Sampling and Pressure-Ablation Variants: Execution Report & Comparison

## GUM-compliant uncertainty quantification for RUL estimation, with inputs drawn from the whole 1 s window and with the Pneumatic pressure channel removed

**Companion document to `case_study_A_report.md` and `case_study_A_report_trimmed.md`.** This report records four further executions of Case Study A that change only Phase 1 (which within-cycle samples, and which channels, feed the model). Architecture, training procedure, train/test split, normalization and propagation code are the same as the baseline run, via unmodified imports from `case_study_A_analysis.py`. It accompanies `case_study_A_analysis_fullwindow.py`, four results CSVs, four validation figures and four run logs (§9).

> **Status update (29 Oct 2026).** The results below use the original budget (full declared input uncertainty propagated as independent noise, LPU-based CSV `u_c`). `case_study_A_report_budget.md` shows that most of the declared uncertainty of Pressure, Velocity and the plain-bearing accelerometer must be systematic and cancels in the normalization for same-instrument use. With the corrected budget, the first-100 input gives $u_c \approx 0.11$–$0.14$ with Pressure and $\approx 0.11$ without, and LPU/MC agree within 4–7%. Removing Pressure therefore matters much less for the same-instrument budget than §4–§7 suggest; it still halves the systematic part relevant for transfer to another instrument (0.77 → 0.38). A blocked split (budget report §3.2) confirms that the accuracy comparisons in §4 are not driven by neighbouring cycles in the training set (RMSE +5–16%).

---

## 1. Why these variants: the baseline uses only the first 50 ms of each cycle

Each row of the ZeMA HDF5 holds **1 s of the return stroke sampled at 2 kHz = 2000 samples per channel** (Dorst et al. 2022, §3.1; dataset attribute `dc:SizeOrDuration`). `zema_emc_annotated.ZeMASamples(datapoints_per_cycle=d)` does **not** downsample: it slices `values[:d]` (same for the uncertainties). With the baseline `datapoints_per_cycle=100`, the model therefore sees only the **first 100 samples = first 50 ms** of the window, i.e. 5% of the available data. This was not stated anywhere in the baseline material ("100 readings/cycle" reads as 100 readings spread over the cycle).

This matters because (i) the baseline's features cover a 50 ms slice (20 Hz frequency resolution), and (ii) several channels change within the 1 s window (e.g. Axial force rises from 5.35 to 6.53 kN, with a step near 0.5 s), so the first 50 ms are not representative of the whole window. The variants below test whether using the whole window helps, and — following the per-channel normalized-uncertainty analysis — what happens when the Pneumatic pressure channel is removed.

### Why Pneumatic pressure is the ablation target

What drives a channel's contribution to $u_{\mathrm{physical}}$ is the dimensionless ratio $u/s$ between its declared standard uncertainty and its cross-cycle standard deviation (the z-score divisor). Normalization leaves this ratio unchanged ($u(z)=u(x)/s$, exact GUM propagation of a linear map), so the dominance of Pressure is a property of the data, not an artifact of normalization. On the baseline inputs (median over the 100 positions):

| Channel | $u$ (dataset) | $s$ (std across cycles) | $u/s$ |
|---|---|---|---|
| **Pneumatic pressure** | 16 830 Pa (constant) | 1 936 Pa | **8.7** (max 13.9) |
| Velocity | 1% of reading (≈2.0 mm/s) | 0.47 mm/s | 4.3 |
| Acceleration plain bearing | 2.832 m/s² (constant) | 1.38 m/s² | 2.1 |
| Acceleration ball bearing / piston rod | 2.832 m/s² (constant) | 3.24 / 4.22 m/s² | 0.87 / 0.67 |
| Axial force | 0.01244 kN (constant) | 0.058 kN | 0.22 |
| Active current | 1% of reading | 0.17 A | 0.17 |
| Motor current (×3) | 1.5% of reading | ≈1.85 A | 0.02 |
| Sound pressure | 2.79·10⁻⁵ Pa (constant) | 0.54 Pa | ≈0 |

The declared Pressure uncertainty is ~9× its cycle-to-cycle variation: metrologically, the trend the network could exploit is buried in the measurement noise. Note that the ZeMA DAQ uncertainties (constant values, or exact 1%/1.5% relative values) are **not documented in Dorst et al. 2022**, which only derives uncertainties for the SmartUp Unit (½ LSB → interpolation → dynamic calibration); they look like datasheet-based Type B values.

---

## 2. Methodology: the variants

All variants are produced by `case_study_A_analysis_fullwindow.py`, which reads the HDF5 directly (same file and hash check as `zema_emc_annotated`) and reproduces the library's channel layout (11 consecutive column blocks, `CHANNEL_NAMES` order).

| Variant | CLI | Samples per channel | Channels | Input dim |
|---|---|---|---|---|
| **Run 1** — decimation over 1 s | *(defaults)* | 200, one every 10 samples at the centre of each block (positions 5, 15, …, 1995) | 11 | 2200 |
| **Run 2** — decimation, no pressure | `--exclude "Pneumatic pressure"` | as Run 1 | 10 | 2000 |
| **Control A** — baseline input, no pressure | `--sampling first --n-points 100 --exclude "Pneumatic pressure"` | first 100 (= `ZeMASamples`) | 10 | 1000 |
| **Control B** — block means over 1 s | `--sampling blockmean` | 200 means of 10 consecutive samples | 11 | 2200 |

**Uncertainty of the inputs.**
- *Decimation* (Runs 1–2) and *first-N* (Control A) keep each selected reading together with its own dataset uncertainty — no assumption needed. Decimation to 200 Hz is done without an anti-aliasing filter, so content above 100 Hz aliases.
- *Block mean* (Control B) acts as a boxcar anti-aliasing filter. The standard uncertainty of a mean of $m$ readings is $\frac{1}{m}\sqrt{\sum_{ij} r_{ij}u_iu_j}$; the correlation $r_{ij}$ is not documented in the dataset, so the conservative choice $r_{ij}=1$ is used, giving $u(\bar x)=\overline{u}$. For the dataset's constant (systematic) uncertainties this is exact; for any random component it overestimates (independent readings would give a factor $1/\sqrt{10}$ less).

**Loader validation** (on 50 cycles, against the `ZeMASamples` cache): `first`/100 reproduces the library's values and uncertainties exactly; `centre` column 0 equals library sample 5; `blockmean` column 0 equals the mean of library samples 0–9 for both values and uncertainties. Run 1 was executed twice and produced a byte-identical CSV.

**One code addition.** The 1 M-sample MC convergence check of `propagate_mc` allocates a $10^6\times n_{\mathrm{in}}$ float64 matrix (≈17.6 GB at 2200 inputs) and was killed for lack of memory. `propagate_mc_chunked` computes the same estimator in chunks of $10^5$ and is used only for that convergence check; the per-cycle $u_{\mathrm{physical}}^{\mathrm{MC}}$ values ($M=10^4$) still come from the unmodified `propagate_mc`.

---

## 3. Environment, data split, model

Unchanged from the baseline (`case_study_A_report.md` §3–§6): same `requirements.txt` and `zema_env`, same `self_test_lpu()` gate, same seed-0 random split (the **same 953 held-out cycles** for every variant), MLP (128, 64, 32) softplus, Adam lr $10^{-3}$, 200 full-batch epochs, float64, 10-seed ensemble for $u_{\mathrm{training}}$, seed-0 model as reference for LPU/MC and the sensitivity ranking. Only the input dimension changes.

*Update:* with the random split, 96% of the test cycles have a neighbouring cycle in the training set. Holding out whole blocks of 50 consecutive cycles raises RMSE from 0.066 to 0.074 (first 100 samples) and from 0.069 to 0.081 (without Pressure), with $R^2$ unchanged at 0.94–0.95 (`case_study_A_report_budget.md` §3.2).

---

## 4. Results

| Variant | RMSE | $R^2$ | MAE | $u_{\mathrm{phys}}$ LPU | $u_{\mathrm{phys}}$ MC | LPU/MC median diff | $u_{\mathrm{training}}$ | $u_c^{\mathrm{MC}}$ | Coverage $k=2$ ($u_c$) | Coverage $k=2$ ($u_{\mathrm{training}}$ only) | $u_c^{\mathrm{MC}}$ / MAE |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Baseline — first 100, 11 ch. | 0.0641 | 0.951 | 0.0461 | 0.641 | 0.984 | 36.5% | 0.0493 | 0.986 | 100% | 89.8% | 21.4× |
| Trimmed — first 100, transient removed | 0.0661 | 0.948 | 0.0483 | 0.655 | 0.910 | 28.3% | 0.0504 | 0.912 | 100% | 89.9% | 18.9× |
| **Run 1** — 200 decimated over 1 s, 11 ch. | 0.1059 | 0.868 | 0.0700 | 0.779 | 1.045 | 31.7% | 0.0719 | 1.048 | 100% | 91.9% | 15.0× |
| **Run 2** — 200 decimated over 1 s, no pressure | 0.0738 | 0.936 | 0.0516 | 0.282 | 0.356 | 22.0% | 0.0586 | 0.362 | 100% | 91.6% | 7.0× |
| **Control A** — first 100, no pressure | 0.0655 | 0.949 | **0.0429** | **0.260** | **0.319** | **20.7%** | **0.0469** | **0.323** | 100% | 90.7% | 7.5× |
| **Control B** — 200 block means over 1 s, 11 ch. | 0.0833 | 0.918 | 0.0603 | 1.168 | 1.982 | 45.0% | 0.0739 | 1.984 | 100% | 94.3% | 32.9× |

As in the baseline, the CSV `u_c` column combines the **LPU** estimate; since every LPU/MC gap stays well above the 5–10% threshold, Monte Carlo remains the primary $u_{\mathrm{physical}}$ and $u_c^{\mathrm{MC}}=\sqrt{u_{\mathrm{phys,MC}}^2+u_{\mathrm{training}}^2}$ is reported above.

### Sensitivity ranking (first-layer $|W^{(1)}|\cdot u_n$ share, seed-0 model)

| Channel | Baseline* | Run 1 | Run 2 | Control A | Control B |
|---|---|---|---|---|---|
| Pneumatic pressure | 51.8% | **53.9%** ($u_n$ 10.2) | — | — | 22.5% ($u_n$ 10.4) |
| Velocity | 25.2% | 24.9% ($u_n$ 4.7) | **54.0%** | **52.3%** ($u_n$ 4.3) | **31.0%** ($u_n$ 14.4) |
| Acceleration plain bearing | 11.5% | 10.2% | 22.3% | 23.9% | 23.4% ($u_n$ 11.0) |
| Acceleration ball bearing | 4.9% | 5.1% | 11.1% | 10.3% | 13.6% ($u_n$ 6.3) |
| Acceleration piston rod | 3.9% | 3.3% | 7.1% | 8.1% | 6.9% ($u_n$ 3.2) |
| Axial force | 1.3% | 1.2% | 2.6% | 2.6% | 0.6% |
| Active current | 1.0% | 1.0% | 2.2% | 2.0% | 1.1% |
| Motor current (×3) | 0.1% each | 0.1% each | 0.2% each | 0.3% each | 0.3% each |
| Sound pressure | ≈0 | ≈0 | ≈0 | ≈0 | ≈0 |

\* From `case_study_A_report.md` §7.

### MC convergence check (held-out test point #0)

| Variant | $M=10^4$ | $M=10^5$ | $M=10^6$ | Δ 10k→100k | Δ 100k→1M |
|---|---|---|---|---|---|
| Run 1 | 0.9850 | 0.9918 | 0.9897 | 0.69% | 0.22% |
| Run 2 | 0.3245 | 0.3240 | 0.3226 | 0.15% | 0.42% |
| Control A | 0.3058 | 0.3106 | 0.3114 | 1.56% | 0.26% |
| Control B | 1.9768 | 1.9845 | 1.9801 | 0.39% | 0.22% |

$M=10^4$ is adequately converged relative to the 20–45% LPU/MC gaps being measured.

---

## 5. Is $u_c$ actionable?

| Check | Baseline† | Run 1 | Run 2 | Control A | Control B |
|---|---|---|---|---|---|
| mean $u_c$ (LPU) vs. mean $\lvert$error$\rvert$ | 0.644 vs 0.046 (14.0×) | 0.783 vs 0.070 (11.2×) | 0.289 vs 0.052 (5.6×) | **0.265 vs 0.043 (6.2×)** | 1.171 vs 0.060 (19.4×) |
| corr($u_{\mathrm{phys}}^{\mathrm{MC}}$, $y_{\mathrm{true}}$) | +0.61 | +0.01 | −0.47 | **+0.45** | −0.38 |
| corr($u_c$, $\lvert$error$\rvert$) | 0.26 | 0.52 | 0.28 | 0.39 | 0.36 |
| early → late $u_{\mathrm{phys}}^{\mathrm{MC}}$ (terciles) | 0.933 → 1.064 | 1.016 → 1.024 | 0.391 → 0.339 | 0.306 → 0.339 | 2.033 → 1.938 |
| early → late mean $\lvert$error$\rvert$ | 0.033 → 0.056 | 0.049 → 0.079 | 0.042 → 0.067 | 0.031 → 0.055 | 0.056 → 0.063 |

† From `case_study_A_report_trimmed.md` §7.

Control A is the only variant that both shrinks the interval substantially and keeps an uncertainty that grows with machine age, as the actual error does. The whole-window variants lose (Run 1) or reverse (Run 2, Control B) that age trend.

---

## 6. Interpretation

**1. Removing Pneumatic pressure costs no accuracy and removes about two thirds of $u_{\mathrm{physical}}$** (Control A vs. baseline). RMSE 0.0655 vs 0.0641 and $R^2$ 0.949 vs 0.951 are within seed noise, MAE even improves (0.043 vs 0.046), while $u_{\mathrm{physical}}^{\mathrm{MC}}$ drops from 0.984 to 0.319 (−68%), the LPU/MC gap narrows from 36.5% to 20.7%, and $u_c$/MAE falls from 21× to 7.5×. The network does not need the Pressure channel to predict, but Pressure carries most of the propagated uncertainty. On the whole window (Run 1 vs. Run 2) Pressure even degrades the prediction (RMSE 0.106 → 0.074).

**2. Sampling the whole window does not help this model.** At equal channels, Run 1 is clearly worse than the baseline (RMSE 0.106 vs 0.064) and Run 2 is worse than Control A (0.074 vs 0.066).

**3. Aliasing explains only part of it** (Control B vs. Run 1). Block means improve RMSE from 0.106 to 0.083 but remain worse than the first 100 contiguous samples (0.064). Two remaining explanations are compatible with the data, and these runs do not separate them: the first 50 ms, contiguous at 2 kHz and phase-locked to the cycle trigger, preserve waveform shape that 200 sparse points lose; and doubling the input dimension (2200) with 3 813 training cycles and unchanged hyperparameters makes the MLP harder to train. To exploit the whole window, frequency-domain features (as in Dorst et al., whose relevant features lie at 250 Hz–1 kHz) are the natural next step rather than raw samples.

**4. Control B's large $u_{\mathrm{physical}}$ (1.98) is mostly the conservative correlation assumption.** Averaging shrinks the cross-cycle spread $s$ of each feature but, with $r_{ij}=1$, not the uncertainty, so $u/s$ rises (Velocity 14.4, Acceleration plain bearing 11.0). This is exact for the systematic (constant) uncertainties and an overestimate for any random component; without correlation information in the dataset the range cannot be narrowed. The same mechanism makes Velocity and the plain-bearing accelerometer, not Pressure, the top contributors in Control B.

**5. Over-coverage persists in every variant.** Coverage at $k=2$ with $u_c$ is 100% everywhere, while the $\pm 2u_{\mathrm{training}}$ interval alone already covers 90–94% of the test cycles, close to the ~95% target. The over-wide intervals come from the declared input uncertainties ($u_{\mathrm{physical}}$), not from the propagation method. After removing Pressure, Velocity ($u/s\approx4.3$) becomes the dominant contributor (≈52–54%), followed by the plain-bearing accelerometer (≈22–24%) — these are the next channels whose declared uncertainty should be scrutinized.

**Caveats.** One seed-0 split; the sensitivity ranking and the actionability correlations use the seed-0 model only. RMSE differences of ~0.002 (baseline vs. Control A vs. trimmed) are within seed noise; differences of 0.01–0.04 (whole-window vs. first-100) are not, but have not been re-confirmed across splits.

---

## 7. Recommendation

1. **Adopt Control A (first 100 samples, without Pneumatic pressure) as the reference configuration** for the paper, or at least report it alongside the baseline: same point accuracy, $u_c$ three times smaller, LPU/MC closer, and the only variant whose uncertainty still trends with machine age.
2. **State explicitly that the inputs are the first 50 ms of the 1 s return-stroke window** (or move to a whole-window representation) — "100 readings per cycle" is otherwise misleading.
3. **Do not switch to raw whole-window samples** with this MLP: neither decimation nor block means beat the first-50 ms input. A whole-window study should use spectral features.
4. **Report the input-uncertainty provenance as a limitation**: the ZeMA DAQ uncertainties are not derived in Dorst et al. 2022, and their correlation structure (needed for any averaging) is undocumented.

*Update:* recommendation 1 was based on the original budget. With the corrected budget (`case_study_A_report_budget.md`) the same-instrument $u_c$ is about the same with and without Pressure (0.110 vs 0.106 with the `noise` model), while the model without Pressure is slightly less accurate under the blocked split (RMSE 0.081 vs 0.074). Keeping all 11 channels and reporting the corrected budget is now the simpler choice; the no-pressure configuration remains useful when the model must be transferred to another instrument (systematic part 0.38 vs 0.77).

---

## 8. Validation figures

`zema_lpu_vs_mc_fullwindow.pdf`, `zema_lpu_vs_mc_fullwindow_no_pressure.pdf`, `zema_lpu_vs_mc_first100_no_pressure.pdf`, `zema_lpu_vs_mc_blockmean.pdf` — same two-panel layout as the baseline `fig:zema_lpu_vs_mc`, generated with the unmodified `make_validation_figure.py`. The ±k=2 band is visibly narrower in the two no-pressure figures but still covers the whole test set.

---

## 9. Files produced by these variants

| File | Role |
|---|---|
| `case_study_A_analysis_fullwindow.py` | Variant pipeline: reads the HDF5 directly, `--sampling centre\|first\|blockmean`, `--n-points`, `--exclude <channel>`; imports `case_study_A_analysis.py` and the diagnostics of `case_study_A_analysis_trimmed.py` unmodified |
| `case_study_A_results_fullwindow.csv` / `pipeline_run_fullwindow.log` / `zema_lpu_vs_mc_fullwindow.pdf` | Run 1 |
| `case_study_A_results_fullwindow_no_pressure.csv` / `pipeline_run_fullwindow_no_pressure.log` / `zema_lpu_vs_mc_fullwindow_no_pressure.pdf` | Run 2 |
| `case_study_A_results_first100_no_pressure.csv` / `pipeline_run_first100_no_pressure.log` / `zema_lpu_vs_mc_first100_no_pressure.pdf` | Control A |
| `case_study_A_results_blockmean.csv` / `pipeline_run_blockmean.log` / `zema_lpu_vs_mc_blockmean.pdf` | Control B |
| `case_study_A_report_fullwindow.md` | This document |

All CSVs share the baseline schema (`y_true, y_hat, u_physical_lpu, u_physical_mc, u_training, u_c, covered_k2`, 953 rows). No baseline or trimmed artifact is overwritten.

---

## 10. Reproducing this report

```bash
source zema_env/bin/activate
python case_study_A_analysis_fullwindow.py                                   # Run 1
python case_study_A_analysis_fullwindow.py --exclude "Pneumatic pressure"    # Run 2
python case_study_A_analysis_fullwindow.py --sampling first --n-points 100 --exclude "Pneumatic pressure"   # Control A
python case_study_A_analysis_fullwindow.py --sampling blockmean              # Control B
for t in fullwindow fullwindow_no_pressure first100_no_pressure blockmean; do
  python make_validation_figure.py --csv case_study_A_results_$t.csv --out zema_lpu_vs_mc_$t.pdf
done
```

Wall-clock time observed: roughly 5–10 minutes per variant on CPU with the dataset already cached (the HDF5 hash check and the 2200-input LPU dominate). Run the variants one at a time: two concurrent runs at 2200 inputs compete for memory during the 1 M-sample convergence check.
