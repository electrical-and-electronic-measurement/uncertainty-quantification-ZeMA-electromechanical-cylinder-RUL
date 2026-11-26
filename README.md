# GUM-based uncertainty evaluation of a neural-network RUL estimator on the ZeMA cylinder data set — code, per-cycle results and run logs

Supplementary material for Sections 5–5.4 of the paper: E. Buchicchio and
P. Carbone, *{{PAPER_TITLE}}*, submitted to *Measurement Science and
Technology*. (File names keep the prefix `case_study_A_` from an earlier
draft of the paper.) This case study applies the
paper's uncertainty-quantification methodology (Phases 1–5) to a real regression
task: estimating remaining useful life (RUL) of an electromechanical cylinder
from multi-sensor time-series data, with a full metrological uncertainty budget
computed both analytically and via Monte Carlo.

## How to cite and licence

Please cite both the paper above and this deposit (doi:{{ZENODO_DOI}});
citation metadata are in [CITATION.cff](CITATION.cff). Source code is released
under the MIT License, all other content (reports, figures, results, logs)
under CC BY 4.0; see [LICENSE](LICENSE). The ZeMA dataset (Dorst et al. 2021,
doi:10.5281/zenodo.5185953, CC BY 4.0) is not redistributed: the scripts
download it from Zenodo.

All figures are original.

## Use case

**Dataset.** ZeMA electromechanical cylinder, annotated (Dorst et al. 2021),
doi:10.5281/zenodo.5185953 — 4,766 run-to-failure cycles, 11 sensor channels
(three accelerometers — plain bearing, piston rod, ball bearing — active
current, axial force, three-phase motor current, pneumatic pressure, sound
pressure, velocity), each reading shipped with its own
metrologically evaluated standard uncertainty. Each cycle stores 1 s of the
return stroke at 2 kHz (2000 samples per channel); the baseline pipeline uses
only the **first 100 samples (first 50 ms)**, because
`ZeMASamples(datapoints_per_cycle=100)` slices `values[:100]` rather than
downsampling — see `case_study_A_report_fullwindow.md` §1.

**Task.** Regression. The measurand is the consumed-lifetime fraction at cycle
*c*: `y_c = c/N ∈ [0,1]`, with `RUL = 1 − y_c`. The label is exact for this
unit (defined from the cycle index); the definitional uncertainty of using it
as a proxy for the RUL of another unit is not evaluated.

**Goal.** Quantify, per test cycle, the standard uncertainty `u(ŷ)` on the
model's prediction by combining:

- **`u_physical`** — sensor input uncertainty propagated through a trained MLP, via
  both the analytical Law of Propagation of Uncertainty (LPU) and Monte Carlo
  (GUM Supplement 1 / JCGM 101:2008), so the two can be cross-checked against
  each other;
- **`u_training`** — training-process variability, estimated statistically as the
  spread across a 10-seed model ensemble (initialization only: training is
  full-batch);
- **`u_calibration`** — any residual variance not explained by the two terms
  above, estimated on a validation set (corrected budget only; 0 in every run);

combined in quadrature, `u_c = sqrt(u_physical² + u_training² + u_calibration²)`, and validated against
empirical coverage of the resulting `±k·u_c` interval.

**Original budget (first run).** LPU and Monte Carlo disagree by a median
**36.5%**, so Monte Carlo was reported as the primary `u_physical` estimate
(per the paper's own decision rule). Coverage at `k=2` is **100%** (vs. an
~95% target), with one dominant sensor channel (Pressure, 51.8% of
first-layer sensitivity): its declared standard uncertainty is ~9× its
cycle-to-cycle variation (`u/s ≈ 8.8`, median over the first 100 samples, training cycles;
≈10.6 over the whole 1 s window), a property of the data that z-score
normalization preserves rather than creates. The point prediction itself is
strong (`R² = 0.951`, 4.5× better RMSE than a naive mean-baseline). See
`case_study_A_report.md` §7. *The diagnosis of the first run (linearity
breaking down at this network depth, over-coverage caused by one amplified
channel) is superseded by the corrected budget below: the 36.5% gap and the
over-coverage come from propagating a mostly systematic declared uncertainty
as independent noise.*

**Follow-up variants.** Removing the Pressure channel keeps the same accuracy
(RMSE 0.0655, `R² = 0.949`) while cutting MC `u_physical` by 68% (0.984 →
0.319) and the LPU/MC gap to 20.7%. Sampling the whole 1 s window instead of
the first 50 ms does not help this MLP (decimation: RMSE 0.106; block means:
0.083). Coverage stays at 100% in every variant, while `u_training` alone
covers 90–94%. See `case_study_A_report_fullwindow.md`. These variants use
the original budget; with the corrected budget below, removing Pressure
changes the same-instrument `u_c` very little (0.110 → 0.106 with the `noise`
model) and only halves the systematic part relevant for transfer.

**Corrected budget.** The original `u_physical` treats every declared input
uncertainty as independent per-sample noise. The data show that most of it
must be common to all cycles (`u > s` for Pressure, Velocity, plain-bearing
acceleration), and a systematic error common to all cycles cancels exactly in
the z-score normalization when the same instrument is used for training and
inference. Splitting `u` into random and systematic parts (a choice of input covariance
consistent with GUM 3.4.3, 5.2.4, F.1.2.3–F.1.2.4 and E.3.6; see
`case_study_A_report_budget.md` §2.1.1), propagating only the random part, adding a validation-based `u_calibration` and combining with Monte Carlo
gives `u_c ≈ 0.11–0.14` instead of ≈0.98. LPU and MC then agree within 4–7%,
and `u_c` is 2.3–2.9× the mean error instead of 21×. Coverage at k=2 is
still 99.5–99.9%, because the data-based estimates of the random part are upper
bounds. Using the manufacturer's noise floor where one is published (only the
Kistler 8704B50M1 accelerometers, `--random-model datasheet`) lowers `u_c` to
0.089–0.093 (LPU/MC within 3–4%, `u_c` ≈ 2× the mean error), but coverage at
k=2 stays at 99.4–99.6%: Velocity and the plain-bearing accelerometer, which
have no noise specification, still hit the data-based cap. The systematic part
(≈0.76–0.78, or ≈0.38 without Pressure) is reported separately, for transfer to
another instrument. See `case_study_A_report_budget.md`; the sensor
specifications are compared with the declared uncertainties in
`case_study_A_dataset_report.md` §4.1.
For comparison (`datasheet`, random split), two intervals that do not
propagate the inputs: the seed ensemble alone has the same correlation with
the error (r = 0.45) but covers 89.8% at k=2; adding a constant term from the
validation residuals (0.039) gives 96.0% with mean `u` = 0.066 (§3.4 of
`case_study_A_report_budget.md`, block "Comparison" of
`pipeline_run_budget_first100_datasheet.log`).
A blocked split (whole blocks of consecutive cycles held out, so no test cycle
has a training neighbour) raises RMSE by only 5–16% (R² 0.94–0.95) and leaves
the budget conclusions unchanged (§3.2 of that report).

## Reproducing the paper's numbers

Run from the repository root after the setup of §0 below. Each budget command
prints its log to stdout (redirect it to `pipeline_run_budget_<variant>.log`)
and writes `case_study_A_results_budget_<variant>.csv`; the first command
prints `pipeline_run.log` and writes `case_study_A_results.csv`.

```bash
python case_study_A_analysis.py                                                                    # preliminary budget (§5.2): 0.98, 36.5%, 100%, MC convergence, MC mean shift
python case_study_A_analysis_budget.py --random-model datasheet                                    # Table 1, Fig. 4, §5.2, §5.4
python case_study_A_analysis_budget.py --random-model noise                                        # 4.7%, u_c 0.110, 99.8%
python case_study_A_analysis_budget.py --random-model bound                                        # 6.3%, u_c 0.136, 99.9%; 98.7/93.9/74%
python case_study_A_analysis_budget.py --random-model datasheet --exclude "Pneumatic pressure"     # §5.3: u_sys 0.383, u_c 0.089
python case_study_A_analysis_budget.py --random-model datasheet --split blocked                    # §5.2 blocked split: RMSE 0.074, R² 0.95, 3.8%, 0.093, 99.3%
python make_validation_figure.py --csv case_study_A_results_budget_first100_datasheet.csv \
       --out zema_lpu_vs_mc_budget_first100_datasheet.pdf                                           # Fig. 4
```

| Paper | File |
|---|---|
| Table 1, Fig. 4, §5.2 (linearity, coverage, residuals, MC convergence), §5.4 | `pipeline_run_budget_first100_datasheet.log`, `case_study_A_results_budget_first100_datasheet.csv` |
| §5.2: 4.7%, `u_c` = 0.110 | `pipeline_run_budget_first100_noise.log` |
| §5.2: 6.3%, `u_c` = 0.136; §5.1: 98.7 / 93.9 / 74% | `pipeline_run_budget_first100_bound.log` |
| §5.2: blocked split | `pipeline_run_budget_first100_datasheet_blocked50.log` |
| §5.3: 0.383, 0.089 | `pipeline_run_budget_first100_no_pressure_datasheet.log` |
| §5.2, §5.4: preliminary budget (0.98, 36.5%, MC convergence, MC mean shift) | `pipeline_run.log`, `case_study_A_results.csv`, `case_study_A_report.md` §7 |
| §5.1: manufacturers' specifications | `case_study_A_dataset_report.md` §4.1 |
| §5.1: 4766 cycles, one every 100, 1% / 1.5% of reading | `dataset_consistency.log` |

**Log provenance.** All `pipeline_run.log` and `pipeline_run_budget_*.log` files and the `case_study_A_results_budget_*.csv` files were regenerated on 26 Nov 2026 from the code of release 1.0.0 (Python 3.10.12, pinned `requirements.txt`). The re-run is deterministic: every line of the previous logs and every value of the previous CSVs is reproduced unchanged; the re-run only adds the MC convergence block, the MC mean shift and the CSV column `mc_bias_indep`. The validation figures `zema_lpu_vs_mc_budget_*.pdf` were not regenerated (a regeneration gives the same content). `pipeline_run.log` and
`case_study_A_results.csv` were first produced in July 2026 with the labels
`u_A`/`u_B`; the labels were renamed in October 2026 to the paper's notation
(`u_physical`/`u_training`) and the run was repeated for this release. The
input standardization uses `std` with divisor *J* (ddof = 0) instead of the
*J* − 1 of the paper's appendix; with *J* = 3242 the difference is 0.015%.

## Code provenance and verification

The code was written with the help of Claude (Anthropic) and reviewed by the
authors. It was verified by: the LPU self-test against a 200 000-sample Monte
Carlo on a random MLP (`--self-test`, agreement < 2%); the agreement of LPU and
MC on the real model (3.6% for the final budget); the MC convergence check in
each log; the notebook `case_study_A_run_analysis.ipynb`, which re-runs the
pipeline cell by cell and reproduces `pipeline_run.log`; and
`check_dataset_consistency.py`, which recomputes every number of the dataset
report from the HDF5 file.

## Tools

| File | Role |
|---|---|
| [case_study_A_analysis.py](case_study_A_analysis.py) | Core pipeline: downloads/loads the ZeMA dataset, z-scores inputs (with exact GUM propagation through normalization), trains a 10-seed MLP ensemble, computes `u_physical` via both `propagate_lpu()` (analytical LPU) and `propagate_mc()` (Monte Carlo), combines `u_c`, and writes `case_study_A_results.csv`. Also contains `self_test_lpu()`, an implementation-correctness gate independent of the dataset. |
| [make_validation_figure.py](make_validation_figure.py) | Reads a results CSV and renders the two-panel validation figure (`fig:zema_lpu_vs_mc`): LPU-vs-MC scatter with identity line, and predictions ±k=2 band vs. ground truth with coverage annotated. |
| [requirements.txt](requirements.txt) / [requirements-dev.txt](requirements-dev.txt) | Pinned versions used to produce every real result (Python 3.10.12): `zema_emc_annotated==0.7.1`, `torch==2.13.0`, `numpy==1.26.4`, `matplotlib==3.10.9`, `h5py`, `pooch`, plus `pandas`, `seaborn`, `ipykernel` for the notebooks. `requirements-dev.txt` adds `nbconvert` and `playwright`, needed only to re-export the notebooks to PDF. |
| [case_study_A_report.md](case_study_A_report.md) | Full execution report: every number, decision, and diagnosis behind the results below. This is the primary reference document. |
| [case_study_A_dataset_report.md](case_study_A_dataset_report.md) | Description of the full HDF5 file (ZeMA DAQ and SmartUp Unit, 24 signals: sensors, units, ranges, uncertainty models, `u/s` ratios, within-window behaviour), a comparison of the declared uncertainties with the manufacturers' specifications of the main sensors (§4.1), and a 22-item consistency check against Dorst et al. 2022, with the dataset issues found (cycle-count typo, axial load below nominal, undocumented ZeMA uncertainties, duplicated SUU uncertainty arrays). |
| [check_dataset_consistency.py](check_dataset_consistency.py) | Prints every number in `case_study_A_dataset_report.md` (output in `dataset_consistency.log`). |
| [case_study_A_eda.ipynb](case_study_A_eda.ipynb) | Exploratory data analysis of the raw ZeMA input data (value/uncertainty distributions per channel, degradation trends, correlations, sanity checks) — run before modeling, independent of `case_study_A_analysis.py`. |
| [case_study_A_eda_outliers.ipynb](case_study_A_eda_outliers.ipynb) | Follow-up to the EDA above: investigates the start-of-cycle spikes visible in Pneumatic pressure/Velocity (shown to be a within-cycle acquisition transient, not a life-stage effect), builds a data-driven detector for genuine early-life ("running-in") outlier cycles on other channels (notably Axial force, Motor current phase 1), and reruns the key EDA views with those cycles excluded. |
| [case_study_A_feature_trimming.ipynb](case_study_A_feature_trimming.ipynb) | Implements and validates the outliers notebook's recommendation to trim the leading acquisition-transient samples (Pneumatic pressure K=7, Velocity K=6, Axial force K=3) before feature extraction: builds the trimmed 1,084-dim feature set, then trains the pipeline's own unmodified MLP (`make_mlp`/`train`, 10-seed ensemble) on raw vs. trimmed features under an identical split and compares test RMSE/R². Includes the exact patch point in `case_study_A_analysis.py` to adopt it. |
| [case_study_A_run_analysis.ipynb](case_study_A_run_analysis.ipynb) | Executes the full GUM pipeline end to end as a cell-by-cell walkthrough (Phases 1–5), reusing every function from `case_study_A_analysis.py` unmodified, for two scenarios: **A** raw features (reproduces the *original-budget* numbers of `pipeline_run.log` exactly, as a fidelity check) and **B** trimmed features via `build_trimmed_dataset()` — recomputing the *full* uncertainty budget (`u_physical` LPU+MC, `u_training`, `u_c`, coverage), not just RMSE, so the trimming recommendation can be judged on the uncertainty quantification itself. Does not overwrite `case_study_A_results.csv`. |
| [case_study_A_analysis_trimmed.py](case_study_A_analysis_trimmed.py) | Standalone, CLI-runnable sibling of `case_study_A_analysis.py` (same imports, unmodified) that trims the acquisition-transient samples before training and writes `case_study_A_results_trimmed.csv`, plus the extra diagnostics (sensitivity ranking, MC convergence, `u_c`-actionability checks) behind `case_study_A_report_trimmed.md`. |
| [case_study_A_report_trimmed.md](case_study_A_report_trimmed.md) | Full execution report for the trimmed-feature variant, structured like `case_study_A_report.md`, with an explicit section-by-section comparison against the baseline (no-trim) results — including a notable finding the point-prediction-only comparison in `case_study_A_feature_trimming.ipynb` couldn't see: trimming reverses the sign of the (weak) age-vs-uncertainty correlation. |
| [case_study_A_analysis_fullwindow.py](case_study_A_analysis_fullwindow.py) | CLI-runnable sibling of `case_study_A_analysis.py` (same imports, unmodified) that reads the HDF5 directly to choose which within-cycle samples feed the model — `--sampling centre` (decimation over the whole 1 s window, default), `first` (what `ZeMASamples` returns) or `blockmean` (block means, conservative fully-correlated uncertainty) — with `--n-points` and `--exclude <channel>` for channel ablations. Writes `case_study_A_results_<variant>.csv` plus the same diagnostics as the trimmed variant. |
| [case_study_A_report_fullwindow.md](case_study_A_report_fullwindow.md) | Execution report for four variants (whole-window decimation with/without Pressure, first 100 samples without Pressure, whole-window block means), compared with the baseline and trimmed runs: per-channel `u/s` analysis, full uncertainty budget, sensitivity ranking, MC convergence, actionability checks and recommendation. |
| [case_study_A_analysis_budget.py](case_study_A_analysis_budget.py) | Corrected uncertainty budget on the baseline model and test cycles: random/systematic split of the declared input uncertainty (`--random-model bound\|noise`, data-based upper bounds of the random part; `datasheet`, which also applies the manufacturer's noise floor where one is published), random part propagated for same-instrument use (LPU + MC), systematic part propagated as a per-channel common shift for instrument transfer, `u_calibration` from a validation set, MC-based `u_c` and coverage; also recomputes the original budget for comparison. Supports `--exclude`, `--sampling`, `--n-points`. |
| [case_study_A_report_budget.md](case_study_A_report_budget.md) | Execution report of the corrected budget (baseline input and without Pressure, all three random models, random and blocked split): why the original budget is not correct, method and verification, results, residual consistency check, recommendation. |
| [case_study_A_eda.pdf](case_study_A_eda.pdf) / [case_study_A_eda_outliers.pdf](case_study_A_eda_outliers.pdf) / [case_study_A_feature_trimming.pdf](case_study_A_feature_trimming.pdf) / [case_study_A_run_analysis.pdf](case_study_A_run_analysis.pdf) | Rendered PDF exports of the four notebooks above, for sharing without a notebook environment. Regenerate with `python export_notebooks_to_pdf.py` after editing the notebooks. |
| [export_notebooks_to_pdf.py](export_notebooks_to_pdf.py) | Renders the EDA notebooks to the PDFs above with wrapped code lines (plain `nbconvert --to webpdf` clips long lines). Needs `requirements-dev.txt` and `playwright install chromium`. |
| [paper_figures/nn_neuron_lpu.tex](paper_figures/nn_neuron_lpu.tex) / [.pdf](paper_figures/nn_neuron_lpu.pdf) | Figure 1 of the paper (`fig:Schematic_representation_artificial_neuron`): (a) artificial neuron of layer *i*, (b) MLP with layer-wise covariance propagation, in the paper's notation (σ_i, **W**_i, **b**_i, **x**^(i), **H**_i). |
| [paper_figures/decision_tree.tex](paper_figures/decision_tree.tex) / [.pdf](paper_figures/decision_tree.pdf) | Figure 2 of the paper (`fig:decision_tree`, five-phase methodology), aligned with the corrected budget; compiles with plain `pdflatex`. |
| [paper_figures/ml_assisted_process.tex](paper_figures/ml_assisted_process.tex) / [.pdf](paper_figures/ml_assisted_process.pdf) | Figure 3 of the paper (`fig:ML_assisted_process`): measurement → dataset → model f̂ → indirect measurement, with the inputs at inference and the data/model uncertainty groups mapped to `u_physical` and `u_model` |

### Real result artifacts

| File | Role |
|---|---|
| [case_study_A_results.csv](case_study_A_results.csv) | Real output, 953 held-out test cycles × 7 columns: `y_true, y_hat, u_physical_lpu, u_physical_mc, u_training, u_c, covered_k2`. |
| [zema_lpu_vs_mc.pdf](zema_lpu_vs_mc.pdf) | Validation figure of the **original (superseded) budget** (36.5% LPU/MC gap), generated from `case_study_A_results.csv`. Figure 4 of the paper is `zema_lpu_vs_mc_budget_first100_datasheet.pdf`. |
| [pipeline_run.log](pipeline_run.log) | Captured stdout of the `case_study_A_analysis.py` run that produced the results above, including the MC convergence check and the mean shift of the MC mean. |
| [case_study_A_report.pdf](case_study_A_report.pdf) / [case_study_A_report.tex](case_study_A_report.tex) | Rendered/typeset versions of the report. |
| [case_study_A_results_trimmed.csv](case_study_A_results_trimmed.csv) | Same schema as `case_study_A_results.csv`, but on the trimmed (1,084-dim) feature set — output of `case_study_A_analysis_trimmed.py`. |
| [zema_lpu_vs_mc_trimmed.pdf](zema_lpu_vs_mc_trimmed.pdf) | The `fig:zema_lpu_vs_mc` validation figure for the trimmed-feature run. |
| [pipeline_run_trimmed.log](pipeline_run_trimmed.log) | Captured stdout of the `case_study_A_analysis_trimmed.py` run behind `case_study_A_report_trimmed.md`. |
| `case_study_A_results_budget_<variant>.csv` / `zema_lpu_vs_mc_budget_<variant>.pdf` / `pipeline_run_budget_<variant>.log` | Outputs of `case_study_A_analysis_budget.py` behind `case_study_A_report_budget.md`: `first100_{bound,noise,datasheet}` and `first100_no_pressure_{bound,noise,datasheet}`, plus the blocked-split runs with suffix `_blocked<size>`. The CSV keeps the 7 standard columns (`u_physical_*` = random part) plus `mc_bias, u_model` (`u_model` = `u_calibration`), `u_sys_lpu, u_sys_mc, u_c_transfer, u_indep_mc, mc_bias_indep` (`mc_bias_indep` = MC mean − f(x) of the original budget). Each log ends with an MC convergence check on test cycle #0 (M = 10⁴/10⁵/10⁶, `u_rand` and `u_sys`). |
| `case_study_A_results_<variant>.csv` / `zema_lpu_vs_mc_<variant>.pdf` / `pipeline_run_<variant>.log` | Results CSV (same schema), validation figure and captured stdout of `case_study_A_analysis_fullwindow.py` for each variant behind `case_study_A_report_fullwindow.md`: `fullwindow`, `fullwindow_no_pressure`, `first100_no_pressure`, `blockmean`. |

## Results summary

### Corrected budget (same 953 test cycles, random split, 15% of training held out for `u_calibration`)

| Input | Random model | RMSE | `u_rand` MC | LPU/MC diff | `u_c` | Coverage k=2 / k=1 | `u_sys` (transfer) |
|---|---|---|---|---|---|---|---|
| first 100, 11 ch. | `bound` | 0.0664 | 0.124 | 6.3% | 0.136 | 99.9% / 96.4% | 0.758 |
| first 100, 11 ch. | `noise` | 0.0664 | 0.095 | 4.7% | 0.110 | 99.8% / 93.1% | 0.767 |
| **first 100, 11 ch.** | **`datasheet`** | **0.0664** | **0.076** | **3.6%** | **0.093** | **99.6% / 89.2%** | **0.782** |
| first 100, no pressure | `bound` | 0.0694 | 0.098 | 4.7% | 0.111 | 99.5% / 95.0% | 0.376 |
| first 100, no pressure | `noise` | 0.0694 | 0.092 | 4.2% | 0.106 | 99.5% / 93.8% | 0.380 |
| first 100, no pressure | `datasheet` | 0.0694 | 0.073 | 3.2% | 0.089 | 99.4% / 90.2% | 0.383 |

`u_calibration` (validation) is 0 in every run. The bold row is the one used in
Section 5.3 of the paper. MC convergence on test cycle #0 (bold row, M = 10⁴ →
10⁵ → 10⁶): `u_rand` changes by 0.75% and 0.28%, `u_sys` by 0.12% and 0.03%
(last block of each `pipeline_run_budget_*.log`; < 2% from 10⁴ to 10⁵ in
every variant). Blocked split and details in
`case_study_A_report_budget.md` §3.

### Original budget (superseded)

Means over the 953-cycle held-out test set:

| Quantity | Value |
|---|---|
| Test RMSE | 0.0641 |
| R² | 0.951 |
| mean `u_physical` (LPU) | 0.6413 |
| mean `u_physical` (Monte Carlo, primary) | 0.9841 |
| LPU vs. MC median relative difference | 36.47% |
| mean `u_training` (10-seed ensemble) | 0.0493 |
| mean `u_c` (combined, MC-based; the CSV `u_c` column uses the LPU value, mean 0.6437) | ≈0.985 |
| Empirical coverage @ k=2 | 100.0% (953/953) |
| MC convergence (10k→100k→1M, test point #0) | Δ 1.75% → 0.08% |
| MC mean − f(x), mean over test cycles | 0.47 (0.468) |

Full uncertainty budget table and per-channel sensitivity ranking (Pressure
51.8%, Velocity 25.2%, plain-bearing acceleration 11.5% — 88.5% combined) are in
`case_study_A_report.md` §7 and §9. These are the original-budget numbers;
the paper uses the corrected budget above.

### Variants (same 953 test cycles, same split and seeds)

| Variant | Input | RMSE | R² | `u_physical` MC | LPU/MC diff | `u_training` | Coverage k=2 (`u_c` / `u_training` only) |
|---|---|---|---|---|---|---|---|
| Baseline | first 100 samples, 11 ch. (1100) | 0.0641 | 0.951 | 0.984 | 36.5% | 0.049 | 100% / 89.8% |
| Trimmed | first 100, transient removed (1084) | 0.0661 | 0.948 | 0.910 | 28.3% | 0.050 | 100% / 89.9% |
| Whole window | 200 decimated over 1 s, 11 ch. (2200) | 0.1059 | 0.868 | 1.045 | 31.7% | 0.072 | 100% / 91.9% |
| Whole window, no pressure | 200 decimated over 1 s, 10 ch. (2000) | 0.0738 | 0.936 | 0.356 | 22.0% | 0.059 | 100% / 91.6% |
| **First 100, no pressure** | first 100, 10 ch. (1000) | **0.0655** | **0.949** | **0.319** | **20.7%** | **0.047** | 100% / 90.7% |
| Block means | 200 means over 1 s, 11 ch. (2200) | 0.0833 | 0.918 | 1.982 | 45.0% | 0.074 | 100% / 94.3% |

Details, sensitivity rankings and caveats in `case_study_A_report_fullwindow.md`.

## Tool usage guidance

### 0. Environment setup

`zema_emc_annotated` pins `numpy<2`, and `numpy==1.26.4` has no wheels for
Python ≥ 3.13, so use a dedicated virtual environment on **Python 3.10**
(the pinned versions in `requirements.txt` were validated against 3.10.12;
3.13+ is known to fail to install).

The recommended way is [uv](https://docs.astral.sh/uv/), which downloads the
required Python version automatically if it is not installed, so no system-wide
Python 3.10 is needed. Name the environment `.venv` (picked up automatically by
VS Code).

Install uv (once):

```bash
# Windows (PowerShell)
winget install --id astral-sh.uv
# macOS / Linux
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Create the environment and install the dependencies:

```bash
uv venv --python 3.10 .venv

# Windows (PowerShell)
.\.venv\Scripts\Activate.ps1
# macOS / Linux
source .venv/bin/activate

uv pip install -r requirements.txt
```

Without activating the environment, point uv at the interpreter explicitly:

```bash
# Windows
uv pip install --python .\.venv\Scripts\python.exe -r requirements.txt
# macOS / Linux
uv pip install --python .venv/bin/python -r requirements.txt
```

Use `uv venv --python 3.10 --clear .venv` to recreate an existing `.venv` that
was built with the wrong Python version. In VS Code, select `.venv` as the
interpreter / notebook kernel.

The reports use an environment named `zema_env`
(`python3.10 -m venv zema_env`); it is equivalent to `.venv`.

Plain `venv` + `pip` also works if Python 3.10 is already installed:

```bash
python3.10 -m venv .venv
# activate as above, then:
pip install -r requirements.txt
```

### 1. Validate the propagation code before touching the dataset

```bash
python case_study_A_analysis.py --self-test
# expected: [self-test] u_LPU=...  u_MC=...  rel.diff=<2%
```

This builds a random softplus MLP, propagates a diagonal input covariance
analytically, compares against a 200,000-sample Monte Carlo run, and hard-
asserts agreement within 2%. It needs no dataset and no download. If it
fails, do not trust any downstream `u_physical` numbers — the bug is in
`propagate_lpu`/`propagate_mc`, not in the case study's data or model.

### 2. Run the full pipeline

```bash
python case_study_A_analysis.py
```

- First run downloads the annotated ZeMA HDF5 (2.67 GB) from Zenodo into a
  local cache — budget roughly an hour on first run; it's cached afterward.
- Trains a 10-seed MLP ensemble (hidden layers `(128, 64, 32)`, softplus,
  float64, Adam lr=1e-3, 200 epochs) at full resolution
  (`datapoints_per_cycle=100` → input dim 1100). Note that
  `datapoints_per_cycle` selects the *first* N of the 2000 samples per cycle
  (the first N/2 ms of the 1 s window), not N samples spread over it; use
  `case_study_A_analysis_fullwindow.py --sampling centre|blockmean` for the
  latter. Expect a few hours on CPU; trivially parallelizable over seeds if
  needed.
- Writes `case_study_A_results.csv` (one row per held-out test cycle).
- Key parameters live in the `Config` dataclass at the top of the script
  (`n_cycles`, `datapoints_per_cycle`, `hidden`, `n_ensemble`, `n_mc`,
  `epochs`, `lr`, `seed`) — edit there to reduce `datapoints_per_cycle` to
  `10` (input dim 110) for faster iteration during development; report
  whichever configuration you finalize.
- If the automatic download is slow or blocked, download the HDF5 manually
  from the Zenodo record (doi:10.5281/zenodo.5185953);
  `ZeMASamples(..., skip_hash_check=True)` accepts a manually placed file in
  the cache directory.

### 2b. (Optional) window-sampling and channel-ablation variants

```bash
python case_study_A_analysis_fullwindow.py                                   # 200 samples decimated over 1 s
python case_study_A_analysis_fullwindow.py --exclude "Pneumatic pressure"    # same, without Pressure
python case_study_A_analysis_fullwindow.py --sampling first --n-points 100 --exclude "Pneumatic pressure"
python case_study_A_analysis_fullwindow.py --sampling blockmean              # 200 block means over 1 s
```

Each writes `case_study_A_results_<variant>.csv` without touching the baseline
artifacts. Run them one at a time: with 2200 inputs the 1 M-sample MC
convergence check is computed in chunks but still uses several GB.

### 3. Generate the validation figure

```bash
# Figure 4 of the paper (corrected budget)
python make_validation_figure.py --csv case_study_A_results_budget_first100_datasheet.csv \
       --out zema_lpu_vs_mc_budget_first100_datasheet.pdf
# original (superseded) budget
python make_validation_figure.py --csv case_study_A_results.csv --out zema_lpu_vs_mc.pdf
```

`--csv` and `--out` default to the original-budget files, so a bare
invocation reproduces `zema_lpu_vs_mc.pdf`.

### 4. Rebuild the paper figures

The three schematic figures in `paper_figures/` are standalone TikZ files that
compile with plain `pdflatex` (a TeX distribution with TikZ, e.g. TeX Live or
MacTeX) to a tightly cropped one-page PDF:

```bash
cd paper_figures
for f in *.tex; do pdflatex -interaction=nonstopmode "$f"; done
rm -f *.aux *.log
```

Each `<name>.tex` writes `<name>.pdf` next to it; copy the PDFs next to the
paper's `.tex`.

### Interpreting a re-run that disagrees with this report

- **LPU vs. MC median relative difference** within a tolerance stated in
  advance → the linearization is adequate and LPU can be reported alongside
  MC; outside it → report MC as primary. The paper uses MC for `u_c` in either
  case. Either outcome is a valid, publishable result — do not tune the
  network to force agreement.
- **Coverage @ k=2** far from ~95%: under-coverage usually means a missing
  uncertainty source (check residual bias first). For over-coverage (as found
  here), first compare each channel's declared `u` with its cross-cycle spread
  `s`: where `u > s` most of `u` is systematic and must not be propagated as
  independent noise (`case_study_A_report_budget.md` §1–§2). Then compare the
  propagated random part with what the residuals allow (§3.1 there).
- **Ensemble `u_training`**: with only 10 seeds, `u_training` itself carries ~24% relative
  sampling noise (χ², 9 dof). If `u_training` turns out to dominate `u_c`, increase
  `n_ensemble` to 20–30 before finalizing the budget table.
