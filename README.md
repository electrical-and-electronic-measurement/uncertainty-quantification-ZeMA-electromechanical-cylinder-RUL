# Case Study A — GUM-Compliant Uncertainty Quantification for RUL Estimation

Supplementary material for Section 5.1 of the paper. This case study applies the
paper's uncertainty-quantification methodology (Phases 1–5) to a real regression
task: estimating remaining useful life (RUL) of an electromechanical cylinder
from multi-sensor time-series data, with a full metrological uncertainty budget
computed both analytically and via Monte Carlo.

## Use case

**Dataset.** ZeMA electromechanical cylinder, annotated (Dorst et al. 2021),
doi:10.5281/zenodo.5185953 — 4,766 run-to-failure cycles, 11 sensor channels
(acceleration x/y/z, active current, force, three-phase motor current,
pressure, sound pressure, velocity), each reading shipped with its own
metrologically evaluated standard uncertainty.

**Task.** Regression. The measurand is the consumed-lifetime fraction at cycle
*k*: `y_k = k/N ∈ [0,1]`, with `RUL = 1 − y_k`. The label is deterministic
(defined purely from the cycle index), so label uncertainty is negligible by
construction.

**Goal.** Quantify, per test cycle, the standard uncertainty `u(ŷ)` on the
model's prediction by combining:

- **`u_physical`** — sensor input uncertainty propagated through a trained MLP, via
  both the analytical Law of Propagation of Uncertainty (LPU) and Monte Carlo
  (GUM Supplement 1 / JCGM 101:2008), so the two can be cross-checked against
  each other;
- **`u_training`** — training-process variability, estimated statistically as the
  spread across a 10-seed model ensemble;

combined in quadrature, `u_c = sqrt(u_physical² + u_training²)`, and validated against
empirical coverage of the resulting `±k·u_c` interval.

**Headline result.** LPU and Monte Carlo disagree by a median **36.5%** — the
first-order linearity assumption breaks down at this network depth and input-
uncertainty magnitude, so Monte Carlo is reported as the primary `u_physical`
estimate (per the paper's own decision rule, not a bug). Coverage at `k=2` is
**100%** (vs. an ~95% target), traced to one dominant, normalization-amplified
sensor channel (Pressure, 51.8% of first-layer sensitivity) rather than a
missing uncertainty source. The point prediction itself remains strong
(`R² = 0.951`, 4.5× better RMSE than a naive mean-baseline) even though the
uncertainty interval is currently too wide to be per-prediction actionable.
See `case_study_A_report.md` §7 for the full diagnosis.

## Tools

| File | Role |
|---|---|
| [case_study_A_analysis.py](case_study_A_analysis.py) | Core pipeline: downloads/loads the ZeMA dataset, z-scores inputs (with exact GUM propagation through normalization), trains a 10-seed MLP ensemble, computes `u_physical` via both `propagate_lpu()` (analytical LPU) and `propagate_mc()` (Monte Carlo), combines `u_c`, and writes `case_study_A_results.csv`. Also contains `self_test_lpu()`, an implementation-correctness gate independent of the dataset. |
| [make_validation_figure.py](make_validation_figure.py) | Reads a results CSV and renders the two-panel validation figure (`fig:zema_lpu_vs_mc`): LPU-vs-MC scatter with identity line, and predictions ±k=2 band vs. ground truth with coverage annotated. |
| [generate_synthetic_case_study_A_data.py](generate_synthetic_case_study_A_data.py) | Dev/testing-only helper. Produces a fake CSV with the same schema as the real pipeline so `make_validation_figure.py` could be exercised before the real dataset finished downloading. Output is explicitly self-labeled `*** SYNTHETIC DATA ***` — never cite it in the paper. |
| [requirements.txt](requirements.txt) | Pinned versions used to produce every real result: `zema_emc_annotated==0.7.1`, `torch==2.13.0`, `numpy==1.26.4`, `matplotlib==3.10.9`. |
| [case_study_A_guide.md](case_study_A_guide.md) | Step-by-step execution guide, mapped explicitly onto the paper's Phase 1–5 methodology. Read this before re-running anything. |
| [case_study_A_report.md](case_study_A_report.md) | Full execution report: every number, decision, and diagnosis behind the results below. This is the primary reference document. |
| [case_study_A_eda.ipynb](case_study_A_eda.ipynb) | Exploratory data analysis of the raw ZeMA input data (value/uncertainty distributions per channel, degradation trends, correlations, sanity checks) — run before modeling, independent of `case_study_A_analysis.py`. Needs `pandas`/`seaborn` in addition to `requirements.txt`. |
| [case_study_A_eda_outliers.ipynb](case_study_A_eda_outliers.ipynb) | Follow-up to the EDA above: investigates the start-of-cycle spikes visible in Pneumatic pressure/Velocity (shown to be a within-cycle acquisition transient, not a life-stage effect), builds a data-driven detector for genuine early-life ("running-in") outlier cycles on other channels (notably Axial force, Motor current phase 1), and reruns the key EDA views with those cycles excluded. |
| [case_study_A_feature_trimming.ipynb](case_study_A_feature_trimming.ipynb) | Implements and validates the outliers notebook's recommendation to trim the leading acquisition-transient samples (Pneumatic pressure K=7, Velocity K=6, Axial force K=3) before feature extraction: builds the trimmed 1,084-dim feature set, then trains the pipeline's own unmodified MLP (`make_mlp`/`train`, 10-seed ensemble) on raw vs. trimmed features under an identical split and compares test RMSE/R². Includes the exact patch point in `case_study_A_analysis.py` to adopt it. |
| [case_study_A_run_analysis.ipynb](case_study_A_run_analysis.ipynb) | Executes the full GUM pipeline end to end as a cell-by-cell walkthrough (Phases 1–5), reusing every function from `case_study_A_analysis.py` unmodified, for two scenarios: **A** raw features (reproduces the headline `README.md` numbers exactly, as a fidelity check) and **B** trimmed features via `build_trimmed_dataset()` — recomputing the *full* uncertainty budget (`u_physical` LPU+MC, `u_training`, `u_c`, coverage), not just RMSE, so the trimming recommendation can be judged on the uncertainty quantification itself. Does not overwrite `case_study_A_results.csv`. |
| [case_study_A_analysis_trimmed.py](case_study_A_analysis_trimmed.py) | Standalone, CLI-runnable sibling of `case_study_A_analysis.py` (same imports, unmodified) that trims the acquisition-transient samples before training and writes `case_study_A_results_trimmed.csv`, plus the extra diagnostics (sensitivity ranking, MC convergence, `u_c`-actionability checks) behind `case_study_A_report_trimmed.md`. |
| [case_study_A_report_trimmed.md](case_study_A_report_trimmed.md) | Full execution report for the trimmed-feature variant, structured like `case_study_A_report.md`, with an explicit section-by-section comparison against the baseline (no-trim) results — including a notable finding the point-prediction-only comparison in `case_study_A_feature_trimming.ipynb` couldn't see: trimming reverses the sign of the (weak) age-vs-uncertainty correlation. |
| [case_study_A_eda.pdf](case_study_A_eda.pdf) / [case_study_A_eda_outliers.pdf](case_study_A_eda_outliers.pdf) / [case_study_A_feature_trimming.pdf](case_study_A_feature_trimming.pdf) / [case_study_A_run_analysis.pdf](case_study_A_run_analysis.pdf) | Rendered PDF exports of the four notebooks above, for sharing without a notebook environment. Regenerate with `python export_notebooks_to_pdf.py` after editing the notebooks. |
| [export_notebooks_to_pdf.py](export_notebooks_to_pdf.py) | Renders the EDA notebooks to the PDFs above with wrapped code lines (plain `nbconvert --to webpdf` clips long lines). Needs `nbconvert` + `playwright` (`playwright install chromium`) on top of `requirements.txt`. |

### Real result artifacts

| File | Role |
|---|---|
| [case_study_A_results.csv](case_study_A_results.csv) | Real output, 953 held-out test cycles × 7 columns: `y_true, y_hat, u_physical_lpu, u_physical_mc, u_training, u_c, covered_k2`. |
| [zema_lpu_vs_mc.pdf](zema_lpu_vs_mc.pdf) | The real `fig:zema_lpu_vs_mc` validation figure, generated from `case_study_A_results.csv`. |
| [pipeline_run.log](pipeline_run.log) | Captured stdout of the actual `case_study_A_analysis.py` run that produced the results above. |
| [case_study_A_report.pdf](case_study_A_report.pdf) / [case_study_A_report.tex](case_study_A_report.tex) | Rendered/typeset versions of the report. |
| [case_study_A_results_trimmed.csv](case_study_A_results_trimmed.csv) | Same schema as `case_study_A_results.csv`, but on the trimmed (1,084-dim) feature set — output of `case_study_A_analysis_trimmed.py`. |
| [zema_lpu_vs_mc_trimmed.pdf](zema_lpu_vs_mc_trimmed.pdf) | The `fig:zema_lpu_vs_mc` validation figure for the trimmed-feature run. |
| [pipeline_run_trimmed.log](pipeline_run_trimmed.log) | Captured stdout of the `case_study_A_analysis_trimmed.py` run behind `case_study_A_report_trimmed.md`. |

### Synthetic (dev-only, not real results)

`case_study_A_results_SYNTHETIC.csv`, `zema_lpu_vs_mc_syntetic.pdf`, and
`make_validation_figure.md` were used only to exercise the plotting pipeline
before the real 2.67 GB dataset finished downloading. Keep them in the working
repo for pipeline testing, but do not bundle them into the paper's
supplementary package and never quote numbers from them.

## Results summary

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
| MC convergence (10k→100k→1M) | Δ 1.75% → 0.08% |

Full uncertainty budget table and per-channel sensitivity ranking (Pressure
51.8%, Velocity 25.2%, Acceleration-x 11.5% — 88.5% combined) are in
`case_study_A_report.md` §7 and §9, with paper-ready LaTeX text in §11.

## Tool usage guidance

### 0. Environment setup

`zema_emc_annotated` pins `numpy<2`, so use a dedicated virtual environment
(Python 3.10 exactly — the pinned versions in `requirements.txt` were
validated against 3.10.12):

```bash
python3.10 -m venv zema_env && source zema_env/bin/activate
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
  (`datapoints_per_cycle=100` → input dim 1100). Expect a few hours on CPU;
  trivially parallelizable over seeds if needed.
- Writes `case_study_A_results.csv` (one row per held-out test cycle).
- Key parameters live in the `Config` dataclass at the top of the script
  (`n_cycles`, `datapoints_per_cycle`, `hidden`, `n_ensemble`, `n_mc`,
  `epochs`, `lr`, `seed`) — edit there to reduce `datapoints_per_cycle` to
  `10` (input dim 110) for faster iteration during development; report
  whichever configuration you finalize.
- If the Zenodo download is slow or blocked, the raw HDF5 is mirrored at
  doi:10.5281/zenodo.5185953; `ZeMASamples(..., skip_hash_check=True)`
  accepts a manually placed file in the cache directory.

### 3. Generate the validation figure

```bash
python make_validation_figure.py --csv case_study_A_results.csv --out zema_lpu_vs_mc.pdf
```

`--csv` and `--out` both default to the real filenames shown above, so a
bare invocation reproduces `zema_lpu_vs_mc.pdf` from the real results. Point
`--csv` at `case_study_A_results_SYNTHETIC.csv` only when smoke-testing the
plotting code itself — never when producing a figure for the paper.

### 4. (Dev-only) regenerate synthetic smoke-test data

```bash
python generate_synthetic_case_study_A_data.py
# writes case_study_A_results_SYNTHETIC.csv
```

Use this only to exercise `make_validation_figure.py` end-to-end without
waiting on the real dataset or a full training run — e.g., when changing the
plotting code. Its own docstring and stdout self-label the output as fake;
never copy a number from it, or a plot made from it, into the paper.

### Interpreting a re-run that disagrees with this report

- **LPU vs. MC median relative difference** below ~5–10% → LPU may be
  reported as primary `u_physical`. Above that → linearity assumption is violated at
  this depth/uncertainty magnitude; report MC as primary instead. Either
  outcome is a valid, publishable result — do not tune the network to force
  agreement.
- **Coverage @ k=2** far from ~95%: under-coverage usually means a missing
  uncertainty source (check residual bias first); over-coverage (as found here)
  usually means the `u_physical` budget is inflated by one channel — check the
  per-channel normalized-uncertainty and first-layer sensitivity breakdown
  (`case_study_A_report.md` §7) before assuming a real over-conservatism.
- **Ensemble `u_training`**: with only 10 seeds, `u_training` itself carries ~24% relative
  sampling noise (χ², 9 dof). If `u_training` turns out to dominate `u_c`, increase
  `n_ensemble` to 20–30 before finalizing the budget table.
