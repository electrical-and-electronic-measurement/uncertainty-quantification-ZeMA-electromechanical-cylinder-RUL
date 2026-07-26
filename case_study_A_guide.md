# Case Study A — Step-by-Step Execution Guide
## GUM-compliant uncertainty quantification for RUL estimation on the ZeMA electromechanical cylinder dataset

This guide walks through the complete execution of Case Study A (Section 5.1 of the restructured paper), from environment setup to the numbers that populate `Table \ref{tab:zema_uncertainty_budget}` and `Figure \ref{fig:zema_lpu_vs_mc}`. Every step maps explicitly onto a phase of the methodology in Section 4 of the paper, so the case study *demonstrates* the framework rather than merely accompanying it.

The companion script `case_study_A_analysis.py` implements everything described here. Its LPU core has already been validated against a 200,000-sample Monte Carlo run (relative difference 0.096% on a smooth synthetic MLP), and its dataset API calls have been verified against `zema_emc_annotated` v0.7.x.

---

## Step 0 — Environment setup (~15 min)

Create a dedicated virtual environment — `zema_emc_annotated` pins `numpy<2`, which will conflict with newer stacks if installed globally:

```bash
python3.10 -m venv zema_env && source zema_env/bin/activate
pip install zema_emc_annotated torch numpy matplotlib
```

Sanity-check the analytical propagation code *before* touching the dataset:

```bash
python case_study_A_analysis.py --self-test
# expected: [self-test] u_LPU=...  u_MC=...  rel.diff=<2%
```

This self-test builds a random softplus MLP, propagates a diagonal input covariance analytically via the layer-wise formula (Eq. `\ref{eq:Law_of_Propagation_of_Uncertainty_for_NN}`), compares against GUM-Supplement-1 Monte Carlo, and asserts agreement within 2%. If this passes, all subsequent LPU numbers are trustworthy.

**Why the script implements the LPU directly instead of importing `pytorch_gum_uncertainty_propagation`:** Ludwig's library pins older PyTorch versions and Python 3.10 exactly; the propagation formula itself is ~40 lines. The script's `propagate_lpu()` follows the identical layer-wise recursion, and you retain the library citation in the paper as the reference implementation. If you prefer to use the library verbatim for the paper's reproducibility claim, install it in a separate Python 3.10 venv (`pip install pytorch-gum-uncertainty-propagation`) and cross-check one test sample against the script — the numbers should agree to machine precision.

---

## Step 1 — Data acquisition (Phase 1: task characterization)

First run of the script downloads the annotated dataset (the raw HDF5, `axis11_2kHz_ZeMA_PTB_SI.h5`, is **2.67 GB** — budget roughly an hour on a typical connection, not the ~200 MB one might expect from a "reading-level" dataset) from Zenodo into a local cache — no manual download needed:

```python
from zema_emc_annotated.dataset import ZeMASamples
from zema_emc_annotated.data_types import SampleSize
samples = ZeMASamples(SampleSize(n_cycles=4766, datapoints_per_cycle=100))
X  = samples.values          # shape (4766, 1100): 11 sensors x 100 readings
UX = samples.uncertainties   # same shape: per-reading STANDARD uncertainties
```

Three decisions to record in the paper (Phase 1 requires explicit task characterization):

1. **Measurand.** The testbed ran the cylinder to end of life, so the consumed-lifetime fraction of cycle *k* out of *N* is `y_k = k/N ∈ [0,1]`, and RUL = 1 − y. This is the convention of the ZeMA ML tutorials (Lulic & Jugo 2019) and of Ludwig's thesis. The label is defined deterministically from the cycle index, so **label uncertainty is negligible** — state this explicitly; it closes off a reviewer question about training-data (label) uncertainty for this specific dataset.
2. **Task type.** Quantitative measurand → regression → standard uncertainty `u(y)` as the dispersion metric. One sentence in the paper.
3. **Input dimension.** `datapoints_per_cycle=100` gives input dimension 1100 (11 × 100), matching the mid-size configurations in Ludwig's runtime tables. You can reduce to 10 (input dim 110) for faster iteration during development; report whichever you finalize.

---

## Step 2 — Uncertainty source identification (Phase 2)

Fill the source inventory that becomes the rows of `Table \ref{tab:zema_uncertainty_budget}`:

| Source | Type | How obtained in this case study |
|---|---|---|
| Sensor input uncertainty (11 channels) | A | Distributed with the dataset (`samples.uncertainties`) — metrologically evaluated per reading |
| Training process variability | B | Std. dev. of predictions across a 10-seed ensemble (Step 4) |
| Architecture selection | B | Optional: repeat ensemble across 2–3 widths; otherwise state it is held fixed and excluded, with justification |
| Generalization / distribution shift | B | Assessed via held-out coverage (Step 6); no separate budget line unless coverage fails |
| Label (RUL definition) | — | Negligible by construction (deterministic cycle index) |

One methodological point worth a sentence in the paper: the dataset provides *per-reading* uncertainties but no cross-sensor covariance information. The script therefore uses a **diagonal input covariance matrix** `U_x = diag(u²(x_i))`. This is an assumption — declare it in the Reporting section (Phase 5 requires stating assumptions), and note that correlated inputs would require only replacing the diagonal with a full matrix in the same propagation formula.

**Normalization is part of the measurement model.** The script z-scores inputs using training-set statistics. Because z-scoring is a linear map, its GUM propagation is exact: `u(x') = u(x)/σ_train`. The script's `normalize_with_uncertainty()` does this; mention it in one line in the paper — it demonstrates that even preprocessing stages are handled within the framework.

---

## Step 3 — Model and training (Phase 3: method selection)

The script trains an MLP with hidden layers (128, 64, 32) and **softplus** activation. The activation choice is the Phase 3 decision in action:

- softplus is C¹ everywhere → analytical LPU applicable (Section 2.2 of the paper);
- the derivative needed for `H(i)` is closed-form: `d/dz softplus_β(z) = sigmoid(βz)`;
- if you want the QuadLU variant instead (to align maximally with Ludwig et al.), add its piecewise derivative to `_activation_derivative()` — three lines, given in the thesis (Eq. 2.21 there).

Record the three Phase-3 criteria as they resolve for this case: white-box access (yes — we own the weights), architectural compatibility (yes — feedforward MLP, smooth activation), computational constraints (LPU chosen as primary; MC retained for validation only). This paragraph writes itself and directly instantiates the paper's decision tree — reference `Figure \ref{fig:decision_tree}`.

Training details to report for reproducibility (Phase 5 reporting requirements): Adam, lr 1e-3, 200 epochs, MSE loss, Xavier init, float64 throughout (float64 matters — it keeps the LPU-vs-MC comparison free of float32 noise).

---

## Step 4 — Ensemble for Type B (Phase 2/4)

The script trains `n_ensemble=10` models differing only in random seed. For each test sample, `u_B` = standard deviation of the 10 predictions (ddof=1). This is the paper's "training process uncertainty" quantified exactly as Phase 2 prescribes ("training an ensemble of models and analyzing the variance of predictions").

Two honest caveats to preempt reviewers:

1. A 10-seed ensemble captures *optimization* variability, not *architecture* uncertainty. Either add a second ensemble axis over widths, or state the limitation — the paper's methodology already distinguishes these sources, so consistency matters.
2. The ensemble std is a Type B *estimate*; with 10 seeds its own relative uncertainty is ~24% (chi-squared, 9 dof). If `u_B` turns out to dominate `u_c`, increase to 20–30 seeds before finalizing the budget table.

---

## Step 5 — Propagation and combination (Phase 4)

For each held-out cycle the script computes, per test sample:

1. `u_A` via `propagate_lpu()` — the layer-wise analytical propagation, Eq. `\ref{eq:Law_of_Propagation_of_Uncertainty_for_NN}`;
2. `u_A^MC` via `propagate_mc()` — 10,000 input samples per GUM Supplement 1 (JCGM 101:2008), same model;
3. `u_c = sqrt(u_A² + u_B²)` — the quadrature combination, Eq. `\ref{eq:combined_std_uncertainty_regression}`.

Outputs land in `case_study_A_results.csv` with one row per test sample: `y_true, y_hat, u_A_lpu, u_A_mc, u_B, u_c, covered_k2`.

**Populating the paper's uncertainty budget table.** Use the *mean over the test set* for each column, and add the sensitivity information: the dominant input channels can be extracted from the first-layer contribution `W(1) diag(u(x)) `— rank sensor channels by their summed absolute contribution and report the top 2–3 in the "Sensitivity" column (e.g., "axial force and pressure dominate"). That single addition turns the table from a formality into an actual diagnostic, which is the point the paper's Reporting section makes ("relative magnitudes of contributions identify dominant sources").

---

## Step 6 — Validation (Phase 5)

Three checks, all computed by the script:

**LPU vs MC agreement.** Median relative difference between `u_A_lpu` and `u_A_mc` across the test set. Expected: low single-digit percent if the linearity assumption holds at the dataset's uncertainty magnitudes (compare with the <2% reported by Lin et al. for the optical sensor — you now have your own number instead of citing theirs). If the difference is large (>5–10%), that is itself a *result*: it means the first-order approximation is inadequate at this network depth, MC becomes the primary method per the paper's own decision rule, and the case study demonstrates the framework's validation phase catching a violated assumption. Either outcome is publishable; do not tune the network to force agreement.

**Empirical coverage at k=2.** Fraction of test samples with `|y_true − y_hat| ≤ 2·u_c`. Target ≈95%. Systematic under-coverage (e.g., 70%) means a source is missing from the budget — the usual suspect here is model bias (MLP misfit), which is *not* a dispersion and signals that residual analysis should be added as a Type B term. Discuss whatever you find; the coverage check is the paper's "gold standard" sentence made concrete.

**Convergence of MC.** Verify `n_mc=10_000` suffices by re-running one sample at 100,000 and confirming `u_A_mc` moves by ≪1%. One sentence in the paper ("convergence verified by...") satisfies the Phase-5 reporting requirement for the MC settings.

**Figure for the paper** (`fig:zema_lpu_vs_mc`): a two-panel figure works well — left panel scatter of `u_A_lpu` vs `u_A_mc` with the identity line (visual linearity check); right panel predictions ±k=2 bands vs true consumed-life over the test cycles, with coverage annotated. Both are three lines of matplotlib from the CSV.

---

## Step 7 — Writing it up

Replace the red TODOs in Section 5.1 with the concrete numbers. Suggested mapping:

- The dataset/tooling paragraph is already written — verify version numbers (`zema_emc_annotated` version, torch version) and add them.
- Table `\ref{tab:zema_uncertainty_budget}`: means from the CSV + dominant-sensor sensitivity ranking.
- One new sentence for the LPU-vs-MC number: "Across the held-out cycles, the analytical and Monte Carlo estimates of u_A agreed to within X% (median), confirming/refuting the first-order linearity assumption at the uncertainty magnitudes present in the data."
- One new sentence for coverage: "The empirical coverage of the k=2 expanded interval was Y% against the ≈95% target..."
- Reporting statement, per Phase 5: state the diagonal-covariance assumption, the label-uncertainty-negligible argument, the normalization propagation, and the MC settings (M=10^4, Gaussian input distributions, convergence check).

## Expected effort

Environment + self-test: under an hour. Dataset download: budget roughly an hour on its own (2.67 GB, one-time, cached afterward). First full run at `datapoints_per_cycle=10`: an afternoon, CPU only (Ludwig's runtime tables put a full LPU pass at input dim 110 in the tens of milliseconds — the ensemble training dominates, not the propagation). Full-resolution run (input dim 1100) + 10-seed ensemble: expect a few hours on CPU; trivially parallel over seeds if needed. Analysis and writing: one day.

## Known risks and fallbacks

If the Zenodo download is slow or blocked, the raw HDF5 is mirrored at doi:10.5281/zenodo.5185953 and `ZeMASamples(..., skip_hash_check=True)` accepts a manually placed file in the cache directory. If `u_B` dominates `u_c` (plausible for a smallish MLP on 4766 samples), that is a finding, not a failure — it quantitatively supports the paper's claim that epistemic sources "should be minimized before being included in the uncertainty budget," and motivates the larger ensemble of Step 4. If coverage is poor, add a residual-based Type B term and report both budgets (before/after) — that before/after comparison is arguably a *stronger* demonstration of the methodology than a clean pass would be.
