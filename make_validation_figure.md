**Panel (a)** — analytical LPU vs. Monte Carlo scatter, identity line, median relative difference annotated in a corner box. This is your linearity-assumption check from Phase 5.

**Panel (b)** — sorted-by-ground-truth predictions with the ±k=2 expanded-uncertainty band shaded, true vs. predicted lines overlaid, empirical coverage annotated against the ≈95% target.

Both panels came out clean on the test run (0.09 s, single-page vector PDF, no warnings after I fixed a docstring escape-sequence issue on the first pass).

Usage once `case_study_A_analysis.py` has produced real output:

```bash
python make_validation_figure.py --csv case_study_A_results.csv --out zema_lpu_vs_mc.pdf
```

Drop `zema_lpu_vs_mc.pdf` next to the `.tex` file and uncomment the `\includegraphics` line in `fig:zema_lpu_vs_mc` — filename already matches.

Notes:

- **Colorblind-safe palette** (Okabe-Ito), consistent line styles (solid truth, dashed prediction) so it reads fine in grayscale print too.
- **Right panel is sorted by `y_true`**, not by cycle index or split order — since the train/test split in the analysis script is a random permutation, plotting by raw index would look like scattered noise; sorting turns it into a readable trend with a visible band.
- Both panels auto-scale to whatever range your real data produces — no hardcoded axis limits to revisit later.