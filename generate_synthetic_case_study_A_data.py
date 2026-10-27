"""
Generates a SYNTHETIC case_study_A_results.csv with the exact schema and file
formatting that case_study_A_analysis.py will produce once run against the
real ZeMA dataset -- so make_validation_figure.py (and the LaTeX figure
pipeline) can be exercised end-to-end before the real experiment is run.

THIS DATA IS FAKE. No dataset, no training, no propagation was run to
produce it -- every column is drawn from a plausible-looking random model
purely so the plotting code has something to consume. Do not copy any
number out of this CSV, or a plot made from it, into the paper.

Row count and the y_true construction mirror the real pipeline exactly
(same n_cycles / test_fraction, same "consumed-lifetime fraction" definition
via linspace + random permutation) so the schema, size, and value range are
representative. The uncertainty and residual columns are synthetic draws,
not physically simulated.

Usage:
    python generate_synthetic_case_study_A_data.py
    # writes case_study_A_results_SYNTHETIC.csv in the working directory
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

# Match case_study_A_analysis.py's Config defaults so row count / value range
# are representative of the real run.
N_CYCLES = 4766
TEST_FRACTION = 0.2
SEED = 42


def make_synthetic_results(seed: int = SEED) -> np.ndarray:
    rng = np.random.default_rng(seed)

    # --- y_true: identical construction to the real pipeline ---
    # (consumed-lifetime fraction y_k = k/N, then a random test split)
    y_full = np.linspace(0.0, 1.0, N_CYCLES)
    idx = rng.permutation(N_CYCLES)
    n_test = int(TEST_FRACTION * N_CYCLES)
    y_true = y_full[idx[:n_test]]

    # --- y_hat: true value + plausible model residual ---
    y_hat = y_true + rng.normal(0.0, 0.02, n_test)

    # --- u_physical via LPU: plausible propagated sensor-uncertainty magnitude ---
    u_physical_lpu = np.abs(rng.normal(0.009, 0.0025, n_test)) + 1e-4

    # --- u_physical via Monte Carlo: LPU +/- a few percent (linearity check) ---
    u_physical_mc = u_physical_lpu * (1.0 + rng.normal(0.0, 0.025, n_test))
    u_physical_mc = np.abs(u_physical_mc) + 1e-4

    # --- u_training: ensemble (training-process) uncertainty, comparable scale ---
    u_training = np.abs(rng.normal(0.012, 0.003, n_test)) + 1e-4

    # --- combined uncertainty, same quadrature rule as the real pipeline ---
    u_c = np.sqrt(u_physical_lpu**2 + u_training**2)

    # --- coverage indicator at k=2 ---
    covered_k2 = (np.abs(y_true - y_hat) <= 2 * u_c).astype(int)

    return np.column_stack(
        [y_true, y_hat, u_physical_lpu, u_physical_mc, u_training, u_c, covered_k2]
    )


def main():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--out", type=Path, default=Path("case_study_A_results_SYNTHETIC.csv")
    )
    p.add_argument("--seed", type=int, default=SEED)
    args = p.parse_args()

    arr = make_synthetic_results(args.seed)

    # Same np.savetxt call as case_study_A_analysis.py -- default comment
    # prefix ("# ") on the header line, so this file round-trips through
    # make_validation_figure.py's np.genfromtxt(..., names=True) exactly the
    # way the real output will.
    np.savetxt(
        args.out,
        arr,
        delimiter=",",
        header="y_true,y_hat,u_physical_lpu,u_physical_mc,u_training,u_c,covered_k2",
    )

    print(f"Wrote {args.out}  ({arr.shape[0]} synthetic rows)")
    print("*** SYNTHETIC DATA -- for pipeline testing only, not real results ***")
    print(f"  synthetic median LPU-vs-MC rel. diff : "
          f"{np.median(np.abs(arr[:,2]-arr[:,3])/arr[:,3]):.2%}")
    print(f"  synthetic empirical coverage @ k=2   : {arr[:,6].mean():.1%}")


if __name__ == "__main__":
    main()
