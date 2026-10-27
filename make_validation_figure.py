r"""
Two-panel validation figure for Case Study A (Section 5.1, fig:zema_lpu_vs_mc).

Panel 1 (left)  : analytical LPU vs. Monte Carlo estimates of u_physical, one point
                  per held-out test cycle, with the identity line and the
                  median relative difference annotated -- the Phase 5
                  "linearity assumption" check.
Panel 2 (right) : predicted consumed-lifetime fraction vs. ground truth over
                  the held-out cycles, with the +-k=2 expanded-uncertainty
                  band, and empirical coverage annotated -- the Phase 5
                  "empirical coverage" check.

Usage
-----
    python make_validation_figure.py \
        --csv case_study_A_results.csv \
        --out zema_lpu_vs_mc.pdf

Expects the CSV produced by case_study_A_analysis.py, with columns:
    y_true, y_hat, u_physical_lpu, u_physical_mc, u_training, u_c, covered_k2

Run with no arguments once case_study_A_results.csv exists in the working
directory; it will write zema_lpu_vs_mc.pdf ready to \includegraphics into
the paper.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# Colorblind-friendly palette (Okabe-Ito)
C_POINT = "#0072B2"
C_IDENTITY = "#999999"
C_TRUTH = "#000000"
C_PRED = "#D55E00"
C_BAND = "#D55E00"


def load_results(csv_path: Path) -> dict[str, np.ndarray]:
    data = np.genfromtxt(csv_path, delimiter=",", names=True)
    cols = {name: data[name] for name in data.dtype.names}
    required = {"y_true", "y_hat", "u_physical_lpu", "u_physical_mc", "u_training", "u_c", "covered_k2"}
    have = {c.lower() for c in cols}
    missing = required - have
    if missing:
        raise ValueError(
            f"CSV is missing expected columns {missing}. "
            f"Found: {sorted(cols)}. Check case_study_A_analysis.py output header."
        )
    # normalise keys to lowercase for robust access regardless of np's casing
    return {k.lower(): v for k, v in cols.items()}


def make_figure(d: dict[str, np.ndarray], out_path: Path) -> None:
    y_true, y_hat = d["y_true"], d["y_hat"]
    u_lpu, u_mc, u_c = d["u_physical_lpu"], d["u_physical_mc"], d["u_c"]
    covered = d["covered_k2"].astype(bool)

    rel_diff = np.abs(u_lpu - u_mc) / u_mc
    median_rel_diff = np.median(rel_diff)
    coverage = covered.mean()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.2))

    # ---------------- Panel 1: LPU vs MC ----------------
    lo = min(u_lpu.min(), u_mc.min()) * 0.9
    hi = max(u_lpu.max(), u_mc.max()) * 1.1
    ax1.plot([lo, hi], [lo, hi], "--", color=C_IDENTITY, lw=1.2, zorder=1,
             label="identity ($u_{LPU}=u_{MC}$)")
    ax1.scatter(u_lpu, u_mc, s=14, color=C_POINT, alpha=0.6,
                edgecolors="none", zorder=2, label="held-out test cycles")
    ax1.set_xlim(lo, hi)
    ax1.set_ylim(lo, hi)
    ax1.set_xlabel(r"$u_{physical}$ analytical LPU")
    ax1.set_ylabel(r"$u_{physical}$ Monte Carlo ($M \geq 10^4$)")
    ax1.set_title("(a) Analytical vs. Monte Carlo propagation")
    ax1.set_aspect("equal", adjustable="box")
    ax1.legend(loc="upper left", frameon=False, fontsize=8)
    ax1.text(
        0.97, 0.05,
        f"median rel. diff. = {median_rel_diff:.1%}",
        transform=ax1.transAxes, ha="right", va="bottom", fontsize=9,
        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=C_IDENTITY, lw=0.8),
    )

    # ---------------- Panel 2: coverage over held-out cycles ----------------
    order = np.argsort(y_true)
    yt, yh, uc = y_true[order], y_hat[order], u_c[order]
    x_axis = np.arange(len(yt))

    ax2.fill_between(x_axis, yh - 2 * uc, yh + 2 * uc,
                      color=C_BAND, alpha=0.2, label=r"$\hat{y} \pm k{=}2\,u_c$")
    ax2.plot(x_axis, yt, color=C_TRUTH, lw=1.3, label="ground truth $y$")
    ax2.plot(x_axis, yh, color=C_PRED, lw=1.0, ls="--", label=r"prediction $\hat{y}$")
    ax2.set_xlabel("held-out test cycles (sorted by ground truth)")
    ax2.set_ylabel("consumed-lifetime fraction")
    ax2.set_title("(b) Prediction with $k{=}2$ expanded uncertainty")
    ax2.legend(loc="upper left", frameon=False, fontsize=8)
    ax2.text(
        0.97, 0.05,
        f"empirical coverage @ $k{{=}}2$ = {coverage:.1%}\n(target $\\approx$95%)",
        transform=ax2.transAxes, ha="right", va="bottom", fontsize=9,
        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=C_IDENTITY, lw=0.8),
    )

    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    print(f"Wrote {out_path}")
    print(f"  median LPU-vs-MC relative difference : {median_rel_diff:.2%}")
    print(f"  empirical coverage @ k=2              : {coverage:.1%}  ({covered.sum()}/{len(covered)})")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--csv", type=Path, default=Path("case_study_A_results.csv"))
    p.add_argument("--out", type=Path, default=Path("zema_lpu_vs_mc.pdf"))
    args = p.parse_args()

    if not args.csv.exists():
        raise SystemExit(
            f"'{args.csv}' not found. Run case_study_A_analysis.py first "
            f"(it writes this CSV in the working directory), then re-run this script."
        )

    d = load_results(args.csv)
    make_figure(d, args.out)


if __name__ == "__main__":
    main()
