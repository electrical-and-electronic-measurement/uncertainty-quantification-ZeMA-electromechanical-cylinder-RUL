"""
Case Study A -- trimmed-feature variant of the GUM uncertainty-quantification
pipeline for RUL estimation on the ZeMA electromechanical cylinder dataset.

Same methodology as `case_study_A_analysis.py` (Phases 1-5), on the same
train/test split, with one change applied before feature extraction: the
leading within-cycle samples of Pneumatic pressure, Velocity and Axial force
are dropped (a measurement/DAQ acquisition transient identified in
`case_study_A_eda_outliers.ipynb` and first validated for point-prediction
accuracy in `case_study_A_feature_trimming.ipynb`). This script recomputes
the *full* GUM uncertainty budget on the trimmed inputs -- not just RMSE --
so the two variants can be compared on equal footing. It imports every core
function from `case_study_A_analysis.py` unmodified; nothing in that file is
changed or duplicated beyond the trimming step itself.

Usage:
    python case_study_A_analysis_trimmed.py
    # writes case_study_A_results_trimmed.csv + prints the §7/§9-style
    # diagnostics used by case_study_A_report_trimmed.md
"""

from __future__ import annotations

import argparse
import math

import numpy as np
import torch

from case_study_A_analysis import (
    Config, load_zema, normalize_with_uncertainty, make_mlp, train,
    propagate_lpu, propagate_mc, self_test_lpu,
)

CHANNEL_NAMES = [
    "Acceleration plain bearing", "Acceleration piston rod", "Acceleration ball bearing",
    "Active current", "Axial force",
    "Motor current phase 1", "Motor current phase 2", "Motor current phase 3",
    "Pneumatic pressure", "Sound pressure", "Velocity",
]
TRIM_CHANNELS = ["Pneumatic pressure", "Velocity", "Axial force"]
BASELINE_FROM = 20
SPIKE_RATIO = 5.0


def channel_block(arr, name, D):
    i = CHANNEL_NAMES.index(name)
    return arr[:, i * D:(i + 1) * D]


def compute_settle_K(X, D):
    """Leading within-cycle run length where avg. deviation from the cycle
    median exceeds SPIKE_RATIO x the steady-state (position>=20) baseline."""
    settle_K = {}
    for name in CHANNEL_NAMES:
        block = channel_block(X, name, D)
        dev = np.abs(block - np.median(block, axis=1, keepdims=True))
        pos_mean_dev = dev.mean(axis=0)
        baseline = np.median(pos_mean_dev[BASELINE_FROM:])
        K = 0
        for p in range(len(pos_mean_dev)):
            if pos_mean_dev[p] > SPIKE_RATIO * baseline:
                K = p + 1
            else:
                break
        settle_K[name] = K
    return settle_K


def build_trimmed_dataset(X, UX, D, trim_channels, settle_K):
    x_parts, u_parts, spans = [], [], {}
    col = 0
    for name in CHANNEL_NAMES:
        block_x = channel_block(X, name, D)
        block_u = channel_block(UX, name, D)
        if name in trim_channels:
            K = settle_K[name]
            block_x, block_u = block_x[:, K:], block_u[:, K:]
        x_parts.append(block_x)
        u_parts.append(block_u)
        spans[name] = (col, col + block_x.shape[1])
        col += block_x.shape[1]
    return np.concatenate(x_parts, axis=1), np.concatenate(u_parts, axis=1), spans


def sensitivity_ranking(model, UXn, spans):
    """Per-channel share of first-layer |W1|.u_n sensitivity (mirrors report §7)."""
    W1 = model[0].weight.detach().numpy()          # (hidden0, in_dim)
    u_n = UXn.mean(axis=0)                           # mean normalized uncertainty per column
    per_col = np.abs(W1).sum(axis=0) * u_n            # (in_dim,)
    total = per_col.sum()
    rows = []
    for name, (lo, hi) in spans.items():
        share = per_col[lo:hi].sum() / total
        rows.append((name, u_n[lo:hi].mean(), share))
    rows.sort(key=lambda r: -r[2])
    return rows


def main_trimmed(cfg: Config):
    X, UX, y = load_zema(cfg)
    D = cfg.datapoints_per_cycle
    N = X.shape[0]

    settle_K = compute_settle_K(X, D)
    print("settling length K per channel:", settle_K)
    X_trim, UX_trim, spans = build_trimmed_dataset(X, UX, D, TRIM_CHANNELS, settle_K)
    print(f"raw input dim {X.shape[1]} -> trimmed input dim {X_trim.shape[1]} "
          f"(dropped {X.shape[1] - X_trim.shape[1]} columns: "
          + ", ".join(f"{n}=-{settle_K[n]}" for n in TRIM_CHANNELS) + ")")

    rng = np.random.default_rng(cfg.seed)
    idx = rng.permutation(N)
    n_test = int(cfg.test_fraction * N)
    te, tr = idx[:n_test], idx[n_test:]

    mean, std = X_trim[tr].mean(0), X_trim[tr].std(0) + 1e-12
    Xn, UXn = normalize_with_uncertainty(X_trim, UX_trim, mean, std)

    # ---- u_training: seed ensemble ----
    models, preds_te = [], []
    for s in range(cfg.n_ensemble):
        m = make_mlp(Xn.shape[1], cfg)
        train(m, Xn[tr], y[tr], cfg, seed=s)
        models.append(m)
        with torch.no_grad():
            preds_te.append(m(torch.tensor(Xn[te], dtype=torch.float64)).squeeze(1).numpy())
        print(f"  seed {s} trained")
    preds_te = np.stack(preds_te)
    u_training = preds_te.std(axis=0, ddof=1)
    model = models[0]

    # ---- u_physical via LPU, validated against MC, on the test set ----
    rows = []
    for j, i in enumerate(te):
        x = torch.tensor(Xn[i], dtype=torch.float64)
        ux = torch.tensor(UXn[i], dtype=torch.float64)
        yhat, U_y = propagate_lpu(model, x, torch.diag(ux ** 2))
        u_physical_lpu = math.sqrt(U_y.item())
        _, u_physical_mc = propagate_mc(model, x, ux, cfg.n_mc, seed=j)
        u_c = math.sqrt(u_physical_lpu ** 2 + u_training[j] ** 2)
        covered = abs(y[i] - yhat.item()) <= 2 * u_c
        rows.append((y[i], yhat.item(), u_physical_lpu, u_physical_mc,
                     u_training[j], u_c, covered))
        if (j + 1) % 200 == 0 or (j + 1) == len(te):
            print(f"  propagated {j + 1}/{len(te)} test cycles")

    r = np.array(rows, dtype=float)
    resid = r[:, 0] - r[:, 1]
    rmse = math.sqrt(np.mean(resid ** 2))
    ss_res, ss_tot = np.sum(resid ** 2), np.sum((r[:, 0] - r[:, 0].mean()) ** 2)
    r2 = 1 - ss_res / ss_tot
    rel_diff = np.abs(r[:, 2] - r[:, 3]) / r[:, 3]
    coverage = r[:, 6].mean()

    print("\n=== Case Study A -- TRIMMED features: results ===")
    print(f"test RMSE                  : {rmse:.4f}")
    print(f"test R2                    : {r2:.4f}")
    print(f"mean u_physical (LPU)      : {r[:, 2].mean():.4e}")
    print(f"mean u_physical (MC)       : {r[:, 3].mean():.4e}")
    print(f"LPU vs MC median rel. diff : {np.median(rel_diff):.2%}  (Phase 5 check)")
    print(f"mean u_training (ensemble) : {r[:, 4].mean():.4e}")
    print(f"mean u_c (combined)        : {r[:, 5].mean():.4e}")
    print(f"empirical coverage @ k=2   : {coverage:.1%}  (target ~95%)")

    cols = "y_true,y_hat,u_physical_lpu,u_physical_mc,u_training,u_c,covered_k2"
    np.savetxt("case_study_A_results_trimmed.csv", r, delimiter=",", header=cols)
    print("\nwrote case_study_A_results_trimmed.csv")

    # ---- extra diagnostics for the report (mirror §7/§7.1 of the baseline report) ----
    print("\n=== Sensitivity ranking (first-layer |W1|.u_n per channel) ===")
    ranking = sensitivity_ranking(model, UXn, spans)
    for name, un_mean, share in ranking:
        width = spans[name][1] - spans[name][0]
        print(f"  {name:28s} width={width:3d}  u_n(mean)={un_mean:8.4f}  share={share:.1%}")

    print("\n=== MC convergence check (held-out test point #0) ===")
    x0 = torch.tensor(Xn[te[0]], dtype=torch.float64)
    ux0 = torch.tensor(UXn[te[0]], dtype=torch.float64)
    mc_conv = {}
    for M in (10_000, 100_000, 1_000_000):
        _, u = propagate_mc(model, x0, ux0, M, seed=0)
        mc_conv[M] = u
        print(f"  n_mc={M:>9,}   u_physical_mc = {u:.6e}")
    d1 = abs(mc_conv[100_000] - mc_conv[10_000]) / mc_conv[10_000]
    d2 = abs(mc_conv[1_000_000] - mc_conv[100_000]) / mc_conv[100_000]
    print(f"  delta 10k->100k = {d1:.2%}   delta 100k->1M = {d2:.2%}")

    print("\n=== Is u_c actionable? (mirrors report §7.1) ===")
    mean_abs_err = np.mean(np.abs(resid))
    print(f"  mean u_c (LPU-based, CSV column) vs mean |error| : {r[:, 5].mean():.4f} vs {mean_abs_err:.4f}"
          f"  ({r[:, 5].mean() / mean_abs_err:.1f}x)")
    corr_uphys_y = np.corrcoef(r[:, 3], r[:, 0])[0, 1]
    corr_uc_err = np.corrcoef(r[:, 5], np.abs(resid))[0, 1]
    print(f"  corr(u_physical_mc, y_true) = {corr_uphys_y:.2f}")
    print(f"  corr(u_c, |error|)          = {corr_uc_err:.2f}")
    terciles = np.quantile(r[:, 0], [1 / 3, 2 / 3])
    lo_mask = r[:, 0] <= terciles[0]
    hi_mask = r[:, 0] >= terciles[1]
    print(f"  early-life mean u_physical_mc = {r[lo_mask, 3].mean():.4f}   "
          f"late-life = {r[hi_mask, 3].mean():.4f}")
    print(f"  early-life mean |error|       = {np.abs(resid)[lo_mask].mean():.4f}   "
          f"late-life = {np.abs(resid)[hi_mask].mean():.4f}")

    return r


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--self-test", action="store_true",
                    help="validate the LPU implementation without the dataset (same as case_study_A_analysis.py)")
    args = p.parse_args()
    if args.self_test:
        self_test_lpu()
    else:
        main_trimmed(Config())
