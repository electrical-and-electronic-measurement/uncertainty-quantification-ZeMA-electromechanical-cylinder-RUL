"""
Case Study A -- full-window variant of the GUM uncertainty-quantification
pipeline for RUL estimation on the ZeMA electromechanical cylinder dataset.

Same methodology as `case_study_A_analysis.py` (Phases 1-5), on the same
train/test split, with one change in the input extraction:
`zema_emc_annotated.ZeMASamples(datapoints_per_cycle=d)` returns the *first*
d of the 2000 samples per cycle (values[:d]), i.e. with d=100 only the first
50 ms of the 1 s return-stroke window. Here the HDF5 file is read directly and
N_POINTS samples are taken uniformly over the whole window (one every
2000/N_POINTS samples, at the centre of each block), together with the
standard uncertainty of the very same samples. No averaging is done, so the
per-sample uncertainties stay exactly those of the dataset and no assumption
on their correlation is needed (decimation is done without an anti-aliasing
filter: content above N_POINTS/2 Hz aliases). It imports every core function
from `case_study_A_analysis.py` and the diagnostics of
`case_study_A_analysis_trimmed.py` unmodified.

Usage:
    python case_study_A_analysis_fullwindow.py
    # writes case_study_A_results_fullwindow.csv + prints the same
    # diagnostics as case_study_A_analysis_trimmed.py
    python case_study_A_analysis_fullwindow.py --exclude "Pneumatic pressure"
    # same, without that channel -> case_study_A_results_fullwindow_no_pressure.csv
    python case_study_A_analysis_fullwindow.py --sampling first --n-points 100 --exclude "Pneumatic pressure"
    # control: baseline input (first 100 samples) without pressure -> ..._first100_no_pressure.csv
    python case_study_A_analysis_fullwindow.py --sampling blockmean
    # control: 200 block means (anti-aliasing) over the 1 s window -> ..._blockmean.csv
"""

from __future__ import annotations

import argparse
import math

import numpy as np
import torch

from case_study_A_analysis import (
    Config, normalize_with_uncertainty, make_mlp, train,
    propagate_lpu, propagate_mc, self_test_lpu,
)
from case_study_A_analysis_trimmed import CHANNEL_NAMES, sensitivity_ranking

N_POINTS = 200          # samples per channel per cycle -> input dim = 11*200
WINDOW = 2000           # samples per channel per cycle in the HDF5 (1 s @ 2 kHz)


def window_indices(n_points: int = N_POINTS, window: int = WINDOW) -> np.ndarray:
    step = window // n_points
    return np.arange(n_points) * step + step // 2       # centre of each block


SAMPLINGS = ("centre", "first", "blockmean")


def sample_window(v, u, n_points: int, sampling: str):
    """Reduces one channel (WINDOW, N) to (n_points, N) values and uncertainties.

    centre    : one sample at the centre of each of n_points blocks (decimation)
    first     : the first n_points samples (what ZeMASamples returns)
    blockmean : mean of each block (anti-aliasing boxcar). The uncertainty of
                a mean of m readings is (1/m) sqrt(sum_ij r_ij u_i u_j); the
                correlation r_ij is not documented in the dataset, so r_ij = 1
                (fully correlated, conservative) -> u(mean) = mean(u_i).
                The dataset's constant uncertainties are systematic by nature,
                for which this is exact.
    """
    if sampling == "centre":
        idx = window_indices(n_points)
        return v[idx], u[idx]
    if sampling == "first":
        return v[:n_points], u[:n_points]
    if sampling == "blockmean":
        m = WINDOW // n_points
        return (v.reshape(n_points, m, -1).mean(1), u.reshape(n_points, m, -1).mean(1))
    raise ValueError(sampling)


def load_zema_fullwindow(cfg: Config, n_points: int = N_POINTS, sampling: str = "centre"):
    """Like `load_zema`, but with the per-cycle samples chosen by `sampling`.

    Channel order and layout (11 consecutive blocks of n_points columns) are
    identical to `zema_emc_annotated.ZeMASamples`, so CHANNEL_NAMES applies.
    """
    import h5py
    from pooch import retrieve
    from zema_emc_annotated.dataset import (
        ZEMA_DATASET_HASH, ZEMA_DATASET_URL, ZEMA_QUANTITIES,
    )

    path = retrieve(url=ZEMA_DATASET_URL, known_hash=ZEMA_DATASET_HASH,
                    progressbar=True)
    x_parts, u_parts = [], []
    with h5py.File(path, "r") as h5f:
        for quantity in ZEMA_QUANTITIES:
            v = h5f["ZeMA_DAQ"][quantity]["qudt:value"][()]     # (2000,N) or (3,2000,N)
            u = h5f["ZeMA_DAQ"][quantity]["qudt:standardUncertainty"][()]
            if v.ndim == 2:
                v, u = v[None], u[None]
            for k in range(v.shape[0]):                         # axes / phases
                vs, us = sample_window(v[k][:, :cfg.n_cycles], u[k][:, :cfg.n_cycles],
                                       n_points, sampling)
                x_parts.append(vs.T)
                u_parts.append(us.T)
    X = np.concatenate(x_parts, axis=1).astype(np.float64)     # (N, 11*n_points)
    UX = np.concatenate(u_parts, axis=1).astype(np.float64)
    y = np.linspace(0.0, 1.0, X.shape[0])                       # consumed life
    return X, UX, y


@torch.no_grad()
def propagate_mc_chunked(model, x, ux_std, n_mc: int, seed: int = 0, chunk: int = 100_000):
    """Same estimator as `propagate_mc`, in chunks so 1M x 2200 inputs fit in memory."""
    g = torch.Generator().manual_seed(seed)
    s1 = s2 = 0.0
    for start in range(0, n_mc, chunk):
        m = min(chunk, n_mc - start)
        eps = torch.randn(m, x.numel(), generator=g, dtype=x.dtype)
        out = model(x.unsqueeze(0) + eps * ux_std.unsqueeze(0))
        s1 += out.sum().item(); s2 += (out ** 2).sum().item()
    mean = s1 / n_mc
    return mean, math.sqrt((s2 - n_mc * mean ** 2) / (n_mc - 1))


def drop_channels(X, UX, D, exclude):
    """Removes the column blocks of the excluded channels; returns X, UX, spans."""
    keep = [name for name in CHANNEL_NAMES if name not in exclude]
    cols = np.concatenate([np.arange(CHANNEL_NAMES.index(n) * D, (CHANNEL_NAMES.index(n) + 1) * D)
                           for n in keep])
    spans = {name: (k * D, (k + 1) * D) for k, name in enumerate(keep)}
    return X[:, cols], UX[:, cols], spans


def main_fullwindow(cfg: Config, exclude: tuple[str, ...] = (),
                    n_points: int = N_POINTS, sampling: str = "centre"):
    unknown = set(exclude) - set(CHANNEL_NAMES)
    assert not unknown, f"unknown channel(s): {unknown}"
    X, UX, y = load_zema_fullwindow(cfg, n_points, sampling)
    D = n_points
    N = X.shape[0]
    X, UX, spans = drop_channels(X, UX, D, exclude)
    print(f"input dim {X.shape[1]} = {len(spans)} channels x {D} samples "
          f"(sampling '{sampling}' of the {WINDOW}-sample window)")
    if exclude:
        print("excluded channels:", ", ".join(exclude))
    tag = {"centre": "fullwindow", "first": f"first{D}", "blockmean": "blockmean"}[sampling]
    if sampling != "first" and D != N_POINTS:
        tag += str(D)
    tag += "_no_" + "_".join(n.split()[-1].lower() for n in exclude) if exclude else ""

    rng = np.random.default_rng(cfg.seed)
    idx = rng.permutation(N)
    n_test = int(cfg.test_fraction * N)
    te, tr = idx[:n_test], idx[n_test:]

    mean, std = X[tr].mean(0), X[tr].std(0) + 1e-12
    Xn, UXn = normalize_with_uncertainty(X, UX, mean, std)

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
    u_c_mc = np.sqrt(r[:, 3] ** 2 + r[:, 4] ** 2)

    print(f"\n=== Case Study A -- {tag} features ({D} samples/channel): results ===")
    print(f"test RMSE                  : {rmse:.4f}")
    print(f"test R2                    : {r2:.4f}")
    print(f"mean u_physical (LPU)      : {r[:, 2].mean():.4e}")
    print(f"mean u_physical (MC)       : {r[:, 3].mean():.4e}")
    print(f"LPU vs MC median rel. diff : {np.median(rel_diff):.2%}  (Phase 5 check)")
    print(f"mean u_training (ensemble) : {r[:, 4].mean():.4e}")
    print(f"mean u_c (combined, LPU)   : {r[:, 5].mean():.4e}")
    print(f"mean u_c (combined, MC)    : {u_c_mc.mean():.4e}")
    print(f"empirical coverage @ k=2   : {r[:, 6].mean():.1%}  (target ~95%)")
    print(f"  (u_training only)        : {np.mean(np.abs(resid) <= 2 * r[:, 4]):.1%}")

    cols = "y_true,y_hat,u_physical_lpu,u_physical_mc,u_training,u_c,covered_k2"
    np.savetxt(f"case_study_A_results_{tag}.csv", r, delimiter=",", header=cols)
    print(f"\nwrote case_study_A_results_{tag}.csv")

    print("\n=== Sensitivity ranking (first-layer |W1|.u_n per channel) ===")
    for name, un_mean, share in sensitivity_ranking(model, UXn, spans):
        print(f"  {name:28s} width={D:3d}  u_n(mean)={un_mean:8.4f}  share={share:.1%}")

    print("\n=== MC convergence check (held-out test point #0) ===")
    x0 = torch.tensor(Xn[te[0]], dtype=torch.float64)
    ux0 = torch.tensor(UXn[te[0]], dtype=torch.float64)
    mc_conv = {}
    for M in (10_000, 100_000, 1_000_000):
        _, u = propagate_mc_chunked(model, x0, ux0, M, seed=0)
        mc_conv[M] = u
        print(f"  n_mc={M:>9,}   u_physical_mc = {u:.6e}")
    d1 = abs(mc_conv[100_000] - mc_conv[10_000]) / mc_conv[10_000]
    d2 = abs(mc_conv[1_000_000] - mc_conv[100_000]) / mc_conv[100_000]
    print(f"  delta 10k->100k = {d1:.2%}   delta 100k->1M = {d2:.2%}")

    print("\n=== Is u_c actionable? ===")
    mean_abs_err = np.mean(np.abs(resid))
    print(f"  mean u_c (LPU-based, CSV column) vs mean |error| : {r[:, 5].mean():.4f} vs {mean_abs_err:.4f}"
          f"  ({r[:, 5].mean() / mean_abs_err:.1f}x)")
    print(f"  corr(u_physical_mc, y_true) = {np.corrcoef(r[:, 3], r[:, 0])[0, 1]:.2f}")
    print(f"  corr(u_c, |error|)          = {np.corrcoef(r[:, 5], np.abs(resid))[0, 1]:.2f}")
    terciles = np.quantile(r[:, 0], [1 / 3, 2 / 3])
    lo_mask, hi_mask = r[:, 0] <= terciles[0], r[:, 0] >= terciles[1]
    print(f"  early-life mean u_physical_mc = {r[lo_mask, 3].mean():.4f}   "
          f"late-life = {r[hi_mask, 3].mean():.4f}")
    print(f"  early-life mean |error|       = {np.abs(resid)[lo_mask].mean():.4f}   "
          f"late-life = {np.abs(resid)[hi_mask].mean():.4f}")

    return r


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--self-test", action="store_true",
                   help="validate the LPU implementation without the dataset (same as case_study_A_analysis.py)")
    p.add_argument("--exclude", nargs="*", default=[], metavar="CHANNEL",
                   help='channels to leave out, e.g. --exclude "Pneumatic pressure"')
    p.add_argument("--sampling", choices=SAMPLINGS, default="centre",
                   help="how the per-cycle samples are taken from the 1 s window")
    p.add_argument("--n-points", type=int, default=N_POINTS,
                   help="samples per channel per cycle")
    args = p.parse_args()
    if args.self_test:
        self_test_lpu()
    else:
        main_fullwindow(Config(), exclude=tuple(args.exclude),
                        n_points=args.n_points, sampling=args.sampling)
