"""
Case Study A -- corrected uncertainty budget for RUL estimation on the ZeMA
electromechanical cylinder dataset.

Same model, training procedure and held-out test cycles as
`case_study_A_analysis.py`; what changes is how the input uncertainty is
interpreted and combined:

1. Random / systematic split of the declared input uncertainty, u^2 = u_rand^2 +
   u_sys^2, per sample. The dataset gives only u, so u_rand is estimated from
   the data (`--random-model`):
     bound : u_rand = min(u, s), s = cross-cycle std of the same column
             (training cycles). Since s^2 = var(true) + u_rand^2, this is an
             upper bound of the part that can vary from cycle to cycle.
     noise : u_rand = min(u, s, sigma_k), sigma_k = white-noise floor of
             channel k from second differences of adjacent samples
             (Type A, robust MAD estimate; an upper bound as well, since
             signal curvature also contributes).
     datasheet : as `noise`, further capped by the sensor noise floor from
             the manufacturer's datasheet where one is published
             (DATASHEET_NOISE). This covers the sensor only, not the rest of
             the acquisition chain, so unlike the other two it is not an
             upper bound for those channels.
2. Same-instrument use (training and inference on the same sensors): a
   systematic error common to all cycles is an affine map per column,
   x' = a_j + b_j x, which the z-score normalization (m, s estimated on the
   same data) removes exactly. Only u_rand is propagated, as independent
   per-sample noise: u_rand_lpu (diagonal LPU) and u_rand_mc (GUM-S1 MC).
3. Transfer to another (re-calibrated) instrument: u_sys is additionally
   propagated as a shift common to all samples of a channel and independent
   between channels (block-diagonal covariance U_k = u_sys,k u_sys,k^T):
   u_sys_lpu, and u_sys_mc with one N(0,1) draw per channel.
4. Model uncertainty from a validation set carved out of the training cycles
   (VAL_FRACTION; the test set is unchanged): u_model^2 = mean(e_val^2) -
   mean(u_training_val^2) - mean(u_rand_mc_val^2), clipped at 0, so the
   residual variance is not double-counted with the other two terms.
5. u_c = sqrt(u_rand_mc^2 + u_training^2 + u_model^2) (Monte Carlo based,
   consistent with the coverage), u_c_transfer = sqrt(u_c^2 + u_sys_mc^2).

For comparison, the original budget (every input independent with its full
declared u, MC) is recomputed on the same model as u_indep_mc.

Usage:
    python case_study_A_analysis_budget.py                       # baseline input, 'bound'
    python case_study_A_analysis_budget.py --random-model noise
    python case_study_A_analysis_budget.py --random-model datasheet
    python case_study_A_analysis_budget.py --exclude "Pneumatic pressure"
    # writes case_study_A_results_budget_<variant>.csv (make_validation_figure.py-compatible)
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
from case_study_A_analysis_fullwindow import (
    SAMPLINGS, load_zema_fullwindow, drop_channels,
)

VAL_FRACTION = 0.15
RANDOM_MODELS = ("bound", "noise", "datasheet")
G = 9.80665

# Sensor noise floors (standard deviation, channel units) published by the manufacturer.
# Kistler 8704B50M1: threshold 0.004 g rms (kistler.com product page; band not stated,
# frequency range 0.5 Hz-10 kHz, wider than the 1 kHz Nyquist band of the 2 kHz data).
# No noise specification is published for the Kistler 8712A5M1 (plain bearing), the
# Festo VPPM (pressure: only accuracy/repeatability of the regulated pressure) or the
# Festo CMMP-AS-C10-11A-P3 (velocity, active current), so those channels keep `noise`.
DATASHEET_NOISE = {
    "Acceleration piston rod": 0.004 * G,
    "Acceleration ball bearing": 0.004 * G,
}
SPLITS = ("random", "blocked")


def make_split(N, cfg: Config, split="random", block_size=50, gap=5):
    """Returns (tr, va, te) cycle indices.

    random  : the baseline split (same 953 test cycles as every earlier run).
    blocked : contiguous blocks of `block_size` cycles are shuffled and assigned
              whole to test / validation / training, so a test cycle no longer
              has its immediate neighbours in the training set; training cycles
              closer than `gap` to a test or validation cycle are dropped (purge).
    """
    rng = np.random.default_rng(cfg.seed)
    if split == "random":
        idx = rng.permutation(N)
        n_test = int(cfg.test_fraction * N)
        te, rest = idx[:n_test], idx[n_test:]
        n_val = int(VAL_FRACTION * len(rest))
        return rest[n_val:], rest[:n_val], te
    blocks = [np.arange(lo, min(lo + block_size, N)) for lo in range(0, N, block_size)]
    order = rng.permutation(len(blocks))
    n_te_blocks = round(cfg.test_fraction * len(blocks))
    n_va_blocks = round(VAL_FRACTION * (len(blocks) - n_te_blocks))
    te = np.concatenate([blocks[b] for b in order[:n_te_blocks]])
    va = np.concatenate([blocks[b] for b in order[n_te_blocks:n_te_blocks + n_va_blocks]])
    tr = np.concatenate([blocks[b] for b in order[n_te_blocks + n_va_blocks:]])
    held = np.zeros(N, dtype=bool)
    held[te] = held[va] = True
    near = np.convolve(held, np.ones(2 * gap + 1), mode="same") > 0
    return tr[~near[tr]], np.sort(va), np.sort(te)


def neighbour_leakage(tr, te):
    """Share of test cycles with at least one / both adjacent cycles in training."""
    t = set(tr.tolist())
    one = np.mean([(i - 1 in t) or (i + 1 in t) for i in te])
    both = np.mean([(i - 1 in t) and (i + 1 in t) for i in te])
    return one, both


def noise_floor(X, spans):
    """Per-channel white-noise std from second differences of adjacent samples.

    For white noise of std sigma, x_i - (x_{i-1} + x_{i+1})/2 has std
    sigma*sqrt(1.5); a MAD estimate keeps genuine signal transients out.
    """
    sigma = {}
    for name, (lo, hi) in spans.items():
        d2 = X[:, lo + 1:hi - 1] - 0.5 * (X[:, lo:hi - 2] + X[:, lo + 2:hi])
        mad = np.median(np.abs(d2 - np.median(d2)))
        sigma[name] = 1.4826 * mad / math.sqrt(1.5)
    return sigma


def split_uncertainty(X, UX, tr, spans, random_model):
    """Returns (U_rand, U_sys) in physical units, with U_rand^2 + U_sys^2 = UX^2."""
    s = X[tr].std(0)
    U_rand = np.minimum(UX, s)
    if random_model in ("noise", "datasheet"):
        for name, sig in noise_floor(X[tr], spans).items():
            lo, hi = spans[name]
            U_rand[:, lo:hi] = np.minimum(U_rand[:, lo:hi], sig)
    if random_model == "datasheet":
        for name, sig in DATASHEET_NOISE.items():
            if name in spans:
                lo, hi = spans[name]
                U_rand[:, lo:hi] = np.minimum(U_rand[:, lo:hi], sig)
    U_sys = np.sqrt(np.maximum(UX ** 2 - U_rand ** 2, 0.0))
    return U_rand, U_sys


def channel_matrix(u_sys, spans):
    """(n_channels, n_in) matrix: row k holds u_sys on channel k's columns, 0 elsewhere."""
    M = torch.zeros(len(spans), u_sys.numel(), dtype=u_sys.dtype)
    for k, (lo, hi) in enumerate(spans.values()):
        M[k, lo:hi] = u_sys[lo:hi]
    return M


@torch.no_grad()
def propagate_sys(model, x, u_sys, spans, n_mc: int, seed: int = 0):
    """Systematic part: fully correlated within a channel, independent between channels."""
    M = channel_matrix(u_sys, spans)
    _, U_y = propagate_lpu(model, x, M.T @ M)              # block-diagonal covariance
    g = torch.Generator().manual_seed(seed)
    eps = torch.randn(n_mc, M.shape[0], generator=g, dtype=x.dtype)
    out = model(x.unsqueeze(0) + eps @ M)
    return math.sqrt(U_y.item()), out.std(unbiased=True).item()


def budget_rows(model, Xn, URn, USn, UXn, idx, u_training, spans, cfg, with_extra=True):
    rows = []
    for j, i in enumerate(idx):
        x = torch.tensor(Xn[i], dtype=torch.float64)
        ur = torch.tensor(URn[i], dtype=torch.float64)
        with torch.no_grad():
            y_hat = model(x.unsqueeze(0)).item()
        _, U_y = propagate_lpu(model, x, torch.diag(ur ** 2))
        u_rand_lpu = math.sqrt(U_y.item())
        mc_mean, u_rand_mc = propagate_mc(model, x, ur, cfg.n_mc, seed=j)
        row = [y_hat, u_rand_lpu, u_rand_mc, mc_mean - y_hat, u_training[j]]
        if with_extra:
            us = torch.tensor(USn[i], dtype=torch.float64)
            u_sys_lpu, u_sys_mc = propagate_sys(model, x, us, spans, cfg.n_mc, seed=j)
            _, u_indep_mc = propagate_mc(model, x, torch.tensor(UXn[i], dtype=torch.float64),
                                         cfg.n_mc, seed=j)
            row += [u_sys_lpu, u_sys_mc, u_indep_mc]
        rows.append(row)
        if with_extra and ((j + 1) % 200 == 0 or (j + 1) == len(idx)):
            print(f"  propagated {j + 1}/{len(idx)} test cycles")
    return np.array(rows, dtype=float)


def main_budget(cfg: Config, exclude=(), n_points=100, sampling="first", random_model="bound",
                split="random", block_size=50, gap=5):
    X, UX, y = load_zema_fullwindow(cfg, n_points, sampling)
    X, UX, spans = drop_channels(X, UX, n_points, exclude)
    N = len(y)
    tag = f"{sampling}{n_points}" + ("_no_" + "_".join(n.split()[-1].lower() for n in exclude)
                                     if exclude else "") + f"_{random_model}"
    if split != "random":
        tag += f"_{split}{block_size}"
    print(f"input dim {X.shape[1]} ({len(spans)} channels x {n_points}, sampling '{sampling}'); "
          f"random model '{random_model}'")

    # random: same held-out test cycles as the baseline; validation carved out of the training part
    tr, va, te = make_split(N, cfg, split, block_size, gap)
    one, both = neighbour_leakage(tr, te)
    print(f"split '{split}'" + (f" (blocks of {block_size}, purge gap {gap})" if split != "random" else "")
          + f": train {len(tr)}, validation {len(va)}, test {len(te)}")
    print(f"test cycles with a training neighbour: one side {one:.1%}, both sides {both:.1%}")

    U_rand, U_sys = split_uncertainty(X, UX, tr, spans, random_model)
    mean, std = X[tr].mean(0), X[tr].std(0) + 1e-12
    Xn, UXn = normalize_with_uncertainty(X, UX, mean, std)
    URn, USn = U_rand / std, U_sys / std

    print("\n=== Random / systematic split of the declared u (share of u^2, mean over samples) ===")
    for name, (lo, hi) in spans.items():
        tot = (UX[:, lo:hi] ** 2).sum()
        print(f"  {name:28s} u_n={np.median(UXn[:, lo:hi]):8.3f}  "
              f"random {(U_rand[:, lo:hi] ** 2).sum() / tot:6.1%}  "
              f"systematic {(U_sys[:, lo:hi] ** 2).sum() / tot:6.1%}  "
              f"u_rand,n={np.median(URn[:, lo:hi]):.4f}")

    # ---- ensemble ----
    models, preds_va, preds_te = [], [], []
    for s in range(cfg.n_ensemble):
        m = make_mlp(Xn.shape[1], cfg)
        train(m, Xn[tr], y[tr], cfg, seed=s)
        models.append(m)
        with torch.no_grad():
            preds_va.append(m(torch.tensor(Xn[va], dtype=torch.float64)).squeeze(1).numpy())
            preds_te.append(m(torch.tensor(Xn[te], dtype=torch.float64)).squeeze(1).numpy())
        print(f"  seed {s} trained")
    u_training_va = np.stack(preds_va).std(axis=0, ddof=1)
    u_training_te = np.stack(preds_te).std(axis=0, ddof=1)
    model = models[0]

    # ---- u_model from the validation set ----
    v = budget_rows(model, Xn, URn, USn, UXn, va, u_training_va, spans, cfg, with_extra=False)
    e_va = y[va] - v[:, 0]
    msq = np.mean(e_va ** 2)
    u_model = math.sqrt(max(0.0, msq - np.mean(u_training_va ** 2) - np.mean(v[:, 2] ** 2)))
    print(f"\nvalidation: RMSE {math.sqrt(msq):.4f}, rms u_training {math.sqrt(np.mean(u_training_va**2)):.4f}, "
          f"rms u_rand_mc {math.sqrt(np.mean(v[:, 2]**2)):.4f}  ->  u_model {u_model:.4f}")

    # ---- test set ----
    t = budget_rows(model, Xn, URn, USn, UXn, te, u_training_te, spans, cfg)
    y_hat, u_rand_lpu, u_rand_mc, mc_bias, u_tr, u_sys_lpu, u_sys_mc, u_indep_mc = t.T
    e = y[te] - y_hat
    u_c = np.sqrt(u_rand_mc ** 2 + u_tr ** 2 + u_model ** 2)
    u_c_transfer = np.sqrt(u_c ** 2 + u_sys_mc ** 2)
    covered = np.abs(e) <= 2 * u_c
    rel = np.abs(u_rand_lpu - u_rand_mc) / np.maximum(u_rand_mc, 1e-300)
    cov = lambda u: np.mean(np.abs(e) <= 2 * u)

    print(f"\n=== Case Study A -- corrected budget [{tag}] ===")
    print(f"test RMSE                          : {math.sqrt(np.mean(e ** 2)):.4f}")
    print(f"test R2                            : {1 - np.sum(e**2) / np.sum((y[te] - y[te].mean())**2):.4f}")
    print(f"mean u_rand (LPU)                  : {u_rand_lpu.mean():.4e}")
    print(f"mean u_rand (MC)                   : {u_rand_mc.mean():.4e}")
    print(f"LPU vs MC median rel. diff (rand)  : {np.median(rel):.2%}")
    print(f"MC mean - f(x) (rand), mean |.|    : {np.mean(np.abs(mc_bias)):.4e}")
    print(f"mean u_training                    : {u_tr.mean():.4e}")
    print(f"u_model (validation)               : {u_model:.4e}")
    print(f"mean u_c (same instrument, MC)     : {u_c.mean():.4e}")
    print(f"mean u_sys (LPU / MC, transfer)    : {u_sys_lpu.mean():.4e} / {u_sys_mc.mean():.4e}")
    print(f"mean u_c (transfer)                : {u_c_transfer.mean():.4e}")
    print(f"mean u_indep (MC, original budget) : {u_indep_mc.mean():.4e}")
    print("coverage @ k=2:")
    print(f"  u_training only                  : {cov(u_tr):.1%}")
    print(f"  u_rand + u_training              : {cov(np.sqrt(u_rand_mc**2 + u_tr**2)):.1%}")
    print(f"  u_c (rand + training + model)    : {covered.mean():.1%}   (target ~95%)")
    print(f"  u_c_transfer                     : {cov(u_c_transfer):.1%}")
    print(f"  original budget (indep + train)  : {cov(np.sqrt(u_indep_mc**2 + u_tr**2)):.1%}")
    print(f"mean u_c / mean |error|            : {u_c.mean() / np.mean(np.abs(e)):.2f}")
    print(f"corr(u_c, |error|)                 : {np.corrcoef(u_c, np.abs(e))[0, 1]:.2f}")

    out = np.c_[y[te], y_hat, u_rand_lpu, u_rand_mc, u_tr, u_c, covered,
                mc_bias, np.full(len(te), u_model), u_sys_lpu, u_sys_mc, u_c_transfer, u_indep_mc]
    cols = ("y_true,y_hat,u_physical_lpu,u_physical_mc,u_training,u_c,covered_k2,"
            "mc_bias,u_model,u_sys_lpu,u_sys_mc,u_c_transfer,u_indep_mc")
    fname = f"case_study_A_results_budget_{tag}.csv"
    np.savetxt(fname, out, delimiter=",", header=cols)
    print(f"\nwrote {fname}  (u_physical_* = random part, same-instrument use)")
    return out


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--self-test", action="store_true",
                   help="validate the LPU implementation without the dataset (same as case_study_A_analysis.py)")
    p.add_argument("--exclude", nargs="*", default=[], metavar="CHANNEL")
    p.add_argument("--sampling", choices=SAMPLINGS, default="first")
    p.add_argument("--n-points", type=int, default=100)
    p.add_argument("--random-model", choices=RANDOM_MODELS, default="bound")
    p.add_argument("--split", choices=SPLITS, default="random")
    p.add_argument("--block-size", type=int, default=50, help="cycles per block (--split blocked)")
    p.add_argument("--gap", type=int, default=5, help="purge gap in cycles (--split blocked)")
    p.add_argument("--n-cycles", type=int, default=None, help="use only the first N cycles (smoke test)")
    args = p.parse_args()
    if args.self_test:
        self_test_lpu()
    else:
        cfg = Config() if args.n_cycles is None else Config(n_cycles=args.n_cycles)
        main_budget(cfg, exclude=tuple(args.exclude), n_points=args.n_points,
                    sampling=args.sampling, random_model=args.random_model,
                    split=args.split, block_size=args.block_size, gap=args.gap)
