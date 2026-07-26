"""
Case Study A -- GUM-compliant uncertainty quantification for RUL estimation
on the ZeMA electromechanical cylinder dataset (Dorst et al. 2021,
doi:10.5281/zenodo.5185953).

Implements the 5-phase methodology of the paper:
  Phase 1: regression task, measurand = consumed-lifetime fraction (proxy for RUL)
  Phase 2: Type A input uncertainties from the annotated dataset;
           Type B model uncertainty from a seed ensemble
  Phase 3: white-box MLP + softplus (C^1) -> analytical LPU applicable
  Phase 4: layer-wise LPU (Eq. LPU-for-NN in the paper) + quadrature combination
  Phase 5: LPU vs Monte Carlo comparison + empirical coverage on held-out cycles

Requirements (Python >= 3.10):
  pip install zema_emc_annotated torch numpy matplotlib

Notes
-----
* The LPU is implemented directly in this script (~40 lines, `propagate_lpu`)
  rather than importing pytorch_gum_uncertainty_propagation, because that
  library pins older torch versions; the implementation below follows the same
  layer-wise formula U_{x(i)} = H W U_{x(i-1)} W^T H and is validated against
  Monte Carlo in `self_test_lpu()` (run with --self-test, no dataset needed).
* First run downloads ~200 MB from Zenodo into ~/.cache (handled by
  zema_emc_annotated).
"""

from __future__ import annotations

import argparse
import dataclasses
import math

import numpy as np
import torch
from torch import nn

# ----------------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------------

@dataclasses.dataclass
class Config:
    n_cycles: int = 4766          # all available cycles (axis11 dataset)
    datapoints_per_cycle: int = 100  # readings per sensor -> input dim = 11*100
    hidden: tuple = (128, 64, 32)
    activation: str = "softplus"  # C^1 -> LPU-compatible (paper Sec. 2.2)
    n_ensemble: int = 10          # seeds for Type B (training-process) uncertainty
    n_mc: int = 10_000            # GUM Supplement 1 samples for validation
    test_fraction: float = 0.2
    epochs: int = 200
    lr: float = 1e-3
    seed: int = 0


# ----------------------------------------------------------------------------
# Phase 1+2 -- data loading: values, Type A uncertainties, target definition
# ----------------------------------------------------------------------------

def load_zema(cfg: Config):
    """Returns X, UX (per-sample standard uncertainties), y.

    Target definition: the testbed experiment ran the cylinder to end of life,
    so the consumed-lifetime fraction of cycle k out of N total is y_k = k/N
    (in [0,1]); RUL = 1 - y. This is the convention of the ZeMA ML tutorials
    (Lulic & Jugo 2019) and of Ludwig (2023). The label is defined
    deterministically from the cycle index -> label uncertainty negligible.
    """
    from zema_emc_annotated.data_types import SampleSize
    from zema_emc_annotated.dataset import ZeMASamples

    samples = ZeMASamples(
        SampleSize(n_cycles=cfg.n_cycles,
                   datapoints_per_cycle=cfg.datapoints_per_cycle),
        normalize=False,  # we normalize manually and propagate uncertainty
    )
    X = np.asarray(samples.values, dtype=np.float64)          # (N, 11*d)
    UX = np.asarray(samples.uncertainties, dtype=np.float64)  # (N, 11*d) std. unc.
    y = np.linspace(0.0, 1.0, X.shape[0])                     # consumed life
    return X, UX, y


def normalize_with_uncertainty(X, UX, mean, std):
    """z-score normalisation is a linear map -> exact GUM propagation:
    u(x') = u(x)/std."""
    return (X - mean) / std, UX / std


# ----------------------------------------------------------------------------
# Model
# ----------------------------------------------------------------------------

def make_mlp(in_dim: int, cfg: Config) -> nn.Sequential:
    acts = {"softplus": nn.Softplus, "sigmoid": nn.Sigmoid, "tanh": nn.Tanh}
    layers, d = [], in_dim
    for h in cfg.hidden:
        layers += [nn.Linear(d, h), acts[cfg.activation]()]
        d = h
    layers += [nn.Linear(d, 1)]  # linear output head (regression)
    return nn.Sequential(*layers)


def train(model, Xtr, ytr, cfg: Config, seed: int):
    torch.manual_seed(seed)
    for m in model.modules():
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight); nn.init.zeros_(m.bias)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    Xt = torch.tensor(Xtr, dtype=torch.float64)
    yt = torch.tensor(ytr, dtype=torch.float64).unsqueeze(1)
    model.double()
    for _ in range(cfg.epochs):
        opt.zero_grad()
        loss = nn.functional.mse_loss(model(Xt), yt)
        loss.backward(); opt.step()
    return model


# ----------------------------------------------------------------------------
# Phase 4a -- analytical LPU:  U_{x(i)} = H(i) W(i) U_{x(i-1)} W(i)^T H(i)
# (Eq. \ref{eq:Law_of_Propagation_of_Uncertainty_for_NN} in the paper)
# ----------------------------------------------------------------------------

def _activation_derivative(act: nn.Module, z: torch.Tensor) -> torch.Tensor:
    if isinstance(act, nn.Softplus):
        return torch.sigmoid(act.beta * z)      # d/dz softplus_b(z) = sigmoid(b z)
    if isinstance(act, nn.Sigmoid):
        s = torch.sigmoid(z); return s * (1 - s)
    if isinstance(act, nn.Tanh):
        return 1 - torch.tanh(z) ** 2
    raise NotImplementedError(type(act))


@torch.no_grad()
def propagate_lpu(model: nn.Sequential, x: torch.Tensor, Ux: torch.Tensor):
    """Layer-wise propagation of the input covariance matrix through an MLP.

    x  : (n_in,)      best estimates of the inputs
    Ux : (n_in,n_in)  input covariance matrix (diag(u^2) if uncorrelated)
    returns (y, Uy)
    """
    a, U = x.clone(), Ux.clone()
    layers = list(model)
    i = 0
    while i < len(layers):
        layer = layers[i]
        if isinstance(layer, nn.Linear):
            W = layer.weight            # (n_out, n_in)
            z = layer(a)                # pre-activation
            U = W @ U @ W.T             # linear stage: C = W
            # if followed by an activation, fold in H = diag(sigma'(z))
            if i + 1 < len(layers) and not isinstance(layers[i + 1], nn.Linear):
                h = _activation_derivative(layers[i + 1], z)
                U = U * torch.outer(h, h)   # H U H^T for diagonal H
                a = layers[i + 1](z)
                i += 2
                continue
            a = z
        i += 1
    return a, U


# ----------------------------------------------------------------------------
# Phase 4b/5 -- Monte Carlo per GUM Supplement 1 (JCGM 101:2008)
# ----------------------------------------------------------------------------

@torch.no_grad()
def propagate_mc(model, x, ux_std, n_mc: int, seed: int = 0):
    g = torch.Generator().manual_seed(seed)
    eps = torch.randn(n_mc, x.numel(), generator=g, dtype=x.dtype)
    samples = model(x.unsqueeze(0) + eps * ux_std.unsqueeze(0))
    return samples.mean().item(), samples.std(unbiased=True).item()


# ----------------------------------------------------------------------------
# Phase 5 -- self test on synthetic data (no dataset download required)
# ----------------------------------------------------------------------------

def self_test_lpu(cfg: Config | None = None, in_dim: int = 11):
    """Validates propagate_lpu against Monte Carlo on a random MLP."""
    cfg = cfg or Config(hidden=(32, 16), epochs=0)
    torch.manual_seed(1)
    model = make_mlp(in_dim, cfg).double()
    x = torch.randn(in_dim, dtype=torch.float64)
    ux = 0.01 * torch.rand(in_dim, dtype=torch.float64) + 0.005
    _, U_lpu = propagate_lpu(model, x, torch.diag(ux ** 2))
    u_lpu = math.sqrt(U_lpu.item())
    _, u_mc = propagate_mc(model, x, ux, n_mc=200_000)
    rel = abs(u_lpu - u_mc) / u_mc
    print(f"[self-test] u_LPU={u_lpu:.6e}  u_MC={u_mc:.6e}  rel.diff={rel:.3%}")
    assert rel < 0.02, "LPU and MC disagree by more than 2% on a smooth MLP"
    return rel


# ----------------------------------------------------------------------------
# Full pipeline
# ----------------------------------------------------------------------------

def main(cfg: Config):
    X, UX, y = load_zema(cfg)

    # chronological split: train on early life, test on late cycles is TOO easy
    # to game (extrapolation) or too hard; the standard tutorial split is random.
    rng = np.random.default_rng(cfg.seed)
    idx = rng.permutation(len(y))
    n_test = int(cfg.test_fraction * len(y))
    te, tr = idx[:n_test], idx[n_test:]

    mean, std = X[tr].mean(0), X[tr].std(0) + 1e-12
    Xn, UXn = normalize_with_uncertainty(X, UX, mean, std)

    # ---- Type B: seed ensemble (training-process uncertainty) ----
    models, preds_te = [], []
    for s in range(cfg.n_ensemble):
        m = make_mlp(Xn.shape[1], cfg)
        train(m, Xn[tr], y[tr], cfg, seed=s)
        models.append(m)
        with torch.no_grad():
            preds_te.append(
                m(torch.tensor(Xn[te], dtype=torch.float64)).squeeze(1).numpy())
    preds_te = np.stack(preds_te)                # (n_ensemble, n_test)
    u_B = preds_te.std(axis=0, ddof=1)           # per-sample Type B std. unc.
    model = models[0]                            # reference model for LPU/MC

    # ---- Type A via LPU, validated against MC, on the test set ----
    rows = []
    for j, i in enumerate(te):
        x = torch.tensor(Xn[i], dtype=torch.float64)
        ux = torch.tensor(UXn[i], dtype=torch.float64)
        yhat, U_y = propagate_lpu(model, x, torch.diag(ux ** 2))
        u_A = math.sqrt(U_y.item())
        _, u_mc = propagate_mc(model, x, ux, cfg.n_mc, seed=j)
        u_c = math.sqrt(u_A ** 2 + u_B[j] ** 2)
        covered = abs(y[i] - yhat.item()) <= 2 * u_c   # k=2 interval
        rows.append((y[i], yhat.item(), u_A, u_mc, u_B[j], u_c, covered))

    r = np.array(rows, dtype=float)
    rel_diff = np.abs(r[:, 2] - r[:, 3]) / r[:, 3]
    print("\n=== Case Study A results (fill Table/Fig in the paper) ===")
    print(f"test RMSE                  : {np.sqrt(np.mean((r[:,0]-r[:,1])**2)):.4f}")
    print(f"mean u_A (LPU)             : {r[:,2].mean():.4e}")
    print(f"mean u_A (Monte Carlo)     : {r[:,3].mean():.4e}")
    print(f"LPU vs MC median rel. diff : {np.median(rel_diff):.2%}  (Phase 5 check)")
    print(f"mean u_B (ensemble)        : {r[:,4].mean():.4e}")
    print(f"mean u_c (combined)        : {r[:,5].mean():.4e}")
    print(f"empirical coverage @ k=2   : {r[:,6].mean():.1%}  (target ~95%)")
    np.savetxt("case_study_A_results.csv", r, delimiter=",",
               header="y_true,y_hat,u_A_lpu,u_A_mc,u_B,u_c,covered_k2")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--self-test", action="store_true",
                   help="validate the LPU implementation without the dataset")
    args = p.parse_args()
    if args.self_test:
        self_test_lpu()
    else:
        main(Config())
