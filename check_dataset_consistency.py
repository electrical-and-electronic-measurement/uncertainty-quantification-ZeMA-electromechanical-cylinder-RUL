"""
Dataset description and consistency checks for the ZeMA EMC dataset
(axis11_2kHz_ZeMA_PTB_SI.h5, doi:10.5281/zenodo.5185953) against
Dorst et al. 2022 (doi:10.1016/j.measen.2022.100376).

Prints every number quoted in `case_study_A_dataset_report.md`:
per-channel inventory (sensor, unit, range, statistics, uncertainty model,
u/s ratio), metadata, cycle-count/duration checks, within-window profiles,
SUU-specific checks and the `zema_emc_annotated` extraction check.

Usage:
    python check_dataset_consistency.py
"""

from __future__ import annotations

import json
import os
import pickle

import h5py
import numpy as np
from pooch import os_cache, retrieve
from zema_emc_annotated.dataset import ZEMA_DATASET_HASH, ZEMA_DATASET_URL

PAPER = {  # values stated in Dorst et al. 2022 (pages as in the article)
    "zema_cycles": 476_560,      # §3.1
    "suu_cycles": 476_617,       # §4.1
    "first_cycle": 51,           # §3.1 / §4.2, 1-based
    "every": 100,                # §3.1
    "duration_h": 387.83,        # §3
    "velocity_mm_s": 200.0,      # §2
    "load_kN": 7.0,              # §2
    "cycle_s": 2.8,              # §2
    "interp_unc_cycle51": 2.236e-3,  # App. A, median sqrt(eq. A.2), m/s^2
}


def label_list(ds):
    s = ds.attrs["si:label"]
    return json.loads(s) if s.startswith("[") else [s.strip('"')]


def channels(h5f):
    out = []
    h5f.visititems(lambda n, o: out.append(n)
                   if isinstance(o, h5py.Group) and "qudt:value" in o else None)
    return out


def uncertainty_model(v, u):
    if np.ptp(u) == 0:
        return f"constant {u.flat[0]:.4g}"
    nz = np.abs(v) > 1e-6
    r = u[nz] / np.abs(v[nz])
    if np.ptp(r) < 1e-9:
        return f"relative {np.median(r):.4g} x |value|"
    return f"time-varying {u.min():.3g}..{u.max():.3g} (median {np.median(u):.3g})"


def main():
    path = retrieve(url=ZEMA_DATASET_URL, known_hash=ZEMA_DATASET_HASH, progressbar=True)
    print(f"file: {path}  ({os.path.getsize(path) / 1e9:.2f} GB)")
    with h5py.File(path, "r") as f:
        print("\n=== Root metadata ===")
        for k, v in f.attrs.items():
            print(f"{k}: {v}")

        print("\n=== Channel inventory ===")
        print("group | label | sensor | unit | shape | min | max | mean | std | uncertainty | u/s | NaN/Inf")
        for name in channels(f):
            g = f[name]
            V, U = g["qudt:value"], g["qudt:standardUncertainty"]
            sensor = g.attrs["sosa:madeBySensor"]
            try:
                sensor = json.loads(sensor)
            except (json.JSONDecodeError, TypeError):
                pass
            for i, lab in enumerate(label_list(V)):
                v = V[i][()] if V.ndim == 3 else V[()]
                u = U[i][()] if U.ndim == 3 else U[()]
                s_cyc = np.median(v.std(axis=1))       # std across cycles, per position
                sens = sensor[i] if isinstance(sensor, list) else sensor
                print(f"{name} | {lab} | {sens} | {g.attrs['si:unit']} | {V.shape} | "
                      f"{v.min():.4g} | {v.max():.4g} | {v.mean():.4g} | {v.std():.4g} | "
                      f"{uncertainty_model(v, u)} | {np.median(u) / s_cyc:.3g} | "
                      f"{int((~np.isfinite(v)).sum() + (~np.isfinite(u)).sum())}")

        print("\n=== Cycle count and duration ===")
        n = f["ZeMA_DAQ/Force/qudt:value"].shape[1]
        expected = len(range(PAPER["first_cycle"], PAPER["zema_cycles"] + 1, PAPER["every"]))
        print(f"cycles in arrays: {n}; expected from paper (51..476560 step 100): {expected}; "
              f"metadata dc:SizeOrDuration says 4776")
        period = PAPER["duration_h"] * 3600 / PAPER["zema_cycles"]
        print(f"mean cycle period implied by paper duration: {period:.3f} s "
              f"(nominal {PAPER['cycle_s']} s + ~0.1 s gap)")
        print(f"one stored cycle every {PAPER['every'] * period / 60:.2f} min")

        print("\n=== Operating point vs paper ===")
        vel = f["ZeMA_DAQ/Velocity/qudt:value"][()]
        frc = f["ZeMA_DAQ/Force/qudt:value"][()]
        print(f"velocity median {np.median(vel):.1f} mm/s (paper ~{PAPER['velocity_mm_s']})")
        fm = frc.mean(axis=0)
        print(f"axial force per-cycle mean {fm.min():.2f}..{fm.max():.2f} kN, "
              f"median {np.median(fm):.2f}, max sample {frc.max():.2f} (paper {PAPER['load_kN']} kN)")
        print(f"pneumatic pressure mean {f['ZeMA_DAQ/Pressure/qudt:value'][()].mean() / 1e5:.2f} bar")

        print("\n=== Within-window profile (mean over cycles, 20 blocks of 50 ms) ===")
        for q in ("Velocity", "Force", "Pressure", "Active_Current"):
            prof = f[f"ZeMA_DAQ/{q}/qudt:value"][()].mean(axis=1).reshape(20, 100).mean(axis=1)
            print(q, np.round(prof, 3).tolist())

        print("\n=== SUU checks ===")
        m = "PTB_SUU/MPU_9250/"
        ref = f[m + "Acceleration/qudt:standardUncertainty"][0]
        for q, k in [("Acceleration", 3), ("Angular_Velocity", 3), ("Magnetic_Flux_Density", 3),
                     ("Temperature", 1)]:
            U = f[m + q + "/qudt:standardUncertainty"]
            for i in range(k):
                ui = U[i] if U.ndim == 3 else U
                print(f"MPU {q}[{i}] uncertainty identical to MPU acceleration X: "
                      f"{np.array_equal(ui[()], ref[()])}")
        bu = f["PTB_SUU/BMA_280/Acceleration/qudt:standardUncertainty"][()]
        print("BMA X/Y/Z uncertainties identical:",
              np.array_equal(bu[0], bu[1]) and np.array_equal(bu[0], bu[2]))
        mu = f[m + "Acceleration/qudt:standardUncertainty"][()]
        print(f"cycle 51 median u: BMA {np.median(bu[:, :, 0]):.4g}, MPU {np.median(mu[:, :, 0]):.4g} m/s^2 "
              f"(paper App. A interpolation term: {PAPER['interp_unc_cycle51']:.4g})")
        k = f["ZeMA_DAQ/Acceleration/qudt:value"][0]
        for s in ("BMA_280", "MPU_9250"):
            a = f[f"PTB_SUU/{s}/Acceleration/qudt:value"][()]
            g_mag = np.linalg.norm(a.mean(axis=1), axis=0).mean()
            r = np.corrcoef(k.std(axis=0), np.linalg.norm(a.std(axis=1), axis=0))[0, 1]
            print(f"{s}: mean |a| = {g_mag:.3f} m/s^2 (g = 9.807); corr of per-cycle RMS "
                  f"with Kistler plain bearing = {r:.3f}")
        mpu_x = f[m + "Acceleration/qudt:value"][0][:, :200]
        print("MPU acc X fraction of equal consecutive samples:",
              round(float(np.mean(np.diff(mpu_x, axis=0) == 0)), 4))
        mag = f[m + "Magnetic_Flux_Density/qudt:value"][0][:, :200]
        smooth = lambda x: np.diff(x, 2, axis=0).std() / x.std()
        print(f"2nd-difference / std: magnetometer {smooth(mag):.3f}, accelerometer {smooth(mpu_x):.3f}")

        print("\n=== zema_emc_annotated extraction ===")
        cache = os_cache("pooch") / "4766_samples_with_100_values_per_sensor.pickle"
        if cache.exists():
            lib = np.asarray(pickle.load(open(cache, "rb")).values)
            raw = f["ZeMA_DAQ/Acceleration/qudt:value"][0][:100, :].T
            print("ZeMASamples(datapoints_per_cycle=100) == first 100 raw samples:",
                  np.array_equal(lib[:, :100], raw))
        else:
            print("library cache not found; run case_study_A_analysis.py once to create it")


if __name__ == "__main__":
    main()
