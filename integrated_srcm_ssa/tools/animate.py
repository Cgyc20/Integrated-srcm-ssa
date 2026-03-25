#!/usr/bin/env python3
"""
animate_npz.py
--------------
Animate one SRCM result file, or overlay two files (Hybrid + SSA).
"""

from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

from srcm_engine.results.io import load_npz
from srcm_engine.results.simulation_results import SimulationResults
from srcm_engine.animation_util import AnimationConfig, animate_results
from srcm_engine.animation_util.animate import animate_overlay


def print_simulation_summary(res: SimulationResults, meta: dict, title: str):
    """Print a clean summary of the simulation data to the terminal."""
    print("\n" + "=" * 50)
    print(f" LOADING: {title}")
    print("=" * 50)
    print(f" Species:    {', '.join(res.species)}")
    print(f" Time Steps: {res.time.shape[0]} (t_start={res.time[0]:.2f}, t_end={res.time[-1]:.2f})")
    print(f" Domain:     L={res.domain.length}, K={res.domain.K} (SSA compartments)")
    print(f" PDE Grid:   Npde={res.domain.n_pde} (multiple={res.domain.pde_multiple})")

    if meta:
        print("-" * 50)
        print(" Metadata found in file:")

        conversion = meta.get("conversion", {})
        if isinstance(conversion, dict) and conversion:
            print("  - conversion:")
            if "DC_threshold" in conversion:
                print(f"      DC_threshold: {conversion['DC_threshold']}")
            if "CD_threshold" in conversion:
                print(f"      CD_threshold: {conversion['CD_threshold']}")
            if "rate" in conversion:
                print(f"      rate: {conversion['rate']}")

        for k, v in meta.items():
            if k == "conversion":
                continue
            if not isinstance(v, (dict, list)):
                print(f"  - {k}: {v}")

    print("=" * 50 + "\n")


def print_npz_keys(npz_path: Path):
    """Small debug helper: print keys without crashing on object arrays."""
    print(f"\n[DEBUG] Keys in {npz_path.name}:")
    d = np.load(npz_path, allow_pickle=False)
    for k in d.files:
        try:
            arr = np.asarray(d[k])
            print(f"  - {k:18s} shape={arr.shape} dtype={arr.dtype}")
        except ValueError:
            print(f"  - {k:18s} <object array>")


def load_ssa_like(npz_path: Path, ref_hybrid: SimulationResults) -> SimulationResults:
    """
    Load an overlay file. First try the standard results loader.
    If that fails, fall back to a minimal SSA-like npz format with keys time + ssa.
    """
    try:
        res, _meta = load_npz(str(npz_path))
        return res
    except Exception:
        pass

    d = np.load(npz_path, allow_pickle=True)
    if "time" not in d.files or "ssa" not in d.files:
        raise KeyError(f"SSA overlay file must contain 'time' and 'ssa'. Found: {d.files}")

    time = np.asarray(d["time"])
    ssa = np.asarray(d["ssa"])
    domain = ref_hybrid.domain
    species = list(ref_hybrid.species)

    if ssa.ndim != 3:
        raise ValueError(f"Expected SSA array with shape (n_species, K, T), got {ssa.shape}")

    n_species, _K, T = ssa.shape
    pde = np.zeros((n_species, ref_hybrid.pde.shape[1], T), dtype=float)

    return SimulationResults(time=time, ssa=ssa, pde=pde, domain=domain, species=species)


def _extract_visual_threshold(meta: dict, manual_threshold: float | None) -> float | None:
    """
    Extract a single threshold value for plotting.

    Priority:
    1. --threshold manual override
    2. old metadata key: threshold_particles
    3. new metadata key: conversion.CD_threshold, if scalar
    4. new metadata key: conversion.DC_threshold, if scalar
    5. None
    """
    if manual_threshold is not None:
        return float(manual_threshold)

    if not isinstance(meta, dict):
        return None

    old_val = meta.get("threshold_particles")
    if old_val is not None:
        return float(old_val)

    conversion = meta.get("conversion", {})
    if not isinstance(conversion, dict):
        return None

    for key in ("CD_threshold", "DC_threshold"):
        val = conversion.get(key)
        if val is None:
            continue
        arr = np.asarray(val)
        if arr.ndim == 0:
            return float(arr)

    return None


def main():
    ap = argparse.ArgumentParser(description="Animate one SRCM npz, or overlay Hybrid + SSA.")
    ap.add_argument("main_npz", type=str, help="Path to main .npz (usually Hybrid)")
    ap.add_argument("overlay_npz", nargs="?", default=None, help="Optional overlay .npz (usually SSA)")
    ap.add_argument("--stride", type=int, default=20, help="Animation stride")
    ap.add_argument("--interval", type=int, default=30, help="Frame interval in ms")
    ap.add_argument("--title", type=str, default=None, help="Custom plot title")
    ap.add_argument("--mass", type=str, default="none", choices=["none", "single", "per_species"])
    ap.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="Manual single threshold line for visualisation only",
    )
    ap.add_argument("--debug-keys", action="store_true", help="Print npz keys")
    args = ap.parse_args()

    main_path = Path(args.main_npz)
    if not main_path.exists():
        raise FileNotFoundError(main_path)

    if args.overlay_npz is not None:
        overlay_path = Path(args.overlay_npz)
        if not overlay_path.exists():
            raise FileNotFoundError(overlay_path)
    else:
        overlay_path = None

    if args.debug_keys:
        print_npz_keys(main_path)
        if overlay_path is not None:
            print_npz_keys(overlay_path)

    plt.close("all")

    # 1. Load main data
    res_main, meta = load_npz(str(main_path))

    # 2. Extract a single threshold for plotting, if available
    threshold_val = _extract_visual_threshold(meta, args.threshold)

    # 3. Print information
    print_simulation_summary(res_main, meta, main_path.name)
    if threshold_val is not None:
        print(f"→ Using threshold guide line at: {threshold_val}")
    else:
        print("→ No scalar threshold available for plotting.")

    # 4. Build config
    cfg = AnimationConfig(
        stride=int(args.stride),
        interval_ms=int(args.interval),
        title=args.title or (
            f"{main_path.name}"
            if overlay_path is None
            else f"Overlay: {main_path.name} vs {overlay_path.name}"
        ),
        mass_plot_mode=args.mass,
        threshold_particles=threshold_val,
        show_threshold=(threshold_val is not None),
    )

    # 5. Animate
    if overlay_path is None:
        animate_results(res_main, cfg=cfg)
    else:
        res_overlay = load_ssa_like(overlay_path, ref_hybrid=res_main)
        print(f"→ Overlaying SSA from: {overlay_path.name}")
        animate_overlay(
            res_main,
            res_overlay,
            cfg=cfg,
            label_main="Hybrid",
            label_overlay="SSA",
        )

    plt.show()


if __name__ == "__main__":
    main()