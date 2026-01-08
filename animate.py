#!/usr/bin/env python3
"""
animate_npz.py
--------------

Animate one SRCM result file, or overlay two files (Hybrid + SSA).

Usage:
  # 1) Animate a single result (hybrid or any srcm_engine-formatted npz)
  python animate_npz.py data/hybrid_results.npz

  # 2) Overlay SSA on Hybrid
  python animate_npz.py data/hybrid_results.npz data/ssa_results.npz
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


def print_npz_keys(npz_path: Path):
    """Small debug helper: prints keys without crashing on object arrays."""
    print(f"\n[NPZ] {npz_path}")
    d = np.load(npz_path, allow_pickle=False)
    print("Keys:")
    for k in d.files:
        try:
            arr = np.asarray(d[k])
            print(f"  - {k:18s} shape={arr.shape} dtype={arr.dtype}")
        except ValueError:
            print(f"  - {k:18s} <object array: not loaded (allow_pickle=False)>")


def load_ssa_like(npz_path: Path, ref_hybrid: SimulationResults) -> SimulationResults:
    """
    Load an SSA file robustly and adapt it to the Hybrid domain grid.
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

    # PDE zeros on Hybrid fine grid
    n_species, _K, T = ssa.shape
    pde = np.zeros((n_species, ref_hybrid.pde.shape[1], T), dtype=float)

    return SimulationResults(time=time, ssa=ssa, pde=pde, domain=domain, species=species)


def main():
    ap = argparse.ArgumentParser(description="Animate one SRCM npz, or overlay Hybrid + SSA.")
    ap.add_argument("main_npz", type=str, help="Path to main .npz (usually Hybrid)")
    ap.add_argument("overlay_npz", nargs="?", default=None, help="Optional overlay .npz (usually SSA)")
    ap.add_argument("--stride", type=int, default=20, help="Animation stride")
    ap.add_argument("--interval", type=int, default=30, help="Frame interval in ms")
    ap.add_argument("--title", type=str, default=None, help="Custom plot title")
    ap.add_argument("--mass", type=str, default="none", choices=["none", "single", "per_species"])
    ap.add_argument("--threshold", type=float, default=None, help="Manual threshold override")
    ap.add_argument("--debug-keys", action="store_true", help="Print npz keys")
    args = ap.parse_args()

    main_path = Path(args.main_npz)
    if not main_path.exists():
        raise FileNotFoundError(main_path)

    overlay_path = Path(args.overlay_npz) if args.overlay_npz else None
    
    if args.debug_keys:
        print_npz_keys(main_path)
        if overlay_path:
            print_npz_keys(overlay_path)

    plt.close("all")

    # 1. Load Main Data and Metadata once
    res_main, meta = load_npz(str(main_path))

    # 2. Threshold extraction (CLI override > Metadata > None)
    threshold_val = args.threshold if args.threshold is not None else meta.get("threshold_particles")
    
    if threshold_val is not None:
        print(f"Using threshold: {threshold_val} particles")

    # 3. Build Config
    cfg = AnimationConfig(
        stride=int(args.stride),
        interval_ms=int(args.interval),
        title=args.title or (f"{main_path.name}" if not overlay_path else f"Overlay: {main_path.name} vs {overlay_path.name}"),
        mass_plot_mode=args.mass,
        threshold_particles=threshold_val,
        show_threshold=(threshold_val is not None)
    )

    # -----------------------------
    # Logic Branch: Single vs Overlay
    # -----------------------------
    if overlay_path is None:
        # Single File Animation
        animate_results(res_main, cfg=cfg)
    else:
        # Overlay Animation
        res_overlay = load_ssa_like(overlay_path, ref_hybrid=res_main)
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