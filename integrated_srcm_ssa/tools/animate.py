# #!/usr/bin/env python3
# """
# animate_npz.py
# --------------
# Animate one SRCM result file, or overlay two files (Hybrid + SSA).
# """

# from __future__ import annotations

# import argparse
# from pathlib import Path
# import numpy as np
# import matplotlib.pyplot as plt

# from srcm_engine.results.io import load_npz
# from srcm_engine.results.simulation_results import SimulationResults
# from srcm_engine.animation_util import AnimationConfig, animate_results
# from srcm_engine.animation_util.animate import animate_overlay


# def print_simulation_summary(res: SimulationResults, meta: dict, title: str):
#     """Print a clean summary of the simulation data to the terminal."""
#     print("\n" + "=" * 50)
#     print(f" LOADING: {title}")
#     print("=" * 50)
#     print(f" Species:    {', '.join(res.species)}")
#     print(f" Time Steps: {res.time.shape[0]} (t_start={res.time[0]:.2f}, t_end={res.time[-1]:.2f})")
#     print(f" Domain:     L={res.domain.length}, K={res.domain.K} (SSA compartments)")
#     print(f" PDE Grid:   Npde={res.domain.n_pde} (multiple={res.domain.pde_multiple})")

#     if meta:
#         print("-" * 50)
#         print(" Metadata found in file:")

#         conversion = meta.get("conversion", {})
#         if isinstance(conversion, dict) and conversion:
#             print("  - conversion:")
#             if "DC_threshold" in conversion:
#                 print(f"      DC_threshold: {conversion['DC_threshold']}")
#             if "CD_threshold" in conversion:
#                 print(f"      CD_threshold: {conversion['CD_threshold']}")
#             if "rate" in conversion:
#                 print(f"      rate: {conversion['rate']}")

#         for k, v in meta.items():
#             if k == "conversion":
#                 continue
#             if not isinstance(v, (dict, list)):
#                 print(f"  - {k}: {v}")

#     print("=" * 50 + "\n")


# def print_npz_keys(npz_path: Path):
#     """Small debug helper: print keys without crashing on object arrays."""
#     print(f"\n[DEBUG] Keys in {npz_path.name}:")
#     d = np.load(npz_path, allow_pickle=False)
#     for k in d.files:
#         try:
#             arr = np.asarray(d[k])
#             print(f"  - {k:18s} shape={arr.shape} dtype={arr.dtype}")
#         except ValueError:
#             print(f"  - {k:18s} <object array>")


# def load_ssa_like(npz_path: Path, ref_hybrid: SimulationResults) -> SimulationResults:
#     """
#     Load an overlay file. First try the standard results loader.
#     If that fails, fall back to a minimal SSA-like npz format with keys time + ssa.
#     """
#     try:
#         res, _meta = load_npz(str(npz_path))
#         return res
#     except Exception:
#         pass

#     d = np.load(npz_path, allow_pickle=True)
#     if "time" not in d.files or "ssa" not in d.files:
#         raise KeyError(f"SSA overlay file must contain 'time' and 'ssa'. Found: {d.files}")

#     time = np.asarray(d["time"])
#     ssa = np.asarray(d["ssa"])
#     domain = ref_hybrid.domain
#     species = list(ref_hybrid.species)

#     if ssa.ndim != 3:
#         raise ValueError(f"Expected SSA array with shape (n_species, K, T), got {ssa.shape}")

#     n_species, _K, T = ssa.shape
#     pde = np.zeros((n_species, ref_hybrid.pde.shape[1], T), dtype=float)

#     return SimulationResults(time=time, ssa=ssa, pde=pde, domain=domain, species=species)


# def _extract_visual_thresholds(meta: dict, manual_threshold: float | None) -> tuple[float | None, float | None]:
#     """
#     Returns (CD_threshold, DC_threshold) for plotting two guide lines.

#     If manual_threshold is given, both are set to it (backward compat).
#     Returns (None, None) if nothing found.
#     """
#     if manual_threshold is not None:
#         return float(manual_threshold), float(manual_threshold)

#     if not isinstance(meta, dict):
#         return None, None

#     # backward compat: old single threshold key
#     old_val = meta.get("threshold_particles")
#     if old_val is not None:
#         v = float(old_val)
#         return v, v

#     conversion = meta.get("conversion", {})
#     if not isinstance(conversion, dict):
#         return None, None

#     def _first_float(val) -> float | None:
#         if val is None:
#             return None
#         if isinstance(val, (int, float)):
#             return float(val)
#         if isinstance(val, dict):
#             return float(next(iter(val.values())))
#         arr = np.asarray(val)
#         if arr.size > 0:
#             return float(arr.flat[0])
#         return None

#     cd = _first_float(conversion.get("CD_threshold"))
#     dc = _first_float(conversion.get("DC_threshold"))
#     return cd, dc

# def main():
#     ap = argparse.ArgumentParser(description="Animate one SRCM npz, or overlay Hybrid + SSA.")
#     ap.add_argument("main_npz", type=str, help="Path to main .npz (usually Hybrid)")
#     ap.add_argument("overlay_npz", nargs="?", default=None, help="Optional overlay .npz (usually SSA)")
#     ap.add_argument("--stride", type=int, default=20, help="Animation stride")
#     ap.add_argument("--interval", type=int, default=30, help="Frame interval in ms")
#     ap.add_argument("--title", type=str, default=None, help="Custom plot title")
#     ap.add_argument("--mass", type=str, default="none", choices=["none", "single", "per_species"])
#     ap.add_argument(
#         "--threshold",
#         type=float,
#         default=None,
#         help="Manual single threshold line for visualisation only",
#     )
#     ap.add_argument("--debug-keys", action="store_true", help="Print npz keys")
#     args = ap.parse_args()

#     main_path = Path(args.main_npz)
#     if not main_path.exists():
#         raise FileNotFoundError(main_path)

#     if args.overlay_npz is not None:
#         overlay_path = Path(args.overlay_npz)
#         if not overlay_path.exists():
#             raise FileNotFoundError(overlay_path)
#     else:
#         overlay_path = None

#     if args.debug_keys:
#         print_npz_keys(main_path)
#         if overlay_path is not None:
#             print_npz_keys(overlay_path)

#     plt.close("all")

#     # 1. Load main data
#     res_main, meta = load_npz(str(main_path))

#     # 2. Extract thresholds for plotting
#     cd_thresh, dc_thresh = _extract_visual_thresholds(meta, args.threshold)

#     # 3. Print information
#     print_simulation_summary(res_main, meta, main_path.name)
#     if cd_thresh is not None or dc_thresh is not None:
#         print(f"→ CD threshold (PDE→SSA): {cd_thresh}")
#         print(f"→ DC threshold (SSA→PDE): {dc_thresh}")
#     else:
#         print("→ No threshold available for plotting.")

#     # 4. Build config
#     cfg = AnimationConfig(
#         stride=int(args.stride),
#         interval_ms=int(args.interval),
#         title=args.title or (
#             f"{main_path.name}"
#             if overlay_path is None
#             else f"Overlay: {main_path.name} vs {overlay_path.name}"
#         ),
#         mass_plot_mode=args.mass,
#         threshold_particles=cd_thresh,       # lower line
#         DC_threshold=dc_thresh,              # upper line  ← need to add to AnimationConfig
#         show_threshold=(cd_thresh is not None or dc_thresh is not None),
#     )


#     # 5. Animate
#     if overlay_path is None:
#         animate_results(res_main, cfg=cfg)
#     else:
#         res_overlay = load_ssa_like(overlay_path, ref_hybrid=res_main)
#         print(f"→ Overlaying SSA from: {overlay_path.name}")
#         animate_overlay(
#             res_main,
#             res_overlay,
#             cfg=cfg,
#             label_main="Hybrid",
#             label_overlay="SSA",
#         )

#     plt.show()


# if __name__ == "__main__":
#     main()


#!/usr/bin/env python3
"""
animate.py
----------
Self-contained animation tool for SRCM simulation results.
Animates one .npz file, or overlays two (Hybrid + SSA).
No dependency on srcm_engine.animation_util.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Literal, Optional, Sequence, Tuple

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import gridspec
from matplotlib.animation import FuncAnimation

from srcm_engine.results.io import load_npz
from srcm_engine.results.simulation_results import SimulationResults


# ============================================================
# Configuration
# ============================================================

@dataclass
class AnimationConfig:
    stride: int = 10
    interval_ms: int = 30
    show_threshold: bool = True
    threshold_particles: Optional[float | Dict[str, float]] = None   # CD threshold (lower, PDE→SSA)
    DC_threshold_particles: Optional[float | Dict[str, float]] = None # DC threshold (upper, SSA→PDE)
    title: str = "SRCM Hybrid Simulation"
    blit: bool = False
    mass_plot_mode: Literal["single", "per_species", "none"] = "single"


# ============================================================
# Style
# ============================================================

def _setup_style() -> Dict[str, str]:
    plt.style.use("dark_background")
    colors = {
        "species_a":       "#4DC3FF",
        "species_b":       "#FF4D6D",
        "species_c":       "#4DFF9E",
        "species_d":       "#FFD700",
        "species_a_light": "#7BD5FF",
        "species_b_light": "#FF7B94",
        "species_c_light": "#7BFFB8",
        "species_d_light": "#FFE44D",
        "background":      "#0A0A12",
        "grid":            "#1E1E2E",
        "text":            "#E8E8F0",
        "mass_ssa_line":   "#00FF88",
        "mass_pde_line":   "#FFAA00",
        "combined_total":  "#FFFFFF",
    }
    mpl.rcParams["figure.figsize"] = [16, 9]
    mpl.rcParams["figure.dpi"] = 100
    mpl.rcParams["font.size"] = 12
    mpl.rcParams["font.family"] = "serif"
    mpl.rcParams["axes.titlepad"] = 20
    return colors


def _species_color(colors: Dict[str, str], i: int) -> str:
    return colors.get(f"species_{chr(97 + i)}", "white")


def _species_light_color(colors: Dict[str, str], i: int) -> str:
    return colors.get(f"species_{chr(97 + i)}_light", "white")


# ============================================================
# Helpers
# ============================================================

def _sync_particle_axis(ax_left: plt.Axes, ax_right: plt.Axes, h: float) -> None:
    y0, y1 = ax_left.get_ylim()
    ax_right.set_ylim(y0 * h, y1 * h)


def _ssa_to_conc_on_pde_grid(ssa: np.ndarray, h: float, pde_multiple: int) -> np.ndarray:
    """(n_species, K, T) -> (n_species, K*pde_multiple, T)"""
    n_species, K, T = ssa.shape
    out = np.zeros((n_species, K * pde_multiple, T), dtype=float)
    for i in range(K):
        s, e = i * pde_multiple, (i + 1) * pde_multiple
        out[:, s:e, :] = ssa[:, i:i+1, :] / h
    return out


def _total_mass(ssa: np.ndarray, pde: np.ndarray, dx: float) -> Dict[str, np.ndarray]:
    ssa_mass = np.sum(ssa, axis=1)
    pde_mass = np.sum(pde, axis=1) * dx
    combined  = ssa_mass + pde_mass
    return {
        "ssa_mass":      ssa_mass,
        "pde_mass":      pde_mass,
        "combined_mass": combined,
        "total_ssa":     np.sum(ssa_mass, axis=0),
        "total_pde":     np.sum(pde_mass, axis=0),
        "total_combined":np.sum(combined,  axis=0),
    }


def _map_frames(t_ref: np.ndarray, t_other: np.ndarray) -> np.ndarray:
    idx = np.searchsorted(t_other, t_ref)
    idx = np.clip(idx, 1, len(t_other) - 1)
    left, right = idx - 1, idx
    return np.where(np.abs(t_other[right] - t_ref) < np.abs(t_other[left] - t_ref), right, left)


def _threshold_conc_map(
    threshold_particles,
    *,
    species: Sequence[str],
    h: float,
) -> Dict[str, float]:
    """Convert threshold (scalar, dict, or list) to {species: concentration}."""
    if threshold_particles is None:
        return {}
    if isinstance(threshold_particles, dict):
        return {sp: float(threshold_particles[sp]) / h for sp in species if sp in threshold_particles}
    if isinstance(threshold_particles, (list, tuple, np.ndarray)):
        if len(threshold_particles) != len(species):
            raise ValueError("threshold_particles length must match number of species")
        return {sp: float(threshold_particles[i]) / h for i, sp in enumerate(species)}
    thr = float(threshold_particles) / h
    return {sp: thr for sp in species}


def _draw_threshold_lines(
    ax: plt.Axes,
    cfg: AnimationConfig,
    species: Sequence[str],
    colors: Dict[str, str],
    h: float,
) -> List[mpl.lines.Line2D]:
    """Draw CD (dotted) and DC (dashed) threshold lines with offset handling."""
    lines: List[mpl.lines.Line2D] = []
    
    # Helper to plot lines with a tiny offset so they don't perfectly overlap
    def _plot_thresh(thresh_map, style, label_prefix, raw_data):
        # Identify if multiple species share the exact same threshold value
        values = list(thresh_map.values())
        unique_vals, counts = np.unique(values, return_counts=True)
        shared_vals = unique_vals[counts > 1]

        for i, sp in enumerate(species):
            if sp not in thresh_map:
                continue
            
            val = thresh_map[sp]
            # If this value is shared, shift it up/down by 1% of the axis height
            offset = 0
            if val in shared_vals:
                # Species 0 goes down, Species 1 goes up slightly
                offset = (i - 0.5) * (ax.get_ylim()[1] * 0.01)

            ln = ax.axhline(
                val + offset,
                color=_species_color(colors, i),
                linestyle=style,
                linewidth=2.0,
                alpha=0.8,
                label=f"{label_prefix} {sp} ({val*h:.1f} ptcl)",
            )
            lines.append(ln)

    # 1. Handle CD (PDE -> SSA) Thresholds
    if cfg.show_threshold and cfg.threshold_particles is not None:
        cd_map = _threshold_conc_map(cfg.threshold_particles, species=species, h=h)
        _plot_thresh(cd_map, ":", "CD", cfg.threshold_particles)

    # 2. Handle DC (SSA -> PDE) Thresholds
    if cfg.show_threshold and cfg.DC_threshold_particles is not None:
        dc_map = _threshold_conc_map(cfg.DC_threshold_particles, species=species, h=h)
        _plot_thresh(dc_map, "--", "DC", cfg.DC_threshold_particles)

    return lines

def _style_ax(ax: plt.Axes, colors: Dict[str, str]) -> None:
    ax.set_facecolor(colors["background"])
    ax.grid(True, alpha=0.2, color=colors["grid"])
    ax.tick_params(colors=colors["text"])


def _legend(ax: plt.Axes, colors: Dict[str, str]) -> None:
    leg = ax.legend(loc="upper right", framealpha=0.9,
                    facecolor=colors["background"], edgecolor=colors["grid"])
    for t in leg.get_texts():
        t.set_color(colors["text"])


# ============================================================
# animate_results  (single SimulationResults)
# ============================================================

def animate_results(res: SimulationResults, cfg: Optional[AnimationConfig] = None):
    if cfg is None:
        cfg = AnimationConfig()

    if len(res.species) not in (1, 2):
        raise ValueError("animate_results supports 1 or 2 species")

    colors = _setup_style()

    time = res.time
    ssa  = res.ssa
    pde  = res.pde

    n_species, K, T = ssa.shape
    _, Npde, _      = pde.shape
    h            = float(res.domain.h)
    dx           = float(res.domain.dx)
    pde_multiple = int(res.domain.pde_multiple)
    L            = float(res.domain.length)

    ssa_x  = np.arange(K) * h / L
    pde_x  = np.linspace(0.0, L, Npde) / L
    bar_w  = h / L

    ssa_conc       = ssa.astype(float) / h
    ssa_conc_pde   = _ssa_to_conc_on_pde_grid(ssa.astype(float), h, pde_multiple)
    combined_conc  = pde + ssa_conc_pde
    masses         = _total_mass(ssa.astype(float), pde.astype(float), dx=dx)

    max_conc = float(np.max(combined_conc))
    if not np.isfinite(max_conc) or max_conc <= 0:
        max_conc = 1.0

    # --- layout ---
    fig = plt.figure(figsize=(16, 9), facecolor=colors["background"])
    if cfg.mass_plot_mode == "per_species":
        gs = gridspec.GridSpec(1 + n_species, 1, height_ratios=[3] + [1] * n_species, hspace=0.3, figure=fig)
    elif cfg.mass_plot_mode == "single":
        gs = gridspec.GridSpec(2, 1, height_ratios=[3, 1], hspace=0.25, figure=fig)
    else:
        gs = gridspec.GridSpec(1, 1, figure=fig)

    ax_main    = fig.add_subplot(gs[0])
    ax_particle = ax_main.twinx()
    _style_ax(ax_main, colors)
    for a in [ax_main, ax_particle]:
        a.tick_params(colors=colors["text"])

    # --- bars + PDE lines ---
    bars: List[mpl.container.BarContainer] = []
    lines_pde: List[mpl.lines.Line2D] = []

    for i, sp in enumerate(res.species):
        bar = ax_main.bar(
            ssa_x, ssa_conc[i, :, 0], width=bar_w, align="edge",
            color=_species_light_color(colors, i), alpha=0.7,
            edgecolor=_species_color(colors, i), linewidth=0.5,
            label=f"SSA {sp}",
        )
        bars.append(bar)
        ln, = ax_main.plot(pde_x, pde[i, :, 0],
                           color=_species_color(colors, i), linewidth=2.5, label=f"PDE {sp}")
        lines_pde.append(ln)

    threshold_lines = _draw_threshold_lines(ax_main, cfg, res.species, colors, h)

    ax_main.set_xlim(0, 1)
    ax_main.set_ylim(0, max_conc * 1.15)
    ax_main.set_ylabel("Concentration (particles / length)", color=colors["text"], fontsize=12)
    ax_particle.set_ylabel("Particles per SSA compartment",  color=colors["text"], fontsize=12)
    ax_main.set_title(cfg.title, color=colors["text"], fontweight="bold")
    _sync_particle_axis(ax_main, ax_particle, h)
    _legend(ax_main, colors)

    time_text = ax_main.text(
        0.02, 0.95, "", transform=ax_main.transAxes,
        color=colors["text"], fontweight="bold",
        bbox=dict(boxstyle="round", facecolor=colors["background"], edgecolor=colors["grid"]),
    )

    # --- optional mass panel ---
    def _style_mass(ax):
        _style_ax(ax, colors)
        ax.tick_params(colors=colors["text"])

    if cfg.mass_plot_mode == "single":
        ax_mass = fig.add_subplot(gs[1])
        _style_mass(ax_mass)
        ax_mass.set_xlabel("Time", color=colors["text"])
        ax_mass.set_ylabel("Mass (particles)", color=colors["text"])
        ax_mass.set_title("Mass evolution", color=colors["text"], fontweight="bold")
        ax_mass.plot(time, masses["total_ssa"],      color=colors["mass_ssa_line"], linestyle="--", linewidth=2, alpha=0.9, label="Total SSA")
        ax_mass.plot(time, masses["total_pde"],      color=colors["mass_pde_line"], linestyle="-.", linewidth=2, alpha=0.9, label="Total PDE")
        ax_mass.plot(time, masses["total_combined"], color=colors["combined_total"], linewidth=3,  alpha=0.9, label="Total Combined")
        leg2 = ax_mass.legend(loc="best", framealpha=0.9, facecolor=colors["background"], edgecolor=colors["grid"])
        for t in leg2.get_texts():
            t.set_color(colors["text"])

    elif cfg.mass_plot_mode == "per_species":
        for i, sp in enumerate(res.species):
            ax_m = fig.add_subplot(gs[1 + i])
            _style_mass(ax_m)
            ax_m.set_ylabel(f"{sp}\nMass", color=colors["text"])
            if i == n_species - 1:
                ax_m.set_xlabel("Time", color=colors["text"])
            ax_m.plot(time, masses["ssa_mass"][i],      color=colors["mass_ssa_line"], linestyle="--", linewidth=2, alpha=0.9, label="SSA")
            ax_m.plot(time, masses["pde_mass"][i],      color=colors["mass_pde_line"], linestyle="-.", linewidth=2, alpha=0.9, label="PDE")
            ax_m.plot(time, masses["combined_mass"][i], color=_species_color(colors, i), linewidth=2.5, alpha=0.9, label="Combined")
            leg2 = ax_m.legend(loc="best", framealpha=0.9, facecolor=colors["background"], edgecolor=colors["grid"])
            for t in leg2.get_texts():
                t.set_color(colors["text"])

    # --- animation ---
    frames = list(range(0, T, max(1, int(cfg.stride))))

    def update(frame_idx: int):
        for i, bar in enumerate(bars):
            for b, hgt in zip(bar, ssa_conc[i, :, frame_idx]):
                b.set_height(float(hgt))
        for i, ln in enumerate(lines_pde):
            ln.set_ydata(pde[i, :, frame_idx])
        _sync_particle_axis(ax_main, ax_particle, h)
        time_text.set_text(f"t = {time[frame_idx]:.4g}   frame={frame_idx}")
        artists: List = []
        for bar in bars:
            artists += list(bar)
        artists += lines_pde + [time_text] + threshold_lines
        return artists

    ani = FuncAnimation(fig, update, frames=frames, interval=int(cfg.interval_ms), blit=bool(cfg.blit))
    plt.tight_layout()
    return ani


# ============================================================
# animate_overlay  (Hybrid + SSA overlay)
# ============================================================

def animate_overlay(
    res_main: SimulationResults,
    res_overlay: SimulationResults,
    *,
    cfg: Optional[AnimationConfig] = None,
    label_main: str = "Hybrid",
    label_overlay: str = "SSA",
):
    if cfg is None:
        cfg = AnimationConfig()

    if len(res_main.species) not in (1, 2):
        raise ValueError("animate_overlay supports 1 or 2 species")
    if list(res_main.species) != list(res_overlay.species):
        raise ValueError(f"Species mismatch: {res_main.species} vs {res_overlay.species}")
    if res_main.domain.length != res_overlay.domain.length or res_main.domain.n_ssa != res_overlay.domain.n_ssa:
        raise ValueError("Domain mismatch between main and overlay")

    colors = _setup_style()

    time  = res_main.time
    ssa   = res_main.ssa
    pde   = res_main.pde
    time2 = res_overlay.time
    ssa2  = res_overlay.ssa

    n_species, K, T = ssa.shape
    h            = float(res_main.domain.h)
    pde_multiple = int(res_main.domain.pde_multiple)
    L            = float(res_main.domain.length)
    Npde         = K * pde_multiple

    tmap    = _map_frames(time, time2)
    ssa_x   = np.arange(K) * h / L
    pde_x   = np.linspace(0.0, L, Npde) / L
    bar_w   = h / L

    ssa_conc  = ssa.astype(float) / h
    ssa2_conc = ssa2.astype(float) / h
    ssa_on_pde = _ssa_to_conc_on_pde_grid(ssa.astype(float), h, pde_multiple)
    combined   = pde + ssa_on_pde

    max_conc = float(np.max([combined.max(), ssa2_conc.max()]))
    if not np.isfinite(max_conc) or max_conc <= 0:
        max_conc = 1.0

    fig = plt.figure(figsize=(16, 9), facecolor=colors["background"])
    ax       = fig.add_subplot(111)
    ax_right = ax.twinx()
    _style_ax(ax, colors)
    for a in [ax, ax_right]:
        a.tick_params(colors=colors["text"])

    bar_main: List[mpl.container.BarContainer] = []
    bar_ov:   List[mpl.container.BarContainer] = []
    line_main: List[mpl.lines.Line2D] = []

    for i, sp in enumerate(res_main.species):
        bm = ax.bar(
            ssa_x, ssa_conc[i, :, 0], width=bar_w, align="edge",
            color=_species_light_color(colors, i), alpha=0.4,
            edgecolor=_species_color(colors, i), linewidth=0.5,
            label=f"{label_main} SSA {sp}",
        )
        bar_main.append(bm)

        bo = ax.bar(
            ssa_x, ssa2_conc[i, :, int(tmap[0])], width=bar_w, align="edge",
            color="none", edgecolor=_species_color(colors, i),
            linewidth=1.5, linestyle="--", alpha=0.8,
            label=f"{label_overlay} SSA {sp}",
        )
        bar_ov.append(bo)

        ln, = ax.plot(pde_x, combined[i, :, 0],
                      color=_species_color(colors, i), linewidth=3.0,
                      label=f"{label_main} combined {sp}")
        line_main.append(ln)

    threshold_lines = _draw_threshold_lines(ax, cfg, res_main.species, colors, h)

    ax.set_xlim(0, 1)
    ax.set_ylim(0, max_conc * 1.15)
    ax.set_xlabel("Scaled Domain [0,1]", color=colors["text"])
    ax.set_ylabel("Concentration", color=colors["text"])
    ax_right.set_ylabel("Particles per SSA box", color=colors["text"])
    ax.set_title(cfg.title, color=colors["text"], fontweight="bold")
    _sync_particle_axis(ax, ax_right, h)
    _legend(ax, colors)

    time_text = ax.text(
        0.02, 0.95, "", transform=ax.transAxes,
        color=colors["text"], fontweight="bold",
        bbox=dict(boxstyle="round", facecolor=colors["background"], edgecolor=colors["grid"]),
    )

    frames = list(range(0, T, max(1, int(cfg.stride))))

    def update(frame_idx: int):
        for i, bm in enumerate(bar_main):
            for b, hgt in zip(bm, ssa_conc[i, :, frame_idx]):
                b.set_height(float(hgt))
        j = int(tmap[frame_idx])
        for i, bo in enumerate(bar_ov):
            for b, hgt in zip(bo, ssa2_conc[i, :, j]):
                b.set_height(float(hgt))
        for i, ln in enumerate(line_main):
            ln.set_ydata(combined[i, :, frame_idx])
        _sync_particle_axis(ax, ax_right, h)
        time_text.set_text(f"t = {time[frame_idx]:.4g}   main={frame_idx}  overlay={j}")
        artists: List = []
        for bm in bar_main:
            artists += list(bm)
        for bo in bar_ov:
            artists += list(bo)
        artists += line_main + [time_text] + threshold_lines
        return artists

    ani = FuncAnimation(fig, update, frames=frames, interval=int(cfg.interval_ms), blit=bool(cfg.blit))
    plt.tight_layout()
    return ani


# ============================================================
# CLI helpers
# ============================================================

def _print_summary(res: SimulationResults, meta: dict, title: str) -> None:
    print("\n" + "=" * 50)
    print(f" LOADING: {title}")
    print("=" * 50)
    print(f" Species:    {', '.join(res.species)}")
    print(f" Time Steps: {res.time.shape[0]}  (t={res.time[0]:.2f} → {res.time[-1]:.2f})")
    print(f" Domain:     L={res.domain.length}, K={res.domain.K}")
    print(f" PDE Grid:   Npde={res.domain.n_pde} (×{res.domain.pde_multiple})")
    if meta:
        print("-" * 50)
        conversion = meta.get("conversion", {})
        if isinstance(conversion, dict) and conversion:
            print("  conversion:")
            for key in ("DC_threshold", "CD_threshold", "rate"):
                if key in conversion:
                    print(f"    {key}: {conversion[key]}")
        for k, v in meta.items():
            if k != "conversion" and not isinstance(v, (dict, list)):
                print(f"  {k}: {v}")
    print("=" * 50 + "\n")


def _print_npz_keys(path: Path) -> None:
    print(f"\n[DEBUG] Keys in {path.name}:")
    d = np.load(path, allow_pickle=False)
    for k in d.files:
        try:
            arr = np.asarray(d[k])
            print(f"  {k:20s} shape={arr.shape} dtype={arr.dtype}")
        except ValueError:
            print(f"  {k:20s} <object array>")


def _load_overlay(npz_path: Path, ref: SimulationResults) -> SimulationResults:
    try:
        res, _ = load_npz(str(npz_path))
        return res
    except Exception:
        pass
    d = np.load(npz_path, allow_pickle=True)
    if "time" not in d.files or "ssa" not in d.files:
        raise KeyError(f"Overlay file must contain 'time' and 'ssa'. Found: {d.files}")
    time = np.asarray(d["time"])
    ssa  = np.asarray(d["ssa"])
    if ssa.ndim != 3:
        raise ValueError(f"Expected ssa shape (n_species, K, T), got {ssa.shape}")
    n_species, _K, T = ssa.shape
    pde = np.zeros((n_species, ref.pde.shape[1], T), dtype=float)
    return SimulationResults(time=time, ssa=ssa, pde=pde, domain=ref.domain, species=list(ref.species))


def _extract_thresholds(meta: dict, manual: float | None) -> tuple[float | None, float | None]:
    """Returns (CD_threshold, DC_threshold) as scalars for plotting."""
    if manual is not None:
        return float(manual), float(manual)
    if not isinstance(meta, dict):
        return None, None

    # backward compat
    old = meta.get("threshold_particles")
    if old is not None:
        v = float(old)
        return v, v

    conversion = meta.get("conversion", {})
    if not isinstance(conversion, dict):
        return None, None

    def _first(val) -> float | None:
        if val is None:
            return None
        if isinstance(val, (int, float)):
            return float(val)
        if isinstance(val, dict):
            return float(next(iter(val.values())))
        arr = np.asarray(val)
        return float(arr.flat[0]) if arr.size > 0 else None

    return _first(conversion.get("CD_threshold")), _first(conversion.get("DC_threshold"))


# ============================================================
# CLI entry point
# ============================================================

def main():
    ap = argparse.ArgumentParser(description="Animate SRCM .npz results.")
    ap.add_argument("main_npz",      type=str,            help="Path to main .npz (Hybrid)")
    ap.add_argument("overlay_npz",   nargs="?", default=None, help="Optional overlay .npz (SSA)")
    ap.add_argument("--stride",      type=int,   default=20)
    ap.add_argument("--interval",    type=int,   default=30,  help="Frame interval ms")
    ap.add_argument("--title",       type=str,   default=None)
    ap.add_argument("--mass",        type=str,   default="none", choices=["none", "single", "per_species"])
    ap.add_argument("--threshold",   type=float, default=None,   help="Manual threshold override")
    ap.add_argument("--debug-keys",  action="store_true")
    args = ap.parse_args()

    main_path = Path(args.main_npz)
    if not main_path.exists():
        raise FileNotFoundError(main_path)

    overlay_path = Path(args.overlay_npz) if args.overlay_npz else None
    if overlay_path and not overlay_path.exists():
        raise FileNotFoundError(overlay_path)

    if args.debug_keys:
        _print_npz_keys(main_path)
        if overlay_path:
            _print_npz_keys(overlay_path)

    plt.close("all")

    res_main, meta = load_npz(str(main_path))
    cd_thresh, dc_thresh = _extract_thresholds(meta, args.threshold)

    _print_summary(res_main, meta, main_path.name)
    print(f"→ CD threshold (PDE→SSA): {cd_thresh}")
    print(f"→ DC threshold (SSA→PDE): {dc_thresh}")

    cfg = AnimationConfig(
        stride=int(args.stride),
        interval_ms=int(args.interval),
        title=args.title or (
            main_path.name if overlay_path is None
            else f"Overlay: {main_path.name} vs {overlay_path.name}"
        ),
        mass_plot_mode=args.mass,
        threshold_particles=cd_thresh,
        DC_threshold_particles=dc_thresh,
        show_threshold=(cd_thresh is not None or dc_thresh is not None),
    )

    if overlay_path is None:
        ani = animate_results(res_main, cfg=cfg)
    else:
        res_overlay = _load_overlay(overlay_path, ref=res_main)
        print(f"→ Overlaying from: {overlay_path.name}")
        ani = animate_overlay(res_main, res_overlay, cfg=cfg, label_main="Hybrid", label_overlay="SSA")

    plt.show()


if __name__ == "__main__":
    main()

    