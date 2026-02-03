import sys
import os
import json
import numpy as np
from pathlib import Path

# -----------------------------
# Pretty terminal helpers (no deps)
# -----------------------------
def _supports_color() -> bool:
    return sys.stdout.isatty() and (sys.platform != "win32" or "WT_SESSION" in os.environ or "TERM" in os.environ)

_COLOR = False
try:
    _COLOR = _supports_color()
except Exception:
    _COLOR = False

RESET = "\033[0m" if _COLOR else ""
BOLD  = "\033[1m"  if _COLOR else ""
DIM   = "\033[2m"  if _COLOR else ""

FG = {
    "cyan":   "\033[36m" if _COLOR else "",
    "green":  "\033[32m" if _COLOR else "",
    "yellow": "\033[33m" if _COLOR else "",
    "red":    "\033[31m" if _COLOR else "",
    "blue":   "\033[34m" if _COLOR else "",
    "mag":    "\033[35m" if _COLOR else "",
    "gray":   "\033[90m" if _COLOR else "",
}

def c(text: str, color: str = None, *, bold=False, dim=False) -> str:
    if not _COLOR:
        return text
    prefix = ""
    if bold: prefix += BOLD
    if dim:  prefix += DIM
    if color: prefix += FG.get(color, "")
    return f"{prefix}{text}{RESET}"

def badge(label: str, color: str) -> str:
    return c(f"[{label}]", color=color, bold=True)

def hr(width: int = 78, char: str = "─") -> str:
    return char * width

def box_title(title: str, width: int = 78) -> str:
    title = f" {title} "
    inner = width - 2
    if len(title) > inner:
        title = title[: inner - 1] + "…"
    pad_left = (inner - len(title)) // 2
    pad_right = inner - len(title) - pad_left
    return f"┌{('─' * pad_left)}{title}{('─' * pad_right)}┐"

def box_line(text: str, width: int = 78) -> str:
    inner = width - 2
    if len(text) > inner:
        text = text[: inner - 1] + "…"
    return f"│{text.ljust(inner)}│"

def box_bottom(width: int = 78) -> str:
    return f"└{('─' * (width - 2))}┘"

def print_table(rows, headers, col_widths, *, indent=0):
    pad = " " * indent
    header_line = " │ ".join(h.ljust(w) for h, w in zip(headers, col_widths))
    sep = "─" * len(header_line)
    print(pad + c(header_line, "gray", bold=True))
    print(pad + c(sep, "gray", dim=True))
    for r in rows:
        line = " │ ".join(str(cell).ljust(w)[:w] for cell, w in zip(r, col_widths))
        print(pad + line)

def try_decode_meta(meta_raw):
    try:
        if hasattr(meta_raw, "shape") and meta_raw.shape == ():
            meta_raw = meta_raw.item()
        if isinstance(meta_raw, (bytes, bytearray)):
            meta_raw = meta_raw.decode("utf-8", errors="replace")
        decoded = json.loads(str(meta_raw))
        return decoded, None
    except Exception as e:
        return None, e

def _get(data, key):
    return data[key] if key in data.files else None

def _shape(arr):
    if arr is None:
        return "—"
    try:
        return str(tuple(int(x) for x in arr.shape))
    except Exception:
        return str(getattr(arr, "shape", "—"))

def _as_species_list(species_arr):
    if species_arr is None:
        return None
    try:
        # common: np.array(dtype=object)
        return [str(x) for x in species_arr.tolist()]
    except Exception:
        return None

def inspect_traj_npz(path: Path, *, width: int = 88, repeat_peek: int | None = None):
    print()
    print(box_title(c("TRAJECTORY NPZ INSPECTOR", "cyan", bold=True), width))
    print(box_line(f"{badge('FILE', 'blue')} {c(path.name, bold=True)}", width))
    print(box_line(f"{DIM}{str(path.resolve())}{RESET}", width))
    print(box_bottom(width))

    try:
        data = np.load(path, allow_pickle=True)
    except Exception as e:
        print(c(f"{badge('ERROR', 'red')} Could not read file: {e}", "red", bold=True))
        return

    # ---------- Key arrays ----------
    time = _get(data, "time")
    ssa = _get(data, "ssa")
    pde = _get(data, "pde")
    species = _as_species_list(_get(data, "species"))

    # ---------- Overview table ----------
    print()
    print(c("▌ FILE CONTENTS", "mag", bold=True))
    rows = []
    for k in data.files:
        v = data[k]
        if getattr(v, "shape", None) == ():
            rows.append((k, "scalar", f"dtype={getattr(v,'dtype','?')}", str(v)))
        else:
            peek = ""
            if k == "species" and species is not None:
                peek = str(species)
            rows.append((k, "array", f"shape={_shape(v)}, dtype={getattr(v,'dtype','?')}", peek))

    print_table(rows, headers=("KEY", "KIND", "DETAILS", "PEEK"), col_widths=(18, 8, 42, 18))

    # ---------- Validate trajectory format ----------
    print()
    print(c("▌ TRAJECTORY SHAPES", "mag", bold=True))

    if ssa is None:
        print(c(f"{badge('ERROR','red')} No 'ssa' array found. This inspector expects trajectory files.", "red", bold=True))
        print(c("Expected keys: time, ssa, (optional) pde, (optional) meta_json, species", "yellow"))
        return

    if ssa.ndim != 4:
        print(c(f"{badge('ERROR','red')} Expected ssa to be 4D (R,S,K,T). Got shape {ssa.shape}", "red", bold=True))
        print(c("Tip: mean files are usually (S,K,T). Use your existing inspector for those.", "yellow"))
        return

    R, S, K, T = map(int, ssa.shape)
    print(f"{badge('SSA','blue')} ssa = {c('(R,S,K,T)', bold=True)} = {ssa.shape}")
    if time is not None:
        print(f"{badge('TIME','blue')} time = {time.shape}  {DIM}({float(time[0]):.3f} → {float(time[-1]):.3f}){RESET}")
        if int(time.shape[0]) != T:
            print(c(f"{badge('WARN','yellow')} time length {len(time)} != SSA T {T}", "yellow"))
    else:
        print(c(f"{badge('WARN','yellow')} No time array found.", "yellow"))

    if pde is None:
        print(f"{badge('PDE','gray')} pde = {DIM}missing{RESET}  (SSA-only trajectories?)")
        Npde = None
    else:
        if pde.ndim != 4:
            print(c(f"{badge('WARN','yellow')} Expected pde to be 4D (R,S,Npde,T). Got {pde.shape}", "yellow"))
            Npde = None
        else:
            Rp, Sp, Npde, Tp = map(int, pde.shape)
            print(f"{badge('PDE','blue')} pde = {c('(R,S,Npde,T)', bold=True)} = {pde.shape}")
            if (Rp, Sp, Tp) != (R, S, T):
                print(c(f"{badge('WARN','yellow')} PDE dims (R,S,T) = {(Rp,Sp,Tp)} don't match SSA {(R,S,T)}", "yellow"))

    if species is not None:
        print(f"{badge('SPEC','green')} {species}")
        if len(species) != S:
            print(c(f"{badge('WARN','yellow')} species list length {len(species)} != SSA species dim {S}", "yellow"))

    # ---------- Metadata ----------
    meta = {}
    if "meta_json" in data.files:
        decoded, err = try_decode_meta(data["meta_json"])
        if decoded is None:
            print()
            print(c("▌ METADATA", "mag", bold=True))
            print(c(f"{badge('WARN','yellow')} Could not decode meta_json: {err}", "yellow"))
        else:
            meta = decoded
            print()
            print(c("▌ METADATA (selected)", "mag", bold=True))
            # keep this compact but useful
            key_order = [
                "run_type", "is_ensemble", "n_repeats",
                "threshold_particles", "conversion_rate",
                "total_time", "dt", "seed", "base_seed",
                "boundary",
            ]
            rows = []
            for k in key_order:
                if k in meta:
                    rows.append((k, meta[k]))
            if "domain" in meta:
                rows.append(("domain", meta["domain"]))
            print_table(rows, headers=("KEY", "VALUE"), col_widths=(22, 60))

    # ---------- Quick stats across repeats ----------
    print()
    print(c("▌ QUICK STATS (across repeats)", "mag", bold=True))

    # Use final time slice, sum over space for each species: totals shape (R,S)
    final_totals = ssa[:, :, :, -1].sum(axis=2)

    mean_tot = final_totals.mean(axis=0)
    std_tot = final_totals.std(axis=0)

    stat_rows = []
    labels = species if species is not None else [f"S{i}" for i in range(S)]
    for i, lab in enumerate(labels):
        stat_rows.append((lab, f"{mean_tot[i]:.3f}", f"{std_tot[i]:.3f}", f"{final_totals[:, i].min():.0f}", f"{final_totals[:, i].max():.0f}"))

    print_table(
        stat_rows,
        headers=("SPECIES", "mean(total@final)", "std", "min", "max"),
        col_widths=(10, 18, 10, 10, 10),
    )

    # ---------- Optional peek at one repeat ----------
    if repeat_peek is not None:
        r = int(repeat_peek)
        if r < 0 or r >= R:
            print(c(f"{badge('WARN','yellow')} repeat_peek {r} out of range [0,{R-1}]", "yellow"))
        else:
            print()
            print(c(f"▌ REPEAT PEEK r={r}", "mag", bold=True))
            # show final spatial profile summary per species: min/max/mean across space
            rep_final = ssa[r, :, :, -1]  # (S,K)
            rows = []
            for i, lab in enumerate(labels):
                x = rep_final[i]
                rows.append((lab, f"{x.min():.0f}", f"{x.max():.0f}", f"{x.mean():.2f}"))
            print_table(rows, headers=("SPECIES", "min(K)", "max(K)", "mean(K)"), col_widths=(10, 10, 10, 10))

    print()
    print(c(hr(width), "gray", dim=True))


def main():
    if len(sys.argv) < 2:
        print("Usage:")
        print("  python inspect_traj.py <file1.npz> [file2.npz ...] [--peek 0]")
        return

    args = sys.argv[1:]
    peek = None
    if "--peek" in args:
        i = args.index("--peek")
        if i + 1 >= len(args):
            print("Error: --peek requires an integer argument")
            return
        peek = int(args[i + 1])
        # remove peek args
        args = args[:i] + args[i+2:]

    paths = [Path(p) for p in args if Path(p).exists()]
    if not paths:
        print("No valid files provided.")
        return

    for p in paths:
        inspect_traj_npz(p, repeat_peek=peek)


if __name__ == "__main__":
    main()
