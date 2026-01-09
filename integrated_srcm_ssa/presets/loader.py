
# integrated_srcm_ssa/presets/loader.py
"""
Preset loader for integrated-srcm-ssa.

What this does:
- Load a preset from YAML/JSON (either packaged presets or a user-provided file path)
- Validate the basic schema
- Build and configure an SRCMRunner (rates, diffusion, conversion, reactions, PDE model)
- Generate initial conditions from simple recipes
- Return (sim, run_cfg) where run_cfg contains kwargs for run_ssa/run_hybrid

Design choices:
- Presets are data-only (YAML/JSON). No arbitrary Python execution.
- PDE "models" are referenced by name and implemented in code (safe + stable).
  You can extend PDE_MODELS with more functions.

Typical usage:

    from integrated_srcm_ssa.presets.loader import load_system

    sim, cfg = load_system("conv_decay_make2A")  # packaged preset name
    res_ssa, meta_ssa = sim.run_ssa(**cfg["ssa"])
    res_h, meta_h = sim.run_hybrid(**cfg["hybrid"])

Or load a file:

    sim, cfg = load_system("/path/to/preset.yaml")

Package layout suggestion:
- integrated_srcm_ssa/presets/data/*.yaml  (bundled preset files)
- integrated_srcm_ssa/presets/loader.py    (this file)
- integrated_srcm_ssa/presets/__init__.py  (re-export helpers)

Note: If you use Poetry, ensure YAML files are included in the wheel/sdist.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, MutableMapping, Optional, Tuple, Union

import json
import importlib.resources as pkg_resources

import numpy as np

try:
    import yaml  # type: ignore
except Exception as e:  # pragma: no cover
    yaml = None  # type: ignore

from integrated_srcm_ssa import SRCMRunner


# -----------------------------
# Errors
# -----------------------------
class PresetError(ValueError):
    """Raised when a preset is malformed or cannot be loaded."""


# -----------------------------
# PDE model registry (safe)
# -----------------------------
PDEFunc = Callable[[np.ndarray, np.ndarray, Mapping[str, float]], Tuple[np.ndarray, np.ndarray]]


def _pde_mass_action_from_terms(
    dA_expr: str, dB_expr: str
) -> PDEFunc:
    """
    Create a PDE drift callable from simple arithmetic expressions in A, B, and r[name].

    Safe-ish evaluator:
    - We do NOT execute arbitrary code.
    - We allow only numpy arrays A, B and scalar rates via mapping r.
    - Implementation uses Python eval with a restricted namespace.
      If you want maximum safety, replace this with an AST-based evaluator.
    """
    # Restricted namespace: no builtins, no access to globals
    allowed_builtins: Dict[str, Any] = {}
    # Expose only A, B and rate names
    def fn(A: np.ndarray, B: np.ndarray, r: Mapping[str, float]) -> Tuple[np.ndarray, np.ndarray]:
        local_vars: Dict[str, Any] = {"A": A, "B": B}
        # Add rates into local vars (alpha, beta, etc.)
        local_vars.update(dict(r))
        try:
            dA = eval(dA_expr, {"__builtins__": allowed_builtins}, local_vars)
            dB = eval(dB_expr, {"__builtins__": allowed_builtins}, local_vars)
        except Exception as e:
            raise PresetError(f"Failed evaluating PDE expressions: {e}") from e
        return dA, dB

    return fn


def pde_model_two_species_mass_action_from_preset(preset: Mapping[str, Any]) -> PDEFunc:
    """
    Default PDE model:
    expects
      preset["pde_model"]["terms"]["dA"], preset["pde_model"]["terms"]["dB"]
    """
    pde = preset.get("pde_model") or {}
    terms = (pde.get("terms") or {})
    dA_expr = terms.get("dA")
    dB_expr = terms.get("dB")
    if not isinstance(dA_expr, str) or not isinstance(dB_expr, str):
        raise PresetError(
            "pde_model.name=mass_action_two_species requires pde_model.terms.dA and pde_model.terms.dB strings."
        )
    return _pde_mass_action_from_terms(dA_expr, dB_expr)


# Registry: map model name -> builder
PDE_MODELS: Dict[str, Callable[[Mapping[str, Any]], PDEFunc]] = {
    "mass_action_two_species": pde_model_two_species_mass_action_from_preset,
}


# -----------------------------
# Initial condition generators
# -----------------------------
def _init_two_patches(
    species: List[str],
    K: int,
    patches: List[Mapping[str, Any]],
) -> Dict[str, np.ndarray]:
    init: Dict[str, np.ndarray] = {s: np.zeros(K, dtype=int) for s in species}

    for p in patches:
        sp = p.get("species")
        if sp not in init:
            raise PresetError(f"Initial condition patch species '{sp}' not in species list {species}.")
        start = int(p.get("start"))
        end = int(p.get("end"))
        value = int(p.get("value"))
        if not (0 <= start <= end <= K):
            raise PresetError(f"Patch range [{start},{end}) is out of bounds for K={K}.")
        init[sp][start:end] = value

    return init


def _init_uniform(
    species: List[str],
    K: int,
    values: Mapping[str, Any],
) -> Dict[str, np.ndarray]:
    init: Dict[str, np.ndarray] = {}
    for s in species:
        v = int(values.get(s, 0))
        init[s] = np.full(K, v, dtype=int)
    return init


def _init_from_explicit_arrays(
    species: List[str],
    K: int,
    arrays: Mapping[str, Any],
) -> Dict[str, np.ndarray]:
    init: Dict[str, np.ndarray] = {}
    for s in species:
        arr = arrays.get(s)
        if arr is None:
            raise PresetError(f"Missing initial_conditions.arrays entry for species '{s}'.")
        a = np.asarray(arr, dtype=int)
        if a.shape != (K,):
            raise PresetError(f"Initial array for '{s}' must have shape (K,) = ({K},), got {a.shape}.")
        init[s] = a
    return init


def build_initial_conditions(preset: Mapping[str, Any]) -> Dict[str, np.ndarray]:
    species = _require_list_str(preset, "species")
    domain = _require_dict(preset, "domain")
    K = int(domain.get("K"))
    ic = _require_dict(preset, "initial_conditions")
    ic_type = ic.get("type")

    if ic_type == "two_patches":
        patches = ic.get("patches")
        if not isinstance(patches, list):
            raise PresetError("initial_conditions.type=two_patches requires initial_conditions.patches list.")
        return _init_two_patches(species, K, patches)

    if ic_type == "uniform":
        values = ic.get("values") or {}
        if not isinstance(values, dict):
            raise PresetError("initial_conditions.type=uniform requires initial_conditions.values dict.")
        return _init_uniform(species, K, values)

    if ic_type == "explicit":
        arrays = ic.get("arrays")
        if not isinstance(arrays, dict):
            raise PresetError("initial_conditions.type=explicit requires initial_conditions.arrays dict of arrays.")
        return _init_from_explicit_arrays(species, K, arrays)

    raise PresetError(
        f"Unknown initial_conditions.type='{ic_type}'. Supported: two_patches, uniform, explicit."
    )


# -----------------------------
# Public API
# -----------------------------
def list_packaged_presets() -> List[str]:
    """
    List preset base names shipped in integrated_srcm_ssa.presets.data.
    Returns names without extension, e.g. ['conv_decay_make2A', '...'].
    """
    names: List[str] = []
    try:
        data_pkg = pkg_resources.files("integrated_srcm_ssa.presets.data")
    except Exception:
        return names

    for p in data_pkg.iterdir():
        if p.is_file() and p.name.lower().endswith((".yaml", ".yml", ".json")):
            names.append(p.stem)
    return sorted(set(names))


def load_preset(name_or_path: str) -> Dict[str, Any]:
    """
    Load a preset by packaged name (e.g. "conv_decay_make2A") OR by file path.
    """
    path = Path(name_or_path)
    if path.exists():
        return _load_from_file(path)

    # Packaged preset name
    for ext in (".yaml", ".yml", ".json"):
        preset = _load_from_package(name_or_path + ext)
        if preset is not None:
            return preset

    available = list_packaged_presets()
    raise PresetError(
        f"Preset '{name_or_path}' not found as a file path or packaged preset. "
        f"Available packaged presets: {available}"
    )


def load_system(name_or_path: str) -> Tuple[SRCMRunner, Dict[str, Dict[str, Any]]]:
    """
    Load preset and build a configured SRCMRunner and run configs.

    Returns:
      sim: SRCMRunner
      cfg: dict with keys 'ssa' and 'hybrid' containing kwargs for run_ssa/run_hybrid
    """
    preset = load_preset(name_or_path)
    validate_preset(preset)

    species = _require_list_str(preset, "species")

    sim = SRCMRunner(species=species)

    # Rates
    rates = _require_dict(preset, "rates")
    sim.define_rates(**rates)

    # Diffusion
    diffusion = _require_dict(preset, "diffusion")
    sim.define_diffusion(**diffusion)

    # Conversion (optional)
    conversion = preset.get("conversion")
    if conversion is not None:
        if not isinstance(conversion, dict):
            raise PresetError("conversion must be a dict if provided.")
        sim.define_conversion(
            threshold=int(conversion.get("threshold")),
            rate=float(conversion.get("rate")),
        )

    # Reactions
    reactions = preset.get("reactions")
    if not isinstance(reactions, list) or len(reactions) == 0:
        raise PresetError("reactions must be a non-empty list.")
    for rxn in reactions:
        if not isinstance(rxn, dict):
            raise PresetError("Each reaction must be a dict.")
        reactants = rxn.get("reactants") or {}
        products = rxn.get("products") or {}
        rate = rxn.get("rate")
        if not isinstance(reactants, dict) or not isinstance(products, dict) or not isinstance(rate, str):
            raise PresetError("Reaction must have reactants dict, products dict, and rate string.")
        sim.add_reaction(reactants, products, rate)

    # PDE model
    pde = preset.get("pde_model")
    if pde is not None:
        if not isinstance(pde, dict):
            raise PresetError("pde_model must be a dict if provided.")
        model_name = pde.get("name")
        if not isinstance(model_name, str):
            raise PresetError("pde_model.name must be a string.")
        builder = PDE_MODELS.get(model_name)
        if builder is None:
            raise PresetError(
                f"Unknown pde_model.name='{model_name}'. Available: {sorted(PDE_MODELS.keys())}"
            )
        sim.set_pde_reactions(builder(preset))

    # Domain + time
    domain = _require_dict(preset, "domain")
    time = _require_dict(preset, "time")

    L = float(domain.get("L"))
    K = int(domain.get("K"))
    total_time = float(time.get("total_time"))
    dt = float(time.get("dt"))
    repeats = int(time.get("repeats", 1))

    init_counts = build_initial_conditions(preset)

    # SSA config
    ssa_cfg: Dict[str, Any] = {
        "L": L,
        "K": K,
        "total_time": total_time,
        "dt": dt,
        "init_counts": init_counts,
        "n_repeats": repeats,
    }

    # Hybrid config
    hybrid: Dict[str, Any] = dict(time.get("hybrid", {})) if isinstance(time.get("hybrid"), dict) else {}
    # Hybrid often needs pde_multiple; allow preset override or default
    pde_multiple = int(hybrid.get("pde_multiple", preset.get("pde_multiple", 8)))
    hybrid_cfg: Dict[str, Any] = {
        "L": L,
        "K": K,
        "pde_multiple": pde_multiple,
        "total_time": total_time,
        "dt": dt,
        "init_counts": init_counts,
        "repeats": repeats,
    }

    return sim, {"ssa": ssa_cfg, "hybrid": hybrid_cfg}


# -----------------------------
# Validation helpers
# -----------------------------
def validate_preset(preset: Mapping[str, Any]) -> None:
    """
    Minimal schema validation (keeps error messages friendly).
    """
    _require_list_str(preset, "species")
    _require_dict(preset, "rates")
    _require_dict(preset, "diffusion")
    _require_dict(preset, "domain")
    _require_dict(preset, "time")
    _require_dict(preset, "initial_conditions")

    domain = preset["domain"]
    if "L" not in domain or "K" not in domain:
        raise PresetError("domain must include L and K.")

    time = preset["time"]
    if "total_time" not in time or "dt" not in time:
        raise PresetError("time must include total_time and dt.")

    reactions = preset.get("reactions")
    if not isinstance(reactions, list) or len(reactions) == 0:
        raise PresetError("reactions must be a non-empty list.")

    # Optional sections
    if preset.get("pde_model") is not None and not isinstance(preset["pde_model"], dict):
        raise PresetError("pde_model must be a dict if provided.")
    if preset.get("conversion") is not None and not isinstance(preset["conversion"], dict):
        raise PresetError("conversion must be a dict if provided.")


def _require_dict(preset: Mapping[str, Any], key: str) -> Dict[str, Any]:
    val = preset.get(key)
    if not isinstance(val, dict):
        raise PresetError(f"'{key}' must be a dict.")
    return dict(val)


def _require_list_str(preset: Mapping[str, Any], key: str) -> List[str]:
    val = preset.get(key)
    if not isinstance(val, list) or not all(isinstance(x, str) for x in val):
        raise PresetError(f"'{key}' must be a list of strings.")
    return list(val)


# -----------------------------
# I/O
# -----------------------------
def _load_from_file(path: Path) -> Dict[str, Any]:
    ext = path.suffix.lower()
    text = path.read_text(encoding="utf-8")
    return _parse_text(ext, text, source=str(path))


def _load_from_package(filename: str) -> Optional[Dict[str, Any]]:
    """
    Try to load preset file from integrated_srcm_ssa.presets.data/<filename>.
    Returns None if not found.
    """
    try:
        data_pkg = pkg_resources.files("integrated_srcm_ssa.presets.data")
        f = data_pkg / filename
        if not f.exists():
            return None
        text = f.read_text(encoding="utf-8")
        return _parse_text(Path(filename).suffix.lower(), text, source=f"packaged:{filename}")
    except ModuleNotFoundError:
        return None
    except FileNotFoundError:
        return None


def _parse_text(ext: str, text: str, source: str) -> Dict[str, Any]:
    if ext in (".json",):
        try:
            obj = json.loads(text)
        except Exception as e:
            raise PresetError(f"Failed parsing JSON preset ({source}): {e}") from e
        if not isinstance(obj, dict):
            raise PresetError(f"Preset root must be a JSON object/dict ({source}).")
        return obj

    if ext in (".yaml", ".yml"):
        if yaml is None:
            raise PresetError(
                "YAML preset requested but PyYAML is not installed. "
                "Install with: pip install pyyaml"
            )
        try:
            obj = yaml.safe_load(text)
        except Exception as e:
            raise PresetError(f"Failed parsing YAML preset ({source}): {e}") from e
        if not isinstance(obj, dict):
            raise PresetError(f"Preset root must be a YAML mapping/dict ({source}).")
        return obj

    raise PresetError(f"Unsupported preset file extension '{ext}' ({source}). Use .yaml/.yml/.json.")

