# integrated_srcm_ssa/presets/__init__.py

"""
Preset systems for integrated-srcm-ssa.

Presets provide declarative, reusable system definitions that configure
SRCMRunner instances without writing boilerplate code.

Typical usage:

    from integrated_srcm_ssa.presets import load_system, list_presets

    sim, cfg = load_system("conv_decay_make2A")
    res_hybrid, meta = sim.run_hybrid(**cfg["hybrid"])
"""

from .loader import (
    load_system,
    load_preset,
    list_packaged_presets as list_presets,
    PresetError,
)

__all__ = [
    "load_system",
    "load_preset",
    "list_presets",
    "PresetError",
]
