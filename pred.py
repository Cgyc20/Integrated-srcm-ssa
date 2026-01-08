import numpy as np
import matplotlib.pyplot as plt

from integrated_srcm_ssa import SRCMRunner
from srcm_engine.results.io import save_npz
from srcm_engine.animation_util import AnimationConfig
from srcm_engine.animation_util.animate import animate_overlay


# =========================
# 1) Setup: conversion + annihilation
# =========================
sim = SRCMRunner(species=["A", "B"])

# Rates (cheap + stable)
sim.define_rates(alpha=0.05, beta=0.05, gamma=0.01)

# Diffusion
sim.define_diffusion(A=0.15, B=0.15)

# Hybrid conversion settings (tweak)
sim.define_conversion(threshold=20, rate=2.0)

# SSA reactions (all <= 2nd order)
sim.add_reaction({"A": 1}, {"B": 1}, "alpha")        # A -> B
sim.add_reaction({"B": 1}, {"A": 1}, "beta")         # B -> A
sim.add_reaction({"A": 1, "B": 1}, {"A"}, "gamma")      # A + B -> ∅


# PDE drift (mean-field)
sim.set_pde_reactions(lambda A, B, r: (
    r["beta"] * B - r["alpha"] * A - r["gamma"] * A * B,
    r["alpha"] * A - r["beta"] * B - r["gamma"] * A * B,
))


# =========================
# 2) Initial conditions: two separated patches
# =========================
K = 40
A_init = np.zeros(K, dtype=int)
B_init = np.zeros(K, dtype=int)

A_init[:] = 0
B_init[:] = 0

A_init[:5] = 80   # even boxes A
B_init[-5:] = 80  # odd boxes B


# Domain + time
L = 20.0
total_time = 30.0
dt = 0.01

# Keep SSA repeats modest; this should already be much faster than autocatalysis models
n_repeats = 10


# =========================
# 3) Run SSA + Hybrid
# =========================
res_ssa, meta_ssa = sim.run_ssa(
    L=L, K=K, total_time=total_time, dt=dt,
    init_counts={"A": A_init, "B": B_init},
    n_repeats=n_repeats
)

res_hybrid, meta_hybrid = sim.run_hybrid(
    L=L, K=K, pde_multiple=8,
    total_time=total_time, dt=dt,
    init_counts={"A": A_init, "B": B_init},
    repeats=n_repeats
)

save_npz(res_ssa, "data/conv_annihil_ssa.npz", meta=meta_ssa)
save_npz(res_hybrid, "data/conv_annihil_hybrid.npz", meta=meta_hybrid)


