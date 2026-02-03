import numpy as np
import matplotlib.pyplot as plt

from integrated_srcm_ssa import SRCMRunner
from srcm_engine.results.io import save_trajectories  # <-- NEW trajectory saver
from srcm_engine.animation_util import AnimationConfig
from srcm_engine.animation_util.animate import animate_overlay


# =========================
# 1) Setup: conversion + decay + collision makes 2A
# =========================
sim = SRCMRunner(species=["A", "B"])

# Rates
# alpha: A -> B
# beta : B -> A
# gamma: A + B -> 2A
# muA  : A -> ∅
# muB  : B -> ∅
sim.define_rates(alpha=0.05, beta=0.05, gamma=0.01, muA=0.02, muB=0.02)

# Diffusion
sim.define_diffusion(A=0.15, B=0.15)

# Hybrid conversion settings
sim.define_conversion(threshold=10, rate=2.0)

# SSA reactions (all <= 2nd order)
sim.add_reaction({"A": 1}, {"B": 1}, "alpha")          # A -> B
sim.add_reaction({"B": 1}, {"A": 1}, "beta")           # B -> A
sim.add_reaction({"A": 1}, {}, "muA")                  # A -> ∅
sim.add_reaction({"B": 1}, {}, "muB")                  # B -> ∅
sim.add_reaction({"A": 1, "B": 1}, {"A": 2}, "gamma")  # A + B -> 2A

# PDE drift (mean-field)
# A+B -> 2A contributes: +gamma*A*B to dA, and -gamma*A*B to dB
sim.set_pde_reactions(lambda A, B, r: (
    r["beta"] * B - r["alpha"] * A - r["muA"] * A + r["gamma"] * A * B,
    r["alpha"] * A - r["beta"] * B - r["muB"] * B - r["gamma"] * A * B,
))


# =========================
# 2) Initial conditions: two separated patches
# =========================
K = 40
A_init = np.zeros(K, dtype=int)
B_init = np.zeros(K, dtype=int)

A_init[:5] = 80
B_init[-5:] = 80


# =========================
# 3) Domain + time
# =========================
L = 20.0
total_time = 30.0
dt = 0.01
n_repeats = 10


# =========================
# 4) Run SSA + Hybrid TRAJECTORIES
# =========================
res_ssa_traj, meta_ssa_traj = sim.run_ssa_trajectories(
    L=L, K=K, total_time=total_time, dt=dt,
    init_counts={"A": A_init, "B": B_init},
    n_repeats=n_repeats,
    parallel=True,
    base_seed=1,
)

res_hybrid_traj, meta_hybrid_traj = sim.run_hybrid_trajectories(
    L=L, K=K, pde_multiple=8,
    total_time=total_time, dt=dt,
    init_counts={"A": A_init, "B": B_init},
    repeats=n_repeats,
    parallel=True,
    seed=1,
)


# =========================
# 5) Save TRAJECTORY ensembles
# =========================
save_trajectories(res_ssa_traj, "data/conv_decay_make2A_ssa.traj.npz", meta=meta_ssa_traj)
save_trajectories(res_hybrid_traj, "data/conv_decay_make2A_hybrid.traj.npz", meta=meta_hybrid_traj)
