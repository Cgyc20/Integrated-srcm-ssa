import numpy as np
import matplotlib.pyplot as plt

from integrated_srcm_ssa import SRCMRunner
from srcm_engine.results.io import save_npz
from srcm_engine.animation_util import AnimationConfig
from srcm_engine.animation_util.animate import animate_overlay


# =========================
# 1) Setup: pure diffusion, single species
# =========================
sim = SRCMRunner(species=["A"])

# Diffusion only
sim.define_diffusion(A=0.2)

# Add TWO ultra-tiny no-op reactions to keep SSA builder happy
# (Prevents the "Reaction" vs "ReactionSystem" edge case)
sim.define_rates(eps=1e-14)

# 👇 Set conversion during setup
sim.define_conversion(threshold=50, rate=2.0)


sim.add_reaction({"A": 1}, {"A": 1}, "eps")   # A -> A (no-op)


# PDE drift is zero (pure diffusion)
# Signature depends on your runner; this matches your earlier style.
sim.set_pde_reactions(lambda A, r: (0.0 * A,))


# =========================
# 2) Initial conditions: mass in central compartment
# =========================
K = 25
center = K // 2
A_init = np.zeros(K, dtype=int)
A_init[center] = 1000

L, total_time, dt = 20.0, 20.0, 0.01
n_repeats = 10


# =========================
# 3) Run SSA + Hybrid
# =========================
res_ssa, meta_ssa = sim.run_ssa(
    L=L, K=K, total_time=total_time, dt=dt,
    init_counts={"A": A_init},
    n_repeats=n_repeats
)

res_hybrid, meta_hybrid = sim.run_hybrid(
    L=L, K=K, pde_multiple=8,
    total_time=total_time, dt=dt,
    init_counts={"A": A_init},
    repeats=n_repeats
)

save_npz(res_ssa, "data/pure_diffusion_ssa.npz", meta=meta_ssa)
save_npz(res_hybrid, "data/pure_diffusion_hybrid.npz", meta=meta_hybrid)

