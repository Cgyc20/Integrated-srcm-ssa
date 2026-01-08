import numpy as np
import matplotlib.pyplot as plt

from integrated_srcm_ssa import SRCMRunner
from srcm_engine.results.io import save_npz
from srcm_engine.animation_util import AnimationConfig
from srcm_engine.animation_util.animate import animate_overlay


# =========================
# 1) Setup activator–inhibitor
# =========================
sim = SRCMRunner(species=["A", "B"])

# Rates (tweak-friendly starter set)
sim.define_rates(
    sA=1.5,      # feed of A (zeroth-order)
    dA=0.20,     # linear decay of A
    sB=0.05,     # small feed of B
    dB=0.35,     # linear decay of B
    k_auto=0.015,   # 2A -> 3A autocatalysis
    k_prodB=0.010,  # 2A -> A + B (A makes B)
    k_conv=0.010    # A + B -> 2B (B converts A)
)

# Diffusion: inhibitor much faster
sim.define_diffusion(A=0.02, B=0.30)
sim.define_conversion(threshold=200, rate=2.0)
# SSA reactions (<=2 reactants; products can be >2)
sim.add_reaction({}, {"A": 1}, "sA")                  # ∅ -> A
sim.add_reaction({"A": 1}, {}, "dA")                  # A -> ∅

sim.add_reaction({}, {"B": 1}, "sB")                  # ∅ -> B
sim.add_reaction({"B": 1}, {}, "dB")                  # B -> ∅

sim.add_reaction({"A": 2}, {"A": 3}, "k_auto")        # 2A -> 3A
sim.add_reaction({"A": 2}, {"A": 1, "B": 1}, "k_prodB")  # 2A -> A + B
sim.add_reaction({"A": 1, "B": 1}, {"B": 2}, "k_conv")   # A + B -> 2B

# PDE drift (mean-field)
sim.set_pde_reactions(lambda A, B, r: (
    - r["dA"] * A + (r["k_auto"] - r["k_prodB"]) * (A**2) - r["k_conv"] * A * B,
     - r["dB"] * B + r["k_prodB"] * (A**2) + r["k_conv"] * A * B
))


# =========================
# 2) Initial conditions
# =========================
K = 16
A_init = np.zeros(K, dtype=int)
B_init = np.zeros(K, dtype=int)

# Baseline + a little bump to seed patterning
A_init[:] = 20
B_init[:] = 5
A_init[K//2 - 2: K//2 + 2] += 10

L, total_time, dt = 20.0, 40.0, 0.05
n_repeats = 10   # keep sane for SSA


# =========================
# 3) Run SSA + Hybrid
# =========================
res_ssa, meta_ssa = sim.run_ssa(
    L=L, K=K, total_time=total_time, dt=dt,
    init_counts={"A": A_init, "B": B_init},
    n_repeats=n_repeats
)

res_hybrid, meta_hybrid = sim.run_hybrid(
    L=L, K=K, pde_multiple=8, total_time=total_time, dt=dt,
    init_counts={"A": A_init, "B": B_init},
    repeats=n_repeats
)


# =========================
# 4) Save
# =========================
save_npz(res_hybrid, "data/ai2_hybrid_results.npz", meta=meta_hybrid)
save_npz(res_ssa, "data/ai2_ssa_results.npz", meta=meta_ssa)


# =========================
