import numpy as np
import matplotlib.pyplot as plt
from integrated_srcm_ssa import SRCMRunner
from srcm_engine.results.io import save_npz
from srcm_engine.animation_util import AnimationConfig
from srcm_engine.animation_util.animate import animate_overlay, animate_results


bar_alpha = 0.01 # ᾱ
bar_beta = 0.1 # β̄
bar_mu = 0.01 # μ̄
bar_D = 0.001 # D̄ (non-dimensional)
omega = 100 # scaling factor for particle SSA

L = 10.0
n_compartments = 40
h = L / n_compartments
total_time = 100.0
dt = 0.005

alpha = bar_alpha
beta = bar_beta / omega # β = β̄ / ω
mu = bar_mu * omega # μ = μ̄ ω


D = bar_D * (L**2) # D = D̄ L²

print(D)
initial_particle_mass = 50


# 1. Setup the System
sim = SRCMRunner(species=["A", "B"])
sim.define_rates(alpha=alpha, beta=beta, mu = mu)
sim.define_diffusion(A=D, B=D)
sim.define_conversion(threshold=20, rate=2.0)

sim.add_reaction({"A": 1, "B":1}, {"A":2}, "beta")
sim.add_reaction({"A": 1}, {}, "alpha")
sim.add_reaction({}, {"B": 1}, "mu")
sim.set_pde_reactions(lambda A, B, r: (
    r["beta"] *A*B - r["alpha"] * A,
    -r["beta"] *A*B
))

# 2. Define Initial State
K = 36
A_init = np.zeros(K, dtype=int)
B_init = np.zeros(K, dtype=int)
A_init[:K//4] = initial_particle_mass
B_init[3*K//4:] = initial_particle_mass

# 3. Run Simulations


n_repeats = 2
# 3. Run Simulations (Capture both results AND meta)
res_ssa, meta_ssa = sim.run_ssa(
    L=L, K=K, total_time=total_time, dt=dt, 
    init_counts={"A": A_init, "B": B_init}, n_repeats=n_repeats
)

res_hybrid, meta_hybrid = sim.run_hybrid(
    L=L, K=K, pde_multiple=8, total_time=total_time, 
    dt=dt, init_counts={"A": A_init, "B": B_init}, repeats=n_repeats
)

# 4. Save Results (Pass the captured meta dictionaries)
# This 'meta_hybrid' now contains your "reactions" list
save_npz(res_hybrid, "data/hybrid_SI_results.npz", meta=meta_hybrid)

# This 'meta_ssa' contains the reaction rules used by the SSA engine
save_npz(res_ssa, "data/ssa_SI_results.npz", meta=meta_ssa)

