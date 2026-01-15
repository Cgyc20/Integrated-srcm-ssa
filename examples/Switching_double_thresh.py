import numpy as np
import matplotlib.pyplot as plt
from integrated_srcm_ssa import SRCMRunner
from srcm_engine.results.io import save_npz
from srcm_engine.animation_util import AnimationConfig
from srcm_engine.animation_util.animate import animate_overlay, animate_results

# 1. Setup the System
sim = SRCMRunner(species=["A", "B"])
sim.define_rates(alpha=0.01, beta=0.01)
sim.define_diffusion(A=0.1, B=0.1)
#sim.define_conversion(threshold=20, rate=2.0)
sim.define_conversion(threshold={"A": 5, "B": 3}, rate=1.0)
sim.add_reaction({"A": 1}, {"B": 1}, "alpha")
sim.add_reaction({"B": 1}, {"A": 1}, "beta")
sim.set_pde_reactions(lambda A, B, r: (
    r["beta"] * B - r["alpha"] * A,
    r["alpha"] * A - r["beta"] * B,
))

# 2. Define Initial State
K = 40
A_init = np.zeros(K, dtype=int)
B_init = np.zeros(K, dtype=int)
A_init[:K//4] = 10 
B_init[3*K//4:] = 10

# 3. Run Simulations
L, total_time, dt = 10.0, 30.0, 0.006

n_repeats = 10
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
save_npz(res_hybrid, "data/hybrid_results.npz", meta=meta_hybrid)

# This 'meta_ssa' contains the reaction rules used by the SSA engine
save_npz(res_ssa, "data/ssa_results.npz", meta=meta_ssa)


# 5. Animate the Comparison
print("→ Launching Overlay Animation...")

# Configure the visual style
cfg = AnimationConfig(
    stride=20,           # Only plot every 20th frame to keep it smooth
    interval_ms=30,
    show_threshold=True,
              # Speed of playback
    title="SRCM Hybrid vs Pure SSA Comparison",
    mass_plot_mode="none"
)

plt.show()