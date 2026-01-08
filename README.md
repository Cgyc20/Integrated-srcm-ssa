## SRCM and SSA repo


This uses two codebases: The `srcm-engine` which is able to run the general srcm on any one dimensional two species system. This also uses the `stochastic-framework` local repo to do the stochastic reactions. 


How to use the simulation runner:
Import all relevant classes:

```python
import numpy as np
import matplotlib.pyplot as plt
from integrated_srcm_ssa import SRCMRunner
from srcm_engine.results.io import save_npz
```

And then to define the SRCM simulation we do the following:

So lets say we want to code in the reaction:

$$
\begin{aligned}
A \xrightarrow{\alpha} B, \\
B \xrightarrow {\beta} A,
\end{aligned}
$$
With some rates. Then we define the following system

```
sim = SRCMRunner(species=["A", "B"])
sim.define_rates(alpha=0.01, beta=0.01)
sim.define_diffusion(A=0.1, B=0.1)

sim.add_reaction({"A": 1}, {"B": 1}, "alpha")
sim.add_reaction({"B": 1}, {"A": 1}, "beta")
```
The SRCM engine will automatically decompose these reactions into the subsequent sub-reactions within the SRCM framework - Please see [The Spatial Regime Hybrid method](https://www.mdpi.com/2227-7390/13/21/3406) (2025, Cameron et al.) for further details. 

Following this we must actually add the corresonding PDE:

$$\begin{aligned}
\frac{\partial \langle A \rangle}{\partial t} &= D_A \nabla^2 \langle A \rangle + \beta \langle B \rangle - \alpha \langle A \rangle \\
\frac{\partial \langle B \rangle}{\partial t} &= D_B \nabla^2 \langle B \rangle + \alpha \langle A \rangle - \beta \langle B \rangle
\end{aligned}$$

```python
sim.set_pde_reactions(lambda A, B, r: (
    r["beta"] * B - r["alpha"] * A,
    r["alpha"] * A - r["beta"] * B,
))
```

We then need to define the conversion threshold and conversion rate (the threshold is in terms of particle per box):
```python
sim.define_conversion(threshold=10, rate=2.0)
```

We then need to add in the domain details, the number of compartments and initial conditions:

```python
# =========================
# Compartments and initial conditions
# =========================
K = 40
A_init = np.zeros(K, dtype=int)
B_init = np.zeros(K, dtype=int)

A_init[:5] = 80
B_init[-5:] = 80
# =========================
# Domain + time
# =========================
L = 20.0
total_time = 30.0
dt = 0.01
n_repeats = 100
```

The finally we need to run the models, both the **SRCM** and the pure **SSA** using the following code:

```python
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
```
And then we finally save the data into which ever folder. The metadata is also saved automatically. 
```python
save_npz(res_ssa, "data/conv_decay_make2A_ssa.npz", meta=meta_ssa)
save_npz(res_hybrid, "data/conv_decay_make2A_hybrid.npz", meta=meta_hybrid)

```

---
## Inspect the npz files

We have a feature to inspect the data to see the meta data such as the rates parameters, the system (i.e the SSA reactions) and so on.

To do this you need to run the `inspect_results.py` file using the following:

```bash
python inspect_results.py path/to/file/filename.npz
```

## Animating Simulation Results

This script lets you animate SRCM simulation results stored in `.npz` files. You can either animate a single result (typically a **Hybrid** simulation) or overlay two results (usually **Hybrid + SSA**) for comparison.

### Basic Usage

To animate a single simulation file:

```bash
python animate_npz.py path/to/simulation.npz
```

This will open a matplotlib animation window showing the evolution of the system over time.

### Overlay Hybrid and SSA Results

To overlay an SSA result on top of a Hybrid simulation:

```bash
python animate_npz.py hybrid_result.npz ssa_result.npz
```

The Hybrid result is used as the reference, and the SSA data is automatically adapted to match the domain.

### Useful Options

* `--stride N`
  Skip frames to speed up the animation (default: `20`).

* `--interval MS`
  Time between frames in milliseconds (default: `30`).

* `--title "My Title"`
  Custom title for the animation window.

* `--mass {none,single,per_species}`
  Enable mass plots if supported by the results.

* `--threshold VALUE`
  Manually override the particle threshold.

* `--debug-keys`
  Print the contents of the `.npz` file(s) for debugging.

### Example

```bash
python animate_npz.py results/hybrid.npz results/ssa.npz --stride 10 --mass per_species
```
