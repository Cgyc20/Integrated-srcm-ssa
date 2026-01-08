# Integrated SRCM–SSA

This repository provides a **Spatial Regime Change Method (SRCM)** wrapper that integrates:

* **`srcm-engine`** — deterministic SRCM + PDE machinery for 1D two-species systems
* **`stochastic-framework`** — stochastic reaction simulation (SSA)

It also includes **command-line tools** to inspect and animate simulation outputs stored as `.npz` files.

Users only need to install **this package** — all required engines are installed automatically.

---

## Installation

Install directly from GitHub:

```bash
pip install "git+https://github.com/Cgyc20/Integrated-srcm-ssa.git"
```

This will automatically install:

* `srcm-engine`
* `stochastic-framework`
* all required Python dependencies

> **Tip:** we recommend installing inside a virtual environment.

---

## Python Usage: Running SRCM and SSA Simulations

### Import required classes

```python
import numpy as np
from integrated_srcm_ssa import SRCMRunner
from srcm_engine.results.io import save_npz
```

---

### Example system

We consider the reversible reaction system:

[
\begin{aligned}
A \xrightarrow{\alpha} B, \
B \xrightarrow{\beta} A
\end{aligned}
]

---

### Define the SRCM simulation

```python
sim = SRCMRunner(species=["A", "B"])

sim.define_rates(alpha=0.01, beta=0.01)
sim.define_diffusion(A=0.1, B=0.1)

sim.add_reaction({"A": 1}, {"B": 1}, "alpha")
sim.add_reaction({"B": 1}, {"A": 1}, "beta")
```

The SRCM engine automatically decomposes reactions into the appropriate sub-reactions required by the hybrid SRCM framework.

For methodological details, see:
**Cameron et al. (2025)** — *The Spatial Regime Hybrid Method*
[https://www.mdpi.com/2227-7390/13/21/3406](https://www.mdpi.com/2227-7390/13/21/3406)

---

### Define the PDE system

[
\begin{aligned}
\frac{\partial \langle A \rangle}{\partial t}
&= D_A \nabla^2 \langle A \rangle + \beta \langle B \rangle - \alpha \langle A \rangle \
\frac{\partial \langle B \rangle}{\partial t}
&= D_B \nabla^2 \langle B \rangle + \alpha \langle A \rangle - \beta \langle B \rangle
\end{aligned}
]

```python
sim.set_pde_reactions(lambda A, B, r: (
    r["beta"] * B - r["alpha"] * A,
    r["alpha"] * A - r["beta"] * B,
))
```

---

### Define conversion parameters

The conversion threshold is measured in **particles per compartment**.

```python
sim.define_conversion(threshold=10, rate=2.0)
```

---

### Domain, initial conditions, and time parameters

```python
K = 40
A_init = np.zeros(K, dtype=int)
B_init = np.zeros(K, dtype=int)

A_init[:5] = 80
B_init[-5:] = 80

L = 20.0
total_time = 30.0
dt = 0.01
n_repeats = 100
```

---

### Run simulations

#### Pure SSA

```python
res_ssa, meta_ssa = sim.run_ssa(
    L=L, K=K,
    total_time=total_time, dt=dt,
    init_counts={"A": A_init, "B": B_init},
    n_repeats=n_repeats
)
```

#### Hybrid SRCM

```python
res_hybrid, meta_hybrid = sim.run_hybrid(
    L=L, K=K, pde_multiple=8,
    total_time=total_time, dt=dt,
    init_counts={"A": A_init, "B": B_init},
    repeats=n_repeats
)
```

---

### Save results

Metadata is saved automatically alongside the simulation output.

```python
save_npz(res_ssa, "data/conv_decay_ssa.npz", meta=meta_ssa)
save_npz(res_hybrid, "data/conv_decay_hybrid.npz", meta=meta_hybrid)
```

---

## Inspecting `.npz` Results

After installation, the inspection tool is available globally.

```bash
srcm-inspect path/to/file.npz
```

This prints:

* array contents and shapes
* reaction definitions
* rate parameters
* diffusion coefficients
* domain and temporal metadata

You may inspect multiple files at once:

```bash
srcm-inspect results/*.npz
```

---

## Animating Simulation Results

Simulation outputs can be animated directly from `.npz` files.

### Animate a single result (typically Hybrid)

```bash
srcm-animate path/to/simulation.npz
```

---

### Overlay Hybrid and SSA results

```bash
srcm-animate hybrid_result.npz ssa_result.npz
```

The hybrid result is used as the spatial reference, and SSA data is automatically adapted to match the domain.

---

### Useful options

* `--stride N`
  Skip frames to speed up animation (default: `20`)

* `--interval MS`
  Time between frames in milliseconds (default: `30`)

* `--title "My Title"`
  Custom animation title

* `--mass {none,single,per_species}`
  Enable mass plots (if supported)

* `--threshold VALUE`
  Override particle conversion threshold

* `--debug-keys`
  Print `.npz` contents for debugging

---

### Example

```bash
srcm-animate results/hybrid.npz results/ssa.npz \
  --stride 10 \
  --mass per_species
```

