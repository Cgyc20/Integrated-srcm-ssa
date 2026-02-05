

# 🧬 Integrated SRCM–SSA

An integrated wrapper for the **Spatial Regime Change Method (SRCM)**, providing a seamless bridge between deterministic PDE machinery and stochastic reaction simulations.


### ✨ Key Features

* **`srcm-engine`**: High-performance deterministic SRCM + PDE machinery for 1D two-species systems.
* **`stochastic-framework`**: Robust Stochastic Simulation Algorithm (SSA) for reaction modeling.
* **Unified CLI**: Built-in tools for inspecting metadata and generating high-quality animations of your simulations.
![Schematic](figures/schematic.png)
---

## 🚀 Quick Start

### Installation

Install the complete suite directly from GitHub. All dependencies and engines are bundled automatically.

```bash
pip install "git+https://github.com/Cgyc20/Integrated-srcm-ssa.git@v1.1.5"
```

> [!TIP]
> We recommend installing within a **virtual environment** (venv/conda) to manage dependencies cleanly.

---

## 🧪 Python API Usage

### 1. Setup & Reaction Logic

Define your species, rates, and the governing physical laws of your system. For example this system:

$$
\begin{aligned} A \xrightarrow{\alpha} B\\
B \xrightarrow{\beta} A 
\end{aligned} 
$$

With corresponding PDE:

$$
\begin{aligned}
\frac{\partial \langle A \rangle}{\partial t} &= D_A \nabla^2 \langle A \rangle + \beta \langle B \rangle - \alpha \langle A \rangle ,
 \\ \frac{\partial \langle B \rangle}{\partial t} &= D_B \nabla^2 \langle B \rangle + \alpha \langle A \rangle - \beta \langle B \rangle.\end{aligned} 
$$

```python
import numpy as np
from integrated_srcm_ssa import SRCMRunner
from srcm_engine.results.io import save_npz

# Initialize for a two-species system (A ⇌ B)
sim = SRCMRunner(species=["A", "B"])
sim.define_rates(alpha=0.01, beta=0.01)
sim.define_diffusion(A=0.1, B=0.1)

# Add reactions
sim.add_reaction({"A": 1}, {"B": 1}, "alpha") # A -> B
sim.add_reaction({"B": 1}, {"A": 1}, "beta")  # B -> A

```

### 2. Define the PDE System

The SRCM handles the transition between discrete particles and continuous densities.

```python
sim.set_pde_reactions(lambda A, B, r: (
    r["beta"] * B - r["alpha"] * A,
    r["alpha"] * A - r["beta"] * B,
))

# Set the threshold (particles per compartment) for regime switching
sim.define_conversion(threshold=10, rate=2.0)

```
We have added  a new feature in the recent release of the `SRCM-engine` and we can have two thresholds. We run this using this feature:

```python
sim.define_conversion(threshold={"A": 5, "B": 3}, rate=1.0)
```

### 3. Execution

Run either a pure stochastic simulation or the integrated hybrid model.

```python
# Run Hybrid SRCM
res_hybrid, meta_hybrid = sim.run_hybrid(
    L=20.0, K=40, pde_multiple=8,
    total_time=30.0, dt=0.01,
    init_counts={"A": A_init, "B": B_init},
    repeats=100
)

# Save results to .npz for analysis
save_npz(res_hybrid, "data/simulation_results.npz", meta=meta_hybrid)

```

---

## 🛠 Command Line Tools

The package installs two powerful global commands to manage your `.npz` output files.

### 🔍 `srcm-inspect`

Quickly view the contents and parameters of a simulation file without opening a notebook.

```bash
# Inspect one or multiple files
srcm-inspect results/hybrid_01.npz results/ssa_01.npz

```

### 🎬 `srcm-animate`

Visualize the dynamics of your simulation. You can overlay Hybrid and SSA results to compare accuracy.

```bash
# Overlay SSA on Hybrid data with custom settings
srcm-animate hybrid.npz ssa.npz --stride 10 --mass per_species --title "Reaction-Diffusion Comparison"

```

| Option | Description | Default |
| --- | --- | --- |
| `--stride` | Skip frames to speed up playback | `20` |
| `--interval` | Milliseconds between frames | `30` |
| `--threshold` | Manually override the particle threshold line | - |
| `--mass` | Plot mass dynamics (`none`, `single`, `per_species`) | `none` |

---


## 🔧 Preset systems (YAML)

The Integrated SRCM–SSA package includes **predefined system presets** written in YAML.
These presets describe complete reaction–diffusion systems (species, reactions, rates, PDE drift, domain, and initial conditions). user can inspect these files for context of the system. 

We would advise only using these as reference The reason we have used YAML files is simply to save data, and so users can download many different systems of choice via the YAML. 

We also advise that you write the python file from the template provided for any other system you wish to run. Presets can be loaded directly and executed without writing boilerplate setup code. 

```python
from integrated_srcm_ssa.presets import load_system

sim, cfg = load_system("my_system.yaml")

res_ssa, meta_ssa = sim.run_ssa(**cfg["ssa"])
res_hybrid, meta_hybrid = sim.run_hybrid(**cfg["hybrid"])

# Choose output directory + filenames
outdir = Path("results/my_experiment")
outdir.mkdir(parents=True, exist_ok=True)

save_npz(res_ssa, outdir / "ssa_run.npz", meta=meta_ssa)
save_npz(res_hybrid, outdir / "hybrid_run.npz", meta=meta_hybrid)


```
Users again can use the `srcm-inspect` and `srcm-animate` to inspect and animate results.

## 📚 Methodology

This framework is based on the research by **Cameron et al. (2025)**.

> **The Spatial Regime Hybrid Method** > *Mathematics 2025, 13(21), 3406* > [Read the full paper here](https://www.mdpi.com/2227-7390/13/21/3406)

