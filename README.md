

# 🧬 Integrated SRCM–SSA

An integrated wrapper for the **Spatial Regime Conversion Method (SRCM)**, providing a seamless bridge between **deterministic PDE models** and **stochastic reaction simulations (SSA)**.

This package combines:
- **`srcm-engine`** — hybrid SSA–PDE simulation framework  
- **`stochastic-framework`** — stochastic simulation (SSA) engine  

to give a unified interface for running, analysing, and visualising spatial hybrid systems.

![Schematic](figures/schematic.png)

---

## ✨ Key Features

- 🔁 Hybrid SSA–PDE simulation with automatic regime switching  
- ⚖️ **Hysteresis-based conversion** (stable switching using two thresholds)  
- ⚡ Parallel execution for ensemble simulations  
- 🎬 Built-in CLI tools for inspection and animation  
- 📦 YAML presets for reproducible systems  

---

## 🚀 Installation

```bash
pip install "git+https://github.com/Cgyc20/Integrated-srcm-ssa.git@v1.3.0"
````

---

## 🧪 Python API Usage

### 1. Setup System

Example system:

$$
A \rightleftharpoons B
$$

```python
import numpy as np
from integrated_srcm_ssa import SRCMRunner
from srcm_engine.results.io import save_npz

sim = SRCMRunner(species=["A", "B"])

sim.define_rates(alpha=0.01, beta=0.01)
sim.define_diffusion(A=0.1, B=0.1)

sim.add_reaction({"A": 1}, {"B": 1}, "alpha")
sim.add_reaction({"B": 1}, {"A": 1}, "beta")
```

---

### 2. Define PDE Dynamics

```python
sim.set_pde_reactions(lambda A, B, r: (
    r["beta"] * B - r["alpha"] * A,
    r["alpha"] * A - r["beta"] * B,
))
```

---

### 3. Define Conversion (Hysteresis)

SRCM uses **two thresholds**:

* `DC_threshold`: discrete → continuous
* `CD_threshold`: continuous → discrete

```python
sim.define_conversion(
    DC_threshold=10,
    CD_threshold=6,
    rate=2.0,
)
```

#### Per-species thresholds

```python
sim.define_conversion(
    DC_threshold={"A": 8, "B": 6},
    CD_threshold={"A": 5, "B": 3},
    rate=1.0,
)
```

---

### 4. Run Simulation

```python
res_hybrid, meta = sim.run_hybrid(
    L=20.0,
    K=40,
    pde_multiple=8,
    total_time=30.0,
    dt=0.01,
    init_counts={"A": A_init, "B": B_init},
    repeats=100,
)
```

---

### 5. Save Results

```python
save_npz(res_hybrid, "data/simulation_results.npz", meta=meta)
```

---

## 🛠 Command Line Tools

### 🔍 Inspect

```bash
srcm-inspect results/hybrid.npz
```

### 🎬 Animate

```bash
srcm-animate hybrid.npz ssa.npz --stride 10 --title "Hybrid vs SSA"
```

| Option       | Description                                 | Default |
| ------------ | ------------------------------------------- | ------- |
| `--stride`   | Frame skipping                              | 20      |
| `--interval` | ms per frame                                | 30      |
| `--mass`     | Plot mass (`none`, `single`, `per_species`) | none    |

---

## 📦 Preset Systems (YAML)

```python
from integrated_srcm_ssa.presets import load_system

sim, cfg = load_system("my_system.yaml")

res_ssa, meta_ssa = sim.run_ssa(**cfg["ssa"])
res_hybrid, meta_hybrid = sim.run_hybrid(**cfg["hybrid"])
```

```python
from pathlib import Path

outdir = Path("results/my_experiment")
outdir.mkdir(parents=True, exist_ok=True)

save_npz(res_ssa, outdir / "ssa.npz", meta=meta_ssa)
save_npz(res_hybrid, outdir / "hybrid.npz", meta=meta_hybrid)
```

---

## 📚 Methodology

Based on:

> Cameron, C. G., Smith, C. A., & Yates, C. A. (2025)
> *The Spatial Regime Conversion Method*
> Mathematics 13(21), 3406
> [https://www.mdpi.com/2227-7390/13/21/3406](https://www.mdpi.com/2227-7390/13/21/3406)

---

## ⚠️ Limitations

* 1D spatial domains only
* reactions limited to order ≤ 2
* explicit PDE time stepping

---

## 🔧 Contributing

Contributions welcome:

* higher-order reactions
* adaptive grids
* GPU acceleration
* improved visualisation

---

## 🧾 Summary

This package provides a unified interface for:

* Pure SSA
* Hybrid SSA–PDE
* Ensemble simulations
* Final-state sampling
* Trajectory analysis

with stable and efficient hybrid dynamics.

```
```
