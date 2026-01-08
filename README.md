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
