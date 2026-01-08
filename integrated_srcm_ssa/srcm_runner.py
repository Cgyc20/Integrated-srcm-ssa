import numpy as np
from stochastic_framework import Reaction, SSA
from srcm_engine.domain import Domain
from srcm_engine.core import HybridModel
from srcm_engine.results.simulation_results import SimulationResults


class SRCMRunner:
    def __init__(self, species: list[str], *, threshold: int = 4, conv_rate: float = 1.0):
        self.species = species
        self.species_map = {s: i for i, s in enumerate(species)}
        self.reactions = []  # (reactants, products, rate_name)
        self.rates = {}
        self.diff_coeffs = {}

        self.hybrid_model = HybridModel(species=species)
        self._reaction_func = None

        # Conversion defaults (can be overridden via define_conversion)
        self.threshold = int(threshold)
        self.conv_rate = float(conv_rate)

    # -------------------------
    # Setup / definition methods
    # -------------------------
    def define_rates(self, **kwargs):
        self.rates.update(kwargs)

    def define_diffusion(self, **kwargs):
        self.diff_coeffs.update(kwargs)

    def define_conversion(self, *, threshold: int | None = None, rate: float | None = None):
        """
        Define SSA <-> PDE conversion settings for hybrid runs.
        threshold: particles per SSA compartment (>= 0)
        rate: conversion rate (>= 0)
        """
        if threshold is not None:
            thr = int(threshold)
            if thr < 0:
                raise ValueError("conversion threshold must be >= 0")
            self.threshold = thr

        if rate is not None:
            cr = float(rate)
            if cr < 0:
                raise ValueError("conversion rate must be >= 0")
            self.conv_rate = cr

    def add_reaction(self, reactants: dict, products: dict, rate_name: str):
        self.reactions.append((reactants, products, rate_name))
        self.hybrid_model.add_reaction(reactants, products, rate_name=rate_name)

    def set_pde_reactions(self, func):
        self._reaction_func = func
        self.hybrid_model.reaction_terms(func)

    # -------------------------
    # Metadata helper
    # -------------------------
    def _get_shared_meta(self, L, K, total_time, dt, repeats):
        return {
            "species": self.species,
            "total_time": float(total_time),
            "dt": float(dt),
            "repeats": int(repeats),
            "diffusion_rates": dict(self.diff_coeffs),
            "reaction_rates": dict(self.rates),
            "reactions": [
                {
                    "reactants": r,
                    "products": p,
                    "rate": self.rates.get(name),
                    "rate_name": name,
                }
                for r, p, name in self.reactions
            ],
            "domain": {"length": L, "K": K, "boundary": "zero-flux"},
        }

    # -------------------------
    # Runners
    # -------------------------
    def run_ssa(self, L, K, total_time, dt, init_counts, n_repeats=10):
        print(f"→ Starting Pure SSA Simulation ({n_repeats} repeats)...")

        rxn = Reaction()
        for r, p, name in self.reactions:
            rxn.add_reaction(r, p, self.rates[name])

        ic = np.zeros((len(self.species), K), dtype=int)
        for spec, arr in init_counts.items():
            ic[self.species_map[spec]] = arr

        ssa_engine = SSA(rxn)
        ssa_engine.set_conditions(
            n_compartments=K,
            domain_length=L,
            total_time=total_time,
            initial_conditions=ic,
            timestep=dt,
            Macroscopic_diffusion_rates=[self.diff_coeffs[s] for s in self.species],
            boundary_conditions="zero-flux",
        )

        avg_out = ssa_engine.run_simulation(n_repeats=n_repeats)
        time = np.asarray(ssa_engine.timevector)
        ssa_data = np.transpose(avg_out, (1, 2, 0))

        domain = Domain(length=L, n_ssa=K, pde_multiple=1, boundary="zero-flux")
        pde_data = np.zeros((len(self.species), K, len(time)))

        res = SimulationResults(time=time, ssa=ssa_data, pde=pde_data, domain=domain, species=self.species)

        meta = self._get_shared_meta(L, K, total_time, dt, n_repeats)
        meta["run_type"] = "pure_ssa"

        print("✅ SSA Simulation Complete.")
        return res, meta

    def run_hybrid(self, L, K, pde_multiple, total_time, dt, init_counts, repeats=10, seed=1, parallel=True):
        print(f"→ Starting SRCM Hybrid Simulation ({repeats} repeats)...")

        # Domain + diffusion + conversion configuration happens here
        self.hybrid_model.domain(L=L, K=K, pde_multiple=pde_multiple, boundary="zero-flux")
        self.hybrid_model.diffusion(**self.diff_coeffs)
        self.hybrid_model.conversion(threshold=self.threshold, rate=self.conv_rate)
        self.hybrid_model.build(rates=self.rates)

        init_ssa = np.zeros((len(self.species), K), dtype=int)
        for spec, arr in init_counts.items():
            init_ssa[self.species_map[spec]] = arr

        init_pde = np.zeros((len(self.species), K * pde_multiple), dtype=float)

        res = self.hybrid_model.run_repeats(
            init_ssa,
            init_pde,
            time=total_time,
            dt=dt,
            repeats=repeats,
            seed=seed,
            parallel=parallel,
        )

        meta = {
            "species": self.species,
            "threshold_particles": self.threshold,
            "conversion_rate": self.conv_rate,
            "diffusion_rates": dict(self.diff_coeffs),
            "reaction_rates": dict(self.rates),
            "total_time": float(total_time),
            "dt": float(dt),
            "repeats": int(repeats),
            "seed": int(seed),
            "reactions": [
                {"reactants": r, "products": p, "rate_name": name, "rate": self.rates.get(name)}
                for r, p, name in self.reactions
            ],
            "domain": {"L": L, "K": K, "pde_multiple": pde_multiple, "boundary": "zero-flux"},
        }

        print("✅ Hybrid Simulation Complete.")
        return res, meta
