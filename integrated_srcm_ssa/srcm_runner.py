import numpy as np
from pathlib import Path
from stochastic_framework import Reaction, SSA
from srcm_engine.domain import Domain
from srcm_engine.core import HybridModel
from srcm_engine.results.simulation_results import SimulationResults
from srcm_engine.results.io import save_results, save_npz

class SRCMRunner:
    def __init__(self, species: list[str]):
        self.species = species
        self.species_map = {s: i for i, s in enumerate(species)}
        self.reactions = [] # Stores (reactants, products, rate_name)
        self.rates = {}
        self.diff_coeffs = {}
        
        self.hybrid_model = HybridModel(species=species)
        self._reaction_func = None
        
        # Track these for metadata
        self.threshold = 4
        self.conv_rate = 1.0

    def define_rates(self, **kwargs):
        self.rates.update(kwargs)

    def define_diffusion(self, **kwargs):
        self.diff_coeffs.update(kwargs)

    def add_reaction(self, reactants: dict, products: dict, rate_name: str):
        self.reactions.append((reactants, products, rate_name))
        self.hybrid_model.add_reaction(reactants, products, rate_name=rate_name)

    def set_pde_reactions(self, func):
        self._reaction_func = func
        self.hybrid_model.reaction_terms(func)

    def _get_shared_meta(self, L, K, total_time, dt, repeats):
        """Helper to build a consistent metadata dictionary."""
        return {
            "species": self.species,
            "total_time": float(total_time),
            "dt": float(dt),
            "repeats": int(repeats),
            "diffusion_rates": self.diff_coeffs,
            "reaction_rates": self.rates,
            "reactions": [
                {
                    "reactants": r, 
                    "products": p, 
                    "rate": self.rates.get(name), 
                    "rate_name": name
                } 
                for r, p, name in self.reactions
            ],
            "domain": {
                "length": L, 
                "K": K, 
                "boundary": "zero-flux"
            }
        }

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
            n_compartments=K, domain_length=L, total_time=total_time,
            initial_conditions=ic, timestep=dt,
            Macroscopic_diffusion_rates=[self.diff_coeffs[s] for s in self.species],
            boundary_conditions="zero-flux"
        )
        
        avg_out = ssa_engine.run_simulation(n_repeats=n_repeats)
        time = np.asarray(ssa_engine.timevector)
        ssa_data = np.transpose(avg_out, (1, 2, 0))
        
        domain = Domain(length=L, n_ssa=K, pde_multiple=1, boundary="zero-flux")
        pde_data = np.zeros((len(self.species), K, len(time)))
        
        res = SimulationResults(time=time, ssa=ssa_data, pde=pde_data, 
                                 domain=domain, species=self.species)
        
        # Build Metadata
        meta = self._get_shared_meta(L, K, total_time, dt, n_repeats)
        meta["run_type"] = "pure_ssa"

        print("✅ SSA Simulation Complete.")
        return res, meta # <--- Returns tuple now

    def run_hybrid(self, L, K, pde_multiple, total_time, dt, init_counts, repeats=10, seed=1):
        print(f"→ Starting SRCM Hybrid Simulation ({repeats} repeats)...")
        self.hybrid_model.domain(L=L, K=K, pde_multiple=pde_multiple, boundary="zero-flux")
        self.hybrid_model.diffusion(**self.diff_coeffs)
        self.hybrid_model.conversion(threshold=self.threshold, rate=self.conv_rate)
        self.hybrid_model.build(rates=self.rates)

        init_ssa = np.zeros((len(self.species), K), dtype=int)
        for spec, arr in init_counts.items():
            init_ssa[self.species_map[spec]] = arr
            
        init_pde = np.zeros((len(self.species), K * pde_multiple), dtype=float)

        res = self.hybrid_model.run_repeats(
            init_ssa, init_pde, time=total_time, dt=dt,
            repeats=repeats, seed=seed, parallel=True
        )

        # Build Metadata
        # 4. Package Metadata
        # We try to find the reactions list in several common engine locations
        # 1. Check '_reactions' (most likely based on your error)
        # 2. Fall back to the internal reaction system if it exists
        # 3. Default to an empty list if nothing is found
        
        # 4. Package Metadata
        engine_reactions = getattr(self.hybrid_model, "_reaction_system", self.hybrid_model)._reactions
        
        # Build Metadata
        meta = {
            "species": self.species,
            "threshold_particles": self.threshold,
            "conversion_rate": self.conv_rate,
            "diffusion_rates": self.diff_coeffs,
            "reaction_rates": self.rates,
            "total_time": float(total_time),
            "dt": float(dt),
            "repeats": int(repeats),
            "seed": int(seed),
            # USE THE LOCAL LIST: It is already JSON-serializable
            "reactions": [
                {
                    "reactants": r, 
                    "products": p, 
                    "rate_name": name,
                    "rate": self.rates.get(name)
                } 
                for r, p, name in self.reactions
            ],
            "domain": {
                "L": L, "K": K, "pde_multiple": pde_multiple, "boundary": "zero-flux"
            }
        }

        print("✅ Hybrid Simulation Complete.")
        return res, meta