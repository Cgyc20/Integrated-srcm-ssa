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
        self.reactions = [] 
        self.rates = {}
        self.diff_coeffs = {}
        
        self.hybrid_model = HybridModel(species=species)
        self._reaction_func = None

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

    def describe(self):
        """Prints the system configuration to terminal."""
        print("\n" + "="*30)
        print("SYSTEM CONFIGURATION")
        print("="*30)
        print(f"Species: {', '.join(self.species)}")
        print(f"Rates: {self.rates}")
        # Call the internal srcm_engine description
        self.hybrid_model.describe_reactions()
        print("="*30 + "\n")

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
        
        print("✅ SSA Simulation Complete.")
        return SimulationResults(time=time, ssa=ssa_data, pde=pde_data, 
                                 domain=domain, species=self.species)

    def run_hybrid(self, L, K, pde_multiple, total_time, dt, init_counts, repeats=10, seed=1):
        print(f"→ Starting SRCM Hybrid Simulation ({repeats} repeats)...")
        self.hybrid_model.domain(L=L, K=K, pde_multiple=pde_multiple, boundary="zero-flux")
        self.hybrid_model.diffusion(**self.diff_coeffs)
        self.hybrid_model.conversion(threshold=4, rate=1.0)
        self.hybrid_model.build(rates=self.rates)

        # Print the reaction mechanics to terminal
        self.hybrid_model.describe_reactions()

        init_ssa = np.zeros((len(self.species), K), dtype=int)
        for spec, arr in init_counts.items():
            init_ssa[self.species_map[spec]] = arr
            
        init_pde = np.zeros((len(self.species), K * pde_multiple), dtype=float)

        res = self.hybrid_model.run_repeats(
            init_ssa, init_pde, time=total_time, dt=dt,
            repeats=repeats, seed=seed, parallel=True
        )
        print("✅ Hybrid Simulation Complete.")
        return res