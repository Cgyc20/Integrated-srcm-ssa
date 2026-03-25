from __future__ import annotations

import numpy as np
from stochastic_framework import Reaction, SSA
from srcm_engine.domain import Domain
from srcm_engine.core import HybridModel
from srcm_engine.results.simulation_results import SimulationResults


def _validate_boundary(boundary: str) -> str:
    boundary = str(boundary)
    if boundary not in ("zero-flux", "periodic"):
        raise ValueError("boundary must be 'zero-flux' or 'periodic'")
    return boundary


def _validate_hysteresis_pair(DC_threshold, CD_threshold) -> None:
    dc = np.asarray(DC_threshold, dtype=float)
    cd = np.asarray(CD_threshold, dtype=float)

    if np.any(dc < 0):
        raise ValueError("DC_threshold must be >= 0")
    if np.any(cd < 0):
        raise ValueError("CD_threshold must be >= 0")

    if dc.ndim == 0 and cd.ndim == 0:
        if float(dc) <= float(cd):
            raise ValueError("DC_threshold must be strictly greater than CD_threshold")
        return

    if dc.ndim == 0:
        dc = np.full_like(cd, float(dc), dtype=float)
    if cd.ndim == 0:
        cd = np.full_like(dc, float(cd), dtype=float)

    if dc.shape != cd.shape:
        raise ValueError("DC_threshold and CD_threshold must have compatible shapes")

    if np.any(dc <= cd):
        raise ValueError("DC_threshold must be strictly greater than CD_threshold for all species")


class SRCMRunner:
    def __init__(
        self,
        species: list[str],
        *,
        DC_threshold: int | float | dict = 6,
        CD_threshold: int | float | dict = 4,
        conv_rate: float = 1.0,
        boundary: str = "zero-flux",
    ):
        self.species = species
        self.species_map = {s: i for i, s in enumerate(species)}
        self.reactions = []  # (reactants, products, rate_name)
        self.rates = {}
        self.diff_coeffs = {}

        self.hybrid_model = HybridModel(species=species)
        self._reaction_func = None

        self.DC_threshold = DC_threshold
        self.CD_threshold = CD_threshold
        self.conv_rate = float(conv_rate)
        _validate_hysteresis_pair(self.DC_threshold, self.CD_threshold)

        self.boundary = _validate_boundary(boundary)

    # -------------------------
    # Setup / definition methods
    # -------------------------
    def define_rates(self, **kwargs):
        self.rates.update(kwargs)

    def define_diffusion(self, **kwargs):
        self.diff_coeffs.update(kwargs)

    def define_boundary(self, boundary: str):
        """Set the default boundary used by run_ssa/run_hybrid unless overridden."""
        self.boundary = _validate_boundary(boundary)

    def define_conversion(
        self,
        *,
        DC_threshold: int | float | dict | None = None,
        CD_threshold: int | float | dict | None = None,
        rate: float | None = None,
        threshold: int | float | dict | None = None,
    ):
        """
        Define SSA <-> PDE conversion settings for hybrid runs.

        Preferred API
        -------------
        DC_threshold : higher threshold for discrete -> continuous conversion
        CD_threshold : lower threshold for continuous -> discrete conversion
        rate         : conversion rate (>= 0)

        Backward compatibility
        ----------------------
        threshold    : if provided, sets both thresholds as:
                       DC_threshold = threshold + 1
                       CD_threshold = threshold
                       This preserves a minimal hysteresis gap.
        """
        new_DC = self.DC_threshold
        new_CD = self.CD_threshold

        if threshold is not None:
            if isinstance(threshold, dict):
                new_CD = dict(threshold)
                new_DC = {k: float(v) + 1.0 for k, v in threshold.items()}
            else:
                thr = float(threshold)
                if thr < 0:
                    raise ValueError("conversion threshold must be >= 0")
                new_CD = thr
                new_DC = thr + 1.0

        if DC_threshold is not None:
            new_DC = DC_threshold
        if CD_threshold is not None:
            new_CD = CD_threshold

        _validate_hysteresis_pair(new_DC, new_CD)
        self.DC_threshold = new_DC
        self.CD_threshold = new_CD

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
    def _get_shared_meta(self, L, K, total_time, dt, repeats, boundary: str):
        return {
            "species": self.species,
            "total_time": float(total_time),
            "dt": float(dt),
            "repeats": int(repeats),
            "boundary": boundary,
            "diffusion_rates": dict(self.diff_coeffs),
            "reaction_rates": dict(self.rates),
            "conversion": {
                "DC_threshold": self.DC_threshold,
                "CD_threshold": self.CD_threshold,
                "rate": self.conv_rate,
            },
            "reactions": [
                {
                    "reactants": r,
                    "products": p,
                    "rate": self.rates.get(name),
                    "rate_name": name,
                }
                for r, p, name in self.reactions
            ],
            "domain": {"length": L, "K": K, "boundary": boundary},
        }

    # -------------------------
    # Runners
    # -------------------------
    def run_ssa(
        self,
        L,
        K,
        total_time,
        dt,
        init_counts,
        n_repeats=10,
        *,
        boundary: str | None = None,
        parallel: bool = False,
        n_jobs: int = -1,
        max_n_jobs: int | None = None,
        base_seed: int | None = None,
    ):
        boundary = _validate_boundary(boundary or self.boundary)
        print(f"→ Starting Pure SSA Simulation ({n_repeats} repeats, boundary={boundary})...")

        rxn = Reaction()
        for r, p, name in self.reactions:
            rxn.add_reaction(r, p, self.rates[name])

        ic = np.zeros((len(self.species), K), dtype=int)
        for spec, arr in init_counts.items():
            ic[self.species_map[spec]] = arr

        ssa_engine = SSA(rxn)
        ssa_engine.set_conditions(
            n_compartments=K,
            domain_length=float(L),
            total_time=float(total_time),
            initial_conditions=ic,
            timestep=float(dt),
            Macroscopic_diffusion_rates=[float(self.diff_coeffs[s]) for s in self.species],
            boundary_conditions=boundary,
        )

        avg_out = ssa_engine.run_simulation(
            n_repeats=int(n_repeats),
            parallel=bool(parallel),
            n_jobs=int(n_jobs) if n_jobs is not None else -1,
            max_n_jobs=max_n_jobs,
            base_seed=base_seed,
        )

        time = np.asarray(ssa_engine.timevector)
        ssa_data = np.transpose(avg_out, (1, 2, 0))

        domain = Domain(length=float(L), n_ssa=int(K), pde_multiple=1, boundary=boundary)
        pde_data = np.zeros((len(self.species), int(K), len(time)))

        res = SimulationResults(time=time, ssa=ssa_data, pde=pde_data, domain=domain, species=self.species)

        meta = self._get_shared_meta(L, K, total_time, dt, n_repeats, boundary)
        meta["run_type"] = "pure_ssa_mean"

        print("✅ SSA Simulation Complete.")
        return res, meta

    def run_hybrid(
        self,
        L,
        K,
        pde_multiple,
        total_time,
        dt,
        init_counts,
        repeats=10,
        seed=1,
        parallel=True,
        *,
        boundary: str | None = None,
    ):
        boundary = _validate_boundary(boundary or self.boundary)
        print(f"→ Starting SRCM Hybrid Simulation ({repeats} repeats, boundary={boundary})...")

        self.hybrid_model.domain(L=float(L), K=int(K), pde_multiple=int(pde_multiple), boundary=boundary)
        self.hybrid_model.diffusion(**self.diff_coeffs)
        self.hybrid_model.conversion(
            DC_threshold=self.DC_threshold,
            CD_threshold=self.CD_threshold,
            rate=self.conv_rate,
        )
        self.hybrid_model.build(rates=self.rates)

        init_ssa = np.zeros((len(self.species), int(K)), dtype=int)
        for spec, arr in init_counts.items():
            init_ssa[self.species_map[spec]] = arr

        init_pde = np.zeros((len(self.species), int(K) * int(pde_multiple)), dtype=float)

        res = self.hybrid_model.run_repeats(
            init_ssa,
            init_pde,
            time=float(total_time),
            dt=float(dt),
            repeats=int(repeats),
            seed=int(seed),
            parallel=bool(parallel),
        )

        meta = self._get_shared_meta(L, K, total_time, dt, repeats, boundary)
        meta.update({
            "run_type": "hybrid_mean",
            "seed": int(seed),
            "domain": {
                "L": float(L),
                "K": int(K),
                "pde_multiple": int(pde_multiple),
                "boundary": boundary,
            },
        })

        print("✅ Hybrid Simulation Complete.")
        return res, meta

    def run_hybrid_trajectories(
        self,
        L,
        K,
        pde_multiple,
        total_time,
        dt,
        init_counts,
        repeats=10,
        seed=1,
        parallel=True,
        *,
        boundary: str | None = None,
        time_stride: int = 1,
    ):
        boundary = _validate_boundary(boundary or self.boundary)
        print(f"→ Starting SRCM Hybrid Trajectory Simulation ({repeats} repeats, boundary={boundary})...")

        self.hybrid_model.domain(L=float(L), K=int(K), pde_multiple=int(pde_multiple), boundary=boundary)
        self.hybrid_model.diffusion(**self.diff_coeffs)
        self.hybrid_model.conversion(
            DC_threshold=self.DC_threshold,
            CD_threshold=self.CD_threshold,
            rate=self.conv_rate,
        )
        self.hybrid_model.build(rates=self.rates)

        init_ssa = np.zeros((len(self.species), int(K)), dtype=int)
        for spec, arr in init_counts.items():
            init_ssa[self.species_map[spec]] = arr

        init_pde = np.zeros((len(self.species), int(K) * int(pde_multiple)), dtype=float)

        res = self.hybrid_model.run_trajectories(
            init_ssa,
            init_pde,
            time=float(total_time),
            dt=float(dt),
            repeats=int(repeats),
            seed=int(seed),
            parallel=bool(parallel),
        )

        time_stride = int(time_stride)
        if time_stride < 1:
            raise ValueError(f"time_stride must be >= 1, got {time_stride}")

        if time_stride > 1:
            res = SimulationResults(
                time=res.time[::time_stride],
                ssa=res.ssa[..., ::time_stride],
                pde=res.pde[..., ::time_stride],
                domain=res.domain,
                species=res.species,
            )

        meta = self._get_shared_meta(L, K, total_time, dt, repeats, boundary)
        meta.update({
            "run_type": "hybrid_trajectories",
            "seed": int(seed),
            "time_stride": int(time_stride),
            "domain": {
                "L": float(L),
                "K": int(K),
                "pde_multiple": int(pde_multiple),
                "boundary": boundary,
            },
        })

        print("✅ Hybrid Trajectory Simulation Complete.")
        return res, meta

    # -------------------------
    # FINAL FRAMES
    # -------------------------
    def run_ssa_final_frames(
        self,
        L,
        K,
        total_time,
        dt,
        init_counts,
        n_repeats=10,
        *,
        boundary: str | None = None,
        seed: int | None = None,
        progress: bool = True,
        save_path: str | None = None,
        parallel: bool = False,
        n_jobs: int = -1,
        max_n_jobs: int | None = None,
    ):
        """
        Returns
        -------
        final_ssa : (repeats, n_species, K) int
        t_final : float
        """
        boundary = _validate_boundary(boundary or self.boundary)
        print(f"→ Starting Pure SSA Final Frames ({n_repeats} repeats, boundary={boundary})...")

        rxn = Reaction()
        for r, p, name in self.reactions:
            rxn.add_reaction(r, p, self.rates[name])

        ic = np.zeros((len(self.species), int(K)), dtype=int)
        for spec, arr in init_counts.items():
            ic[self.species_map[spec]] = arr

        ssa_engine = SSA(rxn)
        ssa_engine.set_conditions(
            n_compartments=int(K),
            domain_length=float(L),
            total_time=float(total_time),
            initial_conditions=ic,
            timestep=float(dt),
            Macroscopic_diffusion_rates=[float(self.diff_coeffs[s]) for s in self.species],
            boundary_conditions=boundary,
        )

        R = int(n_repeats)
        base_seed = int(seed) if seed is not None else None

        final_ssa = ssa_engine.run_final_frames(
            n_repeats=R,
            progress=bool(progress),
            parallel=bool(parallel),
            n_jobs=int(n_jobs),
            max_n_jobs=max_n_jobs,
            base_seed=base_seed,
        )

        t_final = float(ssa_engine.timevector[-1])

        meta = self._get_shared_meta(L, K, total_time, dt, n_repeats, boundary)
        meta["run_type"] = "pure_ssa_final_frames"
        meta["parallel"] = bool(parallel)
        meta["n_jobs"] = int(n_jobs)
        meta["max_n_jobs"] = None if max_n_jobs is None else int(max_n_jobs)
        meta["seed"] = None if seed is None else int(seed)

        if save_path is not None:
            np.savez_compressed(
                save_path,
                final_ssa=final_ssa,
                t_final=t_final,
                species=np.array(self.species, dtype=object),
                meta=meta,
            )

        print("✅ SSA Final Frames Complete.")
        return (final_ssa, t_final), meta

    def run_hybrid_final_frames(
        self,
        L,
        K,
        pde_multiple,
        total_time,
        dt,
        init_counts,
        repeats=10,
        seed=1,
        parallel=True,
        *,
        boundary: str | None = None,
        save_path: str | None = None,
        n_jobs: int = -1,
        prefer: str = "processes",
        progress: bool = True,
    ):
        """
        Requires HybridModel.run_repeats_final(...) to exist.
        """
        boundary = _validate_boundary(boundary or self.boundary)
        print(f"→ Starting SRCM Hybrid Final Frames ({repeats} repeats, boundary={boundary})...")

        self.hybrid_model.domain(L=float(L), K=int(K), pde_multiple=int(pde_multiple), boundary=boundary)
        self.hybrid_model.diffusion(**self.diff_coeffs)
        self.hybrid_model.conversion(
            DC_threshold=self.DC_threshold,
            CD_threshold=self.CD_threshold,
            rate=self.conv_rate,
        )
        self.hybrid_model.build(rates=self.rates)

        init_ssa = np.zeros((len(self.species), int(K)), dtype=int)
        for spec, arr in init_counts.items():
            init_ssa[self.species_map[spec]] = arr

        init_pde = np.zeros((len(self.species), int(K) * int(pde_multiple)), dtype=float)

        final_ssa, final_pde, t_final = self.hybrid_model.run_repeats_final(
            init_ssa,
            init_pde,
            time=float(total_time),
            dt=float(dt),
            repeats=int(repeats),
            seed=int(seed),
            parallel=bool(parallel),
            n_jobs=int(n_jobs),
            prefer=str(prefer),
            progress=bool(progress),
            save_path=save_path,
        )

        meta = self._get_shared_meta(L, K, total_time, dt, repeats, boundary)
        meta.update({
            "run_type": "hybrid_final_frames",
            "seed": int(seed),
            "t_final": float(t_final),
            "saved_path": save_path,
            "domain": {
                "L": float(L),
                "K": int(K),
                "pde_multiple": int(pde_multiple),
                "boundary": boundary,
            },
        })

        print("✅ Hybrid Final Frames Complete.")
        return (final_ssa, final_pde, t_final), meta

    def run_ssa_trajectories(
        self,
        L,
        K,
        total_time,
        dt,
        init_counts,
        n_repeats=10,
        *,
        boundary: str | None = None,
        parallel: bool = False,
        n_jobs: int = -1,
        max_n_jobs: int | None = None,
        base_seed: int | None = None,
        time_stride: int = 1,
    ):
        boundary = _validate_boundary(boundary or self.boundary)
        print(f"→ Starting Pure SSA Trajectory Simulation ({n_repeats} repeats, boundary={boundary})...")

        rxn = Reaction()
        for r, p, name in self.reactions:
            rxn.add_reaction(r, p, self.rates[name])

        ic = np.zeros((len(self.species), int(K)), dtype=int)
        for spec, arr in init_counts.items():
            ic[self.species_map[spec]] = arr

        ssa_engine = SSA(rxn)
        ssa_engine.set_conditions(
            n_compartments=int(K),
            domain_length=float(L),
            total_time=float(total_time),
            initial_conditions=ic,
            timestep=float(dt),
            Macroscopic_diffusion_rates=[float(self.diff_coeffs[s]) for s in self.species],
            boundary_conditions=boundary,
        )

        traj = ssa_engine.run_trajectories(
            n_repeats=int(n_repeats),
            parallel=bool(parallel),
            n_jobs=int(n_jobs) if n_jobs is not None else -1,
            max_n_jobs=max_n_jobs,
            base_seed=base_seed,
            progress=True,
        )

        time = np.asarray(ssa_engine.timevector)
        ssa_data = np.transpose(traj, (0, 2, 3, 1)).astype(float, copy=False)

        time_stride = int(time_stride)
        if time_stride < 1:
            raise ValueError(f"time_stride must be >= 1, got {time_stride}")
        if time_stride > 1:
            time = time[::time_stride]
            ssa_data = ssa_data[..., ::time_stride]

        domain = Domain(length=float(L), n_ssa=int(K), pde_multiple=1, boundary=boundary)
        pde_data = np.zeros_like(ssa_data, dtype=float)

        res = SimulationResults(time=time, ssa=ssa_data, pde=pde_data, domain=domain, species=self.species)

        meta = self._get_shared_meta(L, K, total_time, dt, n_repeats, boundary)
        meta["run_type"] = "pure_ssa_trajectories"
        meta["parallel"] = bool(parallel)
        meta["n_jobs"] = int(n_jobs) if n_jobs is not None else -1
        meta["max_n_jobs"] = None if max_n_jobs is None else int(max_n_jobs)
        meta["base_seed"] = None if base_seed is None else int(base_seed)
        meta["time_stride"] = int(time_stride)

        print("✅ SSA Trajectory Simulation Complete.")
        return res, meta