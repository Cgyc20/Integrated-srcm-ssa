from pathlib import Path
from integrated_srcm_ssa.presets import load_system
from srcm_engine.results.io import save_npz

# Load system from YAML
sim, cfg = load_system("presets/SI_model.yaml")

# Run simulations
res_ssa, meta_ssa = sim.run_ssa(**cfg["ssa"])
res_hybrid, meta_hybrid = sim.run_hybrid(**cfg["hybrid"])

# Choose output directory + filenames
outdir = Path("results/my_experiment")
outdir.mkdir(parents=True, exist_ok=True)

save_npz(res_ssa, outdir / "ssa_run.npz", meta=meta_ssa)
save_npz(res_hybrid, outdir / "hybrid_run.npz", meta=meta_hybrid)
