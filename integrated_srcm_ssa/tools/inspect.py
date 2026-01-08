import sys
import numpy as np
import json
from pathlib import Path

def format_stoich(species_data):
    """Turns {'A': 1, 'B': 2} into 'A + 2B'."""
    if not species_data:
        return "∅"
    if isinstance(species_data, dict):
        parts = [f"{v if v > 1 else ''}{k}" for k, v in species_data.items() if v > 0]
        return " + ".join(parts) if parts else "∅"
    return str(species_data)

def inspect_npz(path):
    print(f"\n{'='*75}")
    print(f" FILE: {path.name}")
    print(f"{'='*75}")
    
    try:
        data = np.load(path, allow_pickle=True)
    except Exception as e:
        print(f"Could not read file: {e}")
        return

    # 1. Array Overview
    print(f"{'KEY':<18} | {'TYPE':<8} | {'DETAILS'}")
    print("-" * 75)
    for k in data.files:
        v = data[k]
        if v.shape == ():
            print(f"{k:<18} | scalar   | {v}")
        else:
            peek = f" -> {v.tolist()}" if k == "species" else ""
            # FIX: Convert v.shape to a string before applying the :<12 alignment
            shape_str = str(v.shape)
            print(f"{k:<18} | array    | shape={shape_str:<15} {peek}")

    # 2. Metadata Deep Dive
    meta = {}
    if "meta_json" in data.files:
        print(f"\n--- METADATA (Decoded) ---")
        try:
            meta = json.loads(str(data["meta_json"]))
            
            # Key Params
            if "threshold_particles" in meta:
                print(f"Hybrid Threshold: {meta['threshold_particles']} particles")
            if "conversion_rate" in meta:
                print(f"Conversion Rate:  {meta['conversion_rate']}")
            
            # Diffusion Rates
            if "diffusion_rates" in meta:
                diffs = ", ".join([f"{k}:{v}" for k, v in meta['diffusion_rates'].items()])
                print(f"Diffusion:        {diffs}")

            # 3. REACTION TABLE
            if "reactions" in meta:
                print(f"\nReaction Framework:")
                print(f"  {'#':<3} | {'Reaction Syntax':<25} | {'Rate Name':<12} | {'Value'}")
                print(f"  {'-'*65}")
                
                for i, r in enumerate(meta["reactions"], 1):
                    try:
                        if isinstance(r, dict):
                            reac = format_stoich(r.get('reactants'))
                            prod = format_stoich(r.get('products'))
                            name = r.get('rate_name', 'n/a')
                            val  = r.get('rate', '??')
                        elif isinstance(r, (list, tuple)):
                            reac = format_stoich(r[0])
                            prod = format_stoich(r[1])
                            name = r[2] if len(r) > 2 else 'n/a'
                            val  = meta.get('reaction_rates', {}).get(name, '??')
                        
                        print(f"  [{i:<1}] | {reac:>10} → {prod:<12} | {name:<12} | {val}")
                    except:
                        print(f"  [{i}] Error parsing: {r}")

        except Exception as e:
            print(f"[Error decoding meta_json]: {e}")

    # 3. Domain Summary
    print(f"\n--- SPATIAL/TEMPORAL ---")
    t_key = next((k for k in data.files if "time" in k.lower()), None)
    if t_key:
        t = data[t_key]
        print(f"Time:   {len(t)} steps | {t[0]:.3f} to {t[-1]:.3f}")
    
    # Try to find K and PDE multiple
    K = data.get('n_ssa') or meta.get('domain', {}).get('K')
    if K is not None:
        mult = data.get('pde_multiple') or meta.get('domain', {}).get('pde_multiple', 1)
        print(f"Domain: K={int(K)} (SSA) | PDE Subgrid={int(K*mult)}")

    print(f"{'='*75}\n")

def main():
    paths = [Path(p) for p in sys.argv[1:] if Path(p).exists()]
    if not paths:
        print("Usage: python inspect_results.py <file.npz>")
        return
    for p in paths:
        inspect_npz(p)

if __name__ == "__main__":
    main()