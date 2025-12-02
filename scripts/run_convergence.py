# ...existing code...
import json
import subprocess
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
# ...existing code...

# -------------------------------------------------------------------
# User-adjustable settings
# -------------------------------------------------------------------

# Use repository root (two levels up from this script) so paths work
REPO_ROOT = Path(__file__).resolve().parents[1]

# Path to your executable (absolute, relative to repo root)
EXECUTABLE = str(REPO_ROOT / "build" / "MEHCscheme")

# Path to a *base* JSON config (relative to repo root)
BASE_CONFIG_PATH = REPO_ROOT / "data" / "config" / "manufactured.json"

# Where to put temporary modified configs (use out/ directory)
TMP_CONFIG_DIR = REPO_ROOT / "out" / "convergence_configs"

# Which orders and refinements to test
ORDERS = [1]
REFINEMENTS = [0, 1, 2, 3]
# ...existing code...

def load_base_config(path: str | Path) -> dict:
    p = Path(path)
    with p.open("r") as f:
        return json.load(f)


def write_config(cfg: dict, path: str | Path) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w") as f:
        json.dump(cfg, f, indent=4)


def run_simulation(config_path: str | Path) -> str:
    """
    Run MEHCscheme with the given config file and return stdout as string.
    Assumes your executable accepts -c <config> (or --config <config>).
    """
    # pass the option flag and the config path as separate list items
    cmd = [EXECUTABLE, "-c", str(config_path)]
    print("Running:", " ".join(cmd))

    try:
        result = subprocess.run(
            cmd,
            text=True,
            check=False,
        )
    except FileNotFoundError as e:
        raise RuntimeError(f"Failed to run {EXECUTABLE}: {e}")

    return result.stdout
# ...existing code...

def run_convergence():
    base_cfg = load_base_config(BASE_CONFIG_PATH)

    # Ensure temp config directory exists
    TMP_CONFIG_DIR.mkdir(parents=True, exist_ok=True)

    # Data structure: errors[order] = [(h, error), ...]
    errors = {order: [] for order in ORDERS}
    base_dt = base_cfg["dt"]
    for order in ORDERS:
        print(f"\n=== Order {order} ===")
        for ref in REFINEMENTS:
            print(f"  - Running refinement {ref}...")

            cfg = dict(base_cfg)  # shallow copy is fine for this flat JSON
            cfg["order"] = order
            cfg["refinements"] = ref
            cfg["dt"] = base_dt # halve dt with each refinement   

            # Build a specific outputfile name if you like:
            cfg["outputfile"] = f"conv_order{order}_ref{ref}"

            tmp_config_path = TMP_CONFIG_DIR / f"config_order{order}_ref{ref}.json"
            write_config(cfg, tmp_config_path)

            # Run the simulation
            out = run_simulation(tmp_config_path)


# ...existing code...
def main() -> None:
    """Entry point for the convergence script."""
    run_convergence()


if __name__ == "__main__":
    main()
# ...existing code...