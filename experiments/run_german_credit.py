import argparse, runpy, os
from pathlib import Path

SCRIPTS = {
    "convergence": "convergence.py",
    "local": "local_analysis.py",
    "global": "global_analysis.py",
    "msweep": "m_sweep.py",
    "within_pot": "within_pot.py",
}

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--task", choices=SCRIPTS.keys(), default="convergence")
    args = p.parse_args()

    base = Path(__file__).parent / "german_credit"
    os.chdir(base)  # helps relative paths inside your scripts
    runpy.run_path(str(base / SCRIPTS[args.task]), run_name="__main__")

if __name__ == "__main__":
    main()