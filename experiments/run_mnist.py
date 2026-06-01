import argparse, runpy, os
from pathlib import Path

SCRIPTS = {
    "train": "train_mnist.py",
    "patchtest": "patchtest.py",
    "global": "global.py",
    "heatmaps": "heatmaps.py",
    "globalheat": "globalheat.py",
    "equal_split": "equal_split.py",
    "micro_game": "micro_game.py",
}

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--task", choices=SCRIPTS.keys(), default="patchtest")
    args = p.parse_args()

    base = Path(__file__).parent / "mnist"
    os.chdir(base)
    runpy.run_path(str(base / SCRIPTS[args.task]), run_name="__main__")

if __name__ == "__main__":
    main()