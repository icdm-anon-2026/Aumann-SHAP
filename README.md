# Aumann-SHAP Experiments

This repository contains the code required to reproduce the experiments presented in the **Aumann-SHAP** study.

It includes evaluation pipelines for:
- **German Credit** (tabular counterfactual attribution + interaction analysis)
- **MNIST (1 → 7)** (pixel attributions, heatmaps, patch test, global explanations)
<img width="1393" height="331" alt="image" src="https://github.com/user-attachments/assets/e8e5f9d5-4b84-447d-a8e3-9af4aa3fa148" />

---

## Installation

1) Clone the repository:
https://github.com/ecml-anon-2026/Aumann-SHAP

2) Install dependencies (from the repository root folder):
pip install -r requirements.txt

---

## Quickstart (Primary runs)

German Credit (primary):
python experiments/run_german_credit.py --task local

MNIST (primary):
python experiments/run_mnist.py --task patchtest

---

## Available tasks

German Credit:
- local
- within_pot
- global
- msweep
- convergence

MNIST:
- train
- equal_split
- micro_game
- heatmaps
- patchtest
- global
- globalheat

---

## Reproducibility / Execution order (IMPORTANT)

Some scripts depend on files created by earlier scripts (caches/artifacts/checkpoints).  
Use the documented order here:
- docs/REPRODUCIBILITY.md
- experiments/german_credit/README.md
- experiments/mnist/README.md

---

## Notes

- No datasets are committed to this repository.
- German Credit uses a pretrained cached model included at:
  experiments/german_credit/cache/models_split_rs1.joblib
- MNIST training can be skipped if the checkpoint already exists.

---

## Package (optional)

You can install this repository as a local package:

pip install -e .

Minimal usage:

- Exact grid-state backend (tabular / small k): returns (Totals, Within-pot)
  from aumann_shap import explain
  totals, within_pot = explain(model, x0, x1, backend="grid_state", m=5)

- Monte Carlo backend (large k / pixels): returns (Totals, None)
  totals, within_pot = explain(model, x0, x1, backend="mc", m=10, n_perms=200, seed=0)

(Within-pot is only returned for the exact grid-state backend.)
## Usage for ML (Tabular data)

### 1) Minimal example (exact grid-state micro-game Shapley)

```python
import numpy as np
import pandas as pd
from aumann_shap import explain

# Any callable f(pd.Series) -> float works
def g(x: pd.Series) -> float:
    return float(np.tanh(0.8*x["x1"] + 0.4*x["x2"] + 0.6*x["x1"]*x["x2"]))

x0 = pd.Series({"x1": 0.0, "x2": 0.0})
x1 = pd.Series({"x1": 1.0, "x2": 1.0})

totals, within_pot = explain(g, x0, x1, backend="grid_state", m=5)

print("Totals (sum to Δ):")
print(totals)
print("\nWithin-pot:")
print(within_pot)
```
---

### 2) Monte Carlo micro-game Shapley (Totals only)

```python
import numpy as np
import pandas as pd
from aumann_shap import explain

def g(x: pd.Series) -> float:
    return float(np.tanh(0.8*x["x1"] + 0.4*x["x2"] + 0.6*x["x1"]*x["x2"]))

x0 = pd.Series({"x1": 0.0, "x2": 0.0})
x1 = pd.Series({"x1": 1.0, "x2": 1.0})

totals, within_pot = explain(
    g, x0, x1,
    backend="mc",
    m=10,
    n_perms=300,
    seed=0
)

print("Totals (MC estimate):")
print(totals)
print("within_pot:", within_pot)  # None for MC
```
If your model supports batch prediction (sklearn/xgboost/torch), explain uses it automatically; otherwise it falls back to per-row evaluation (may be slower).
---
## Usage for Vision (MNIST)
requires running train_mnist.py (produces the resnet18_mnist_1vs7.pt).
```python
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
from torchvision.models import resnet18
from matplotlib.colors import TwoSlopeNorm
from aumann_shap import explain

# -----------------------------
# Config
# -----------------------------
CKPT_PATH = "resnet18_mnist_1vs7.pt"
idx0, idx1 = 2, 1809
eps_changed = 0.05

m = 10
n_perms = 50
seed = 0

MEAN, STD = 0.1307, 0.3081
device = "cuda" if torch.cuda.is_available() else "cpu"

# -----------------------------
# Load MNIST endpoints in [0,1]
# -----------------------------
test_raw = datasets.MNIST(
    root="data", train=False, download=True, transform=transforms.ToTensor()
)
x0_t, y0 = test_raw[idx0]
x1_t, y1 = test_raw[idx1]
x0 = x0_t[0].numpy()
x1 = x1_t[0].numpy()

# Use only "changed" pixels as the counterfactual endpoint (x_end)
mask = np.abs(x1 - x0) > eps_changed
x_end = x0.copy()
x_end[mask] = x1[mask]

# -----------------------------
# Load trained ResNet-18 (1 vs 7)
# -----------------------------
def build_model():
    mdl = resnet18(weights=None)
    mdl.conv1 = nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1, bias=False)
    mdl.maxpool = nn.Identity()
    mdl.fc = nn.Linear(mdl.fc.in_features, 2)  # 0="1", 1="7"
    return mdl

model = build_model().to(device)
ckpt = torch.load(CKPT_PATH, map_location=device)
model.load_state_dict(ckpt["model_state"])
model.eval()

@torch.no_grad()
def model_batch(X: np.ndarray) -> np.ndarray:
    """
    X: (B, 784) float array in [0,1]
    returns: (B,) probabilities P(7|x)
    """
    X = torch.from_numpy(X.astype(np.float32)).to(device).view(-1, 1, 28, 28)
    X = (X - MEAN) / STD
    logits = model(X)
    p7 = torch.softmax(logits, dim=1)[:, 1]
    return p7.detach().cpu().numpy()

# -----------------------------
# Aumann-SHAP (MC micro-game Shapley totals)
# -----------------------------
totals, _ = explain(
    model=model,                         # only used for schema; scoring comes from model_batch
    x0=x0.reshape(-1),
    x1=x_end.reshape(-1),
    backend="mc",
    model_batch=model_batch,
    m=m,
    n_perms=n_perms,
    seed=seed,
)

heat = totals.reshape(28, 28).astype(float)

# Keep ONLY changed pixels (everything else becomes NaN -> white)
heat_only_changed = heat.copy()
heat_only_changed[~mask] = np.nan

# Diverging colormap with white at 0, and NaNs rendered white
cmap = plt.get_cmap("RdBu").copy()
cmap.set_bad(color="white")

# Symmetric scaling so 0 is the white center
vmax = np.nanmax(np.abs(heat_only_changed)) + 1e-12
norm = TwoSlopeNorm(vcenter=0.0, vmin=-vmax, vmax=vmax)

plt.figure(figsize=(10, 3))
plt.subplot(1, 3, 1); plt.imshow(x0, cmap="gray", vmin=0, vmax=1);    plt.title("Baseline $x_0$"); plt.axis("off")
plt.subplot(1, 3, 2); plt.imshow(x_end, cmap="gray", vmin=0, vmax=1); plt.title("Endpoint $x_{end}$"); plt.axis("off")

plt.subplot(1, 3, 3)
plt.imshow(heat_only_changed, cmap=cmap, norm=norm, interpolation="nearest")
plt.title("Micro-game Shapley (changed pixels only)")
plt.axis("off")
plt.colorbar(fraction=0.046, pad=0.04)

plt.tight_layout()
plt.show()
```
<p align="center">
  <img src="https://github.com/user-attachments/assets/85bd4f42-26da-4277-9596-3400a05847ca" width="800" height="200" />
</p>

## Project structure

docs/                 reproducibility + anonymity notes  
experiments/          entrypoints + dataset-specific scripts  
src/aumann_shap/      package source code  
requirements.txt      dependencies  
LICENSE               MIT license
