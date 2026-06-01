import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
from torchvision.models import resnet18

# -----------------------------
# 0) Load RAW MNIST endpoints
# -----------------------------
test_raw = datasets.MNIST(root="data", train=False, download=True, transform=transforms.ToTensor())

idx0 = 2      # baseline x0
idx1 = 1809   # counterfactual x1

x0, y0 = test_raw[idx0]  # [1,28,28] in [0,1]
x1, y1 = test_raw[idx1]

x0 = x0[0]  # [28,28]
x1 = x1[0]  # [28,28]

# -----------------------------
# Select FEATURES = ALL changed pixels (|x1-x0| > eps)
# -----------------------------
eps = 0.05
diff = (x1 - x0).abs()
mask_feats = (diff > eps)                    # [28,28] bool
feat_coords = torch.nonzero(mask_feats)      # [k,2]
k = feat_coords.shape[0]
print(f"All changed pixels with |x1-x0| > {eps}: k={k}")

# -----------------------------
# 1) Load trained ResNet
# -----------------------------
CKPT_PATH = "resnet18_mnist_1vs7.pt"
MEAN, STD = 0.1307, 0.3081
device = "cuda" if torch.cuda.is_available() else "cpu"

def build_model():
    m = resnet18(weights=None)
    m.conv1 = nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1, bias=False)
    m.maxpool = nn.Identity()
    m.fc = nn.Linear(m.fc.in_features, 2)  # 0="1", 1="7"
    return m

model = build_model().to(device)
ckpt = torch.load(CKPT_PATH, map_location=device)
model.load_state_dict(ckpt["model_state"])
model.eval()

def normalize(x01):
    return (x01 - MEAN) / STD

@torch.no_grad()
def g_p7(batch_x01):
    """g(x) = P(7|x) for the binary model (class 1 = '7')."""
    logits = model(normalize(batch_x01).to(device))
    probs = torch.softmax(logits, dim=1)
    return probs[:, 1].detach().cpu()  # [B]

# Build x_end = x0 but with ALL changed pixels switched to x1 values
x0_full = x0.unsqueeze(0).unsqueeze(0)  # [1,1,28,28]
x_end = x0.clone()
for (r, c) in feat_coords.tolist():
    x_end[r, c] = x1[r, c]
x_end_full = x_end.unsqueeze(0).unsqueeze(0)

p7_x0 = float(g_p7(x0_full)[0].item())
p7_xend = float(g_p7(x_end_full)[0].item())
print(f"P7(x0)   = {p7_x0:.6f}")
print(f"P7(xend) = {p7_xend:.6f}  (x0 with ALL changed pixels swapped)")
print(f"Delta = {p7_xend - p7_x0:.6f}")

# -----------------------------
# 2) Equal-split Shapley (corner-game) via permutation sampling
# -----------------------------
N_PERMS = 400  # increase if you want smoother (800/1500/etc.)
attrib = torch.zeros(k, dtype=torch.float64)

x0_base = x0.clone()
x1_target = x1.clone()

for t in range(N_PERMS):
    perm = torch.randperm(k)

    # Build the (k+1) prefix images in ONE batch
    batch = torch.zeros((k + 1, 1, 28, 28), dtype=torch.float32)
    cur = x0_base.clone()
    batch[0, 0] = cur

    for s in range(1, k + 1):
        rr, cc = feat_coords[perm[s - 1]].tolist()
        cur = cur.clone()
        cur[rr, cc] = x1_target[rr, cc]
        batch[s, 0] = cur

    vals = g_p7(batch)                 # [k+1]
    marg = vals[1:] - vals[:-1]        # [k]
    attrib[perm] += marg.double()

attrib /= float(N_PERMS)

print(f"Sum Shapley attributions = {attrib.sum().item():.6f} (should match Delta above)")

# -----------------------------
# 3) Heatmap (28x28) from attributions
# -----------------------------
heat = torch.zeros((28, 28), dtype=torch.float32)
for j, (r, c) in enumerate(feat_coords.tolist()):
    heat[r, c] = float(attrib[j].item())

heat_np = heat.numpy()
vmax = float(np.max(np.abs(heat_np)) + 1e-12)

# -----------------------------
# 4) Plot
# -----------------------------
plt.figure(figsize=(12, 3))

plt.subplot(1, 4, 1)
plt.imshow(x0.numpy(), cmap="gray", vmin=0, vmax=1)
plt.title(f"x0 (idx={idx0}, y={y0})\nP7={p7_x0:.3f}")
plt.axis("off")

plt.subplot(1, 4, 2)
plt.imshow(x1.numpy(), cmap="gray", vmin=0, vmax=1)
plt.title(f"x1 (idx={idx1}, y={y1})")
plt.axis("off")

plt.subplot(1, 4, 3)
plt.imshow(heat_np, cmap="RdBu", vmin=-vmax, vmax=vmax)
plt.title("Equal-split (Shapley)\nBLUE = pushes to 7")
plt.axis("off")
plt.colorbar(fraction=0.046, pad=0.04)

plt.subplot(1, 4, 4)
plt.imshow(x1.numpy(), cmap="gray", vmin=0, vmax=1)
plt.imshow(heat_np, cmap="RdBu", vmin=-vmax, vmax=vmax, alpha=0.75)
plt.title("Overlay on x1")
plt.axis("off")

plt.tight_layout()
plt.show()

# -----------------------------
# 5) SAVE CACHE
# -----------------------------
import os

CACHE_DIR = "cache_attribs"
os.makedirs(CACHE_DIR, exist_ok=True)

cache_path = os.path.join(
    CACHE_DIR,
    f"eqsplit_idx{idx0}_to_{idx1}_eps{eps}_perms{N_PERMS}.pt"
)

save_obj = {
    "method": "equal_split_shapley",
    "idx0": idx0,
    "idx1": idx1,
    "eps_changed": float(eps),
    "N_PERMS": int(N_PERMS),
    "y0": int(y0),
    "y1": int(y1),
    "p7_x0": float(p7_x0),
    "p7_xend": float(p7_xend),
    "Delta": float(p7_xend - p7_x0),

    # endpoints + mask
    "x0": x0.detach().cpu(),          # [28,28]
    "x1": x1.detach().cpu(),          # [28,28]
    "x_end": x_end.detach().cpu(),    # [28,28] (x0 with changed pixels swapped)
    "mask_feats": mask_feats.detach().cpu(),      # [28,28] bool
    "coords": feat_coords.detach().cpu(),         # [k,2]

    # attributions
    "attrib_vec": attrib.detach().cpu(),          # [k] float64 (per changed pixel)
    "heat": heat.detach().cpu(),                  # [28,28] float32
}

torch.save(save_obj, cache_path)
print("Saved cache:", cache_path)

# also save heatmap as .npy for quick loading in plotting scripts
np.save(cache_path.replace(".pt", "_heat.npy"), heat_np)
