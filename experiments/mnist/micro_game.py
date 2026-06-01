
import os
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
from torchvision.models import resnet18

# -----------------------------
# CONFIG
# -----------------------------
CKPT_PATH = "resnet18_mnist_1vs7.pt"
idx0 = 2
idx1 = 1809

eps_changed = 0.05     # changed pixel threshold
m = 10                 # micro-steps per pixel
N_PERMS = 200          # Monte Carlo permutations of micro-players
CHUNK = 512            # forward-pass chunk size for big batches
SAVE_DIR = "micro_game_results"
os.makedirs(SAVE_DIR, exist_ok=True)

MEAN, STD = 0.1307, 0.3081
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Device:", device)

# -----------------------------
# 0) Load RAW MNIST endpoints (pixels in [0,1])
# -----------------------------
test_raw = datasets.MNIST(root="data", train=False, download=True, transform=transforms.ToTensor())
x0, y0 = test_raw[idx0]
x1, y1 = test_raw[idx1]
x0 = x0[0].clone()  # [28,28]
x1 = x1[0].clone()  # [28,28]

# Changed pixels define the feature set N
diff = (x1 - x0).abs()
mask_feats = diff > eps_changed
coords = torch.nonzero(mask_feats, as_tuple=False)   # [k,2]
k = coords.shape[0]
print(f"All changed pixels with |x1-x0| > {eps_changed}: k={k}")

# Delta only on changed pixels
delta = torch.zeros_like(x0)
delta[mask_feats] = (x1 - x0)[mask_feats]

# Path end for this chosen feature set (x0 with changed pixels moved fully to x1)
x_end = (x0 + delta).clone()

# -----------------------------
# 1) Load trained ResNet
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
print("Loaded checkpoint:", CKPT_PATH)
print("Best test acc:", ckpt.get("best_test_acc", None))

def normalize(x01):
    return (x01 - MEAN) / STD

@torch.no_grad()
def g_p7(batch_x01):
    """g(x)=P(7|x) for a batch [B,1,28,28] in [0,1]. returns [B] on CPU."""
    logits = model(normalize(batch_x01).to(device))
    probs = torch.softmax(logits, dim=1)
    return probs[:, 1].detach().cpu()

@torch.no_grad()
def p7_single(x01_28x28):
    b = x01_28x28.unsqueeze(0).unsqueeze(0)  # [1,1,28,28]
    return float(g_p7(b)[0].item())

p7_x0 = p7_single(x0)
p7_xend = p7_single(x_end)
Delta = p7_xend - p7_x0
print(f"P7(x0)   = {p7_x0:.6f}")
print(f"P7(xend) = {p7_xend:.6f}  (x0 with ALL changed pixels moved to x1)")
print(f"Delta    = {Delta:.6f}")

# -----------------------------
# 2) GLOBAL MICRO-GAME Shapley on micro-players (i,s)
#    Player set: N' = {(i,s): i=1..k, s=1..m}  so n = k*m
#    v(A) = g(x_{p(A)}) - g(x0)
#    Permutation estimator over random shuffles of the multiset of replicas
# -----------------------------
n = k * m
print(f"Micro-player count: n = k*m = {k}*{m} = {n}")

# Precompute per-pixel step increment: each time pixel i appears, add delta_i/m
delta_vec = delta[mask_feats].float()          # [k]
delta_step_vec = (delta_vec / float(m)).float()  # [k]

# Multiset of micro-players encoded as pixel indices [0..k-1] repeated m times
base_micro = torch.repeat_interleave(torch.arange(k, dtype=torch.long), repeats=m)  # [n]

attrib = torch.zeros(k, dtype=torch.float64)  # final per-pixel totals (sum over replicas)

for tperm in range(N_PERMS):
    # random permutation of micro-players (multiset shuffle)
    order = base_micro[torch.randperm(n)]  # [n], entries in {0..k-1}

    # Build the (n+1) prefix images in one tensor batch
    # x_0 = x0, x_t = x0 + sum_{j in first t micro-players} (delta_j/m) at that pixel
    batch = torch.empty((n + 1, 1, 28, 28), dtype=torch.float32)
    cur = x0.clone().float()
    batch[0, 0] = cur

    for s in range(1, n + 1):
        pi = int(order[s - 1].item())     # pixel index in [0..k-1]
        rr, cc = coords[pi].tolist()      # pixel location
        cur[rr, cc] += float(delta_step_vec[pi].item())
        batch[s, 0] = cur

    # Evaluate g on the whole path batch (chunked)
    vals = torch.empty(n + 1, dtype=torch.float32)
    for start in range(0, n + 1, CHUNK):
        end = min(n + 1, start + CHUNK)
        vals[start:end] = g_p7(batch[start:end])

    marg = (vals[1:] - vals[:-1]).double()  # [n] marginal contributions per micro-step

    # Aggregate micro-steps to per-pixel totals for this permutation
    attrib.index_add_(0, order, marg)

    if (tperm + 1) % max(1, N_PERMS // 10) == 0:
        print(f"  perms {tperm+1:>4d}/{N_PERMS} done")

# Average over permutations (Shapley MC estimator)
attrib /= float(N_PERMS)

print(f"Sum micro-game Shapley totals = {attrib.sum().item():.6f} (should approx match Delta)")
print(f"Gap (sum - Delta)             = {(attrib.sum().item() - Delta):.6f}")

# -----------------------------
# 3) Heatmap (28x28)
# -----------------------------
heat = torch.zeros((28, 28), dtype=torch.float32)
for j, (r, c) in enumerate(coords.tolist()):
    heat[r, c] = float(attrib[j].item())

heat_np = heat.numpy()
vmax = float(np.max(np.abs(heat_np)) + 1e-12)

# -----------------------------
# 4) Plot + save
# -----------------------------
plt.figure(figsize=(12, 3))

plt.subplot(1, 4, 1)
plt.imshow(x0.numpy(), cmap="gray", vmin=0, vmax=1)
plt.title(f"x0 (idx={idx0}, y={y0})\nP7={p7_x0:.3f}")
plt.axis("off")

plt.subplot(1, 4, 2)
plt.imshow(x_end.numpy(), cmap="gray", vmin=0, vmax=1)
plt.title(f"xend (m={m}) uses changed pixels\nP7={p7_xend:.3f}")
plt.axis("off")

plt.subplot(1, 4, 3)
plt.imshow(heat_np, cmap="RdBu", vmin=-vmax, vmax=vmax)
plt.title(f"Global micro-game Shapley\n(m={m}, perms={N_PERMS})")
plt.axis("off")
plt.colorbar(fraction=0.046, pad=0.04)

plt.subplot(1, 4, 4)
plt.imshow(x_end.numpy(), cmap="gray", vmin=0, vmax=1)
plt.imshow(heat_np, cmap="RdBu", vmin=-vmax, vmax=vmax, alpha=0.75)
plt.title("Overlay on xend")
plt.axis("off")

plt.tight_layout()
out_path = os.path.join(SAVE_DIR, f"micro_game_shap_m{m}_perms{N_PERMS}_idx{idx0}_to_{idx1}.png")
plt.savefig(out_path, dpi=200)
print("Saved:", out_path)
plt.show()


# -----------------------------
# 5) SAVE CACHE
# -----------------------------
CACHE_DIR = "cache_attribs"
os.makedirs(CACHE_DIR, exist_ok=True)

cache_path = os.path.join(
    CACHE_DIR,
    f"microgame_idx{idx0}_to_{idx1}_eps{eps_changed}_m{m}_perms{N_PERMS}.pt"
)

save_obj = {
    "method": "global_microgame_shapley",
    "idx0": idx0,
    "idx1": idx1,
    "eps_changed": float(eps_changed),
    "m": int(m),
    "N_PERMS": int(N_PERMS),
    "y0": int(y0),
    "y1": int(y1),
    "p7_x0": float(p7_x0),
    "p7_xend": float(p7_xend),
    "Delta": float(Delta),

    # endpoints + mask
    "x0": x0.detach().cpu(),          # [28,28]
    "x1": x1.detach().cpu(),          # [28,28]
    "x_end": x_end.detach().cpu(),    # [28,28] (x0 + delta)
    "mask_feats": mask_feats.detach().cpu(),      # [28,28] bool
    "coords": coords.detach().cpu(),              # [k,2]
    "delta": delta.detach().cpu(),                # [28,28]

    # attributions
    "attrib_vec": attrib.detach().cpu(),          # [k] float64
    "heat": heat.detach().cpu(),                  # [28,28] float32
}

torch.save(save_obj, cache_path)
print("Saved cache:", cache_path)

# also save heatmap as .npy
np.save(cache_path.replace(".pt", "_heat.npy"), heat_np)