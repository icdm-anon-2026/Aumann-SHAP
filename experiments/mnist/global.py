import os
import math
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
from torchvision.models import resnet18

# -----------------------------
# CONFIG
# -----------------------------
CKPT_PATH = "resnet18_mnist_1vs7.pt"
OUT_DIR = "figures_global"
os.makedirs(OUT_DIR, exist_ok=True)

CACHE_DIR = "cache_global"
os.makedirs(CACHE_DIR, exist_ok=True)

SEED = 123
torch.manual_seed(SEED)
np.random.seed(SEED)

# Pairing
MAX_PAIRS = 200          # set None to try all test "1"s (can be expensive)
PAIR_BATCH = 64          # batch size for NN pairing

# Changed-pixel support
EPS_CHANGED = 0.05

# Equal-split Shapley (MC over changed pixels)
EQ_N_PERMS = 80          # increase for lower MC variance
EQ_CHUNK = 256           # model batch chunk

# Micro-game Shapley (MC over micro-players)
MICRO_M = 4              # micro-steps per pixel (m); increase for finer grid, more cost
MICRO_N_PERMS = 40       # increase for lower MC variance
MICRO_CHUNK = 256        # model batch chunk

# Equal Surplus (deterministic) on corner-game
# (no MC parameters)

# Plot settings
PLOT_ABS = True          # plot mean |attribution| (recommended for global aggregation)
SAVE_DPI = 280

MEAN, STD = 0.1307, 0.3081
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Device:", device)

# -----------------------------
# Model + g(x)=P(7|x)
# -----------------------------
def build_model():
    mdl = resnet18(weights=None)
    mdl.conv1 = nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1, bias=False)
    mdl.maxpool = nn.Identity()
    mdl.fc = nn.Linear(mdl.fc.in_features, 2)  # 0="1", 1="7"
    return mdl

def normalize(x01_bchw):
    return (x01_bchw - MEAN) / STD

@torch.no_grad()
def g_p7_batch(model, xs_28x28, chunk=256):
    """
    xs_28x28: torch.FloatTensor [B,28,28] in [0,1]
    returns: torch.FloatTensor [B] on CPU
    """
    B = xs_28x28.shape[0]
    out = torch.empty((B,), dtype=torch.float32)
    for s in range(0, B, chunk):
        e = min(B, s + chunk)
        b = xs_28x28[s:e].unsqueeze(1)  # [b,1,28,28]
        logits = model(normalize(b).to(device))
        probs = torch.softmax(logits, dim=1)[:, 1]
        out[s:e] = probs.detach().cpu()
    return out

@torch.no_grad()
def g_p7_single(model, x_28x28):
    return float(g_p7_batch(model, x_28x28.unsqueeze(0), chunk=1)[0].item())

# -----------------------------
# Data: MNIST test set (raw pixels in [0,1])
# -----------------------------
test_raw = datasets.MNIST(root="data", train=False, download=True, transform=transforms.ToTensor())
X = torch.stack([test_raw[i][0][0] for i in range(len(test_raw))], dim=0)  # [10000,28,28]
Y = torch.tensor([test_raw[i][1] for i in range(len(test_raw))], dtype=torch.long)

idx_ones = torch.where(Y == 1)[0]
idx_sevs = torch.where(Y == 7)[0]
X1 = X[idx_ones]  # [n1,28,28]
X7 = X[idx_sevs]  # [n7,28,28]

print(f"Test-set: n1={len(idx_ones)} (digit=1), n7={len(idx_sevs)} (digit=7)")

# -----------------------------
# Pairing: for each "1", nearest "7" in L2 pixel space
# -----------------------------
def pair_nearest_7_for_1s(X1, X7, max_pairs=None, batch=64):
    """
    Returns:
      pairs: list of tuples (i1, i7) where i1 indexes X1, i7 indexes X7 (local indices)
    """
    n1 = X1.shape[0]
    n7 = X7.shape[0]

    # optionally subsample 1s (deterministically)
    if max_pairs is not None and max_pairs < n1:
        # take a fixed permutation for reproducibility
        perm = torch.randperm(n1)
        sel = perm[:max_pairs]
        X1_sel = X1[sel]
        sel_map = sel
    else:
        X1_sel = X1
        sel_map = None

    A = X1_sel.reshape(X1_sel.shape[0], -1).float().to(device)  # [B,784]
    B = X7.reshape(n7, -1).float().to(device)                   # [n7,784]
    B_sq = (B * B).sum(dim=1)                                   # [n7]

    pairs = []
    for s in range(0, A.shape[0], batch):
        e = min(A.shape[0], s + batch)
        Ab = A[s:e]                         # [b,784]
        Ab_sq = (Ab * Ab).sum(dim=1)        # [b]

        # dist^2 = ||a||^2 + ||b||^2 - 2 a·b
        dot = Ab @ B.t()                    # [b,n7]
        dist2 = Ab_sq[:, None] + B_sq[None, :] - 2.0 * dot
        nn = torch.argmin(dist2, dim=1)     # [b]

        for j in range(nn.numel()):
            i1_local = (s + j)
            i7_local = int(nn[j].item())
            if sel_map is not None:
                i1_local = int(sel_map[i1_local].item())
            pairs.append((int(i1_local), i7_local))

    return pairs

pairs = pair_nearest_7_for_1s(X1, X7, max_pairs=MAX_PAIRS, batch=PAIR_BATCH)
print(f"Paired {len(pairs)} baselines using NN(1→7)")

# Save pairing indices (global indices in original test set) for reuse
pair_cache_path = os.path.join(CACHE_DIR, f"pairs_nn1to7_max{MAX_PAIRS}_seed{SEED}.pt")
torch.save({
    "seed": SEED,
    "max_pairs": MAX_PAIRS,
    "pairs_local": pairs,
    "idx_ones": idx_ones,
    "idx_sevs": idx_sevs,
}, pair_cache_path)
print("Saved pairing cache:", pair_cache_path)

# -----------------------------
# Attribution primitives
# -----------------------------
def changed_support(x0, x1, eps):
    diff = (x1 - x0).abs()
    mask = diff > eps
    coords = torch.nonzero(mask, as_tuple=False)  # [k,2]
    return mask, coords

def x_end_from_mask(x0, x1, mask):
    out = x0.clone()
    out[mask] = x1[mask]
    return out

def equal_split_shapley_heat(model, x0, x1, coords, n_perms, chunk):
    """
    Corner-game Shapley on changed pixels via permutation sampling.
    Returns heat [28,28] float32 and attrib_vec [k] float64
    """
    k = coords.shape[0]
    if k == 0:
        return torch.zeros((28, 28), dtype=torch.float32), torch.zeros((0,), dtype=torch.float64)

    attrib = torch.zeros(k, dtype=torch.float64)

    for _ in range(n_perms):
        perm = torch.randperm(k)

        batch = torch.empty((k + 1, 28, 28), dtype=torch.float32)
        cur = x0.clone().float()
        batch[0] = cur

        for s in range(1, k + 1):
            rr, cc = coords[perm[s - 1]].tolist()
            cur = cur.clone()
            cur[int(rr), int(cc)] = x1[int(rr), int(cc)]
            batch[s] = cur

        vals = g_p7_batch(model, batch, chunk=chunk)     # [k+1] float32
        marg = (vals[1:] - vals[:-1]).double()          # [k] float64
        attrib[perm] += marg

    attrib /= float(n_perms)

    heat = torch.zeros((28, 28), dtype=torch.float32)
    for j, (r, c) in enumerate(coords.tolist()):
        heat[int(r), int(c)] = float(attrib[j].item())
    return heat, attrib

def micro_game_shapley_heat(model, x0, x1, mask, coords, m, n_perms, chunk):
    """
    Global micro-game Shapley on micro-players (i,s) via permutation sampling.
    Returns heat [28,28] float32 and attrib_vec [k] float64 (per changed pixel total)
    """
    k = coords.shape[0]
    if k == 0:
        return torch.zeros((28, 28), dtype=torch.float32), torch.zeros((0,), dtype=torch.float64)

    delta = torch.zeros_like(x0)
    delta[mask] = (x1 - x0)[mask]
    delta_vec = delta[mask].float()                      # [k]
    delta_step = (delta_vec / float(m)).float()         # [k]

    n = k * m
    base_micro = torch.repeat_interleave(torch.arange(k, dtype=torch.long), repeats=m)  # [n]

    attrib = torch.zeros(k, dtype=torch.float64)

    for _ in range(n_perms):
        order = base_micro[torch.randperm(n)]  # [n] entries in {0..k-1}

        batch = torch.empty((n + 1, 28, 28), dtype=torch.float32)
        cur = x0.clone().float()
        batch[0] = cur

        for s in range(1, n + 1):
            pi = int(order[s - 1].item())
            rr, cc = coords[pi].tolist()
            cur[ int(rr), int(cc) ] += float(delta_step[pi].item())
            batch[s] = cur

        vals = g_p7_batch(model, batch, chunk=chunk)          # [n+1]
        marg = (vals[1:] - vals[:-1]).double()                # [n]
        attrib.index_add_(0, order, marg)

    attrib /= float(n_perms)

    heat = torch.zeros((28, 28), dtype=torch.float32)
    for j, (r, c) in enumerate(coords.tolist()):
        heat[int(r), int(c)] = float(attrib[j].item())
    return heat, attrib

def equal_surplus_heat(model, x0, x1, coords, eps_mask, chunk):
    """
    Equal Surplus in Δ-units on the corner-value game induced by the changed set N:
      g_i = v({i}) - v(∅)
      R = Δ - sum_i g_i
      ES_i = g_i + R/|N|
    Deterministic given the model and endpoints.
    Returns heat [28,28] float32 and attrib_vec [k] float64
    """
    k = coords.shape[0]
    if k == 0:
        return torch.zeros((28, 28), dtype=torch.float32), torch.zeros((0,), dtype=torch.float64)

    # baseline score and endpoint score (on x_end supported on N)
    x_end = x_end_from_mask(x0, x1, eps_mask)
    p0 = g_p7_single(model, x0)
    pend = g_p7_single(model, x_end)
    Delta = float(pend - p0)

    # singleton images: x^{ {i} }
    batch = torch.empty((k, 28, 28), dtype=torch.float32)
    for j in range(k):
        rr, cc = coords[j].tolist()
        xi = x0.clone().float()
        xi[int(rr), int(cc)] = x1[int(rr), int(cc)]
        batch[j] = xi

    vals = g_p7_batch(model, batch, chunk=chunk).double().numpy()  # [k]
    g_i = vals - p0                                                # [k]
    R = Delta - float(g_i.sum())
    es = g_i + (R / float(k))                                      # [k]

    attrib = torch.from_numpy(es).to(torch.float64)

    heat = torch.zeros((28, 28), dtype=torch.float32)
    for j, (r, c) in enumerate(coords.tolist()):
        heat[int(r), int(c)] = float(attrib[j].item())
    return heat, attrib

# -----------------------------
# Load model
# -----------------------------
model = build_model().to(device)
ckpt = torch.load(CKPT_PATH, map_location=device)
model.load_state_dict(ckpt["model_state"])
model.eval()
print("Loaded:", CKPT_PATH, "| best_test_acc:", ckpt.get("best_test_acc", None))

# -----------------------------
# Run global aggregation
# -----------------------------
sum_heat_eq = torch.zeros((28, 28), dtype=torch.float64)
sum_heat_mc = torch.zeros((28, 28), dtype=torch.float64)
sum_heat_es = torch.zeros((28, 28), dtype=torch.float64)

sum_abs_eq = torch.zeros((28, 28), dtype=torch.float64)
sum_abs_mc = torch.zeros((28, 28), dtype=torch.float64)
sum_abs_es = torch.zeros((28, 28), dtype=torch.float64)

counts = 0
stats = {
    "k_changed": [],
    "Delta": [],
}

for t, (i1_local, i7_local) in enumerate(pairs, start=1):
    x0 = X1[i1_local].clone()
    x1 = X7[i7_local].clone()

    mask, coords = changed_support(x0, x1, EPS_CHANGED)
    k = coords.shape[0]
    if k == 0:
        continue

    x_end = x_end_from_mask(x0, x1, mask)
    Delta = g_p7_single(model, x_end) - g_p7_single(model, x0)

    # Equal-split (MC)
    heat_eq, _ = equal_split_shapley_heat(
        model, x0, x1, coords,
        n_perms=EQ_N_PERMS, chunk=EQ_CHUNK
    )

    # Micro-game (MC)
    heat_mc, _ = micro_game_shapley_heat(
        model, x0, x1, mask, coords,
        m=MICRO_M, n_perms=MICRO_N_PERMS, chunk=MICRO_CHUNK
    )

    # Equal Surplus (deterministic)
    heat_es, _ = equal_surplus_heat(
        model, x0, x1, coords, eps_mask=mask, chunk=EQ_CHUNK
    )

    # accumulate
    sum_heat_eq += heat_eq.double()
    sum_heat_mc += heat_mc.double()
    sum_heat_es += heat_es.double()

    sum_abs_eq += heat_eq.double().abs()
    sum_abs_mc += heat_mc.double().abs()
    sum_abs_es += heat_es.double().abs()

    counts += 1
    stats["k_changed"].append(int(k))
    stats["Delta"].append(float(Delta))

    if t % max(1, len(pairs) // 10) == 0:
        print(f"  processed {t}/{len(pairs)} pairs (kept {counts})")

print(f"Done. Used {counts} pairs with k>0.")
print(f"Mean k_changed = {np.mean(stats['k_changed']):.1f} | median = {np.median(stats['k_changed']):.0f}")
print(f"Mean Delta     = {np.mean(stats['Delta']):.4f} | median = {np.median(stats['Delta']):.4f}")

# Means
mean_heat_eq = (sum_heat_eq / max(1, counts)).float()
mean_heat_mc = (sum_heat_mc / max(1, counts)).float()
mean_heat_es = (sum_heat_es / max(1, counts)).float()

mean_abs_eq = (sum_abs_eq / max(1, counts)).float()
mean_abs_mc = (sum_abs_mc / max(1, counts)).float()
mean_abs_es = (sum_abs_es / max(1, counts)).float()

# Save cache
out_cache = os.path.join(
    CACHE_DIR,
    f"global_meanheat_nn1to7_pairs{counts}_eps{EPS_CHANGED}_"
    f"eqPerms{EQ_N_PERMS}_microM{MICRO_M}_microPerms{MICRO_N_PERMS}_seed{SEED}.pt"
)
torch.save({
    "counts": counts,
    "eps_changed": float(EPS_CHANGED),
    "eq_n_perms": int(EQ_N_PERMS),
    "micro_m": int(MICRO_M),
    "micro_n_perms": int(MICRO_N_PERMS),
    "seed": int(SEED),
    "pairing": "NN among test 7s (L2 in pixel space)",
    "stats": stats,
    "mean_heat_eq": mean_heat_eq,
    "mean_heat_mc": mean_heat_mc,
    "mean_heat_es": mean_heat_es,
    "mean_abs_eq": mean_abs_eq,
    "mean_abs_mc": mean_abs_mc,
    "mean_abs_es": mean_abs_es,
}, out_cache)
print("Saved global cache:", out_cache)

# Also save npy for quick plotting elsewhere
np.save(out_cache.replace(".pt", "_mean_abs_eq.npy"), mean_abs_eq.numpy())
np.save(out_cache.replace(".pt", "_mean_abs_mc.npy"), mean_abs_mc.numpy())
np.save(out_cache.replace(".pt", "_mean_abs_es.npy"), mean_abs_es.numpy())

# -----------------------------
# Plot: side-by-side global heatmaps
# -----------------------------
if PLOT_ABS:
    A = mean_abs_mc.numpy()
    B = mean_abs_eq.numpy()
    C = mean_abs_es.numpy()
    titles = ["Micro-game (mean |attrib|)", "Equal-split (mean |attrib|)", "Equal Surplus (mean |attrib|)"]
    cmap = "Reds"
    vmin, vmax = 0.0, float(np.max([A.max(), B.max(), C.max()]) + 1e-12)
else:
    A = mean_heat_mc.numpy()
    B = mean_heat_eq.numpy()
    C = mean_heat_es.numpy()
    titles = ["Micro-game (mean attrib)", "Equal-split (mean attrib)", "Equal Surplus (mean attrib)"]
    cmap = "RdBu"
    vmax = float(np.max(np.abs([A, B, C])) + 1e-12)
    vmin = -vmax

fig, axes = plt.subplots(1, 3, figsize=(10.8, 3.4))
for ax, M, title in zip(axes, [A, B, C], titles):
    im = ax.imshow(M, cmap=cmap, vmin=vmin, vmax=vmax, interpolation="nearest")
    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.axis("off")

# single shared colorbar
cbar = fig.colorbar(im, ax=axes.ravel().tolist(), fraction=0.028, pad=0.02)
cbar.ax.tick_params(labelsize=9)

fig.suptitle(
    f"MNIST global 1→7 mean heatmaps (NN pairing, pairs={counts}, eps={EPS_CHANGED})",
    fontsize=12, fontweight="bold"
)
fig.tight_layout(rect=[0, 0, 1, 0.92])

out_fig = os.path.join(
    OUT_DIR,
    f"mnist_global_meanheat_nn1to7_pairs{counts}_eps{EPS_CHANGED}_"
    f"eqPerms{EQ_N_PERMS}_microM{MICRO_M}_microPerms{MICRO_N_PERMS}_abs{int(PLOT_ABS)}.png"
)
fig.savefig(out_fig, dpi=SAVE_DPI, bbox_inches="tight")
print("Saved figure:", out_fig)
plt.show()

