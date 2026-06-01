import os, glob
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torchvision.models import resnet18

# -----------------------------
# CONFIG
# -----------------------------
IDX0, IDX1 = 2, 1809
K_MAX  = 40     # curve runs K=0..K_MAX
K_SHOW = 12     # image row shows patches at this K

# Random baseline
N_RAND = 200          # number of random trials for band
RAND_SEED = 123       # reproducibility
BAND_PCTL = (10, 90)  # shaded band percentiles

CACHE_DIR = "cache_attribs"
CKPT_PATH = "resnet18_mnist_1vs7.pt"
OUT_DIR = "figures"
os.makedirs(OUT_DIR, exist_ok=True)

MEAN, STD = 0.1307, 0.3081
device = "cuda" if torch.cuda.is_available() else "cpu"

# -----------------------------
# Helpers
# -----------------------------
def latest_match(patterns):
    files = []
    for p in patterns:
        files.extend(glob.glob(p))
    if not files:
        raise FileNotFoundError(
            "No cache files found.\n"
            f"Looked in: {CACHE_DIR}\n"
            f"Patterns: {patterns}\n"
            "Make sure you ran the caching scripts and that CACHE_DIR matches.\n"
        )
    files = sorted(files, key=lambda f: os.path.getmtime(f), reverse=True)
    return files[0]

def build_model():
    mdl = resnet18(weights=None)
    mdl.conv1 = nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1, bias=False)
    mdl.maxpool = nn.Identity()
    mdl.fc = nn.Linear(mdl.fc.in_features, 2)  # 0="1", 1="7"
    return mdl

def normalize(x01_bchw):
    return (x01_bchw - MEAN) / STD

@torch.no_grad()
def p7_batch(model, xs_28x28, chunk=512):
    """
    xs_28x28: torch.FloatTensor [B,28,28] in [0,1]
    returns: np.float64 [B]
    """
    B = xs_28x28.shape[0]
    out = np.empty((B,), dtype=np.float64)
    for s in range(0, B, chunk):
        e = min(B, s + chunk)
        b = xs_28x28[s:e].unsqueeze(1)  # [b,1,28,28]
        logits = model(normalize(b).to(device))
        probs = torch.softmax(logits, dim=1)[:, 1]
        out[s:e] = probs.detach().cpu().double().numpy()
    return out

def build_patch_sequence(x0, x1, coords_order, K_max):
    """
    coords_order: [k_changed,2] order to apply patches
    returns imgs: [K_max+1,28,28], imgs[K] applies first K coords.
    """
    imgs = torch.empty((K_max + 1, 28, 28), dtype=torch.float32)
    cur = x0.clone().float()
    imgs[0] = cur
    for k in range(1, K_max + 1):
        r, c = coords_order[k - 1].tolist()
        cur[int(r), int(c)] = x1[int(r), int(c)]
        imgs[k] = cur
    return imgs

def first_k_reaching(curve, thr):
    idx = np.where(curve >= thr)[0]
    return int(idx[0]) if idx.size > 0 else None

def auc_over_budget(curve):
    return float(np.trapezoid(curve, dx=1.0) / (len(curve) - 1))

def ranked_coords_by_abs_attrib(cache_obj):
    coords = cache_obj["coords"].long()
    attrib = cache_obj["attrib_vec"].double()
    order = torch.argsort(attrib.abs(), descending=True)
    return coords[order]

def ranked_coords_by_abs_delta(x0, x1, coords_pool):
    diff = (x1 - x0).abs()
    rr = coords_pool[:, 0].long()
    cc = coords_pool[:, 1].long()
    scores = diff[rr, cc].double()
    order = torch.argsort(scores, descending=True)
    return coords_pool[order]

@torch.no_grad()
def equal_surplus_delta_units(x0, x1, coords_pool, model, chunk=512):
    """
    Equal Surplus on the CORNER game (players = changed pixels in coords_pool),
    in Δ-units exactly:

      g_i = g(x^{ {i} }) - g(x0)
      Δ   = g(x_end) - g(x0)   where x_end patches all changed pixels
      R   = Δ - sum_i g_i
      ES_i = g_i + R/|N|
    """
    k = coords_pool.shape[0]
    coords_pool = coords_pool.long()

    rr = coords_pool[:, 0]
    cc = coords_pool[:, 1]

    # x_end = x0 patched with ALL changed coords
    x_end = x0.clone()
    x_end[rr, cc] = x1[rr, cc]

    # baseline + endpoint
    p0 = float(p7_batch(model, x0.unsqueeze(0), chunk=chunk)[0])
    pend = float(p7_batch(model, x_end.unsqueeze(0), chunk=chunk)[0])
    Delta_total = pend - p0

    # singleton hybrids x^{ {i} }
    imgs_single = x0.unsqueeze(0).repeat(k, 1, 1)
    imgs_single[torch.arange(k), rr, cc] = x1[rr, cc]

    p_single = p7_batch(model, imgs_single, chunk=chunk)
    g_vec = torch.from_numpy(p_single).double() - p0

    R = float(Delta_total - g_vec.sum().item())
    ES_vec = g_vec + (R / float(k))

    return ES_vec, Delta_total, p0, pend

# -----------------------------
# Load caches
# -----------------------------
eq_file = latest_match([
    os.path.join(CACHE_DIR, f"eqsplit_idx{IDX0}_to_{IDX1}_eps*_perms*.pt"),
])
micro_file = latest_match([
    os.path.join(CACHE_DIR, f"microgame_idx{IDX0}_to_{IDX1}_eps*_m*_perms*.pt"),
])

eq = torch.load(eq_file, map_location="cpu")
mc = torch.load(micro_file, map_location="cpu")

x0 = eq["x0"].float()
x1 = eq["x1"].float()
coords_pool = eq["coords"].long()

coords_eq_sorted    = ranked_coords_by_abs_attrib(eq)
coords_mc_sorted    = ranked_coords_by_abs_attrib(mc)
coords_delta_sorted = ranked_coords_by_abs_delta(x0, x1, coords_pool)

# preliminary safety
K_MAX = int(min(K_MAX,
                coords_eq_sorted.shape[0],
                coords_mc_sorted.shape[0],
                coords_delta_sorted.shape[0],
                coords_pool.shape[0]))
K_SHOW = int(max(0, min(K_SHOW, K_MAX)))

# -----------------------------
# Load model
# -----------------------------
model = build_model().to(device)
ckpt = torch.load(CKPT_PATH, map_location=device)
model.load_state_dict(ckpt["model_state"])
model.eval()

# -----------------------------
# Equal Surplus (corner-game): exact
# -----------------------------
ES_vec, Delta_total, p0, pend = equal_surplus_delta_units(
    x0=x0, x1=x1, coords_pool=coords_pool, model=model, chunk=512
)
order_es = torch.argsort(ES_vec.abs(), descending=True)
coords_es_sorted = coords_pool[order_es]

# final safety including ES
K_MAX = int(min(K_MAX, coords_es_sorted.shape[0]))
K_SHOW = int(max(0, min(K_SHOW, K_MAX)))
Ks = np.arange(0, K_MAX + 1, dtype=int)

# -----------------------------
# Build patched sequences + evaluate curves
# -----------------------------
imgs_micro = build_patch_sequence(x0, x1, coords_mc_sorted,    K_MAX)
imgs_equal = build_patch_sequence(x0, x1, coords_eq_sorted,    K_MAX)
imgs_delta = build_patch_sequence(x0, x1, coords_delta_sorted, K_MAX)
imgs_es    = build_patch_sequence(x0, x1, coords_es_sorted,    K_MAX)

p7_micro = p7_batch(model, imgs_micro)
p7_equal = p7_batch(model, imgs_equal)
p7_delta = p7_batch(model, imgs_delta)
p7_es    = p7_batch(model, imgs_es)

# -----------------------------
# Random(K) baseline: random order over SAME changed-pixel pool
# -----------------------------
k_changed = coords_pool.shape[0]
gen = torch.Generator().manual_seed(RAND_SEED)

p7_rand_all = np.empty((N_RAND, K_MAX + 1), dtype=np.float64)
for t in range(N_RAND):
    perm = torch.randperm(k_changed, generator=gen)
    coords_rand = coords_pool[perm]
    imgs_rand = build_patch_sequence(x0, x1, coords_rand, K_MAX)
    p7_rand_all[t] = p7_batch(model, imgs_rand)

p7_rand_mean = p7_rand_all.mean(axis=0)
lo_q, hi_q = BAND_PCTL
p7_rand_lo = np.percentile(p7_rand_all, lo_q, axis=0)
p7_rand_hi = np.percentile(p7_rand_all, hi_q, axis=0)

# -----------------------------
# Efficiency metrics
# -----------------------------
metrics = {}
for name, curve in [
    ("Micro-game", p7_micro),
    ("Equal-split", p7_equal),
    ("Equal Surplus", p7_es),
    ("|x1-x0| (delta)", p7_delta),
    (f"Random (mean of {N_RAND})", p7_rand_mean),
]:
    metrics[name] = {
        "K@0.5": first_k_reaching(curve, 0.5),
        "K@0.9": first_k_reaching(curve, 0.9),
        "AUC": auc_over_budget(curve),
    }

print("\n=== Budget efficiency metrics (smaller K is better; larger AUC is better) ===")
for name, m in metrics.items():
    print(
        f"{name:>18s} | "
        f"K@0.5={m['K@0.5']}  "
        f"K@0.9={m['K@0.9']}  "
        f"AUC={m['AUC']:.4f}"
    )

# -----------------------------
# Image row for K_SHOW (now 4 images)
# -----------------------------
x_micro_show = imgs_micro[K_SHOW]
x_equal_show = imgs_equal[K_SHOW]
x_es_show    = imgs_es[K_SHOW]

p_micro_show = float(p7_micro[K_SHOW])
p_equal_show = float(p7_equal[K_SHOW])
p_es_show    = float(p7_es[K_SHOW])

# -----------------------------
# Plot
# -----------------------------
fig = plt.figure(figsize=(14.6, 5.0))
gs = fig.add_gridspec(nrows=2, ncols=4, height_ratios=[1.0, 0.85], hspace=0.22, wspace=0.18)

# Top row
axes_top = [fig.add_subplot(gs[0, j]) for j in range(4)]
for ax in axes_top:
    ax.axis("off")

axes_top[0].imshow(x0.numpy(), cmap="gray", vmin=0, vmax=1, interpolation="nearest")
axes_top[0].set_title(r"$x^0$ (baseline)", fontsize=12, fontweight="bold")

axes_top[1].imshow(x_micro_show.numpy(), cmap="gray", vmin=0, vmax=1, interpolation="nearest")
axes_top[1].set_title(rf"micro (Top-{K_SHOW})" + "\n" + rf"$P_7={p_micro_show:.3f}$",
                      fontsize=12, fontweight="bold")

axes_top[2].imshow(x_equal_show.numpy(), cmap="gray", vmin=0, vmax=1, interpolation="nearest")
axes_top[2].set_title(rf"equal (Top-{K_SHOW})" + "\n" + rf"$P_7={p_equal_show:.3f}$",
                      fontsize=12, fontweight="bold")

axes_top[3].imshow(x_es_show.numpy(), cmap="gray", vmin=0, vmax=1, interpolation="nearest")
axes_top[3].set_title(rf"ES (Top-{K_SHOW})" + "\n" + rf"$P_7={p_es_show:.3f}$",
                      fontsize=12, fontweight="bold")

# Bottom: curves span all 4 columns
axc = fig.add_subplot(gs[1, :])

# Random band behind
axc.fill_between(Ks, p7_rand_lo, p7_rand_hi, alpha=0.16, linewidth=0,
                 label=f"Random ({lo_q}–{hi_q} pct.)")
axc.plot(Ks, p7_rand_mean, linewidth=2, linestyle="--", label="Random (mean)")

# Main curves
axc.plot(Ks, p7_equal, marker="o", markersize=3, linewidth=2, label="Equal-split")
axc.plot(Ks, p7_micro, marker="o", markersize=3, linewidth=2, label="Micro-game")
axc.plot(Ks, p7_es, linewidth=2, linestyle="-", label="Equal Surplus")
axc.plot(Ks, p7_delta, linewidth=2, linestyle="-.", label=r"Top-$K$ by $|x^1-x^0|$")

# Reference lines
axc.axhline(0.5, linestyle="--", linewidth=1)
axc.axhline(0.9, linestyle="--", linewidth=1)

axc.set_xlim(0, K_MAX)
axc.set_ylim(0.0, 1.02)
axc.set_xlabel("Pixel budget K (Top-K by rule)", fontsize=11)
axc.set_ylabel(r"$P_7$ after patching", fontsize=11)
axc.set_title("Counterfactual patch test: efficiency vs pixel budget",
              fontsize=13, fontweight="bold")
axc.legend(frameon=False, fontsize=10, ncol=2)

for spine in ["top", "right"]:
    axc.spines[spine].set_visible(False)

out_path = os.path.join(
    OUT_DIR,
    f"mnist_1to7_patch_budget_curve_withES_top4_rand{N_RAND}_Kmax{K_MAX}_Kshow{K_SHOW}_idx{IDX0}_to_{IDX1}.png"
)
fig.savefig(out_path, dpi=280, bbox_inches="tight")
print("\nSaved:", out_path)
plt.show()