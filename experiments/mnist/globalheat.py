import os, glob
import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

# -----------------------------
# CONFIG
# -----------------------------
CACHE_DIR = "cache_global"
FIG_DIR   = "figures_global"
os.makedirs(FIG_DIR, exist_ok=True)

# If you want a specific file, set CACHE_PATH explicitly:
CACHE_PATH = None  # e.g. r"cache_global\global_meanheat_nn1to7_pairs200_eps0.05_....pt"

# Contrast control
CLIP_Q = 0.995   # clip vmax at this quantile (0.99–0.999 recommended)
EPS = 1e-12

# Curve controls
K_MAX = 200      # x-axis max for “top-K pixels explain mass” curve (<= 784)

# -----------------------------
# Helpers
# -----------------------------
def latest_pt(path_glob):
    files = sorted(glob.glob(path_glob), key=os.path.getmtime, reverse=True)
    if not files:
        raise FileNotFoundError(f"No cache found matching: {path_glob}")
    return files[0]

def to_np(x):
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.array(x)

def find_first_key(d, candidates):
    """Return first existing key from candidates, else None."""
    for k in candidates:
        if k in d:
            return k
    return None

def find_by_substrings(d, must_have=(), any_of=()):
    """
    Fallback: find a key that contains ALL 'must_have' substrings and at least one of 'any_of' (if provided).
    """
    keys = list(d.keys())
    for k in keys:
        ks = str(k).lower()
        if all(s.lower() in ks for s in must_have):
            if (not any_of) or any(s.lower() in ks for s in any_of):
                return k
    return None

def extract_heatmaps(obj):
    # ✅ your cache keys:
    #   mean_heat_mc, mean_heat_eq, mean_heat_es

    k_micro = "mean_heat_mc"
    k_equal = "mean_heat_eq"
    k_es    = "mean_heat_es"

    if k_micro not in obj or k_equal not in obj or k_es not in obj:
        print("\n[DEBUG] Cache keys:\n", "\n".join(map(str, obj.keys())))
        raise KeyError(f"Missing keys: micro={k_micro in obj}, equal={k_equal in obj}, es={k_es in obj}")

    H_micro = to_np(obj[k_micro]).astype(np.float64)
    H_equal = to_np(obj[k_equal]).astype(np.float64)
    H_es    = to_np(obj[k_es]).astype(np.float64)

    for name, H in [("micro", H_micro), ("equal", H_equal), ("es", H_es)]:
        if H.shape != (28, 28):
            raise ValueError(f"{name} heatmap has shape {H.shape}, expected (28,28).")

    meta = {
        "pairs": obj.get("stats", {}).get("pairs_used", None),
        "eps": obj.get("eps_changed", None),
    }
    return H_micro, H_equal, H_es, meta

def concentration_curve(H):
    """
    H: mean attribution heatmap (28x28).
    Returns:
      ranked: sorted values (descending, by value not abs since mean is usually >=0 here)
      cum_mass: cumulative sum / total
    """
    v = H.reshape(-1)
    # If some negatives exist, use abs to measure "importance mass"
    mass = np.abs(v)
    order = np.argsort(mass)[::-1]
    ranked = mass[order]
    total = ranked.sum() + EPS
    cum_mass = np.cumsum(ranked) / total
    return ranked, cum_mass

def ranked_profile(H):
    """
    Returns sorted |mean attribution| values for a profile plot.
    """
    v = np.abs(H.reshape(-1))
    v = np.sort(v)[::-1]
    return v
k_micro = "mean_abs_mc"
k_equal = "mean_abs_eq"
k_es    = "mean_abs_es"
# -----------------------------
# Load cache
# -----------------------------
if CACHE_PATH is None:
    CACHE_PATH = latest_pt(os.path.join(CACHE_DIR, "*.pt"))

obj = torch.load(CACHE_PATH, map_location="cpu")
H_micro, H_equal, H_es, meta = extract_heatmaps(obj)

# -----------------------------
# Choose display scaling
# -----------------------------
# Use one shared vmax from pooled values, clipped at quantile to avoid one hot pixel nuking contrast
pool = np.concatenate([np.abs(H_micro).ravel(), np.abs(H_equal).ravel(), np.abs(H_es).ravel()])
vmax = float(np.quantile(pool, CLIP_Q) + EPS)

# If the maps are essentially nonnegative, use sequential colormap; otherwise use diverging
has_neg = (H_micro.min() < -1e-9) or (H_equal.min() < -1e-9) or (H_es.min() < -1e-9)
if has_neg:
    cmap = "RdBu_r"
    vmin = -vmax
else:
    cmap = "Reds"
    vmin = 0.0

# -----------------------------
# Curves
# -----------------------------
rank_m, cum_m = concentration_curve(H_micro)
rank_e, cum_e = concentration_curve(H_equal)
rank_s, cum_s = concentration_curve(H_es)

prof_m = ranked_profile(H_micro)
prof_e = ranked_profile(H_equal)
prof_s = ranked_profile(H_es)

K_MAX = int(min(K_MAX, 28 * 28))
Ks = np.arange(1, K_MAX + 1)

# -----------------------------
# Plot — SPLIT INTO TWO FIGURES
# -----------------------------
plt.rcParams.update({
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 11,
})

pairs_txt = meta.get("pairs", None)
eps_txt   = meta.get("eps", None)
subtitle = "MNIST global 1→7 mean explanations (NN pairing)"
if pairs_txt is not None:
    subtitle += f" • pairs={pairs_txt}"
if eps_txt is not None:
    subtitle += f" • eps={eps_txt}"
subtitle += f" • clip q={CLIP_Q}"

# ========= FIG 1: TOP ROW (heatmaps + colorbar) =========
fig_top = plt.figure(figsize=(13.8, 3.0))
gs_top = GridSpec(
    nrows=1, ncols=4,
    width_ratios=[1, 1, 1, 0.05],
    wspace=0.20
)

axH1 = fig_top.add_subplot(gs_top[0, 0])
axH2 = fig_top.add_subplot(gs_top[0, 1])
axH3 = fig_top.add_subplot(gs_top[0, 2])
cax  = fig_top.add_subplot(gs_top[0, 3])

im1 = axH1.imshow(H_micro, cmap=cmap, vmin=vmin, vmax=vmax, interpolation="nearest")
im2 = axH2.imshow(H_equal, cmap=cmap, vmin=vmin, vmax=vmax, interpolation="nearest")
im3 = axH3.imshow(H_es,    cmap=cmap, vmin=vmin, vmax=vmax, interpolation="nearest")

for ax in (axH1, axH2, axH3):
    ax.set_xticks([]); ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)

axH1.set_title("Micro-game (mean attribution)", fontweight="bold")
axH2.set_title("Equal-split (mean attribution)", fontweight="bold")
axH3.set_title("Equal Surplus (mean attribution)", fontweight="bold")

cb = fig_top.colorbar(im3, cax=cax)
cb.set_label("Mean |attribution| (global 1→7)", rotation=90)

fig_top.suptitle(subtitle, fontsize=14, fontweight="bold", y=0.98)

out_top = os.path.join(FIG_DIR, "mnist_global_meanheat_toprow.png")
fig_top.savefig(out_top, dpi=260, bbox_inches="tight")
print("Saved TOP figure:", out_top)
plt.show()
plt.close(fig_top)


# ========= FIG 2A: Curve 1 alone (cumulative mass vs K) =========
fig_cum = plt.figure(figsize=(8.6, 3.8))
axC1 = fig_cum.add_subplot(111)

axC1.plot(Ks, cum_m[:K_MAX], linewidth=2.5, label="Micro-game")
axC1.plot(Ks, cum_e[:K_MAX], linewidth=2.5, label="Equal-split")
axC1.plot(Ks, cum_s[:K_MAX], linewidth=2.5, label="Equal Surplus")
axC1.plot(Ks, Ks / (28*28), linestyle="--", linewidth=1.5, label="Uniform baseline")

axC1.set_xlim(1, K_MAX)
axC1.set_ylim(0.0, 1.02)
axC1.set_xlabel("Pixel budget K (top-K by global mean importance)")
axC1.set_ylabel("Cumulative attribution mass captured")
axC1.set_title("Global concentration: how quickly top pixels explain the mean shift", fontweight="bold")
axC1.legend(frameon=False, ncol=2)
for spine in ["top", "right"]:
    axC1.spines[spine].set_visible(False)

fig_cum.suptitle(subtitle, fontsize=13, fontweight="bold", y=0.98)

out_cum = os.path.join(FIG_DIR, "mnist_global_curve_cum_mass.png")
fig_cum.savefig(out_cum, dpi=260, bbox_inches="tight")
print("Saved curve (cum mass):", out_cum)
plt.show()
plt.close(fig_cum)


# ========= FIG 2B: Curve 2 alone (ranked profile) =========
fig_prof = plt.figure(figsize=(8.6, 3.8))
axC2 = fig_prof.add_subplot(111)

axC2.plot(Ks, prof_m[:K_MAX], linewidth=2.2, label="Micro-game")
axC2.plot(Ks, prof_e[:K_MAX], linewidth=2.2, label="Equal-split")
axC2.plot(Ks, prof_s[:K_MAX], linewidth=2.2, label="Equal Surplus")

axC2.set_xlim(1, K_MAX)
axC2.set_yscale("log")
axC2.set_xlabel("Pixel rank (sorted by global mean importance)")
axC2.set_ylabel("Mean |attribution| (log scale)")
axC2.set_title("Ranked importance profile", fontweight="bold")
axC2.legend(frameon=False, fontsize=10)
for spine in ["top", "right"]:
    axC2.spines[spine].set_visible(False)

fig_prof.suptitle(subtitle, fontsize=13, fontweight="bold", y=0.98)

out_prof = os.path.join(FIG_DIR, "mnist_global_curve_ranked_profile.png")
fig_prof.savefig(out_prof, dpi=260, bbox_inches="tight")
print("Saved curve (ranked profile):", out_prof)
plt.show()
plt.close(fig_prof)