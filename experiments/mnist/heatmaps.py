import os
import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm, ListedColormap
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

# -----------------------------
# CONFIG
# -----------------------------
EQ_FILE    = r"cache_attribs/eqsplit_idx2_to_1809_eps0.05_perms400.pt"
MICRO_FILE = r"cache_attribs/microgame_idx2_to_1809_eps0.05_m10_perms200.pt"

OUT_DIR = "figures"
os.makedirs(OUT_DIR, exist_ok=True)
OUT_PATH = os.path.join(OUT_DIR, "mnist_1to7_all_maps_4x4.png")

DPI = 350

# Heatmap robust scaling
VMAX_QUANTILE = 0.995
HEAT_CMAP = "RdBu"   # BLUE positive, RED negative

# Important pixels rule
TOP_K = 25
TOP_Q = None  # e.g. 0.20 for top 20%

# Redistribution mask (optional)
TAU_DIFF = 0.015  # set None to disable

# -----------------------------
# Load caches
# -----------------------------
eq = torch.load(EQ_FILE, map_location="cpu")
micro = torch.load(MICRO_FILE, map_location="cpu")

x0 = eq["x0"].float().squeeze()
x_end = micro.get("x_end", eq.get("x_end", eq.get("x1"))).float().squeeze()

heat_eq = eq["heat"].float().squeeze()
heat_micro = micro["heat"].float().squeeze()

# -----------------------------
# Shared symmetric scale for heatmaps (eq + micro)
# -----------------------------
abs_all = torch.cat([heat_eq.abs().reshape(-1), heat_micro.abs().reshape(-1)], dim=0).numpy()
abs_all = abs_all[abs_all > 0]
vmax_h = float(np.quantile(abs_all, VMAX_QUANTILE)) if abs_all.size else 1e-12
vmax_h = max(vmax_h, 1e-12)
norm_h = TwoSlopeNorm(vcenter=0.0, vmin=-vmax_h, vmax=vmax_h)

# -----------------------------
# Important-pixel binary masks (largest |attribution|)
# -----------------------------
def topk_mask(heat_28x28: torch.Tensor, k: int) -> torch.Tensor:
    a = heat_28x28.abs().reshape(-1)
    k = int(max(1, min(k, a.numel())))
    thr = torch.kthvalue(a, a.numel() - k + 1).values.item()
    return (heat_28x28.abs() >= thr)

def topq_mask(heat_28x28: torch.Tensor, q: float) -> torch.Tensor:
    q = float(np.clip(q, 0.0, 1.0))
    a = heat_28x28.abs().reshape(-1).numpy()
    thr = float(np.quantile(a, 1.0 - q))
    return (heat_28x28.abs() >= thr)

if TOP_Q is not None:
    mask_eq = topq_mask(heat_eq, TOP_Q)
    mask_micro = topq_mask(heat_micro, TOP_Q)
    rule_str = f"Top {int(TOP_Q*100)}%"
else:
    mask_eq = topk_mask(heat_eq, TOP_K)
    mask_micro = topk_mask(heat_micro, TOP_K)
    rule_str = f"Top-{TOP_K}"

mask_cmap = ListedColormap([(0, 0, 0, 0), (0.90, 0.10, 0.10, 0.85)])  # transparent -> red

# -----------------------------
# Redistribution map: D = micro - equal
# -----------------------------
D = (heat_micro - heat_eq).numpy()
vmax_d = float(np.max(np.abs(D)) + 1e-12)

# -----------------------------
# Plot helpers
# -----------------------------
def add_inset_colorbar(ax, im, ticks, label=None):
    cax = inset_axes(ax, width="4%", height="70%", loc="center right", borderpad=1.0)
    cb = plt.colorbar(im, cax=cax)
    cb.set_ticks(ticks)
    cb.ax.tick_params(labelsize=9)
    if label:
        cb.set_label(label, fontsize=9)
    return cb

# -----------------------------
# Figure: 2 rows x 4 cols (8 panels)
# -----------------------------
plt.close("all")
plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 11,
    "axes.titleweight": "semibold",
})

fig = plt.figure(figsize=(15.2, 7.2), constrained_layout=False)
gs = fig.add_gridspec(nrows=2, ncols=4, wspace=0.2, hspace=0.25)

# Row 1
ax_x0     = fig.add_subplot(gs[0, 0])
ax_xend   = fig.add_subplot(gs[0, 1])
ax_imp_eq = fig.add_subplot(gs[0, 2])
ax_imp_mc = fig.add_subplot(gs[0, 3])

# Row 2
ax_h_eq = fig.add_subplot(gs[1, 0])
ax_h_mc = fig.add_subplot(gs[1, 1])
ax_D    = fig.add_subplot(gs[1, 2])
ax_ov   = fig.add_subplot(gs[1, 3])

# --- Row 1: baseline / endpoint
ax_x0.imshow(x0.numpy(), cmap="gray", vmin=0, vmax=1)
ax_x0.set_title(r"Baseline $x_0$")
ax_x0.axis("off")

ax_xend.imshow(x_end.numpy(), cmap="gray", vmin=0, vmax=1)
ax_xend.set_title(r"Path end $x_{end}$")
ax_xend.axis("off")

# --- Row 1: important pixels (red on x_end)
ax_imp_eq.imshow(x_end.numpy(), cmap="gray", vmin=0, vmax=1)
ax_imp_eq.imshow(mask_eq.numpy().astype(int), cmap=mask_cmap, interpolation="nearest")
ax_imp_eq.set_title(f"Equal-split ({rule_str})")
ax_imp_eq.axis("off")

ax_imp_mc.imshow(x_end.numpy(), cmap="gray", vmin=0, vmax=1)
ax_imp_mc.imshow(mask_micro.numpy().astype(int), cmap=mask_cmap, interpolation="nearest")
ax_imp_mc.set_title(f"Micro-game ({rule_str})")
ax_imp_mc.axis("off")

# --- Row 2: heatmaps
im_eq = ax_h_eq.imshow(heat_eq.numpy(), cmap=HEAT_CMAP, norm=norm_h, interpolation="nearest")
ax_h_eq.set_title("Equal-split Shapley")
ax_h_eq.axis("off")

im_mc = ax_h_mc.imshow(heat_micro.numpy(), cmap=HEAT_CMAP, norm=norm_h, interpolation="nearest")
ax_h_mc.set_title("Micro-game Shapley")
ax_h_mc.axis("off")

# --- Shared horizontal colorbar for the two heatmaps (below both)
pos_eq = ax_h_eq.get_position()
pos_mc = ax_h_mc.get_position()

left = pos_eq.x0
right = pos_mc.x1
bottom = min(pos_eq.y0, pos_mc.y0) - 0.022   # very close to the panels
height = 0.018

cax_shared = fig.add_axes([left, bottom, right - left, height])
cb_shared = fig.colorbar(im_mc, cax=cax_shared, orientation="horizontal")
cb_shared.set_ticks(np.linspace(-vmax_h, vmax_h, 5))
cb_shared.ax.tick_params(labelsize=9, pad=1)
cb_shared.set_label("attrib", fontsize=9, labelpad=2)

# --- Row 2: D and overlay
im_d = ax_D.imshow(D, cmap=HEAT_CMAP, vmin=-vmax_d, vmax=vmax_d, interpolation="nearest")
ax_D.set_title(r"$D = \mathrm{micro} - \mathrm{equal}$")
ax_D.axis("off")

ax_ov.imshow(x_end.numpy(), cmap="gray", vmin=0, vmax=1, interpolation="nearest")
ax_ov.imshow(D, cmap=HEAT_CMAP, vmin=-vmax_d, vmax=vmax_d, alpha=0.85, interpolation="nearest")
ax_ov.set_title(r"Overlay of $D$ on $x_{end}$")
ax_ov.axis("off")

# Optional big-redistribution mask overlay
if TAU_DIFF is not None and TAU_DIFF > 0:
    big = (np.abs(D) >= TAU_DIFF)
    overlay = np.zeros((*big.shape, 4), dtype=np.float32)
    overlay[big] = np.array([1.0, 0.0, 0.0, 0.60], dtype=np.float32)
    ax_ov.imshow(overlay, interpolation="nearest")
    ax_ov.text(
        0.02, 0.98, rf"$|D|\geq {TAU_DIFF:.3f}$",
        transform=ax_ov.transAxes, va="top", ha="left",
        fontsize=9, color="white",
        bbox=dict(boxstyle="round,pad=0.25", facecolor="black", alpha=0.35, edgecolor="none"),
    )

add_inset_colorbar(ax_D, im_d, ticks=np.linspace(-vmax_d, vmax_d, 5), label="Δ attrib")

fig.savefig(OUT_PATH, dpi=DPI, bbox_inches="tight")
print("Saved:", OUT_PATH)
plt.show()
