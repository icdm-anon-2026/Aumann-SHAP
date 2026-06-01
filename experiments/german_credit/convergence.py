# msweep_fig3style_show.py  (updated to accept share_pp_of_delta)

from __future__ import annotations

import os, re
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator, FuncFormatter
from matplotlib.lines import Line2D

mpl.rcParams.update(
    {
        "figure.dpi": 150,
        "font.size": 11,
        "axes.labelsize": 12,
        "axes.titlesize": 12,
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,
        "legend.fontsize": 11,
        "lines.linewidth": 2.0,
        "lines.markersize": 5.5,
    }
)

# -----------------------
# USER SETTINGS
# -----------------------
CACHE_DIR = "./cache"
CSV_NAME  = "msweep_long_idx242_rs1_thr30_t080.csv"

VALUE_KEY = "xgboost"  # "logistic" | "mlp" | "xgboost"
METHODS_ORDER = ["DiCE-like", "Growing Spheres", "Genetic"]
STOPPING_RULE_TEXT = "stopping rule: 3 consecutive steps with max |Δshare| ≤ 0.1 pp"

FIX_YLIM = False
YLIM = (-5, 80)

def stable_feature_sort(feats):
    def key(f):
        m = re.match(r"X(\d+)$", str(f).strip())
        return (0, int(m.group(1))) if m else (1, str(f))
    return sorted(feats, key=key)

def pct_formatter(x, pos):
    return f"{x:.0f}"

def compute_unified_ylim(df_):
    y = df_["share_pp"].astype(float).values
    lo, hi = float(np.nanmin(y)), float(np.nanmax(y))
    pad = max(1.5, 0.08 * (hi - lo + 1e-9))
    lo2, hi2 = lo - pad, hi + pad
    lo2 = min(lo2, -1.0)
    hi2 = max(hi2,  1.0)
    return lo2, hi2

def style_map_for_features(feats):
    linestyles = ["-", "--", "-.", ":"]
    markers    = ["o", "s", "D", "^", "v", "P", "X", ">", "<", "h", "*", "d"]
    feats = stable_feature_sort(feats)
    style = {}
    for i, f in enumerate(feats):
        style[f] = dict(
            color=f"C{i % 10}",
            linestyle=linestyles[i % len(linestyles)],
            marker=markers[i % len(markers)],
            markerfacecolor="white",
            markeredgewidth=0.8,
        )
    return style

def panel_plot(ax, df_panel, style_map, title, ylim=None):
    ax.grid(True, which="both", linestyle="--", linewidth=0.6, alpha=0.55)
    ax.axhline(0.0, color="0.45", linewidth=1.0, alpha=0.85, zorder=0)

    feats = stable_feature_sort(df_panel["feature"].unique())
    for feat in feats:
        d = df_panel[df_panel["feature"] == feat].sort_values("m")
        ax.plot(d["m"], d["share_pp"], label=str(feat), **style_map[feat])

    ax.set_title(title)
    ax.set_xlabel("m")
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.yaxis.set_major_formatter(FuncFormatter(pct_formatter))

    mmin = int(df_panel["m"].min())
    mmax = int(df_panel["m"].max())
    ax.set_xlim(mmin - 0.2, mmax + 0.2)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    if ylim is not None:
        ax.set_ylim(*ylim)

def main():
    csv_path = os.path.join(CACHE_DIR, CSV_NAME)
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Missing CSV:\n{csv_path}")

    df = pd.read_csv(csv_path)

    required_core = {"method", "value_key", "m", "feature"}
    missing_core = sorted(list(required_core - set(df.columns)))
    if missing_core:
        raise ValueError(f"CSV missing columns: {missing_core}\nFound: {list(df.columns)}")

    df = df.copy()
    df["m"] = df["m"].astype(int)

    # Accept any of these:
    # - share_pp (already percentage points)
    # - share_pp_of_delta (already percentage points)
    # - share_of_delta (fraction, convert to pp)
    if "share_pp" in df.columns:
        df["share_pp"] = df["share_pp"].astype(float)
    elif "share_pp_of_delta" in df.columns:
        df["share_pp"] = df["share_pp_of_delta"].astype(float)
    elif "share_of_delta" in df.columns:
        df["share_pp"] = 100.0 * df["share_of_delta"].astype(float)
    else:
        raise ValueError(
            "CSV must contain one of: 'share_pp', 'share_pp_of_delta', 'share_of_delta'. "
            f"Found: {list(df.columns)}"
        )

    df_vk = df[df["value_key"] == VALUE_KEY].copy()
    if df_vk.empty:
        raise ValueError(f"No rows for value_key='{VALUE_KEY}'. Available: {sorted(df['value_key'].unique())}")

    feat_union = stable_feature_sort(df_vk["feature"].unique())
    style_map = style_map_for_features(feat_union)
    ylim = YLIM if FIX_YLIM else compute_unified_ylim(df_vk)

    fig, axes = plt.subplots(1, 3, figsize=(13.2, 4.8), sharey=True)
    fig.subplots_adjust(top=0.82, bottom=0.28, left=0.07, right=0.98, wspace=0.26)

    fig.suptitle(f"Convergence in m (shares of ΔV) — model: {VALUE_KEY}", y=0.97)
    fig.text(0.5, 0.90, STOPPING_RULE_TEXT, ha="center", va="center")

    for ax, method in zip(axes, METHODS_ORDER):
        df_panel = df_vk[df_vk["method"] == method].copy()
        if df_panel.empty:
            ax.axis("off")
            continue
        panel_plot(ax, df_panel, style_map, title=method, ylim=ylim)

    axes[0].set_ylabel("share of ΔV (pp)")

    handles = [Line2D([0], [0], label=str(f), **style_map[f]) for f in feat_union]
    fig.legend(
        handles=handles,
        labels=[str(f) for f in feat_union],
        loc="lower center",
        bbox_to_anchor=(0.5, 0.03),
        ncol=min(6, max(1, len(feat_union))),
        frameon=True,
        borderpad=0.4,
        labelspacing=0.4,
        handlelength=2.4,
        markerscale=1.0,
    )

    plt.show()

if __name__ == "__main__":
    main()