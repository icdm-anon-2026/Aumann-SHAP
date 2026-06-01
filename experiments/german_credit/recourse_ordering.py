import os, json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from joblib import load

# ── settings ──────────────────────────────────────────────────────────
CACHE_DIR        = "./cache"
MODEL_CACHE_FILE = os.path.join(CACHE_DIR, "models_split_rs1.joblib")
RUN_TAG          = "GLOBAL_rs1_thr30_t080_diceN8000_m5_tau0005_seed123"
ENDPOINTS_FILE   = os.path.join(CACHE_DIR, f"global_cf_endpoints_{RUN_TAG}.jsonl")
LONG_FILE        = os.path.join(CACHE_DIR, f"global_long_{RUN_TAG}.csv")
OUT_FIGURE       = os.path.join(CACHE_DIR, "tabular_budget_curve.pdf")
THRESHOLDS       = [0.5, 0.8]

# ── load model ────────────────────────────────────────────────────────
import warnings
warnings.filterwarnings("ignore")
models    = load(MODEL_CACHE_FILE)
xgb_model = models["xgboost"]

def predict_xgb(x_dict):
    df = pd.DataFrame([x_dict])
    return float(xgb_model.predict_proba(df)[:, 1][0])

# ── load endpoints ────────────────────────────────────────────────────
endpoints = {}
with open(ENDPOINTS_FILE) as f:
    for line in f:
        rec = json.loads(line)
        endpoints[rec["idx"]] = rec

# ── load attribution scores ───────────────────────────────────────────
long_df = pd.read_csv(LONG_FILE)
methods = {"micro": "S_micro", "equal": "phi_eq", "ES": "ES"}

def get_scores(idx, method_col):
    sub = long_df[long_df["idx"] == idx][["feature", method_col]]
    return dict(zip(sub["feature"], sub[method_col]))

# ── recourse ordering loop ────────────────────────────────────────────
max_k   = max(len(rec["changed"]) for rec in endpoints.values())
curves  = {m: [[] for _ in range(max_k + 1)] for m in methods}
kat     = {m: {t: [] for t in THRESHOLDS} for m in methods}

for idx, rec in endpoints.items():
    x0      = rec["x0"]
    xcf     = rec["xcf"]
    changed = rec["changed"]
    k       = len(changed)
    if k == 0:
        continue

    for mname, mcol in methods.items():
        scores = get_scores(idx, mcol)
        ranked = sorted(changed, key=lambda c: scores.get(c, 0.0), reverse=True)

        x_current = dict(x0)
        curves[mname][0].append(predict_xgb(x_current))

        for step, feat in enumerate(ranked, start=1):
            x_current[feat] = xcf[feat]
            curves[mname][step].append(predict_xgb(x_current))

        for t in THRESHOLDS:
            x_c = dict(x0)
            hit = None
            for step, feat in enumerate(ranked, start=1):
                x_c[feat] = xcf[feat]
                if predict_xgb(x_c) >= t:
                    hit = step
                    break
            if hit is not None:
                kat[mname][t].append(hit)

# ── print results ─────────────────────────────────────────────────────
print("\nRecourse ordering results")
print("="*52)
for mname in methods:
    print(f"\n{mname}:")
    for t in THRESHOLDS:
        vals = kat[mname][t]
        mean = np.mean(vals) if vals else float("nan")
        print(f"  K@{t}: mean={mean:.3f}  (n={len(vals)})")

# ── plot ──────────────────────────────────────────────────────────────
colors = {"micro": "#2ca02c", "equal": "#d62728", "ES": "#1f77b4"}
labels = {"micro": "Micro-game Shapley", "equal": "Equal-split Shapley", "ES": "Equal Surplus"}

fig, ax = plt.subplots(figsize=(5, 3.5))

for mname in methods:
    xs, ys = [], []
    for step in range(max_k + 1):
        vals = curves[mname][step]
        if len(vals) == 0:
            break
        xs.append(step)
        ys.append(np.mean(vals))
    ax.plot(xs, ys, marker="o", markersize=4,
            color=colors[mname], label=labels[mname], linewidth=1.8)

for t in THRESHOLDS:
    ax.axhline(t, color="grey", linestyle="--", linewidth=0.8, alpha=0.7)
    ax.text(max_k * 0.98, t + 0.012, f"p={t}", ha="right", fontsize=8, color="grey")

ax.set_xlabel("Number of feature changes applied (K)", fontsize=9)
ax.set_ylabel("Mean $p_{\\mathrm{xgboost}}$", fontsize=9)
ax.set_title("Recourse ordering test — German Credit", fontsize=9)
ax.legend(fontsize=8)
ax.set_xticks(range(max_k + 1))
ax.grid(True, alpha=0.3)
fig.tight_layout()
fig.savefig(OUT_FIGURE, dpi=200, bbox_inches="tight")
print(f"\n[Saved] figure -> {OUT_FIGURE}")
plt.show()