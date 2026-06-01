import warnings; warnings.filterwarnings("ignore")
import os, json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from joblib import load

CACHE_DIR = "./cache_adult"
saved        = load(f"{CACHE_DIR}/adult_models.joblib")
xgb          = saved["model"]
feature_cols = saved["feature_cols"]

def predict(x_dict):
    df = pd.DataFrame([x_dict], columns=feature_cols)
    return float(xgb.predict_proba(df)[:, 1][0])

# load data
endpoints = []
with open(f"{CACHE_DIR}/adult_endpoints.jsonl") as f:
    for line in f:
        endpoints.append(json.loads(line))

long_df = pd.read_csv(f"{CACHE_DIR}/adult_long.csv")

def get_scores(idx, col):
    sub = long_df[long_df["idx"] == idx][["feature", col]]
    return dict(zip(sub["feature"], sub[col]))

# ordering test
THRESHOLDS = [0.5, 0.7]
methods    = {"micro": "S_micro", "equal": "phi_eq", "ES": "ES"}
max_k      = max(len(r["changed"]) for r in endpoints)

curves = {m: [[] for _ in range(max_k+1)] for m in methods}
kat    = {m: {t: [] for t in THRESHOLDS} for m in methods}

for rec in endpoints:
    idx     = rec["idx"]
    x0      = rec["x0"]
    xcf     = rec["xcf"]
    changed = rec["changed"]

    for mname, mcol in methods.items():
        scores = get_scores(idx, mcol)
        ranked = sorted(changed,
                        key=lambda c: scores.get(c, 0.0),
                        reverse=True)

        x_cur = x0.copy()
        curves[mname][0].append(predict(x_cur))

        for step, feat in enumerate(ranked, 1):
            x_cur[feat] = xcf[feat]
            curves[mname][step].append(predict(x_cur))

        for t in THRESHOLDS:
            x_c = x0.copy()
            for step, feat in enumerate(ranked, 1):
                x_c[feat] = xcf[feat]
                if predict(x_c) >= t:
                    kat[mname][t].append(step)
                    break

# results
print("\nRecourse ordering results — Adult Income")
print("="*52)
for mname in ["micro", "equal", "ES"]:
    print(f"\n{mname}:")
    for t in THRESHOLDS:
        vals = kat[mname][t]
        print(f"  K@{t}: mean={np.mean(vals):.3f}  (n={len(vals)})")

print("\nAUC of mean score curve:")
for mname in ["micro", "equal", "ES"]:
    ys = []
    for step in range(max_k+1):
        v = curves[mname][step]
        if not v: break
        ys.append(np.mean(v))
    print(f"  {mname}: {np.trapz(ys):.4f}")

# plot
colors = {"micro":"#2ca02c","equal":"#d62728","ES":"#1f77b4"}
labels = {"micro":"Micro-game Shapley",
          "equal":"Equal-split Shapley","ES":"Equal Surplus"}

fig, ax = plt.subplots(figsize=(5, 3.5))
for mname in ["micro","equal","ES"]:
    xs, ys = [], []
    for step in range(max_k+1):
        v = curves[mname][step]
        if not v: break
        xs.append(step); ys.append(np.mean(v))
    ax.plot(xs, ys, marker="o", markersize=4,
            color=colors[mname], label=labels[mname], linewidth=1.8)

for t in THRESHOLDS:
    ax.axhline(t, color="grey", linestyle="--", linewidth=0.8, alpha=0.7)
    ax.text(max_k*0.97, t+0.012, f"p={t}",
            ha="right", fontsize=8, color="grey")

k_vals = [len(r["changed"]) for r in endpoints]
ax.set_xlabel("Number of feature changes applied (K)", fontsize=9)
ax.set_ylabel("Mean predicted probability (income >50K)", fontsize=9)
ax.set_title(f"Recourse ordering — Adult Income "
             f"($\\bar{{k}}={np.mean(k_vals):.1f}$, $n={len(endpoints)}$)",
             fontsize=9)
ax.legend(fontsize=8)
ax.set_xticks(range(max_k+1))
ax.grid(True, alpha=0.3)
fig.tight_layout()
fig.savefig(f"{CACHE_DIR}/adult_budget_curve.pdf", dpi=200,
            bbox_inches="tight")
print(f"\n[Saved] {CACHE_DIR}/adult_budget_curve.pdf")
plt.show()