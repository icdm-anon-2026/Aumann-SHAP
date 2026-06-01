import warnings; warnings.filterwarnings("ignore")
import os, json, numpy as np, pandas as pd
from joblib import load
from math import lgamma, exp
from itertools import product

CACHE_DIR = "./cache_adult"
os.makedirs(CACHE_DIR, exist_ok=True)

# ── 1) LOAD ────────────────────────────────────────────────────────────
print("Loading model...")
saved = load(f"{CACHE_DIR}/adult_models.joblib")
xgb          = saved["model"]
feature_cols = saved["feature_cols"]
X_test       = saved["X_test"].reset_index(drop=True)
y_test       = saved["y_test"].reset_index(drop=True)

def predict(x_dict):
    df = pd.DataFrame([x_dict], columns=feature_cols)
    return float(xgb.predict_proba(df)[:, 1][0])

# ── 2) META ────────────────────────────────────────────────────────────
X_all = pd.concat([saved["X_train"], saved["X_test"]])
bounds   = {c: (float(X_all[c].min()), float(X_all[c].max())) for c in feature_cols}
ranges   = {c: max(bounds[c][1]-bounds[c][0], 1.0) for c in feature_cols}
IMMUTABLE = {"race", "sex", "native_country", "relationship"}
actionable = [c for c in feature_cols if c not in IMMUTABLE]

def sanitize(x):
    x = x.copy()
    for c in feature_cols:
        lo, hi = bounds[c]
        x[c] = float(np.clip(round(float(x[c])), lo, hi))
    return x

# ── 3) CF GENERATION ───────────────────────────────────────────────────
TARGET  = 0.70
THR_LOW = 0.25
N_SAMPLES = 3000
MAX_CHG = 8
MIN_CHG = 3   # force dense CFs so ordering test has room to work
RADIUS  = 0.35

def random_perturb(x0, rng):
    x = x0.copy()
    m = int(rng.integers(MIN_CHG, MAX_CHG+1))
    chosen = list(rng.choice(actionable,
                             size=min(m, len(actionable)),
                             replace=False))
    for c in chosen:
        lo, hi = bounds[c]
        step = rng.normal(0, RADIUS * ranges[c])
        step = float(int(np.sign(step) * max(1, abs(step))))
        x[c] = float(np.clip(round(x[c]+step), lo, hi))
    return sanitize(x)

probs      = xgb.predict_proba(X_test)[:, 1]
candidates = X_test[probs < THR_LOW].reset_index(drop=True).head(150)
print(f"Candidates (p < {THR_LOW}): {len(candidates)}")

records = []
for idx in range(len(candidates)):
    x0  = sanitize(candidates.iloc[idx].to_dict())
    rng = np.random.default_rng(42 + idx)
    best = None
    for _ in range(N_SAMPLES):
        x1 = random_perturb(x0, rng)
        p1 = predict(x1)
        if p1 >= TARGET:
            d = sum(abs(x1[c]-x0[c])/ranges[c] for c in feature_cols)
            if best is None or d < best["dist"]:
                best = {"x0": x0, "xcf": x1, "dist": d, "p": p1}
    if best is not None:
        changed = [c for c in feature_cols
                   if abs(x0[c]-best["xcf"][c]) > 1e-9]
        if len(changed) >= MIN_CHG:
            best["changed"] = changed
            records.append(best)

    if (idx+1) % 50 == 0:
        print(f"  {idx+1}/{len(candidates)}  found={len(records)}")

k_vals = [len(r["changed"]) for r in records]
print(f"\nFound {len(records)} CFs")
print(f"Mean k={np.mean(k_vals):.2f}  "
      f"min={min(k_vals)}  max={max(k_vals)}")

# ── 4) ATTRIBUTION ─────────────────────────────────────────────────────
def _lc(n, k):
    if k < 0 or k > n: return float("-inf")
    return lgamma(n+1) - lgamma(k+1) - lgamma(n-k+1)

def corner_values(x0, xcf, changed):
    k = len(changed)
    v = {}
    for mask in range(1 << k):
        x = x0.copy()
        for i in range(k):
            if (mask >> i) & 1:
                x[changed[i]] = xcf[changed[i]]
        v[mask] = predict(x)
    return v

def mobius(v, k):
    pots = {}
    for mask in range(1, 1 << k):
        t = 0.0
        sub = mask
        while True:
            s = bin(mask).count('1') - bin(sub).count('1')
            t += (1.0 if s%2==0 else -1.0) * v[sub]
            if sub == 0: break
            sub = (sub-1) & mask
        pots[mask] = t
    return pots

def equal_split(changed, pots):
    k   = len(changed)
    S   = {c: 0.0 for c in changed}
    for mask, pv in pots.items():
        mems = [changed[i] for i in range(k) if (mask>>i)&1]
        for c in mems:
            S[c] += pv / len(mems)
    return S

def es_split(changed, v):
    k  = len(changed)
    v0 = v[0]; vN = v[(1<<k)-1]
    g  = {c: v[1<<i]-v0 for i,c in enumerate(changed)}
    R  = (vN-v0) - sum(g.values())
    return {c: g[c]+R/k for c in changed}

def micro_pot_share(u, x0, xcf, m=5):
    k = len(u); n = k*m
    shape = (m+1,)*k
    g_t = np.zeros(shape)
    for p in product(range(m+1), repeat=k):
        x = x0.copy()
        for j, feat in enumerate(u):
            x[feat] = x0[feat] + (p[j]/m)*(xcf[feat]-x0[feat])
        g_t[p] = predict(x)
    r_t = np.zeros(shape)
    for p in product(range(m+1), repeat=k):
        t = 0.0
        for Sm in range(1 << k):
            s    = bin(Sm).count('1')
            sign = 1.0 if (k-s)%2==0 else -1.0
            q    = list(p)
            for j in range(k):
                if not (Sm>>j)&1: q[j] = 0
            t += sign * g_t[tuple(q)]
        r_t[p] = t
    logC = [_lc(m, pj) for pj in range(m+1)]
    sh   = np.zeros(k)
    for i in range(k):
        acc = 0.0
        for p in product(range(m+1), repeat=k):
            if p[i] >= m: continue
            ps  = sum(p)
            lw  = lgamma(ps+1) + lgamma(n-ps) - lgamma(n+1)
            lm  = sum(logC[pj] for pj in p) + np.log(m - p[i])
            pn  = list(p); pn[i] += 1
            acc += exp(lw+lm) * (r_t[tuple(pn)] - r_t[p])
        sh[i] = acc
    s = sh.sum()
    if abs(s) > 1e-12:
        sh *= r_t[(m,)*k] / s
    return {u[i]: sh[i] for i in range(k)}

def micro_total(x0, xcf, changed, m=5):
    k     = len(changed)
    v     = corner_values(x0, xcf, changed)
    pots  = mobius(v, k)
    S     = {c: 0.0 for c in changed}
    delta = abs(v[(1<<k)-1] - v[0])
    for mask, pv in pots.items():
        mems = [changed[i] for i in range(k) if (mask>>i)&1]
        if len(mems) == 1:
            S[mems[0]] += pv
            continue
        if abs(pv) / max(delta, 1e-12) < 0.005:
            for c in mems: S[c] += pv/len(mems)
        else:
            sh = micro_pot_share(mems, x0, xcf, m)
            for c in mems: S[c] += sh[c]
    return S

# ── 5) COMPUTE + SAVE ──────────────────────────────────────────────────
print("\nComputing attributions...")
rows = []
for n_idx, rec in enumerate(records):
    x0      = rec["x0"]
    xcf     = rec["xcf"]
    changed = rec["changed"]
    k       = len(changed)
    v       = corner_values(x0, xcf, changed)
    pots    = mobius(v, k)

    phi_eq = equal_split(changed, pots)
    ES     = es_split(changed, v)
    S_mic  = micro_total(x0, xcf, changed)

    for c in changed:
        rows.append({
            "idx":     n_idx,
            "feature": c,
            "k":       k,
            "v0":      v[0],
            "vN":      v[(1<<k)-1],
            "phi_eq":  phi_eq[c],
            "S_micro": S_mic[c],
            "ES":      ES[c],
        })

    if (n_idx+1) % 20 == 0:
        print(f"  {n_idx+1}/{len(records)}")

long_df = pd.DataFrame(rows)
long_df.to_csv(f"{CACHE_DIR}/adult_long.csv", index=False)

# save endpoints
with open(f"{CACHE_DIR}/adult_endpoints.jsonl", "w") as f:
    for rec in records:
        f.write(json.dumps({
            "idx":     records.index(rec),
            "x0":      rec["x0"],
            "xcf":     rec["xcf"],
            "changed": rec["changed"],
            "v0":      predict(rec["x0"]),
            "vN":      predict(rec["xcf"]),
        }) + "\n")

print(f"\n[Saved] {CACHE_DIR}/adult_long.csv")
print(f"[Saved] {CACHE_DIR}/adult_endpoints.jsonl")
print("DONE.")