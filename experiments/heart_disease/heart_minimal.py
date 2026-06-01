import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
import math
from itertools import combinations
import warnings
warnings.filterwarnings('ignore')

# ---------------------------------------------------------
# CONFIG
# ---------------------------------------------------------
URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data"
COLS = ['age','sex','cp','trestbps','chol','fbs','restecg','thalach','exang','oldpeak','slope','ca','thal','target']
NUMERIC = ['age','trestbps','chol','thalach','oldpeak']  # we only analyze pots among these
M = 5
LOW_THR, HIGH_THR = 0.2, 0.8

# ---------------------------------------------------------
# 1. LOAD
# ---------------------------------------------------------
df = pd.read_csv(URL, header=None, names=COLS, na_values='?')
df = df.dropna().reset_index(drop=True)
df['target'] = (df['target'] > 0).astype(int)

X = df.drop('target', axis=1)
y = df['target']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

# ---------------------------------------------------------
# 2. TRAIN
# ---------------------------------------------------------
model = XGBClassifier(
    n_estimators=100, max_depth=4, learning_rate=0.1,
    subsample=0.8, colsample_bytree=0.8,
    random_state=42, eval_metric='logloss', use_label_encoder=False
)
model.fit(X_train, y_train)

p_train = model.predict_proba(X_train)[:, 1]
p_test  = model.predict_proba(X_test)[:, 1]

# ---------------------------------------------------------
# 3. FIND COUNTERFACTUAL PAIRS (numeric-only changed features, k=2 or 3)
# ---------------------------------------------------------
cand0 = X_test[(y_test == 0) & (p_test < LOW_THR)].copy()
cand1 = X_train[p_train > HIGH_THR].copy()

def dist(a, b, feats):
    return np.sqrt(((a[feats] - b[feats])**2).sum())

pairs = []
for idx0, row0 in cand0.head(40).iterrows():
    best_d, best_idx1 = np.inf, None
    for idx1, row1 in cand1.iterrows():
        d = dist(row0, row1, NUMERIC)
        if d < best_d:
            best_d, best_idx1 = d, idx1
    if best_idx1 is None:
        continue
    changed = [f for f in NUMERIC if abs(row0[f] - X_train.loc[best_idx1, f]) > 1e-9]
    if 2 <= len(changed) <= 3:
        pairs.append({
            'idx0': idx0, 'idx1': best_idx1,
            'x0': row0.to_dict(), 'x1': X_train.loc[best_idx1].to_dict(),
            'changed': changed, 'k': len(changed), 'dist': best_d
        })
    if len(pairs) >= 3:
        break

print(f"Selected {len(pairs)} counterfactual pairs with k in {{2,3}}\n")

# ---------------------------------------------------------
# 4. ATTribution helpers
# ---------------------------------------------------------
def g_eval(x_dict):
    """Evaluate model probability at a full feature dict."""
    arr = np.array([x_dict[c] for c in COLS[:-1]]).reshape(1, -1)
    return model.predict_proba(arr)[0, 1]

def harsanyi_phi(changed, x0, x1):
    k = len(changed)
    phi = 0.0
    for mask in range(2**k):
        bits = [(mask >> i) & 1 for i in range(k)]
        x = dict(x0)
        for i in range(k):
            if bits[i]:
                x[changed[i]] = x1[changed[i]]
        sign = (-1)**(k - sum(bits))
        phi += sign * g_eval(x)
    return phi

def residual_table(changed, x0, x1, m):
    k = len(changed)
    shape = [m+1]*k
    r = np.zeros(shape)
    for p_idx in np.ndindex(*shape):
        p = np.array(p_idx)
        t = p / m
        val = 0.0
        for Tmask in range(2**k):
            Tbits = [(Tmask >> i) & 1 for i in range(k)]
            x = dict(x0)
            for i in range(k):
                if Tbits[i]:
                    # i in T: use slider value t_i
                    x[changed[i]] = x0[changed[i]] + t[i]*(x1[changed[i]] - x0[changed[i]])
                else:
                    # i not in T: baseline
                    x[changed[i]] = x0[changed[i]]
            sign = (-1)**(k - sum(Tbits))
            val += sign * g_eval(x)
        r[p_idx] = val
    return r

def brute_shapley(changed, x0, x1, m):
    k = len(changed)
    n = k * m
    r = residual_table(changed, x0, x1, m)
    
    # cache v by grid state
    v_cache = {}
    for p_idx in np.ndindex(*[m+1]*k):
        v_cache[tuple(p_idx)] = r[p_idx]
    
    players = [(i, s) for i in range(k) for s in range(1, m+1)]
    
    # precompute v for all 2^n coalitions
    v_sub = {}
    for mask in range(2**n):
        p = [0]*k
        idx = 0
        temp = mask
        while temp:
            if temp & 1:
                i, _ = players[idx]
                p[i] += 1
            idx += 1
            temp >>= 1
        v_sub[mask] = v_cache[tuple(p)]
    
    phi_players = np.zeros(n)
    fact = [math.factorial(i) for i in range(n+1)]
    
    for j in range(n):
        total = 0.0
        for mask in range(2**n):
            if not (mask & (1 << j)):
                sz = bin(mask).count('1')
                w = fact[sz] * fact[n - sz - 1] / fact[n]
                total += w * (v_sub[mask | (1 << j)] - v_sub[mask])
        phi_players[j] = total
    
    S = np.zeros(k)
    for j in range(n):
        i, _ = players[j]
        S[i] += phi_players[j]
    return S

# Add this right before the "HEART DISEASE MINIMAL RESULTS" print block
print("\n=== PAIR 1 DETAILS ===")
p = pairs[0]
print("Changed features:", p['changed'])
print("Baseline x0:")
for f in p['changed']:
    print(f"  {f}: {p['x0'][f]:.4f}")
print("Counterfactual x1:")
for f in p['changed']:
    print(f"  {f}: {p['x1'][f]:.4f}")
print("Delta (x1 - x0):")
for f in p['changed']:
    print(f"  {f}: {p['x1'][f] - p['x0'][f]:+.4f}")
print("Baseline label:", y_test.loc[p['idx0']])
print("Counterfactual label:", y_train.loc[p['idx1']])

# ---------------------------------------------------------
# 5. RUN & EMIT LATEX
# ---------------------------------------------------------
print("=" * 60)
print("HEART DISEASE MINIMAL RESULTS (paste into LaTeX)")
print("=" * 60)

for pid, pair in enumerate(pairs[:3], 1):
    ch = pair['changed']
    k = pair['k']
    x0, x1 = pair['x0'], pair['x1']
    
    phi_u = harsanyi_phi(ch, x0, x1)
    S_micro = brute_shapley(ch, x0, x1, M)
    eq = phi_u / k
    
    print(f"\n--- Pair {pid}: interaction pot $u=\\{{{','.join(ch)}\\}}$, $\\phi_u={phi_u:.4f}$ ---")
    print("Feature & Equal-split & Aumann-SHAP & Diff \\\\")
    for i, f in enumerate(ch):
        print(f"{f} & {eq:.4f} & {S_micro[i]:.4f} & {S_micro[i]-eq:+.4f} \\\\")
    print(f"Sum check: {S_micro.sum():.4f} (should equal {phi_u:.4f})")
    