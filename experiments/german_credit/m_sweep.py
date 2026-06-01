# micro_msweep_from_artifact.py
# Read your local-analysis artifact (x0 + CF endpoints) and run per-model m-sweep for MICRO totals.
# Outputs CSVs into ./cache/

import os, json, warnings
import numpy as np
import pandas as pd
from itertools import product
from math import comb

from sklearn.model_selection import train_test_split
from joblib import load

# ============================================================
# SETTINGS
# ============================================================
CACHE_DIR = "./cache"
os.makedirs(CACHE_DIR, exist_ok=True)

ARTIFACT_PATH = os.path.join(CACHE_DIR, "artifact_rs1_thr30_t080.json")
MODEL_CACHE_FILE = os.path.join(CACHE_DIR, "models_split_rs1.joblib")

thr_low = 0.30
target  = 0.80
random_state_split = 1

# sweep settings
VALUE_KEYS = ["logistic", "mlp", "xgboost"]   # per-model probs
M_START = 5
M_MAX   = 15
EPS_PP  = 0.10   # convergence threshold in percentage points of Δ share
STREAK_N = 3

MAX_K_CHANGED = 12
SUPPRESS_XGB_PICKLE_WARNING = True

# ============================================================
# WARNINGS (optional)
# ============================================================
if SUPPRESS_XGB_PICKLE_WARNING:
    warnings.filterwarnings(
        "ignore",
        message=r".*If you are loading a serialized model.*",
        category=UserWarning
    )

# ============================================================
# HELPERS: artifact parsing
# ============================================================
def _norm(s: str) -> str:
    return "".join(ch for ch in str(s).lower() if ch.isalnum())

def _looks_like_xvec(d: dict) -> bool:
    if not isinstance(d, dict):
        return False
    ks = list(d.keys())
    return len(ks) >= 20 and all(str(k).startswith("X") for k in ks)

def _get_any(d: dict, keys: list[str]):
    for k in keys:
        if isinstance(d, dict) and k in d and d[k] is not None:
            return d[k]
    return None

def load_artifact(path: str) -> dict:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing artifact: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

METHOD_ALIASES = {
    "DiCE-like": ["DiCE-like", "dice", "cf_dice", "dice_like"],
    "Growing Spheres": ["Growing Spheres", "GrowingSpheres", "gs", "cf_gs"],
    "Genetic": ["Genetic", "ga", "cf_ga", "genetic"],
}

def extract_method_xcf(artifact: dict, method_name: str) -> dict | None:
    # search in common containers first
    containers = []
    for k in ["methods", "counterfactuals", "cfs", "cf", "results"]:
        if isinstance(artifact.get(k, None), dict):
            containers.append(artifact[k])
    containers.append(artifact)  # fallback

    aliases = [_norm(a) for a in METHOD_ALIASES.get(method_name, [method_name])]

    for root in containers:
        if not isinstance(root, dict):
            continue

        # exact-key match
        for k, v in root.items():
            if _norm(k) in aliases:
                # v might be xcf directly or a block holding it
                if _looks_like_xvec(v):
                    return v
                xcf = _get_any(v, ["xcf", "x_cf", "x1", "x", "cf"])
                if _looks_like_xvec(xcf):
                    return xcf

        # alias-key match (non-exact keys)
        for k, v in root.items():
            if any(a in _norm(k) for a in aliases):
                if _looks_like_xvec(v):
                    return v
                xcf = _get_any(v, ["xcf", "x_cf", "x1", "x", "cf"])
                if _looks_like_xvec(xcf):
                    return xcf

    return None

def extract_x0_and_idx(artifact: dict):
    idx = artifact.get("idx", None)
    x0 = artifact.get("x0", None)

    if x0 is None:
        base = artifact.get("baseline", None)
        if isinstance(base, dict):
            idx = idx if idx is not None else base.get("idx", None)
            x0 = base.get("x0", None) or base.get("x", None)

    if not _looks_like_xvec(x0):
        raise ValueError("Could not find x0 in artifact (expected dict with keys X1..X27).")
    if idx is None:
        idx = -1
    return int(idx), x0

# ============================================================
# DATASET (for meta + sanitize)
# ============================================================
URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/statlog/german/german.data"
raw = pd.read_csv(URL, sep=r"\s+", header=None)
raw.columns = [
    "checking", "duration", "credit_history", "purpose", "amount", "savings",
    "employment", "installment_rate", "personal_status_sex", "other_debtors",
    "residence", "property", "age", "other_plans", "housing",
    "existing_credits", "job", "liable_people", "telephone", "foreign_worker",
    "y_raw"
]
y = (raw["y_raw"] == 1).astype(int)

X1 = raw["duration"]
X2 = raw["amount"]
X3 = raw["installment_rate"]
X4 = raw["residence"]
X5 = raw["existing_credits"]
X6 = raw["liable_people"]

X7  = (raw["telephone"] == "A192").astype(int)
X8  = raw["checking"].isin(["A12", "A13", "A14"]).astype(int)
X9  = (raw["checking"] == "A13").astype(int)

X10 = raw["savings"].isin(["A62", "A63", "A64"]).astype(int)
X11 = raw["savings"].isin(["A63", "A64"]).astype(int)

X12 = (raw["credit_history"] == "A33").astype(int)
X13 = (raw["credit_history"] == "A30").astype(int)
X14 = (raw["credit_history"] == "A34").astype(int)

X15 = (raw["other_plans"] == "A141").astype(int)

X16 = (raw["other_debtors"] == "A102").astype(int)
X17 = (raw["other_debtors"] == "A103").astype(int)

X18 = (raw["employment"] == "A71").astype(int)
X19 = (raw["employment"] == "A72").astype(int)
X20 = raw["employment"].isin(["A74", "A75"]).astype(int)

X21 = raw["personal_status_sex"].isin(["A91", "A93", "A94"]).astype(int)
X23 = raw["personal_status_sex"].isin(["A93", "A95"]).astype(int)

X22 = (raw["foreign_worker"] == "A201").astype(int)
X24 = raw["age"]

X25 = (raw["housing"] == "A152").astype(int)
X26 = (raw["housing"] == "A151").astype(int)

X27 = raw["job"].isin(["A173", "A174"]).astype(int)

X = pd.DataFrame({
    "X1": X1, "X2": X2, "X3": X3, "X4": X4, "X5": X5, "X6": X6, "X7": X7,
    "X8": X8, "X9": X9, "X10": X10, "X11": X11, "X12": X12, "X13": X13,
    "X14": X14, "X15": X15, "X16": X16, "X17": X17, "X18": X18, "X19": X19,
    "X20": X20, "X21": X21, "X22": X22, "X23": X23, "X24": X24, "X25": X25,
    "X26": X26, "X27": X27,
})

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=random_state_split, stratify=y
)

def build_meta(X_ref: pd.DataFrame):
    cols = list(X_ref.columns)
    bounds = {c: (float(X_ref[c].min()), float(X_ref[c].max())) for c in cols}
    uniq = {c: np.sort(X_ref[c].unique()) for c in cols}
    binary_cols = [c for c in cols if set(uniq[c]).issubset({0, 1})]
    ranges = {}
    for c in cols:
        lo, hi = bounds[c]
        r = hi - lo
        ranges[c] = r if r > 0 else 1.0
    return {"cols": cols, "bounds": bounds, "binary_cols": binary_cols, "ranges": ranges}

meta = build_meta(X_train)

def sanitize(x: pd.Series, meta_) -> pd.Series:
    x = x.astype(float).copy()
    for c in meta_["cols"]:
        lo, hi = meta_["bounds"][c]
        v = float(x[c])
        v = min(max(v, lo), hi)
        if c in meta_["binary_cols"]:
            v = 1.0 if v >= 0.5 else 0.0
        else:
            v = float(int(round(v)))
        x[c] = v
    return x

def series_from_dict(d: dict, cols: list[str]) -> pd.Series:
    return pd.Series({c: float(d[c]) for c in cols})

# ============================================================
# MODELS + EVAL
# ============================================================
if not os.path.exists(MODEL_CACHE_FILE):
    raise FileNotFoundError(f"Missing models file: {MODEL_CACHE_FILE}")

models = load(MODEL_CACHE_FILE)
print(f"[Loaded fitted models from] {MODEL_CACHE_FILE}")

def row_to_df_float(x_row: pd.Series) -> pd.DataFrame:
    x = x_row.astype(float)
    return pd.DataFrame([x.values], columns=x.index)

def predict_probs_all_float(models_dict: dict, x_row: pd.Series) -> dict:
    X1row = row_to_df_float(x_row)
    out = {}
    for name, m in models_dict.items():
        out[name] = float(m.predict_proba(X1row)[:, 1][0])
    out["p_min3"] = float(min(out.values()))
    out["p_max3"] = float(max(out.values()))
    return out

def value_scalar_float(models_dict: dict, x_row: pd.Series, key: str) -> float:
    probs = predict_probs_all_float(models_dict, x_row)
    return float(probs[key])

# ============================================================
# CORNER GAME + POTS
# ============================================================
def changed_features_between(x0: pd.Series, x1: pd.Series, cols: list[str]) -> list[str]:
    return [c for c in cols if float(x0[c]) != float(x1[c])]

def corner_values_v(models_dict, x0: pd.Series, xcf: pd.Series, changed: list[str], value_key: str) -> dict:
    k = len(changed)
    v = {}
    for mask in range(1 << k):
        x = x0.astype(float).copy()
        for i in range(k):
            if (mask >> i) & 1:
                c = changed[i]
                x[c] = float(xcf[c])
        v[mask] = value_scalar_float(models_dict, x, key=value_key)
    return v

def mobius_pots_from_v(v: dict, k: int) -> dict:
    pots = {}
    for mask in range(1, 1 << k):
        total = 0.0
        sub = mask
        while True:
            sign = -1.0 if ((mask.bit_count() - sub.bit_count()) % 2 == 1) else 1.0
            total += sign * v[sub]
            if sub == 0:
                break
            sub = (sub - 1) & mask
        pots[mask] = float(total)
    return pots

# ============================================================
# MICRO: build g-table + r-table for a pot u
# ============================================================
def cube_point_numeric(x0: pd.Series, x1: pd.Series, u: list[str], p_tuple: tuple[int, ...], m: int) -> pd.Series:
    x = x0.astype(float).copy()
    for j, feat in enumerate(u):
        tj = float(p_tuple[j]) / float(m)
        x[feat] = float(x0[feat]) + tj * (float(x1[feat]) - float(x0[feat]))
    return x

def build_g_table(models_dict, x0: pd.Series, xcf: pd.Series, u: list[str], m: int, value_key: str) -> np.ndarray:
    k = len(u)
    shape = (m + 1,) * k
    g = np.zeros(shape, dtype=float)
    for p in product(range(m + 1), repeat=k):
        x = cube_point_numeric(x0, xcf, u, p, m)
        g[p] = value_scalar_float(models_dict, x, key=value_key)
    return g

def build_r_table_from_g(g: np.ndarray) -> np.ndarray:
    shape = g.shape
    k = len(shape)
    m = shape[0] - 1
    r = np.zeros_like(g)
    for p in product(range(m + 1), repeat=k):
        total = 0.0
        for Smask in range(1 << k):
            s = Smask.bit_count()
            sign = -1.0 if ((k - s) % 2 == 1) else 1.0
            q = list(p)
            for j in range(k):
                if ((Smask >> j) & 1) == 0:
                    q[j] = 0
            total += sign * g[tuple(q)]
        r[p] = float(total)
    return r

def micro_shapley_shares_from_r_table(u: list[str], r: np.ndarray, m: int) -> dict:
    u = list(u)
    d = len(u)
    n = d * m

    fact = [1.0] * (n + 1)
    for k in range(2, n + 1):
        fact[k] = fact[k - 1] * float(k)
    denom = fact[n]

    shares = {feat: 0.0 for feat in u}

    for i_idx, i_feat in enumerate(u):
        phi_rep = 0.0
        ranges = [range(m + 1)] * d
        ranges[i_idx] = range(m)  # a_i in 0..m-1

        for a in product(*ranges):
            a = list(a)
            A_size = sum(a)

            count_subsets = comb(m - 1, a[i_idx])
            for j in range(d):
                if j == i_idx:
                    continue
                count_subsets *= comb(m, a[j])

            weight = (float(count_subsets) * fact[A_size] * fact[n - A_size - 1]) / denom

            a_next = a.copy()
            a_next[i_idx] += 1

            w0 = float(r[tuple(a)])
            w1 = float(r[tuple(a_next)])
            phi_rep += weight * (w1 - w0)

        shares[i_feat] = float(m) * float(phi_rep)

    # efficiency inside pot: scale to pot value (numerical drift guard)
    pot_val = float(r[(m,) * d])
    ssum = float(sum(shares.values()))
    if abs(ssum) > 1e-12:
        scale = pot_val / ssum
        for c in shares:
            shares[c] *= float(scale)

    return shares

# ============================================================
# MICRO TOTALS AT GIVEN m (for one CF + value_key)
# ============================================================
def micro_totals_at_m(models_dict, x0: pd.Series, xcf: pd.Series, changed: list[str], v: dict, pots: dict,
                      m: int, value_key: str) -> tuple[dict, float]:
    k = len(changed)
    v0 = float(v[0])
    vN = float(v[(1 << k) - 1])
    delta = float(vN - v0)

    S = {c: 0.0 for c in changed}
    # singleton pots
    for i, c in enumerate(changed):
        S[c] += float(pots.get(1 << i, 0.0))

    EPS_POT = 1e-12
    for mask, pot_val in pots.items():
        if mask.bit_count() < 2:
            continue
        if abs(float(pot_val)) < EPS_POT:
            continue

        members = [changed[i] for i in range(k) if (mask >> i) & 1]

        g = build_g_table(models_dict, x0, xcf, members, m=m, value_key=value_key)
        r = build_r_table_from_g(g)
        shares_u = micro_shapley_shares_from_r_table(members, r, m=m)

        for c in members:
            S[c] += float(shares_u.get(c, 0.0))

    return S, delta

def shares_pp(S: dict, feats: list[str], delta: float) -> np.ndarray:
    if abs(delta) < 1e-12:
        return np.zeros(len(feats), dtype=float)
    return np.array([float(S.get(f, 0.0)) / float(delta) for f in feats], dtype=float) * 100.0

# ============================================================
# MAIN: load artifact and sweep
# ============================================================
artifact = load_artifact(ARTIFACT_PATH)
idx, x0_dict = extract_x0_and_idx(artifact)

x0 = sanitize(series_from_dict(x0_dict, meta["cols"]), meta)

methods = [
    ("DiCE-like", extract_method_xcf(artifact, "DiCE-like")),
    ("Growing Spheres", extract_method_xcf(artifact, "Growing Spheres")),
    ("Genetic", extract_method_xcf(artifact, "Genetic")),
]

run_tag = artifact.get("run_tag", f"rs{random_state_split}_thr{int(thr_low*100):02d}_t{int(target*100):03d}")
out_dir = CACHE_DIR

print(f"Using artifact: {ARTIFACT_PATH}")
print(f"idx={idx} | run_tag={run_tag}")

all_rows = []
final_rows = []

for method_name, xcf_dict in methods:
    if xcf_dict is None:
        print(f"\n[{method_name}] missing xcf in artifact; skipping.")
        continue

    xcf = sanitize(series_from_dict(xcf_dict, meta["cols"]), meta)
    changed = changed_features_between(x0, xcf, meta["cols"])
    if len(changed) == 0:
        print(f"\n[{method_name}] no changes (xcf==x0); skipping sweep.")
        continue
    if len(changed) > MAX_K_CHANGED:
        print(f"\n[{method_name}] too many changed features k={len(changed)} > {MAX_K_CHANGED}; skipping.")
        continue

    feats = sorted(changed)

    print("\n" + "="*92)
    print(f"{method_name}: m-sweep micro totals (artifact endpoints) | k={len(changed)}")
    print("="*92)

    for value_key in VALUE_KEYS:
        v = corner_values_v(models, x0, xcf, changed, value_key=value_key)
        pots = mobius_pots_from_v(v, len(changed))

        streak = 0
        prev = None
        last_m = None
        last_delta = None

        print("\n" + "-"*92)
        print(f"value={value_key}: sweep m={M_START}..{M_MAX} (stop after {STREAK_N} hits ≤ {EPS_PP} pp)")
        print("-"*92)

        for m in range(M_START, M_MAX + 1):
            S, delta = micro_totals_at_m(models, x0, xcf, changed, v, pots, m=m, value_key=value_key)
            sh = shares_pp(S, feats, delta)

            # store long rows
            for f, sval, spp in zip(feats, [S.get(f, 0.0) for f in feats], sh):
                all_rows.append({
                    "idx": idx,
                    "method": method_name,
                    "value_key": value_key,
                    "m": int(m),
                    "feature": f,
                    "S_micro": float(sval),
                    "deltaV": float(delta),
                    "share_pp_of_delta": float(spp),
                })

            # convergence check
            if prev is not None:
                diff = sh - prev
                max_pp = float(np.max(np.abs(diff)))
                hit = (max_pp <= EPS_PP)
                streak = (streak + 1) if hit else 0
                print(f"[m={m}] ΔV={delta:.6f} | max|Δshare|={max_pp:.4f} pp -> streak={streak}/{STREAK_N}")
                if streak >= STREAK_N:
                    last_m = m
                    last_delta = delta
                    break
            else:
                print(f"[m={m}] ΔV={delta:.6f}")

            prev = sh.copy()
            last_m = m
            last_delta = delta

        # save final rows (last_m)
        final_rows.append({
            "idx": idx,
            "method": method_name,
            "value_key": value_key,
            "m_last": int(last_m),
            "deltaV": float(last_delta),
        })

# ============================================================
# SAVE
# ============================================================
df_long = pd.DataFrame(all_rows)
df_final = pd.DataFrame(final_rows)

out_long = os.path.join(out_dir, f"msweep_long_idx{idx}_{run_tag}.csv")
out_final = os.path.join(out_dir, f"msweep_final_idx{idx}_{run_tag}.csv")

df_long.to_csv(out_long, index=False)
df_final.to_csv(out_final, index=False)

print(f"\n[Saved] {out_long}")
print(f"[Saved] {out_final}")
print("\nDONE.")