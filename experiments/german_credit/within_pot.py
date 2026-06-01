# within-pot.py  (cache-based, no artifact needed)
# Uses:
#   ./cache/models_split_rs1.joblib
#   ./cache/cf_t080_bad30_idx_242.json
# Prints + saves within-pot splits: equal-in-pot vs micro-in-pot for each CF method.

import os, json
import numpy as np
import pandas as pd
from itertools import product
from math import comb
from joblib import load

# =========================
# Settings
# =========================
CACHE_DIR = "./cache"
MODEL_CACHE_FILE = os.path.join(CACHE_DIR, "models_split_rs1.joblib")
CF_CACHE_FILE    = os.path.join(CACHE_DIR, "cf_t080_bad30_idx_242.json")

M_GLOBAL = 5
MAX_K_CHANGED = 12
VALUE_KEY = "p_min3"   # same as your paper/table S2

os.makedirs(CACHE_DIR, exist_ok=True)

# =========================
# Feature labels (for printing)
# =========================
FEATURE_LABELS = {
    "X1": "duration (months)",
    "X2": "amount (credit amount)",
    "X3": "installment_rate",
    "X4": "residence (years at current address)",
    "X5": "existing_credits",
    "X6": "liable_people",
    "X7": "telephone == A192",
    "X8": "checking in {A12,A13,A14}",
    "X9": "checking == A13",
    "X10": "savings in {A62,A63,A64}",
    "X11": "savings in {A63,A64}",
    "X12": "credit_history == A33",
    "X13": "credit_history == A30",
    "X14": "credit_history == A34",
    "X15": "other_plans == A141",
    "X16": "other_debtors == A102",
    "X17": "other_debtors == A103",
    "X18": "employment == A71",
    "X19": "employment == A72",
    "X20": "employment in {A74,A75}",
    "X21": "personal_status_sex in {A91,A93,A94}",
    "X22": "foreign_worker == A201",
    "X23": "personal_status_sex in {A93,A95}",
    "X24": "age",
    "X25": "housing == A152",
    "X26": "housing == A151",
    "X27": "job in {A173,A174}",
}

def pretty_feat(c: str) -> str:
    return f"{c} ({FEATURE_LABELS.get(c, c)})"

# =========================
# Load cached models + CF endpoints
# =========================
if not os.path.exists(MODEL_CACHE_FILE):
    raise FileNotFoundError(f"Missing {MODEL_CACHE_FILE}. Put it in ./cache/.")

if not os.path.exists(CF_CACHE_FILE):
    raise FileNotFoundError(f"Missing {CF_CACHE_FILE}. Put it in ./cache/.")

models = load(MODEL_CACHE_FILE)

with open(CF_CACHE_FILE, "r", encoding="utf-8") as f:
    cf_payload = json.load(f)

x0 = pd.Series(cf_payload["x0"])
xcf_dice = None if cf_payload.get("cf_dice") is None else pd.Series(cf_payload["cf_dice"])
xcf_gs   = None if cf_payload.get("cf_gs")   is None else pd.Series(cf_payload["cf_gs"])
xcf_ga   = None if cf_payload.get("cf_ga")   is None else pd.Series(cf_payload["cf_ga"])

# =========================
# Minimal meta/sanitize (just to be safe)
# =========================
# We rebuild bounds from endpoints only (no dataset needed here).
ALL_COLS = list(x0.index)

def build_meta_from_endpoints(x0_: pd.Series, x1_: pd.Series):
    cols = list(x0_.index)
    bounds = {}
    ranges = {}
    binary_cols = []
    for c in cols:
        lo = float(min(x0_[c], x1_[c]))
        hi = float(max(x0_[c], x1_[c]))
        bounds[c] = (lo, hi)
        r = hi - lo
        ranges[c] = r if r > 0 else 1.0
        if set([float(x0_[c]), float(x1_[c])]).issubset({0.0, 1.0}):
            binary_cols.append(c)
    return {"cols": cols, "bounds": bounds, "ranges": ranges, "binary_cols": binary_cols}

def sanitize(x: pd.Series, meta) -> pd.Series:
    x = x.astype(float).copy()
    for c in meta["cols"]:
        lo, hi = meta["bounds"][c]
        v = float(x[c])
        v = min(max(v, lo), hi)
        if c in meta["binary_cols"]:
            v = 1.0 if v >= 0.5 else 0.0
        else:
            v = float(int(round(v)))
        x[c] = v
    return x

# =========================
# Model evaluation
# =========================
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

def value_scalar_float(models_dict, x_row: pd.Series, key="p_min3") -> float:
    return float(predict_probs_all_float(models_dict, x_row)[key])

# =========================
# Equal-split pots + within-pot equal split
# =========================
def changed_features_between(xa: pd.Series, xb: pd.Series, cols: list) -> list:
    return [c for c in cols if float(xa[c]) != float(xb[c])]

def v_subset_float(models_dict, x0_: pd.Series, xcf_: pd.Series, subset, value_key="p_min3") -> float:
    x = x0_.astype(float).copy()
    for c in subset:
        x[c] = float(xcf_[c])
    return value_scalar_float(models_dict, x, key=value_key)

def corner_values_v(models_dict, x0_: pd.Series, xcf_: pd.Series, changed: list, value_key="p_min3"):
    k = len(changed)
    v = {}
    for mask in range(1 << k):
        subset = [changed[i] for i in range(k) if (mask >> i) & 1]
        v[mask] = v_subset_float(models_dict, x0_, xcf_, subset, value_key=value_key)
    return v

def mobius_pots_from_v(v: dict, k: int) -> dict:
    phi = {}
    for mask in range(1, 1 << k):
        total = 0.0
        sub = mask
        while True:
            sign = -1.0 if ((mask.bit_count() - sub.bit_count()) % 2 == 1) else 1.0
            total += sign * v[sub]
            if sub == 0:
                break
            sub = (sub - 1) & mask
        phi[mask] = float(total)
    return phi

# =========================
# Micro-game within-pot shares
# =========================
def cube_point_numeric(x0_: pd.Series, x1_: pd.Series, t: dict, changed: list) -> pd.Series:
    x = x0_.astype(float).copy()
    for c in changed:
        tj = float(t.get(c, 0.0))
        x[c] = float(x0_[c]) + tj * (float(x1_[c]) - float(x0_[c]))
    return x

def g_u_numeric(models_dict, x0_: pd.Series, xcf_: pd.Series, u: list, t_u: dict, value_key="p_min3") -> float:
    x_t = cube_point_numeric(x0_, xcf_, t_u, changed=u)
    return value_scalar_float(models_dict, x_t, key=value_key)

def residual_table_r_u(models_dict, x0_: pd.Series, xcf_: pd.Series, u: list, m: int = M_GLOBAL, value_key="p_min3") -> dict:
    u = list(u)
    d = len(u)
    r = {}
    for p_tuple in product(range(m + 1), repeat=d):
        t_all = {u[j]: float(p_tuple[j]) / float(m) for j in range(d)}
        total = 0.0
        for Smask in range(1 << d):
            S = [u[j] for j in range(d) if (Smask >> j) & 1]
            sign = -1.0 if ((d - len(S)) % 2 == 1) else 1.0
            t_S = {c: t_all[c] for c in S}
            total += sign * g_u_numeric(models_dict, x0_, xcf_, u=S, t_u=t_S, value_key=value_key)
        r[p_tuple] = float(total)
    return r

def micro_shapley_shares_from_r_table(u: list, r_table: dict, m: int = M_GLOBAL) -> dict:
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
        ranges[i_idx] = range(m)

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

            w0 = r_table[tuple(a)]
            w1 = r_table[tuple(a_next)]
            phi_rep += weight * (w1 - w0)

        shares[i_feat] = float(m) * float(phi_rep)

    return shares

def pot_shares_micro(models_dict, x0_: pd.Series, xcf_: pd.Series, changed: list, pots: dict, m: int, value_key: str):
    # returns pot_shares[mask] = {feat: share_in_that_pot}
    pot_shares = {}
    for mask, pot_val in pots.items():
        if int(mask).bit_count() < 2:
            continue
        u = [changed[i] for i in range(len(changed)) if ((mask >> i) & 1)]
        r_table = residual_table_r_u(models_dict, x0_, xcf_, u=u, m=m, value_key=value_key)
        shares_u = micro_shapley_shares_from_r_table(u=u, r_table=r_table, m=m)
        pot_shares[int(mask)] = shares_u
    return pot_shares

# =========================
# Build + print within-pot tables
# =========================
def within_pot_tables(method_name: str, x0_: pd.Series, xcf_: pd.Series):
    meta = build_meta_from_endpoints(x0_, xcf_)
    x0s = sanitize(x0_.copy(), meta)
    x1s = sanitize(xcf_.copy(), meta)

    changed = changed_features_between(x0s, x1s, meta["cols"])
    k = len(changed)
    if k == 0:
        print(f"\n[{method_name}] no changed features.")
        return None, None

    if k > MAX_K_CHANGED:
        raise ValueError(f"k={k} too large for safety MAX_K_CHANGED={MAX_K_CHANGED}")

    v = corner_values_v(models, x0s, x1s, changed, value_key=VALUE_KEY)
    pots = mobius_pots_from_v(v, k)  # mask -> pot value

    ps_micro = pot_shares_micro(models, x0s, x1s, changed, pots, m=M_GLOBAL, value_key=VALUE_KEY)

    rows = []
    for mask, pot_val in pots.items():
        mask = int(mask)
        s = mask.bit_count()
        if s < 2:
            continue
        members = [changed[i] for i in range(k) if ((mask >> i) & 1)]
        eq_each = float(pot_val) / float(len(members))
        shares_u = ps_micro.get(mask, {})
        for c in members:
            mic = float(shares_u.get(c, np.nan))
            rows.append({
                "method": method_name,
                "pot_mask": mask,
                "pot_size": len(members),
                "pot_value": float(pot_val),
                "pot": " + ".join(pretty_feat(x) for x in members),
                "feature": pretty_feat(c),
                "equal_in_pot": eq_each,
                "micro_in_pot": mic,
                "micro_minus_equal": mic - eq_each,
            })

    df_long = pd.DataFrame(rows)
    df_pot = (
        df_long.groupby(["method", "pot_mask", "pot_size", "pot_value", "pot"], as_index=False)
        .agg(equal_sum=("equal_in_pot", "sum"), micro_sum=("micro_in_pot", "sum"))
    )
    df_pot["equal_ok"] = (df_pot["equal_sum"] - df_pot["pot_value"]).abs() <= 1e-6
    df_pot["micro_ok"] = (df_pot["micro_sum"] - df_pot["pot_value"]).abs() <= 1e-6
    df_pot["abs_pot"] = df_pot["pot_value"].abs()

    df_pot_show = df_pot.sort_values("abs_pot", ascending=False).drop(columns=["abs_pot"])
    print("\n" + "=" * 88)
    print(f"WITHIN-POT SUMMARY — {method_name}")
    print("=" * 88)
    print(df_pot_show.to_string(index=False))

    print("\n" + "-" * 88)
    print(f"WITHIN-POT DETAILS — {method_name}")
    print("-" * 88)
    df_long = df_long.sort_values(["pot_value", "pot", "feature"], ascending=[False, True, True])
    print(df_long.to_string(index=False))

    # save
    tag = "rs1_thr30_t080"
    out_sum  = os.path.join(CACHE_DIR, f"within_pot_summary_{method_name.replace(' ','_')}_{tag}.csv")
    out_long = os.path.join(CACHE_DIR, f"within_pot_long_{method_name.replace(' ','_')}_{tag}.csv")
    df_pot_show.to_csv(out_sum, index=False)
    df_long.to_csv(out_long, index=False)
    print(f"\n[Saved] {out_sum}")
    print(f"[Saved] {out_long}")

    return df_pot_show, df_long

# =========================
# Run for your 3 CFs
# =========================
pairs = [
    ("DiCE-like", xcf_dice),
    ("Growing Spheres", xcf_gs),
    ("Genetic", xcf_ga),
]

for name, xcf in pairs:
    if xcf is None:
        print(f"\n[{name}] missing CF in {CF_CACHE_FILE}")
        continue
    within_pot_tables(name, x0, xcf)

print("\nDONE.")