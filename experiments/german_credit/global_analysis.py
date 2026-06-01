import os, sys, json, atexit
import numpy as np
import pandas as pd
from itertools import product
from math import lgamma, exp
from joblib import load

# =============================
# SETTINGS
# =============================
thr_low = 0.30
target  = 0.80
random_state_split = 1

DICE_N_SAMPLES = 8000
DICE_RADIUS    = 0.55
DICE_MAX_CHG   = 12
DICE_SEED_BASE = 123

M_GLOBAL = 5
TAU_REL  = 0.005          # tiny-pot fallback threshold (relative to |Δ|)

CACHE_DIR = "./cache"
os.makedirs(CACHE_DIR, exist_ok=True)

RUN_TAG = (
    f"GLOBAL_rs{random_state_split}_thr{int(thr_low*100):02d}_t{int(target*100):03d}"
    f"_diceN{DICE_N_SAMPLES}_m{M_GLOBAL}_tau{int(TAU_REL*1000):04d}_seed{DICE_SEED_BASE}"
)

MODEL_CACHE_FILE = os.path.join(CACHE_DIR, f"models_split_rs{random_state_split}.joblib")

OUT_LOG       = os.path.join(CACHE_DIR, f"OUTPUT_{RUN_TAG}.txt")
OUT_ENDPOINTS = os.path.join(CACHE_DIR, f"global_cf_endpoints_{RUN_TAG}.jsonl")
OUT_META      = os.path.join(CACHE_DIR, f"global_meta_{RUN_TAG}.csv")
OUT_LONG      = os.path.join(CACHE_DIR, f"global_long_{RUN_TAG}.csv")
OUT_AVG       = os.path.join(CACHE_DIR, f"global_avg_{RUN_TAG}.csv")
OUT_ARTIFACT  = os.path.join(CACHE_DIR, f"global_artifact_{RUN_TAG}.json")

# =============================
# LOG TO FILE (always)
# =============================
class _Tee:
    def __init__(self, *files): self.files = files
    def write(self, data):
        for f in self.files: f.write(data)
    def flush(self):
        for f in self.files: f.flush()

_log_f = open(OUT_LOG, "w", encoding="utf-8")
sys.stdout = _Tee(sys.stdout, _log_f)
sys.stderr = _Tee(sys.stderr, _log_f)
atexit.register(_log_f.close)

# =============================
# 0) LOAD DATASET + BUILD 27 FEATURES
# =============================
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

german_27 = pd.DataFrame({
    "X1": X1, "X2": X2, "X3": X3, "X4": X4, "X5": X5, "X6": X6, "X7": X7,
    "X8": X8, "X9": X9, "X10": X10, "X11": X11, "X12": X12, "X13": X13,
    "X14": X14, "X15": X15, "X16": X16, "X17": X17, "X18": X18, "X19": X19,
    "X20": X20, "X21": X21, "X22": X22, "X23": X23, "X24": X24, "X25": X25,
    "X26": X26, "X27": X27,
    "y": y
})
assert german_27.shape == (1000, 28)

X = german_27.drop(columns=["y"])
y = german_27["y"].astype(int)

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
def pretty_feat(c): return f"{c} ({FEATURE_LABELS.get(c,c)})"

# =============================
# 1) LOAD TRAINED MODELS (NO RETRAIN)
# =============================
if not os.path.exists(MODEL_CACHE_FILE):
    raise FileNotFoundError(
        f"Missing cached models: {MODEL_CACHE_FILE}\n"
        f"Put your old models_split_rs{random_state_split}.joblib in ./cache (same folder as this script)."
    )

models = load(MODEL_CACHE_FILE)
print(f"[Loaded fitted models from] {MODEL_CACHE_FILE}")

if "xgboost" not in models:
    raise KeyError("models cache missing 'xgboost' key")

xgb_model = models["xgboost"]

def row_to_df_float(x_row: pd.Series) -> pd.DataFrame:
    x = x_row.astype(float)
    return pd.DataFrame([x.values], columns=x.index)

def predict_prob_xgb(x_row: pd.Series) -> float:
    X1row = row_to_df_float(x_row)
    return float(xgb_model.predict_proba(X1row)[:, 1][0])

# =============================
# 2) META / SANITIZE / DIST
# =============================
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

meta = build_meta(X)

def sanitize(x: pd.Series) -> pd.Series:
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

def norm_l1(x0: pd.Series, x1: pd.Series) -> float:
    return float(sum(abs(float(x1[c]) - float(x0[c])) / meta["ranges"][c] for c in meta["cols"]))

# =============================
# 3) CANDIDATE POOL (XGB ONLY)
# =============================
proba_all = X.copy()
proba_all["p_xgboost"] = xgb_model.predict_proba(X)[:, 1]
pool = proba_all[proba_all["p_xgboost"] < thr_low].copy()
candidate_indices = list(pool.index)

print("\n" + "="*92)
print("GLOBAL RUN: XGB ONLY + seeded DiCE endpoints + (EQ-SPLIT vs MICRO(grid-state) vs ES)")
print("="*92)
print(f"thr_low={thr_low} | target={target} | candidates={len(candidate_indices)}")
print(f"dice: N={DICE_N_SAMPLES}, radius={DICE_RADIUS}, max_chg={DICE_MAX_CHG}, seed_base={DICE_SEED_BASE}")
print(f"micro: m={M_GLOBAL}, tau_rel={TAU_REL}")

if len(candidate_indices) == 0:
    raise ValueError("No candidates. Increase thr_low.")

# =============================
# 4) DICE-LIKE (DETERMINISTIC)
# =============================
def random_perturb(x0: pd.Series, rng: np.random.Generator, radius=0.30, max_changes=8):
    x = x0.astype(float).copy()
    cols = meta["cols"]

    m = int(rng.integers(2, max_changes + 1))
    chosen_cols = list(rng.choice(cols, size=min(m, len(cols)), replace=False))

    for c in chosen_cols:
        r = meta["ranges"][c]
        if c in meta["binary_cols"]:
            x[c] = 1.0 - float(int(x[c]))
        else:
            step = rng.normal(0, radius * r)
            step = float(int(np.sign(step) * max(1, abs(step))))
            x[c] = float(x[c]) + step
        x = sanitize(x)
    return x

def dice_like_xgb(x0: pd.Series, seed: int):
    rng = np.random.default_rng(seed)
    best = None
    for _ in range(DICE_N_SAMPLES):
        x1 = random_perturb(x0, rng, radius=DICE_RADIUS, max_changes=DICE_MAX_CHG)
        p1 = predict_prob_xgb(x1)
        if p1 >= target:
            d = norm_l1(x0, x1)
            cand = {"x": x1, "dist": d, "p": p1}
            if (best is None) or (cand["dist"] < best["dist"]):
                best = cand
    return best

# =============================
# 5) CORNER GAME / POTS / EQ-SPLIT / ES
# =============================
def changed_features_between(x0: pd.Series, x1: pd.Series) -> list[str]:
    return [c for c in meta["cols"] if float(x0[c]) != float(x1[c])]

def corner_value(x0: pd.Series, xcf: pd.Series, subset: list[str]) -> float:
    x = x0.astype(float).copy()
    for c in subset:
        x[c] = float(xcf[c])
    return predict_prob_xgb(x)

def corner_values_v(x0: pd.Series, xcf: pd.Series, changed: list[str]) -> dict[int, float]:
    k = len(changed)
    v = {}
    for mask in range(1 << k):
        subset = [changed[i] for i in range(k) if (mask >> i) & 1]
        v[mask] = float(corner_value(x0, xcf, subset))
    return v

def mobius_pots_from_v(v: dict[int, float], k: int) -> dict[int, float]:
    m = {}
    for mask in range(1, 1 << k):
        total = 0.0
        sub = mask
        while True:
            sign = -1.0 if ((mask.bit_count() - sub.bit_count()) % 2 == 1) else 1.0
            total += sign * v[sub]
            if sub == 0:
                break
            sub = (sub - 1) & mask
        m[mask] = float(total)
    return m

def equal_split_from_pots(changed: list[str], pots: dict[int, float]) -> dict[str, float]:
    phi = {c: 0.0 for c in changed}
    k = len(changed)
    for mask, pot_val in pots.items():
        s = mask.bit_count()
        members = [changed[i] for i in range(k) if (mask >> i) & 1]
        each = float(pot_val) / float(s)
        for c in members:
            phi[c] += each
    return phi

def equal_surplus_from_v(changed: list[str], v: dict[int, float]) -> dict[str, float]:
    k = len(changed)
    if k == 0:
        return {}
    v0 = float(v[0])
    vN = float(v[(1 << k) - 1])
    delta = vN - v0

    g = {c: float(v[1 << i] - v0) for i, c in enumerate(changed)}
    R = float(delta - sum(g.values()))
    each = float(R) / float(k)
    return {c: float(g[c] + each) for c in changed}

# =============================
# 6) MICRO (GRID-STATE CLOSED FORM) FOR ONE POT
# =============================
def _log_comb(n: int, k: int) -> float:
    if k < 0 or k > n:
        return float("-inf")
    return lgamma(n + 1) - lgamma(k + 1) - lgamma(n - k + 1)

def _cube_point_numeric(x0: pd.Series, xcf: pd.Series, u: list[str], p: tuple[int, ...], m: int) -> pd.Series:
    x = x0.astype(float).copy()
    for j, feat in enumerate(u):
        tj = float(p[j]) / float(m)
        x[feat] = float(x0[feat]) + tj * (float(xcf[feat]) - float(x0[feat]))
    return x

def _build_g_table(u: list[str], x0: pd.Series, xcf: pd.Series, m: int) -> np.ndarray:
    k = len(u)
    shape = (m + 1,) * k
    g = np.zeros(shape, dtype=float)
    for p in product(range(m + 1), repeat=k):
        x = _cube_point_numeric(x0, xcf, u, p, m)
        g[p] = predict_prob_xgb(x)
    return g

def _build_r_table_from_g(g: np.ndarray) -> np.ndarray:
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

def micro_shapley_gridstate_for_pot(u: list[str], x0: pd.Series, xcf: pd.Series, m: int) -> dict[str, float]:
    u = list(u)
    k = len(u)
    n = k * m

    g = _build_g_table(u, x0, xcf, m)
    r = _build_r_table_from_g(g)

    logC = [_log_comb(m, pj) for pj in range(m + 1)]
    shares = {feat: 0.0 for feat in u}

    for i_idx, i_feat in enumerate(u):
        acc = 0.0
        for p in product(range(m + 1), repeat=k):
            if p[i_idx] >= m:
                continue
            p_sum = int(sum(p))

            # shapley weight part: |p|!(n-|p|-1)! / n!
            log_w = lgamma(p_sum + 1) + lgamma(n - p_sum) - lgamma(n + 1)

            # multiplicity: prod_j C(m, p_j) and (m - p_i)
            log_mult = 0.0
            for pj in p:
                log_mult += logC[pj]
            log_mult += np.log(float(m - p[i_idx]))

            weight = exp(log_w + log_mult)

            p_next = list(p)
            p_next[i_idx] += 1
            delta = float(r[tuple(p_next)] - r[p])

            acc += weight * delta

        shares[i_feat] = float(acc)

    # enforce efficiency inside pot
    pot_val = float(r[(m,) * k])
    ssum = float(sum(shares.values()))
    if abs(ssum) > 1e-12:
        scale = pot_val / ssum
        for c in shares:
            shares[c] *= float(scale)

    return shares

def micro_totals_for_candidate(x0: pd.Series, xcf: pd.Series, changed: list[str], v: dict, pots: dict) -> tuple[dict, float]:
    k = len(changed)
    if k == 0:
        return {}, 0.0

    v0 = float(v[0])
    vN = float(v[(1 << k) - 1])
    delta_total = float(vN - v0)
    denom = max(abs(delta_total), 1e-12)

    S = {c: 0.0 for c in changed}

    # singleton pots
    for i, c in enumerate(changed):
        S[c] += float(pots.get(1 << i, 0.0))

    fb = 0
    inter = 0

    for mask, pot_val in pots.items():
        s = mask.bit_count()
        if s < 2:
            continue
        inter += 1

        members = [changed[i] for i in range(k) if (mask >> i) & 1]

        if abs(float(pot_val)) / denom < float(TAU_REL):
            each = float(pot_val) / float(s)
            for c in members:
                S[c] += each
            fb += 1
            continue

        shares_u = micro_shapley_gridstate_for_pot(members, x0, xcf, M_GLOBAL)
        for c in members:
            S[c] += float(shares_u.get(c, 0.0))

    fb_rate = float(fb / inter) if inter > 0 else 0.0
    return S, fb_rate

# =============================
# 7) MAIN LOOP + SAVE ENDPOINTS
# =============================
rows_meta = []
rows_long = []
endpoints_written = 0

with open(OUT_ENDPOINTS, "w", encoding="utf-8") as f_end:
    for t, idx in enumerate(candidate_indices, start=1):
        idx_int = int(idx)

        x0 = sanitize(X.loc[idx_int].copy())
        v0 = float(predict_prob_xgb(x0))

        cf = dice_like_xgb(x0, seed=(DICE_SEED_BASE + idx_int))
        if cf is None:
            rows_meta.append({
                "idx": idx_int,
                "found_cf": False,
                "k_changed": None,
                "v0_xgb": v0,
                "vN_xgb": None,
                "delta_xgb": None,
                "dist_l1": None,
                "fallback_rate": None,
            })
            if (t % 25) == 0 or t == len(candidate_indices):
                print(f"  processed {t}/{len(candidate_indices)}")
            continue

        xcf = sanitize(cf["x"].copy())
        vN = float(predict_prob_xgb(xcf))
        delta = float(vN - v0)
        dist = float(cf["dist"])

        changed = changed_features_between(x0, xcf)
        k_changed = int(len(changed))

        # corner game + pots on changed set
        v = corner_values_v(x0, xcf, changed)
        pots = mobius_pots_from_v(v, k_changed)

        phi_eq = equal_split_from_pots(changed, pots)
        ES = equal_surplus_from_v(changed, v)
        S_micro, fb_rate = micro_totals_for_candidate(x0, xcf, changed, v, pots)

        # save endpoints (jsonl)
        rec = {
            "idx": idx_int,
            "x0": {k: float(vv) for k, vv in x0.to_dict().items()},
            "xcf": {k: float(vv) for k, vv in xcf.to_dict().items()},
            "changed": changed,
            "v0_xgb": v0,
            "vN_xgb": vN,
            "delta_xgb": delta,
            "dist_l1": dist,
        }
        f_end.write(json.dumps(rec) + "\n")
        endpoints_written += 1

        # meta
        rows_meta.append({
            "idx": idx_int,
            "found_cf": True,
            "k_changed": k_changed,
            "v0_xgb": v0,
            "vN_xgb": vN,
            "delta_xgb": delta,
            "dist_l1": dist,
            "fallback_rate": fb_rate,
        })

        # long rows (only changed)
        for c in changed:
            rows_long.append({
                "idx": idx_int,
                "feature": c,
                "feature_pretty": pretty_feat(c),
                "phi_eq": float(phi_eq.get(c, 0.0)),
                "S_micro": float(S_micro.get(c, 0.0)),
                "ES": float(ES.get(c, 0.0)),
            })

        if (t % 25) == 0 or t == len(candidate_indices):
            print(f"  processed {t}/{len(candidate_indices)}")

meta_df = pd.DataFrame(rows_meta)
long_df = pd.DataFrame(rows_long)

print("\n" + "-"*92)
print("BATCH SUMMARY")
print("-"*92)
print("Candidates:", len(candidate_indices))
print("Found CF:", int(meta_df["found_cf"].sum()), "/", len(meta_df))
print("Endpoints written:", endpoints_written)

# =============================
# 8) AVERAGES (effective CF only)
# =============================
ALL_FEATURES = list(meta["cols"])
effective_mask = (meta_df["found_cf"] == True) & (meta_df["delta_xgb"].fillna(0.0).abs() > 1e-12)
effective_ids = meta_df.loc[effective_mask, "idx"].astype(int).tolist()

if long_df.empty or len(effective_ids) == 0:
    print("\nNo effective CF pairs to summarize.")
    meta_df.to_csv(OUT_META, index=False)
    print(f"[Saved] meta -> {OUT_META}")
    # still save artifact
    with open(OUT_ARTIFACT, "w", encoding="utf-8") as f:
        json.dump({
            "run_tag": RUN_TAG,
            "settings": {
                "thr_low": thr_low, "target": target, "random_state_split": random_state_split,
                "dice": {"N": DICE_N_SAMPLES, "radius": DICE_RADIUS, "max_changes": DICE_MAX_CHG, "seed_base": DICE_SEED_BASE},
                "micro": {"m": M_GLOBAL, "tau_rel": TAU_REL},
                "model_cache_file": MODEL_CACHE_FILE,
            },
            "counts": {
                "candidates": len(candidate_indices),
                "found_cf": int(meta_df["found_cf"].sum()),
                "effective": len(effective_ids),
            },
            "outputs": {
                "log": OUT_LOG,
                "endpoints": OUT_ENDPOINTS,
                "meta": OUT_META,
                "long": OUT_LONG,
                "avg": OUT_AVG,
            }
        }, f, indent=2)
    print(f"[Saved] artifact -> {OUT_ARTIFACT}")
    print("\nDONE.")
    raise SystemExit

wide_eq = (long_df.pivot_table(index="idx", columns="feature", values="phi_eq", aggfunc="sum")
           .reindex(index=effective_ids).fillna(0.0).reindex(columns=ALL_FEATURES, fill_value=0.0))
wide_mi = (long_df.pivot_table(index="idx", columns="feature", values="S_micro", aggfunc="sum")
           .reindex(index=effective_ids).fillna(0.0).reindex(columns=ALL_FEATURES, fill_value=0.0))
wide_es = (long_df.pivot_table(index="idx", columns="feature", values="ES", aggfunc="sum")
           .reindex(index=effective_ids).fillna(0.0).reindex(columns=ALL_FEATURES, fill_value=0.0))

avg_df = pd.DataFrame({
    "feature": ALL_FEATURES,
    "feature_pretty": [pretty_feat(c) for c in ALL_FEATURES],
    "avg_phi_eq": wide_eq.mean(axis=0).values,
    "avg_S_micro": wide_mi.mean(axis=0).values,
    "avg_ES": wide_es.mean(axis=0).values,
})
avg_df["avg(S_micro-phi_eq)"] = avg_df["avg_S_micro"] - avg_df["avg_phi_eq"]
avg_df["avg(S_micro-ES)"] = avg_df["avg_S_micro"] - avg_df["avg_ES"]

avg_df["abs_micro"] = avg_df["avg_S_micro"].abs()
avg_df = avg_df.sort_values("abs_micro", ascending=False).drop(columns=["abs_micro"])

print("\n" + "-"*92)
print(f"TOP FEATURES BY |average MICRO(grid-state) contribution| (effective CF only, n={len(effective_ids)})")
print("-"*92)
print(avg_df.head(20).to_string(index=False))

# =============================
# 9) SAVE OUTPUTS
# =============================
meta_df.to_csv(OUT_META, index=False)
long_df.to_csv(OUT_LONG, index=False)
avg_df.to_csv(OUT_AVG, index=False)

with open(OUT_ARTIFACT, "w", encoding="utf-8") as f:
    json.dump({
        "run_tag": RUN_TAG,
        "settings": {
            "thr_low": thr_low, "target": target, "random_state_split": random_state_split,
            "dice": {"N": DICE_N_SAMPLES, "radius": DICE_RADIUS, "max_changes": DICE_MAX_CHG, "seed_base": DICE_SEED_BASE},
            "micro": {"m": M_GLOBAL, "tau_rel": TAU_REL},
            "model_cache_file": MODEL_CACHE_FILE,
        },
        "counts": {
            "candidates": len(candidate_indices),
            "found_cf": int(meta_df["found_cf"].sum()),
            "effective": int(len(effective_ids)),
        },
        "outputs": {
            "log": OUT_LOG,
            "endpoints": OUT_ENDPOINTS,
            "meta": OUT_META,
            "long": OUT_LONG,
            "avg": OUT_AVG,
        }
    }, f, indent=2)

print(f"\n[Saved] endpoints -> {OUT_ENDPOINTS}")
print(f"[Saved] meta      -> {OUT_META}")
print(f"[Saved] long      -> {OUT_LONG}")
print(f"[Saved] avg       -> {OUT_AVG}")
print(f"[Saved] artifact  -> {OUT_ARTIFACT}")
print(f"[Saved] log       -> {OUT_LOG}")

print("\nDONE.")