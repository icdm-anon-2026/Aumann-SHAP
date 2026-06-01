import os, json, sys, atexit, random
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import roc_auc_score, accuracy_score, log_loss

from xgboost import XGBClassifier
from joblib import dump, load

from itertools import product
from math import comb

# ============================================================
# Settings
# ============================================================
thr_low = 0.30
target  = 0.80
random_state_split = 1

M_GLOBAL = 5
MAX_K_CHANGED = 12

CACHE_DIR = "./cache"
os.makedirs(CACHE_DIR, exist_ok=True)

# If these exist ( file : models_split_rs1.joblib needs to be in same directory)
MODEL_CACHE_FILE = os.path.join(CACHE_DIR, f"models_split_rs{random_state_split}.joblib")
FORCE_MODEL_RETRAIN = False

# CF cache (per chosen idx + settings)
FORCE_CF_RECOMPUTE = False

# Repro seed (for python/numpy; CF RNG already fixed via default_rng(random_state))
GLOBAL_SEED = 1
random.seed(GLOBAL_SEED)
np.random.seed(GLOBAL_SEED)

RUN_TAG = f"rs{random_state_split}_thr{int(thr_low*100):02d}_t{int(target*100):03d}"
OUTPUT_LOG = os.path.join(CACHE_DIR, f"OUTPUT_{RUN_TAG}.txt")
ARTIFACT_JSON = os.path.join(CACHE_DIR, f"artifact_{RUN_TAG}.json")

# ============================================================
# Simple tee logging (no sys.unraisablehook mess)
# ============================================================
class _Tee:
    def __init__(self, *files):
        self.files = files
    def write(self, data):
        for f in self.files:
            f.write(data)
    def flush(self):
        for f in self.files:
            f.flush()

_orig_out, _orig_err = sys.stdout, sys.stderr
_log_f = open(OUTPUT_LOG, "w", encoding="utf-8")
sys.stdout = _Tee(_orig_out, _log_f)
sys.stderr = _Tee(_orig_err, _log_f)

def _restore_streams():
    try:
        sys.stdout = _orig_out
        sys.stderr = _orig_err
    finally:
        try:
            _log_f.close()
        except Exception:
            pass

atexit.register(_restore_streams)

# ============================================================
# 0) Load dataset + build the 27 features
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

german_27 = pd.DataFrame({
    "X1": X1, "X2": X2, "X3": X3, "X4": X4, "X5": X5, "X6": X6, "X7": X7,
    "X8": X8, "X9": X9, "X10": X10, "X11": X11, "X12": X12, "X13": X13,
    "X14": X14, "X15": X15, "X16": X16, "X17": X17, "X18": X18, "X19": X19,
    "X20": X20, "X21": X21, "X22": X22, "X23": X23, "X24": X24, "X25": X25,
    "X26": X26, "X27": X27,
    "y": y
})

X = german_27.drop(columns=["y"])
y = german_27["y"].astype(int)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=random_state_split, stratify=y
)

# ============================================================
# 1) Build models (cache fitted)
# ============================================================
def build_models():
    logit = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=5000, solver="lbfgs", C=1e6))
    ])

    mlp = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", MLPClassifier(
            hidden_layer_sizes=(64, 32),
            activation="relu",
            alpha=1e-4,
            max_iter=2000,
            random_state=1,
            early_stopping=True,
            n_iter_no_change=20
        ))
    ])

    # keep EXACT params (don’t “improve determinism” here if you want paper compatibility)
    xgb = XGBClassifier(
        n_estimators=500,
        learning_rate=0.05,
        max_depth=3,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_lambda=1.0,
        objective="binary:logistic",
        eval_metric="logloss",
        random_state=1,
    )
    return {"logistic": logit, "mlp": mlp, "xgboost": xgb}

if (not FORCE_MODEL_RETRAIN) and os.path.exists(MODEL_CACHE_FILE):
    models = load(MODEL_CACHE_FILE)
    print(f"\n[Loaded fitted models from] {MODEL_CACHE_FILE}")
else:
    models = build_models()
    rows = []
    for name, model in models.items():
        model.fit(X_train, y_train)
        p_test = model.predict_proba(X_test)[:, 1]
        yhat = (p_test >= 0.5).astype(int)
        rows.append({
            "model": name,
            "AUC": roc_auc_score(y_test, p_test),
            "Accuracy@0.5": accuracy_score(y_test, yhat),
            "LogLoss": log_loss(y_test, p_test),
            "Mean p̂": float(np.mean(p_test)),
        })
    results = pd.DataFrame(rows).sort_values("AUC", ascending=False)
    print("\n=== Model performance on test set ===")
    print(results.to_string(index=False))
    dump(models, MODEL_CACHE_FILE)
    print(f"\n[Saved fitted models to] {MODEL_CACHE_FILE}")

# ============================================================
# 2) Predicted probabilities table (test set)
# ============================================================
def row_to_df(x_row: pd.Series) -> pd.DataFrame:
    x = x_row.astype(float)
    return pd.DataFrame([x.values], columns=x.index)

def predict_probs_all(models_dict: dict, x_row: pd.Series) -> dict:
    X1row = row_to_df(x_row)
    out = {}
    for name, m in models_dict.items():
        out[name] = float(m.predict_proba(X1row)[:, 1][0])
    out["p_min3"] = float(min(out.values()))
    out["p_max3"] = float(max(out.values()))
    return out

proba_df = X_test.copy()
proba_df["y_true"] = y_test.values
for name, model in models.items():
    proba_df[f"p_{name}"] = model.predict_proba(X_test)[:, 1]
pcols = [f"p_{name}" for name in models.keys()]
proba_df["p_max3"] = proba_df[pcols].max(axis=1)
proba_df["p_min3"] = proba_df[pcols].min(axis=1)

# ============================================================
# 3) Choose a candidate (deterministic)
# ============================================================
pool = proba_df[proba_df["p_max3"] < thr_low].copy()
print(f"\nHow many satisfy ALL three < {thr_low:.2f}:", len(pool))
if len(pool) == 0:
    raise ValueError(f"No test applicant has ALL THREE probs < {thr_low:.2f}.")

chosen = pool.sort_values("p_max3", ascending=False).head(1)
idx = int(chosen.index[0])

print("\nCHOSEN PERSON (all three < thr_low, closest below):")
print(chosen[["y_true"] + pcols + ["p_min3", "p_max3"]])

x0 = X_test.loc[idx].copy()

# ============================================================
# 4) CF utilities
# ============================================================
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

def pretty_feat(c): return f"{c} ({FEATURE_LABELS.get(c, c)})"

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

def norm_l1(x0, x1, meta) -> float:
    d = 0.0
    for c in meta["cols"]:
        d += abs(float(x1[c]) - float(x0[c])) / meta["ranges"][c]
    return float(d)

def diff_table(x0: pd.Series, x1: pd.Series) -> pd.DataFrame:
    rows = []
    for c in x0.index:
        if float(x0[c]) != float(x1[c]):
            rows.append({"feature": pretty_feat(c), "before": int(x0[c]), "after": int(x1[c])})
    return pd.DataFrame(rows)

def random_perturb(x0: pd.Series, meta, rng, radius=0.30, max_changes=8, actionable=None):
    x = x0.astype(float).copy()
    cols = meta["cols"] if actionable is None else [c for c in meta["cols"] if c in actionable]
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
        x = sanitize(x, meta)
    return x

# ============================================================
# 5) CF methods
# ============================================================
def dice_like(models_dict, x0, meta, target=0.80, n_samples=15000, radius=0.55, max_changes=12,
              actionable=None, random_state=1):
    rng = np.random.default_rng(random_state)
    best = None
    for _ in range(n_samples):
        x1 = random_perturb(x0, meta, rng, radius=radius, max_changes=max_changes, actionable=actionable)
        probs = predict_probs_all(models_dict, x1)
        if probs["p_min3"] >= target:
            d = norm_l1(x0, x1, meta)
            cand = {"x": x1, "dist": d, "probs": probs}
            if (best is None) or (cand["dist"] < best["dist"]):
                best = cand
    return best

def growing_spheres(models_dict, x0, meta, target=0.80,
                    radii=(0.05,0.10,0.18,0.25,0.35,0.45,0.60,0.80,1.00),
                    per_radius=4000, max_changes=12,
                    actionable=None, random_state=1):
    rng = np.random.default_rng(random_state)
    for rad in radii:
        best = None
        for _ in range(per_radius):
            x1 = random_perturb(x0, meta, rng, radius=rad, max_changes=max_changes, actionable=actionable)
            probs = predict_probs_all(models_dict, x1)
            if probs["p_min3"] >= target:
                d = norm_l1(x0, x1, meta)
                cand = {"x": x1, "dist": d, "probs": probs, "radius": rad}
                if (best is None) or (cand["dist"] < best["dist"]):
                    best = cand
        if best is not None:
            return best
    return None

def genetic(models_dict, x0, meta, target=0.80,
            pop_size=240, generations=100,
            init_radius=0.65, max_changes=12,
            mutation_rate=0.35, crossover_rate=0.35,
            actionable=None, random_state=1):
    rng = np.random.default_rng(random_state)

    def fitness(x):
        probs = predict_probs_all(models_dict, x)
        pmin = probs["p_min3"]
        d = norm_l1(x0, x, meta)
        if pmin < target:
            return d + 30.0 * (target - pmin)
        return d - 0.20 * (pmin - target)

    pop = [random_perturb(x0, meta, rng, radius=init_radius, max_changes=max_changes, actionable=actionable)
           for _ in range(pop_size)]
    fit = np.array([fitness(x) for x in pop], dtype=float)

    cols = meta["cols"] if actionable is None else [c for c in meta["cols"] if c in actionable]

    for _ in range(generations):
        elite_k = max(10, pop_size // 10)
        elite_idx = np.argsort(fit)[:elite_k]
        elites = [pop[i] for i in elite_idx]

        new_pop = elites.copy()
        while len(new_pop) < pop_size:
            a, b = rng.integers(0, pop_size, size=2)
            p1 = pop[a] if fit[a] < fit[b] else pop[b]
            a, b = rng.integers(0, pop_size, size=2)
            p2 = pop[a] if fit[a] < fit[b] else pop[b]

            child = p1.copy()

            if rng.random() < crossover_rate:
                swap_k = int(rng.integers(2, min(10, len(cols)) + 1))
                swap_cols = rng.choice(cols, size=swap_k, replace=False)
                for c in swap_cols:
                    child[c] = p2[c]

            if rng.random() < mutation_rate:
                child = random_perturb(child, meta, rng, radius=init_radius,
                                       max_changes=max_changes, actionable=actionable)

            child = sanitize(child, meta)
            new_pop.append(child)

        pop = new_pop[:pop_size]
        fit = np.array([fitness(x) for x in pop], dtype=float)

    best = None
    for x in pop:
        probs = predict_probs_all(models_dict, x)
        if probs["p_min3"] >= target:
            cand = {"x": x, "dist": norm_l1(x0, x, meta), "probs": probs}
            if (best is None) or (cand["dist"] < best["dist"]):
                best = cand
    return best

# ============================================================
# 6) CF cache
# ============================================================
def cf_cache_filename(idx, target, thr_low):
    t = int(round(target * 100))
    b = int(round(thr_low * 100))
    return os.path.join(CACHE_DIR, f"cf_t{t:03d}_bad{b:02d}_idx_{int(idx)}.json")

def save_cf_cache(idx, target, thr_low, x0, cf_dice, cf_gs, cf_ga):
    fn = cf_cache_filename(idx, target, thr_low)
    payload = {
        "idx": int(idx),
        "target": float(target),
        "thr_low": float(thr_low),
        "x0": x0.to_dict(),
        "cf_dice": None if cf_dice is None else cf_dice["x"].to_dict(),
        "cf_gs":   None if cf_gs   is None else cf_gs["x"].to_dict(),
        "cf_ga":   None if cf_ga   is None else cf_ga["x"].to_dict(),
    }
    with open(fn, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print(f"\n[Saved counterfactuals to] {fn}")
    return fn

def load_cf_cache(idx, target, thr_low):
    fn = cf_cache_filename(idx, target, thr_low)
    if not os.path.exists(fn):
        return None
    with open(fn, "r", encoding="utf-8") as f:
        payload = json.load(f)
    x0 = pd.Series(payload["x0"])
    xcf_dice = None if payload["cf_dice"] is None else pd.Series(payload["cf_dice"])
    xcf_gs   = None if payload["cf_gs"]   is None else pd.Series(payload["cf_gs"])
    xcf_ga   = None if payload["cf_ga"]   is None else pd.Series(payload["cf_ga"])
    return x0, xcf_dice, xcf_gs, xcf_ga, fn

# ============================================================
# 7) Run CF search
# ============================================================
meta = build_meta(X_train)

x0 = sanitize(x0.copy(), meta)
base_probs = predict_probs_all(models, x0)
print(f"\n[Chosen idx={idx}] baseline p_min3={base_probs['p_min3']:.3f} | per-model:",
      {k: round(v, 3) for k, v in base_probs.items()})

cached = None if FORCE_CF_RECOMPUTE else load_cf_cache(idx, target, thr_low)

if cached is not None:
    x0_cached, xcf_dice, xcf_gs, xcf_ga, fn = cached
    print(f"\n[Loaded counterfactuals from] {fn}")

    x0 = sanitize(x0_cached.copy(), meta)
    cf_dice = None if xcf_dice is None else {"x": sanitize(xcf_dice.copy(), meta)}
    cf_gs   = None if xcf_gs   is None else {"x": sanitize(xcf_gs.copy(), meta)}
    cf_ga   = None if xcf_ga   is None else {"x": sanitize(xcf_ga.copy(), meta)}
    cf_file = fn
else:
    cf_dice = dice_like(models, x0, meta, target=target, random_state=1)
    cf_gs   = growing_spheres(models, x0, meta, target=target, random_state=1)
    cf_ga   = genetic(models, x0, meta, target=target, random_state=1)
    cf_file = save_cf_cache(idx, target, thr_low, x0, cf_dice, cf_gs, cf_ga)

for cf in (cf_dice, cf_gs, cf_ga):
    if cf is not None and "x" in cf:
        cf["x"] = sanitize(cf["x"].copy(), meta)
        cf["probs"] = predict_probs_all(models, cf["x"])
        cf["dist"] = norm_l1(x0, cf["x"], meta)

# ============================================================
# 8) Pretty print CF results
# ============================================================
def report_cf(title, cf):
    print("\n" + "="*70)
    print(title)
    print("="*70)
    if cf is None:
        print("No counterfactual found.")
        return
    probs = cf["probs"]
    print("Target:", target)
    print("Per-model probs:", {k: round(v, 3) for k, v in probs.items()})
    print("p_min3:", round(probs["p_min3"], 3), "| L1-norm distance:", round(cf["dist"], 4))
    dt = diff_table(x0, cf["x"])
    if len(dt) == 0:
        print("\nNo changes (already at target?)")
    else:
        print("\nChanged features:")
        print(dt.to_string(index=False))

report_cf("DiCE-like random search CF", cf_dice)
report_cf("Growing Spheres CF", cf_gs)
report_cf("Genetic algorithm CF", cf_ga)

# ============================================================
# 9) Equal-split (totals)
# ============================================================
def value_scalar(models_dict, x_row: pd.Series, key="p_min3") -> float:
    probs = predict_probs_all(models_dict, x_row)
    return float(probs[key])

def equal_split_general(models_dict, x0: pd.Series, xcf: pd.Series, meta, value_key="p_min3", max_k=12):
    x0  = sanitize(x0.copy(), meta)
    xcf = sanitize(xcf.copy(), meta)

    changed = [c for c in meta["cols"] if float(x0[c]) != float(xcf[c])]
    k = len(changed)

    v0 = value_scalar(models_dict, x0, key=value_key)
    vN = value_scalar(models_dict, xcf, key=value_key)

    if k == 0:
        return {"changed": [], "phi": {}, "v0": v0, "vN": vN, "pots": {}}
    if k > max_k:
        raise ValueError(f"Too many changed features (k={k}). max_k={max_k}")

    def x_from_mask(mask: int) -> pd.Series:
        x = x0.copy()
        for i in range(k):
            if (mask >> i) & 1:
                x[changed[i]] = xcf[changed[i]]
        return sanitize(x, meta)

    v = {mask: value_scalar(models_dict, x_from_mask(mask), key=value_key) for mask in range(1 << k)}

    m_pots = {}
    for mask in range(1, 1 << k):
        total = 0.0
        sub = mask
        while True:
            sign = -1.0 if ((mask.bit_count() - sub.bit_count()) % 2 == 1) else 1.0
            total += sign * v[sub]
            if sub == 0:
                break
            sub = (sub - 1) & mask
        m_pots[mask] = float(total)

    phi = {c: 0.0 for c in changed}
    for mask, pot in m_pots.items():
        s = mask.bit_count()
        for i in range(k):
            if (mask >> i) & 1:
                phi[changed[i]] += pot / s

    return {"changed": changed, "phi": phi, "v0": float(v[0]), "vN": float(v[(1 << k) - 1]), "pots": m_pots}

# ============================================================
# 9b) Equal Surplus (Δ-version)
# ============================================================
def equal_surplus_delta(models_dict, x0: pd.Series, xcf: pd.Series, meta, value_key="p_min3", max_k=12):
    x0  = sanitize(x0.copy(), meta)
    xcf = sanitize(xcf.copy(), meta)

    changed = [c for c in meta["cols"] if float(x0[c]) != float(xcf[c])]
    k = len(changed)

    v0 = value_scalar(models_dict, x0, key=value_key)
    vN = value_scalar(models_dict, xcf, key=value_key)
    delta = float(vN - v0)

    if k == 0:
        return {"changed": [], "ES": {}, "v0": v0, "vN": vN, "delta": delta}
    if k > max_k:
        raise ValueError(f"Too many changed features (k={k}). max_k={max_k}")

    v_single = {}
    for c in changed:
        x = x0.copy()
        x[c] = xcf[c]
        x = sanitize(x, meta)
        v_single[c] = value_scalar(models_dict, x, key=value_key)

    g = {c: float(v_single[c] - v0) for c in changed}
    R = float(delta - sum(g.values()))
    ES = {c: float(g[c] + R / float(k)) for c in changed}
    return {"changed": changed, "ES": ES, "v0": v0, "vN": vN, "delta": delta, "R": R}

# ============================================================
# 10) Micro-game Shapley (totals + within-pot breakdown)
# ============================================================
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

def changed_features_between(x0: pd.Series, x1: pd.Series, cols: list) -> list:
    return [c for c in cols if float(x0[c]) != float(x1[c])]

def cube_point_numeric(x0: pd.Series, x1: pd.Series, t: dict, changed: list) -> pd.Series:
    x = x0.astype(float).copy()
    for c in changed:
        tj = float(t.get(c, 0.0))
        x[c] = float(x0[c]) + tj * (float(x1[c]) - float(x0[c]))
    return x

def g_u_numeric(models_dict, x0: pd.Series, xcf: pd.Series, u: list, t_u: dict, value_key="p_min3") -> float:
    x_t = cube_point_numeric(x0, xcf, t_u, changed=u)
    return value_scalar_float(models_dict, x_t, key=value_key)

def v_subset_float(models_dict, x0: pd.Series, xcf: pd.Series, subset, value_key="p_min3") -> float:
    x = x0.astype(float).copy()
    for c in subset:
        x[c] = float(xcf[c])
    return value_scalar_float(models_dict, x, key=value_key)

def corner_values_v(models_dict, x0: pd.Series, xcf: pd.Series, changed: list, value_key="p_min3"):
    k = len(changed)
    v = {}
    for mask in range(1 << k):
        subset = [changed[i] for i in range(k) if (mask >> i) & 1]
        v[mask] = v_subset_float(models_dict, x0, xcf, subset, value_key=value_key)
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

def residual_table_r_u(models_dict, x0: pd.Series, xcf: pd.Series, u: list, m: int = M_GLOBAL, value_key="p_min3") -> dict:
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
            total += sign * g_u_numeric(models_dict, x0, xcf, u=S, t_u=t_S, value_key=value_key)
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

def microgame_shapley_attribution(models_dict, x0: pd.Series, xcf: pd.Series, meta, m: int = M_GLOBAL, value_key: str = "p_min3"):
    x0 = sanitize(x0.copy(), meta)
    xcf = sanitize(xcf.copy(), meta)

    cols = meta["cols"]
    changed = changed_features_between(x0, xcf, cols)
    k = len(changed)

    if k == 0:
        v0 = value_scalar_float(models_dict, x0.astype(float), key=value_key)
        return changed, v0, v0, {}, {}, {}

    if k > MAX_K_CHANGED:
        raise ValueError(f"Too many changed features: k={k}, MAX_K_CHANGED={MAX_K_CHANGED}")

    v = corner_values_v(models_dict, x0, xcf, changed, value_key=value_key)
    pots = mobius_pots_from_v(v, k)

    S = {c: 0.0 for c in changed}
    for i, c in enumerate(changed):
        S[c] += float(pots.get(1 << i, 0.0))

    breakdown = {}
    EPS_POT = 1e-12
    for mask, pot_val in pots.items():
        if mask.bit_count() < 2:
            continue
        if abs(pot_val) < EPS_POT:
            continue
        u = [changed[i] for i in range(k) if (mask >> i) & 1]
        r_table = residual_table_r_u(models_dict, x0, xcf, u=u, m=m, value_key=value_key)
        shares_u = micro_shapley_shares_from_r_table(u=u, r_table=r_table, m=m)
        breakdown[tuple(u)] = shares_u
        for c in u:
            S[c] += float(shares_u[c])

    return changed, float(v[0]), float(v[(1 << k) - 1]), pots, S, breakdown

# ============================================================
# 11) Totals table + save artifacts
# ============================================================
def totals_bundle(models_dict, x0, xcf, meta, m=M_GLOBAL, value_key="p_min3"):
    res_eq = equal_split_general(models_dict, x0, xcf, meta, value_key=value_key, max_k=MAX_K_CHANGED)
    res_es = equal_surplus_delta(models_dict, x0, xcf, meta, value_key=value_key, max_k=MAX_K_CHANGED)
    changed_mg, v0, vN, pots_mg, S_micro, breakdown = microgame_shapley_attribution(
        models_dict, x0, xcf, meta, m=m, value_key=value_key
    )

    feats = sorted(set(res_eq.get("changed", [])) | set(res_es.get("changed", [])) | set(changed_mg))
    phi_eq = res_eq.get("phi", {})
    ES     = res_es.get("ES", {})

    rows = []
    for c in feats:
        pe = float(phi_eq.get(c, 0.0))
        sm = float(S_micro.get(c, 0.0))
        es = float(ES.get(c, 0.0))
        rows.append({
            "feature": pretty_feat(c),
            "phi_eq": pe,
            "S_micro": sm,
            "ES_delta": es,
            "micro_minus_eq": sm - pe,
            "micro_minus_ES": sm - es,
        })

    df = pd.DataFrame(rows)
    df["abs_S"] = df["S_micro"].abs()
    df = df.sort_values("abs_S", ascending=False).drop(columns=["abs_S"])

    # map within-pot breakdown to mask -> shares for later
    changed_list = res_eq.get("changed", [])
    pot_shares_by_mask = {}
    for u_tuple, shares_u in breakdown.items():
        mask = 0
        for i, c in enumerate(changed_list):
            if c in u_tuple:
                mask |= (1 << i)
        pot_shares_by_mask[int(mask)] = {k: float(v) for k, v in shares_u.items()}

    bundle = {
        "df": df,
        "equal_split": {
            "changed": res_eq.get("changed", []),
            "phi": {k: float(v) for k, v in res_eq.get("phi", {}).items()},
            "pots": {str(int(k)): float(v) for k, v in res_eq.get("pots", {}).items()},
            "v0": float(res_eq.get("v0", 0.0)),
            "vN": float(res_eq.get("vN", 0.0)),
        },
        "equal_surplus": {
            "changed": res_es.get("changed", []),
            "ES": {k: float(v) for k, v in res_es.get("ES", {}).items()},
            "delta": float(res_es.get("delta", 0.0)),
            "R": float(res_es.get("R", 0.0)),
            "v0": float(res_es.get("v0", 0.0)),
            "vN": float(res_es.get("vN", 0.0)),
        },
        "micro_game": {
            "changed": changed_mg,
            "v0": float(v0),
            "vN": float(vN),
            "pots": {str(int(k)): float(v) for k, v in pots_mg.items()},
            "S_micro": {k: float(v) for k, v in S_micro.items()},
            "pot_shares_by_mask": {str(k): v for k, v in pot_shares_by_mask.items()},
        }
    }
    return bundle

# ============================================================
# 12) Run + print + save
# ============================================================
x1_dice = None if cf_dice is None else cf_dice["x"]
x1_gs   = None if cf_gs   is None else cf_gs["x"]
x1_ga   = None if cf_ga   is None else cf_ga["x"]

methods = [
    ("DiCE-like", x1_dice),
    ("Growing Spheres", x1_gs),
    ("Genetic", x1_ga),
]

artifact = {
    "run_tag": RUN_TAG,
    "settings": {
        "thr_low": thr_low,
        "target": target,
        "random_state_split": random_state_split,
        "M_GLOBAL": M_GLOBAL,
        "MAX_K_CHANGED": MAX_K_CHANGED,
        "GLOBAL_SEED": GLOBAL_SEED,
        "MODEL_CACHE_FILE": MODEL_CACHE_FILE,
        "CF_CACHE_FILE": cf_file,
    },
    "idx": int(idx),
    "baseline_probs": {k: float(v) for k, v in base_probs.items()},
    "x0": {k: float(v) for k, v in x0.to_dict().items()},
    "counterfactuals": {},
    "totals": {},
}

for method_name, xcf in methods:
    if xcf is None:
        continue

    print("\n" + "#"*80)
    print(f"TOTALS COMPARISON TABLE — {method_name}")
    print("#"*80)

    bundle = totals_bundle(models, x0, xcf, meta, m=M_GLOBAL, value_key="p_min3")
    df = bundle["df"]
    print(df.to_string(index=False))

    # save per-method CSV
    safe = method_name.lower().replace(" ", "_").replace("-", "_")
    out_csv = os.path.join(CACHE_DIR, f"totals_{safe}_{RUN_TAG}.csv")
    df.to_csv(out_csv, index=False)

    # store into master artifact
    artifact["counterfactuals"][method_name] = {
        "xcf": {k: float(v) for k, v in xcf.to_dict().items()},
    }
    artifact["totals"][method_name] = {
        "table_rows": df.to_dict(orient="records"),
        "equal_split": bundle["equal_split"],
        "equal_surplus": bundle["equal_surplus"],
        "micro_game": bundle["micro_game"],
        "totals_csv": out_csv,
    }

with open(ARTIFACT_JSON, "w", encoding="utf-8") as f:
    json.dump(artifact, f, indent=2)

print(f"\n[Saved] master artifact -> {ARTIFACT_JSON}")
print(f"[Saved] log -> {OUTPUT_LOG}")
print("\nDONE.")