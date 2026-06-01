import warnings; warnings.filterwarnings("ignore")
import os, numpy as np, pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, roc_auc_score
from xgboost import XGBClassifier
from joblib import dump

os.makedirs("./cache_adult", exist_ok=True)

# ── 1) LOAD ────────────────────────────────────────────────────────────
print("Loading data...")
URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data"
cols = ["age","workclass","fnlwgt","education","education_num",
        "marital_status","occupation","relationship","race","sex",
        "capital_gain","capital_loss","hours_per_week","native_country","income"]
df = pd.read_csv(URL, names=cols, sep=",\s*", engine="python", na_values="?")
df = df.dropna().reset_index(drop=True)
print(f"Loaded {len(df)} rows after dropping NaN")

# ── 2) ENCODE ──────────────────────────────────────────────────────────
cat_cols = ["workclass","education","marital_status","occupation",
            "relationship","race","sex","native_country"]
encoders = {}
for c in cat_cols:
    encoders[c] = LabelEncoder()
    df[c] = encoders[c].fit_transform(df[c].astype(str))

feature_cols = ["age","workclass","education_num","marital_status",
                "occupation","relationship","race","sex",
                "capital_gain","capital_loss","hours_per_week",
                "native_country","education"]

X = df[feature_cols].astype(float)
y = (df["income"].str.contains(">50K")).astype(int)

print(f"Features: {feature_cols}")
print(f"Class balance: {y.mean():.3f} positive")

# ── 3) SPLIT ───────────────────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)
print(f"Train: {len(X_train)}  Test: {len(X_test)}")

# ── 4) TRAIN ───────────────────────────────────────────────────────────
print("\nTraining XGBoost...")
xgb = XGBClassifier(n_estimators=200, max_depth=5, learning_rate=0.1,
                    eval_metric="logloss", random_state=42)
xgb.fit(X_train, y_train)

acc = accuracy_score(y_test, xgb.predict(X_test))
auc = roc_auc_score(y_test, xgb.predict_proba(X_test)[:,1])
print(f"Test accuracy: {acc:.4f}")
print(f"Test AUC:      {auc:.4f}")

# ── 5) SAVE ────────────────────────────────────────────────────────────
dump({"model": xgb, "encoders": encoders,
      "feature_cols": feature_cols,
      "X_train": X_train, "X_test": X_test,
      "y_train": y_train, "y_test": y_test},
     "./cache_adult/adult_models.joblib")

print("\n[Saved] ./cache_adult/adult_models.joblib")
print("DONE.")