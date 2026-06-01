import numpy as np
import pandas as pd
from aumann_shap import explain

# ----- grid_state smoke (tabular) -----
def model_series(x: pd.Series) -> float:
    return float(x.sum())

x0 = {"a": 0.0, "b": 0.0, "c": 0.0}
x1 = {"a": 1.0, "b": 2.0, "c": 0.0}
tot, within = explain(model_series, x0, x1, m=3, backend="grid_state")
print("grid_state totals:", tot.to_dict())
print("grid_state within_pot rows:", 0 if within is None else len(within))

# ----- mc smoke (vector) -----
def model_batch(X: np.ndarray) -> np.ndarray:
    return X.sum(axis=1)  # [B]

x0v = np.array([0, 0, 0, 0], dtype=np.float32)
x1v = np.array([1, 0, 2, 0], dtype=np.float32)
tot2, within2 = explain(lambda x: None, x0v, x1v, m=5, backend="mc", model_batch=model_batch, n_perms=50, seed=0)
print("mc totals:", tot2)
print("mc within_pot:", within2)