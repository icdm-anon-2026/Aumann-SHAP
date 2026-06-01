import numpy as np
import pandas as pd
from aumann_shap import explain

def g(x: pd.Series) -> float:
    return float(np.tanh(0.8*x["x1"] + 0.4*x["x2"] + 0.6*x["x1"]*x["x2"]))

x0 = pd.Series({"x1": 0.0, "x2": 0.0})
x1 = pd.Series({"x1": 1.0, "x2": 1.0})

totals, within_pot = explain(g, x0, x1, backend="mc", m=10, n_perms=20, seed=0)

assert within_pot is None
assert len(totals) == 2
print("MC autobatch fallback smoke test OK:", totals)