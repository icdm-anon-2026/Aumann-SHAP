from __future__ import annotations

from typing import Callable, Dict, Iterable, Optional, Tuple, Union
import numpy as np
import pandas as pd
import warnings

from .tabular_gridstate import explain_tabular_gridstate


def _to_series(x) -> pd.Series:
    if isinstance(x, pd.Series):
        return x.astype(float)
    if isinstance(x, dict):
        return pd.Series(x).astype(float)
    raise TypeError("For grid_state backend, x0/x1 must be pandas.Series or dict.")


def _mc_totals_microplayers(
    model_batch: Callable[[np.ndarray], np.ndarray],
    x0: np.ndarray,
    x1: np.ndarray,
    m: int,
    n_perms: int,
    seed: int,
    changed_tol: float,
) -> np.ndarray:
    """
    Monte-Carlo micro-game Shapley totals (NO within-pot).
    - Players are micro-steps; we estimate totals directly via permutation paths.
    - Works for large-k (pixels) because it avoids 2^k corner enumeration.
    """
    rng = np.random.default_rng(seed)

    x0 = np.asarray(x0, dtype=np.float32)
    x1 = np.asarray(x1, dtype=np.float32)
    assert x0.shape == x1.shape, "x0 and x1 must have same shape"

    orig_shape = x0.shape
    x0f = x0.reshape(-1)
    x1f = x1.reshape(-1)

    changed = np.where(np.abs(x1f - x0f) > float(changed_tol))[0]
    d = x0f.size
    k = int(changed.size)

    totals_flat = np.zeros(d, dtype=np.float64)
    if k == 0:
        return totals_flat.reshape(orig_shape)

    delta = (x1f[changed] - x0f[changed]).astype(np.float32)
    delta_step = delta / float(m)

    # multiset of micro-players: indices 0..k-1 repeated m times
    base_micro = np.repeat(np.arange(k, dtype=np.int32), repeats=m)  # length n=k*m
    n = int(base_micro.size)

    for _ in range(int(n_perms)):
        order = base_micro[rng.permutation(n)]  # entries in [0..k-1]

        X = np.empty((n + 1, d), dtype=np.float32)
        cur = x0f.copy()
        X[0] = cur

        for t in range(1, n + 1):
            pi = int(order[t - 1])          # which changed feature
            feat_idx = int(changed[pi])     # which coordinate in x
            cur[feat_idx] += float(delta_step[pi])
            X[t] = cur

        vals = np.asarray(model_batch(X), dtype=np.float64).reshape(-1)
        if vals.shape[0] != (n + 1):
            raise ValueError("model_batch must return one score per row of X (shape [n+1]).")

        marg = vals[1:] - vals[:-1]  # [n]
        for t in range(n):
            pi = int(order[t])
            feat_idx = int(changed[pi])
            totals_flat[feat_idx] += float(marg[t])

    totals_flat /= float(n_perms)
    return totals_flat.reshape(orig_shape)
def _auto_model_batch(model: Callable, x0_schema):
    """
    Try to build an efficient batch scorer automatically.
    Returns model_batch(X)->scores where X is np.ndarray shape (B, d).
    Falls back to a safe per-row loop (slower) if no vectorized API exists.
    """
    cols = None
    if isinstance(x0_schema, pd.Series):
        cols = list(x0_schema.index)
    elif isinstance(x0_schema, dict):
        cols = list(x0_schema.keys())

    # sklearn / xgboost style
    if hasattr(model, "predict_proba"):
        def _batch(X: np.ndarray) -> np.ndarray:
            X_in = pd.DataFrame(X, columns=cols) if cols is not None else X
            P = model.predict_proba(X_in)
            P = np.asarray(P)
            return P[:, -1] if P.ndim == 2 else P.reshape(-1)
        return _batch

    if hasattr(model, "predict"):
        def _batch(X: np.ndarray) -> np.ndarray:
            X_in = pd.DataFrame(X, columns=cols) if cols is not None else X
            y = model.predict(X_in)
            return np.asarray(y).reshape(-1)
        return _batch
    # torch.nn.Module style (vision / deep models)
    try:
        import torch
        import torch.nn as nn
    except Exception:
        torch = None
        nn = None

    if torch is not None and nn is not None and isinstance(model, nn.Module) and isinstance(x0_schema, (np.ndarray, torch.Tensor)):
        # x0_schema carries the unflattened shape, e.g. (28,28) or (1,28,28)
        shape = tuple(np.asarray(x0_schema).shape)
        params = list(model.parameters())
        device = params[0].device if params else torch.device("cpu")
        model.eval()

        @torch.no_grad()
        def _batch(X: np.ndarray) -> np.ndarray:
            X = np.asarray(X, dtype=np.float32)
            B, d = X.shape
            img = torch.from_numpy(X.reshape((B,) + shape)).to(device)

            # If (B,H,W) -> make it (B,1,H,W)
            if img.ndim == 3:
                img = img.unsqueeze(1)

            logits = model(img)

            # Support binary or 2-class heads
            if logits.ndim == 1:
                probs = torch.sigmoid(logits)
            elif logits.shape[1] == 1:
                probs = torch.sigmoid(logits[:, 0])
            else:
                probs = torch.softmax(logits, dim=1)[:, -1]

            return probs.detach().cpu().numpy()

        return _batch
    # generic callable fallback (correct but slower)
    warnings.warn(
        "backend='mc': no vectorized predictor detected; falling back to per-row calls (may be slow).",
        UserWarning,
    )

    def _batch(X: np.ndarray) -> np.ndarray:
        out = np.empty((X.shape[0],), dtype=float)
        if cols is None:
            for i in range(X.shape[0]):
                out[i] = float(model(X[i]))
        else:
            for i in range(X.shape[0]):
                out[i] = float(model(pd.Series(X[i], index=cols)))
        return out

    return _batch

def explain(
    model: Callable,
    x0,
    x1,
    *,
    m: Union[int, Dict[str, int]] = 5,
    backend: str = "auto",               # "grid_state" | "mc" | "auto"
    categorical: Optional[Iterable[str]] = None,  # only used for grid_state
    tau_rel: float = 0.0,                # only used for grid_state
    k_max_exact: int = 15,               # auto cutoff
    # MC options:
    model_batch: Optional[Callable[[np.ndarray], np.ndarray]] = None,
    n_perms: int = 200,
    seed: int = 0,
    changed_tol: float = 0.0,
) -> Tuple[object, Optional[pd.DataFrame]]:
    """
    Returns (totals, within_pot).

    grid_state:
      - totals: pandas.Series (only changed feats)
      - within_pot: pandas.DataFrame

    mc:
      - totals: numpy array with same shape as x0/x1
      - within_pot: None

    Notes:
      - Within-pot requires interaction pots (Möbius) which needs 2^k corners.
        That is feasible only for small k, so we expose within-pot only in grid_state.
    """
    if backend not in {"auto", "grid_state", "mc"}:
        raise ValueError("backend must be one of: 'auto', 'grid_state', 'mc'.")

    # If grid_state or auto, we need tabular inputs (Series/dict) to compute k
    if backend in {"auto", "grid_state"}:
        x0s = _to_series(x0)
        x1s = _to_series(x1)
        changed = [c for c in x0s.index if float(x0s[c]) != float(x1s[c])]
        k = len(changed)

        if backend == "auto" and k > int(k_max_exact):
            backend = "mc"
            # Keep schema for MC (feature names) so we can build per-row Series if needed
            x0 = x0s
            x1 = x1s
        else:
            res = explain_tabular_gridstate(
                model=model,
                x0=x0s,
                x1=x1s,
                m=m,
                categorical=categorical,
                tau_rel=float(tau_rel),
            )
            return res.totals, res.within_pot

    # MC backend
    if model_batch is None:
        model_batch = _auto_model_batch(model, x0)

    m_int = int(m) if isinstance(m, int) else int(min(m.values()))
    totals = _mc_totals_microplayers(
        model_batch=model_batch,
        x0=x0,
        x1=x1,
        m=m_int,
        n_perms=int(n_perms),
        seed=int(seed),
        changed_tol=float(changed_tol),
    )
    return totals, None