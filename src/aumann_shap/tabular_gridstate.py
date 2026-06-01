from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from math import exp, lgamma
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd


@dataclass
class TabularExplanation:
    totals: pd.Series
    within_pot: pd.DataFrame
    pots: Dict[Tuple[str, ...], float]
    changed: List[str]
    meta: Dict


def _log_comb(n: int, k: int) -> float:
    if k < 0 or k > n:
        return float("-inf")
    return lgamma(n + 1) - lgamma(k + 1) - lgamma(n - k + 1)


def _changed_features(x0: pd.Series, x1: pd.Series) -> List[str]:
    cols = list(x0.index)
    return [c for c in cols if float(x0[c]) != float(x1[c])]


def _mix_point_numeric(x0: pd.Series, x1: pd.Series, feats: List[str], t: Sequence[float]) -> pd.Series:
    x = x0.astype(float).copy()
    for j, f in enumerate(feats):
        x[f] = float(x0[f]) + float(t[j]) * (float(x1[f]) - float(x0[f]))
    return x


def _mix_eval(
    model: Callable[[pd.Series], float],
    x0: pd.Series,
    x1: pd.Series,
    feats: List[str],
    t: Sequence[float],
    categorical: Optional[set] = None,
) -> float:
    """
    Endpoint-mixture evaluation for categorical coords:
    - numeric coords: interpolated
    - categorical coords: averaged over endpoints with weights t / (1-t)
    Preserves all cube corners exactly.
    """
    categorical = categorical or set()
    cat_feats = [f for f in feats if f in categorical]
    num_feats = [f for f in feats if f not in categorical]

    # Base numeric interpolation first
    x_num = _mix_point_numeric(x0, x1, num_feats, [t[feats.index(f)] for f in num_feats])

    if not cat_feats:
        # all numeric
        x_full = x_num
        for f in feats:
            if f in num_feats:
                continue
            # unreachable here
        return float(model(x_full))

    # Mixture over cat endpoint combinations
    # z=0 -> baseline value, z=1 -> counterfactual value
    total = 0.0
    for z in product([0, 1], repeat=len(cat_feats)):
        w = 1.0
        x = x_num.copy()
        for j, f in enumerate(cat_feats):
            tj = float(t[feats.index(f)])
            if z[j] == 0:
                w *= (1.0 - tj)
                x[f] = float(x0[f])
            else:
                w *= tj
                x[f] = float(x1[f])
        total += w * float(model(x))
    return float(total)


def _corner_values(model: Callable[[pd.Series], float], x0: pd.Series, x1: pd.Series, changed: List[str]) -> Dict[int, float]:
    k = len(changed)
    v: Dict[int, float] = {}
    # v(0)
    v[0] = float(model(x0))
    for mask in range(1, 1 << k):
        x = x0.astype(float).copy()
        for i in range(k):
            if (mask >> i) & 1:
                f = changed[i]
                x[f] = float(x1[f])
        v[mask] = float(model(x))
    return v


def _mobius_pots(v: Dict[int, float], k: int) -> Dict[int, float]:
    pots: Dict[int, float] = {}
    for mask in range(1, 1 << k):
        total = 0.0
        sub = mask
        while True:
            sign = -1.0 if ((mask.bit_count() - sub.bit_count()) % 2 == 1) else 1.0
            total += sign * float(v[sub])
            if sub == 0:
                break
            sub = (sub - 1) & mask
        pots[mask] = float(total)
    return pots


def _build_g_table(
    model: Callable[[pd.Series], float],
    x0: pd.Series,
    x1: pd.Series,
    u: List[str],
    m: int,
    categorical: Optional[set] = None,
) -> np.ndarray:
    k = len(u)
    shape = (m + 1,) * k
    g = np.zeros(shape, dtype=float)
    for p in product(range(m + 1), repeat=k):
        t = [pi / float(m) for pi in p]
        g[p] = _mix_eval(model, x0, x1, u, t, categorical=categorical)
    return g


def _build_r_table(g: np.ndarray) -> np.ndarray:
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


def micro_shapley_gridstate_for_pot(
    model: Callable[[pd.Series], float],
    x0: pd.Series,
    x1: pd.Series,
    u: List[str],
    m: int,
    categorical: Optional[Iterable[str]] = None,
) -> Dict[str, float]:
    u = list(u)
    k = len(u)
    n = k * m
    catset = set(categorical) if categorical is not None else set()

    g = _build_g_table(model, x0, x1, u, m, categorical=catset)
    r = _build_r_table(g)

    logC = [_log_comb(m, pj) for pj in range(m + 1)]
    shares = {feat: 0.0 for feat in u}

    for i_idx, i_feat in enumerate(u):
        acc = 0.0
        for p in product(range(m + 1), repeat=k):
            if p[i_idx] >= m:
                continue
            p_sum = int(sum(p))

            # |p|!(n-|p|-1)! / n!
            log_w = lgamma(p_sum + 1) + lgamma(n - p_sum) - lgamma(n + 1)

            # prod_j C(m, p_j) * (m - p_i)
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

    # Efficiency rescale inside pot
    pot_val = float(r[(m,) * k])
    ssum = float(sum(shares.values()))
    if abs(ssum) > 1e-12:
        scale = pot_val / ssum
        for c in shares:
            shares[c] *= float(scale)

    return shares


def explain_tabular_gridstate(
    model: Callable[[pd.Series], float],
    x0: Union[pd.Series, Dict],
    x1: Union[pd.Series, Dict],
    m: Union[int, Dict[str, int]] = 5,
    categorical: Optional[Iterable[str]] = None,
    tau_rel: float = 0.0,
) -> TabularExplanation:
    """
    Exact micro-game Shapley on the grid-state closed form (tabular, small k).

    - model: callable(pd.Series)->float (e.g., probability score)
    - x0, x1: baseline/counterfactual (Series or dict)
    - m: uniform int or per-feature dict {feat: m_i} (currently uniform is used inside each pot; per-feature can be added later)
    - categorical: optional list of features to treat with endpoint-mixture
    - tau_rel: tiny-pot fallback; if >0, pots with |phi_u|/|Delta| < tau_rel are equal-split

    Returns totals (per feature), within-pot table, and pots.
    """
    x0s = pd.Series(x0).astype(float)
    x1s = pd.Series(x1).astype(float)
    assert list(x0s.index) == list(x1s.index), "x0/x1 must have same features in same order"

    changed = _changed_features(x0s, x1s)
    k = len(changed)

    v = _corner_values(model, x0s, x1s, changed)
    pots_mask = _mobius_pots(v, k)

    v0 = float(v[0])
    vN = float(v[(1 << k) - 1]) if k > 0 else float(v0)
    Delta = float(vN - v0)
    denom = max(abs(Delta), 1e-12)

    totals = {c: 0.0 for c in changed}
    within_rows = []
    pots_named: Dict[Tuple[str, ...], float] = {}

    # singleton pots
    for i, c in enumerate(changed):
        phi_i = float(pots_mask.get(1 << i, 0.0))
        totals[c] += phi_i
        pots_named[(c,)] = phi_i

    # interaction pots
    for mask, phi_u in pots_mask.items():
        s = mask.bit_count()
        if s < 2:
            continue
        members = [changed[i] for i in range(k) if (mask >> i) & 1]
        ukey = tuple(members)
        pots_named[ukey] = float(phi_u)

        # tiny-pot equal split fallback
        if tau_rel > 0.0 and abs(float(phi_u)) / denom < float(tau_rel):
            each = float(phi_u) / float(s)
            for c in members:
                totals[c] += each
                within_rows.append({"pot": ukey, "feature": c, "share": each, "phi_u": float(phi_u), "mode": "equal_split_fallback"})
            continue

        m_u = int(m) if isinstance(m, int) else int(min(m.get(c, 1) for c in members))
        shares_u = micro_shapley_gridstate_for_pot(model, x0s, x1s, members, m=m_u, categorical=categorical)

        for c in members:
            sc = float(shares_u.get(c, 0.0))
            totals[c] += sc
            within_rows.append({"pot": ukey, "feature": c, "share": sc, "phi_u": float(phi_u), "mode": f"micro_shapley_m{m_u}"})

    totals_series = pd.Series(totals).reindex(changed).fillna(0.0)
    within_df = pd.DataFrame(within_rows)

    return TabularExplanation(
        totals=totals_series,
        within_pot=within_df,
        pots=pots_named,
        changed=changed,
        meta={"Delta": Delta, "v0": v0, "vN": vN, "m": m, "tau_rel": tau_rel},
    )