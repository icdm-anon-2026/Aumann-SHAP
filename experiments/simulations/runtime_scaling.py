from __future__ import annotations

import math
import statistics
import time
from dataclasses import dataclass
from typing import Callable, List, Optional

import numpy as np
import matplotlib.pyplot as plt


# -----------------------------
# Helpers: log combinatorics
# -----------------------------
def log_fact(n: int) -> float:
    return math.lgamma(n + 1)


def log_comb(n: int, k: int) -> float:
    if k < 0 or k > n:
        return float("-inf")
    return log_fact(n) - log_fact(k) - log_fact(n - k)


# -----------------------------
# Synthetic residual table r_p over {0..m}^k
# -----------------------------
def make_residual_table(k: int, m: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    shape = (m + 1,) * k
    return rng.standard_normal(size=shape, dtype=np.float64)


# -----------------------------
# Reduced method (grid-state closed form): Shapley within-pot
# -----------------------------
def reduced_within_pot_shapley(r: np.ndarray, k: int, m: int) -> np.ndarray:
    """
    Grid-state closed form for within-pot Shapley (uniform resolution m for all k dims).
    Returns vector of length k.
    """
    n = k * m

    # Shapley size weights: |p|! (n-|p|-1)! / n!  (log)
    log_size_weight = [
        log_fact(t) + log_fact(n - t - 1) - log_fact(n) for t in range(n)
    ]

    # Precompute log binomials log C(m, p_j)
    log_binom = [log_comb(m, pj) for pj in range(m + 1)]

    shares = np.zeros(k, dtype=np.float64)

    for p in np.ndindex(*(m + 1 for _ in range(k))):
        t = 0
        sum_log_binoms = 0.0
        for pj in p:
            t += pj
            sum_log_binoms += log_binom[pj]
        if t >= n:
            continue

        base_log = log_size_weight[t] + sum_log_binoms
        r_p = r[p]

        for i in range(k):
            pi = p[i]
            if pi >= m:
                continue

            p_plus = list(p)
            p_plus[i] = pi + 1
            delta = r[tuple(p_plus)] - r_p

            # multiplicity factor (m - pi)
            logw = base_log + math.log(m - pi)
            shares[i] += math.exp(logw) * delta

    return shares


# -----------------------------
# Naive method (micro-coalitions): exact Shapley on micro-game
# -----------------------------
def naive_within_pot_shapley(r: np.ndarray, k: int, m: int) -> np.ndarray:
    """
    Exact micro-game enumeration (2^(n-1) per micro-player).
    Uses symmetry: compute Shapley for one replica of feature i, multiply by m.
    """
    n = k * m
    w_by_s = [
        math.exp(log_fact(s) + log_fact(n - s - 1) - log_fact(n)) for s in range(n)
    ]

    # Micro-player list: feature index repeated m times
    features = []
    for j in range(k):
        features.extend([j] * m)

    shares = np.zeros(k, dtype=np.float64)

    for i in range(k):
        # remove one representative replica of feature i
        removed_idx = i * m
        others = features[:removed_idx] + features[removed_idx + 1 :]
        n1 = n - 1

        # bitmask per feature
        feat_masks = [0] * k
        for bitpos, feat in enumerate(others):
            feat_masks[feat] |= (1 << bitpos)

        total_micro = 0.0
        for mask in range(1 << n1):
            s = mask.bit_count()
            counts = [(mask & feat_masks[j]).bit_count() for j in range(k)]

            vA = r[tuple(counts)]
            counts[i] += 1
            vAplus = r[tuple(counts)]

            total_micro += w_by_s[s] * (vAplus - vA)

        shares[i] = m * total_micro

    return shares


# -----------------------------
# Timing helpers
# -----------------------------
def calibrate_loops(func: Callable[[], object], min_total_s: float = 0.15, max_loops: int = 5000) -> int:
    func()  # warmup
    loops = 1
    while loops < max_loops:
        t0 = time.perf_counter()
        for _ in range(loops):
            func()
        dt = time.perf_counter() - t0
        if dt >= min_total_s:
            return loops
        ratio = min_total_s / max(dt, 1e-12)
        loops = min(max_loops, max(loops + 1, int(math.ceil(loops * ratio))))
    return loops


def time_median(func: Callable[[], object], repeats: int = 5, min_total_s: float = 0.15) -> float:
    loops = calibrate_loops(func, min_total_s=min_total_s)
    per_call = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        for _ in range(loops):
            func()
        dt = time.perf_counter() - t0
        per_call.append(dt / loops)
    return statistics.median(per_call)


@dataclass
class Row:
    k: int
    m: int
    n: int
    G: int
    t_red_s: float
    t_naive_s: Optional[float]


def fit_c_naive(rows: List[Row]) -> Optional[float]:
    vals = []
    for r in rows:
        if r.t_naive_s is None:
            continue
        n = r.n
        vals.append(r.t_naive_s / (n * (2.0 ** n)))
    return statistics.median(vals) if vals else None


def fit_c_red(rows: List[Row]) -> float:
    vals = [r.t_red_s / (r.k * r.G) for r in rows]
    return statistics.median(vals)


def run_case(k: int, m: int, seed: int, repeats: int, max_naive_n: int) -> Row:
    n = k * m
    G = (m + 1) ** k
    r = make_residual_table(k, m, seed=seed)

    def f_red():
        return reduced_within_pot_shapley(r, k, m)

    t_red = time_median(f_red, repeats=repeats)

    t_naive = None
    if n <= max_naive_n:
        def f_na():
            return naive_within_pot_shapley(r, k, m)
        t_naive = time_median(f_na, repeats=max(3, repeats // 2))

        # (optional) correctness sanity check for feasible n
        sh_na = f_na()
        sh_re = f_red()
        maxdiff = float(np.max(np.abs(sh_na - sh_re)))
        if maxdiff > 1e-8:
            print(f"[warn] max|diff|={maxdiff:.2e} at (k={k}, m={m})")

    return Row(k=k, m=m, n=n, G=G, t_red_s=t_red, t_naive_s=t_naive)


def main():
    repeats = 5
    seed_base = 0
    max_naive_n = 20  # keep micro-coalition enumeration feasible

    # Scaling vs m (k fixed)
    k_fixed = 3
    m_max = 50
    m_values = list(range(2, m_max + 1))

    # Scaling vs k (m fixed)
    m_fixed = 4
    k_values = list(range(2, 9))

    rows_m: List[Row] = []
    print(f"Running runtime vs m (k={k_fixed}, m=2..{m_max}) ...")
    for m in m_values:
        rows_m.append(run_case(k_fixed, m, seed=seed_base + 1000 * k_fixed + m,
                               repeats=repeats, max_naive_n=max_naive_n))

    rows_k: List[Row] = []
    print(f"Running runtime vs k (m={m_fixed}, k=2..{max(k_values)}) ...")
    for k in k_values:
        rows_k.append(run_case(k, m_fixed, seed=seed_base + 1000 * k + m_fixed,
                               repeats=repeats, max_naive_n=max_naive_n))

    # Fits
    c_naive = fit_c_naive(rows_m)  # may be None
    c_red_m = fit_c_red(rows_m)
    c_red_k = fit_c_red(rows_k)

    # Arrays
    ms = np.array([r.m for r in rows_m], dtype=float)
    t_red = np.array([r.t_red_s for r in rows_m], dtype=float)
    ms_na = np.array([r.m for r in rows_m if r.t_naive_s is not None], dtype=float)
    t_na = np.array([r.t_naive_s for r in rows_m if r.t_naive_s is not None], dtype=float)

    ks = np.array([r.k for r in rows_k], dtype=float)
    t_red_k = np.array([r.t_red_s for r in rows_k], dtype=float)
    ks_na = np.array([r.k for r in rows_k if r.t_naive_s is not None], dtype=float)
    t_na_k = np.array([r.t_naive_s for r in rows_k if r.t_naive_s is not None], dtype=float)

    # ----------------------------
    # Figure 1: runtime vs m
    # ----------------------------
    fig, ax = plt.subplots(figsize=(6.2, 4.2), constrained_layout=True)
    ax.plot(ms, t_red, marker="o", label="Grid-state (measured)")
    if len(ms_na) > 0:
        ax.plot(ms_na, t_na, marker="o", label="Micro-coalitions (measured)")

    ax.plot(ms, c_red_m * (k_fixed * ((ms + 1.0) ** k_fixed)),
            linestyle="--", label=r"Fit: $c\,k(m{+}1)^k$")

    if c_naive is not None:
        n = k_fixed * ms
        ax.plot(ms, c_naive * (n * (2.0 ** n)),
                linestyle="--", label=r"Fit: $c\,n2^n$")

    ax.set_yscale("log")
    ax.set_xlabel("Grid resolution $m$")
    ax.set_ylabel("Median runtime per pot (s) [log]")
    ax.set_title(f"Runtime vs $m$ (fixed $k={k_fixed}$)")
    ax.grid(True, which="both", linestyle="--", linewidth=0.6)
    ax.legend(loc="best", frameon=True)

    # ----------------------------
    # Figure 2: runtime vs k
    # ----------------------------
    fig2, ax2 = plt.subplots(figsize=(6.2, 4.2), constrained_layout=True)
    ax2.plot(ks, t_red_k, marker="o", label="Grid-state (measured)")
    if len(ks_na) > 0:
        ax2.plot(ks_na, t_na_k, marker="o", label="Micro-coalitions (measured)")

    ax2.plot(ks, c_red_k * (ks * ((m_fixed + 1.0) ** ks)),
             linestyle="--", label=r"Fit: $c\,k(m{+}1)^k$")

    if c_naive is not None:
        n_k = ks * m_fixed
        ax2.plot(ks, c_naive * (n_k * (2.0 ** n_k)),
                 linestyle="--", label=r"Fit: $c\,n2^n$")

    ax2.set_yscale("log")
    ax2.set_xlabel("Pot size $k$")
    ax2.set_ylabel("Median runtime per pot (s) [log]")
    ax2.set_title(f"Runtime vs $k$ (fixed $m={m_fixed}$)")
    ax2.grid(True, which="both", linestyle="--", linewidth=0.6)
    ax2.legend(loc="best", frameon=True)

    plt.show()


if __name__ == "__main__":
    main()