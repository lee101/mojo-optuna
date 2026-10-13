"""Overfitting statistics: deflated Sharpe, CSCV/PBO, nested walk-forward, symbol-selection penalty."""

from __future__ import annotations

import itertools
import math
from typing import Any, Callable, Sequence

import numpy as np

EULER = 0.5772156649015329


def _norm_cdf(x):
    return 0.5 * (1 + math.erf(x / math.sqrt(2)))


def _norm_ppf(p: float) -> float:
    from scipy.stats import norm

    return float(norm.ppf(p))


def agg(values: Sequence[float], how: str) -> float:
    v = np.asarray([float(x) for x in values], dtype=float)
    if v.size == 0:
        return math.nan
    if how == "mean":
        return float(v.mean())
    if how == "median":
        return float(np.median(v))
    if how == "min":
        return float(v.min())
    if how == "max":
        return float(v.max())
    if how == "sum":
        return float(v.sum())
    if how == "std":
        return float(v.std(ddof=1)) if v.size > 1 else 0.0
    if how == "last":
        return float(v[-1])
    if how == "first":
        return float(v[0])
    if how == "count":
        return float(v.size)
    if how == "sharpe":
        s = v.std(ddof=1) if v.size > 1 else 0.0
        return float(v.mean() / s) if s > 0 else (math.inf if v.mean() > 0 else 0.0)
    if how == "lcb":  # mean - 1 standard error
        se = v.std(ddof=1) / math.sqrt(v.size) if v.size > 1 else 0.0
        return float(v.mean() - se)
    if how.startswith("q") and how[1:].isdigit():
        return float(np.quantile(v, int(how[1:]) / 100.0))
    raise ValueError(f"unknown aggregation {how!r}")


# ---------------------------------------------------------------- deflated Sharpe

def sharpe(r: Sequence[float]) -> float:
    r = np.asarray(r, float)
    if r.size < 2:
        return 0.0
    s = r.std(ddof=1)
    return float(r.mean() / s) if s > 0 else 0.0


def expected_max_sharpe(sr_var: float, n: float) -> float:
    if n <= 1:
        return 0.0
    return math.sqrt(max(sr_var, 0.0)) * (
        (1 - EULER) * _norm_ppf(1 - 1.0 / n) + EULER * _norm_ppf(1 - 1.0 / (n * math.e)))


def deflated_sharpe(returns: Sequence[float], trial_sharpes: Sequence[float], n_trials: float | None = None) -> dict:
    """Bailey & Lopez de Prado (2014). `returns`: the selected strategy's per-period returns."""
    r = np.asarray(returns, float)
    T = r.size
    sr = sharpe(r)
    srs = np.asarray([s for s in trial_sharpes if math.isfinite(s)], float)
    n = float(n_trials if n_trials is not None else max(1, srs.size))
    v = float(srs.var(ddof=1)) if srs.size > 1 else 0.0
    sr0 = expected_max_sharpe(v, n)
    if T < 3:
        return {"sr": sr, "sr0": sr0, "dsr": math.nan, "psr": math.nan, "T": T, "n_trials": n}
    m = r.mean()
    sd = r.std()
    g3 = float(((r - m) ** 3).mean() / sd ** 3) if sd > 0 else 0.0
    g4 = float(((r - m) ** 4).mean() / sd ** 4) if sd > 0 else 3.0
    den = 1 - g3 * sr + (g4 - 1) / 4 * sr * sr
    den = math.sqrt(den) if den > 0 else 1e-9
    dsr = _norm_cdf((sr - sr0) * math.sqrt(T - 1) / den)
    psr = _norm_cdf(sr * math.sqrt(T - 1) / den)
    return {"sr": sr, "sr0": sr0, "dsr": dsr, "psr": psr, "T": T, "n_trials": n, "sr_var": v,
            "skew": g3, "kurt": g4}


def effective_trials(M: np.ndarray) -> float:
    """N_eff = rho + (1 - rho) N from the mean pairwise correlation of trial return columns."""
    n = M.shape[1]
    if n < 3 or M.shape[0] < 3:
        return float(n)
    sd = M.std(0)
    keep = sd > 0
    if keep.sum() < 3:
        return float(n)
    C = np.corrcoef(M[:, keep].T)
    iu = np.triu_indices_from(C, 1)
    rho = float(np.clip(np.nanmean(C[iu]), 0, 1))
    return rho + (1 - rho) * n


# ---------------------------------------------------------------- CSCV / PBO

def _blocks(M: np.ndarray, max_blocks: int) -> np.ndarray:
    T = M.shape[0]
    S = min(T, max_blocks)
    S -= S % 2
    if S < 2:
        raise ValueError("CSCV needs at least 2 folds")
    parts = np.array_split(np.arange(T), S)
    return np.vstack([M[p].mean(0) for p in parts])


def cscv_np(B: np.ndarray, masks: np.ndarray):
    S, N = B.shape
    half = S // 2
    out_l = np.empty(len(masks))
    out_is = np.empty(len(masks))
    out_oos = np.empty(len(masks))
    for c, m in enumerate(masks):
        sel = np.array([(m >> s) & 1 for s in range(S)], bool)
        is_ = B[sel].sum(0) / half
        oos = B[~sel].sum(0) / half
        j = int(np.argmax(is_))
        rank = (oos < oos[j]).sum() + 0.5 * ((oos == oos[j]).sum() - 1) + 1
        w = rank / (N + 1)
        out_l[c] = math.log(w / (1 - w))
        out_is[c], out_oos[c] = is_[j], oos[j]
    return out_l, out_is, out_oos


def pbo(M: np.ndarray, max_blocks: int = 16, use_mojo: bool = True) -> dict:
    """Probability of backtest overfitting (Bailey et al. 2015).

    M: [period/fold, trial] performance, higher is better.
    """
    M = np.asarray(M, float)
    if M.ndim != 2 or M.shape[1] < 2:
        return {"pbo": math.nan, "n_trials": int(M.shape[1]) if M.ndim == 2 else 0}
    B = _blocks(M, max_blocks)
    S, N = B.shape
    masks = np.array([sum(1 << i for i in c) for c in itertools.combinations(range(S), S // 2)], dtype=np.int64)
    res = None
    if use_mojo:
        try:
            from .kernels import cscv

            res = cscv(B, masks)
        except Exception:
            res = None
    if res is None:
        res = cscv_np(B, masks)
    lam, is_, oos = res
    slope = float(np.polyfit(is_, oos, 1)[0]) if np.ptp(is_) > 0 else math.nan
    return {
        "pbo": float((lam <= 0).mean()),
        "logit_mean": float(lam.mean()),
        "blocks": S,
        "combos": int(len(masks)),
        "n_trials": N,
        "oos_of_best_mean": float(oos.mean()),
        "is_of_best_mean": float(is_.mean()),
        "p_oos_best_below_median": float((lam <= 0).mean()),
        "p_oos_best_loss": float((oos < 0).mean()),
        "degradation_slope": slope,
    }


# ---------------------------------------------------------------- nested walk-forward

def inner_folds(folds: Sequence[int], outer: int, mode: str = "before", purge: int = 0) -> list[int]:
    if mode == "before":
        return [f for f in folds if f < outer - purge]
    if mode == "other":
        return [f for f in folds if abs(f - outer) > purge]
    raise ValueError("nested inner mode must be 'before' or 'other'")


def nested_select(
    fold_metric: np.ndarray,
    folds: Sequence[int],
    outer: Sequence[int],
    select: Callable[[np.ndarray], np.ndarray],
    *,
    mode: str = "before",
    purge: int = 0,
    min_inner: int = 2,
    baseline: int | None = None,
    feasible: Callable[[list[int]], np.ndarray] | None = None,
) -> dict:
    """Post-hoc nested walk-forward on a [fold, trial] matrix (higher = better per fold).

    For each outer fold, pick the trial with the best `select(inner rows)` score and
    record where it lands on the outer fold.
    """
    folds = list(folds)
    rows = []
    for k in outer:
        inn = inner_folds(folds, k, mode, purge)
        if len(inn) < min_inner or k not in folds:
            continue
        idx = [folds.index(f) for f in inn]
        sc = select(fold_metric[idx])
        if feasible is not None:
            ok = feasible(inn)
            sc = np.where(ok, sc, -np.inf)
        if not np.isfinite(sc).any():
            continue
        j = int(np.nanargmax(sc))
        o = fold_metric[folds.index(k)]
        pct = float((o < o[j]).mean() + 0.5 * (o == o[j]).mean() - 0.5 / len(o))
        row = {"outer": k, "inner": inn, "trial_idx": j, "inner_score": float(sc[j]),
               "inner_mean": float(fold_metric[idx, j].mean()), "outer_value": float(o[j]),
               "outer_median_trial": float(np.median(o)), "outer_percentile": pct}
        if baseline is not None:
            row["baseline_inner_mean"] = float(fold_metric[idx, baseline].mean())
            row["baseline_outer"] = float(o[baseline])
        rows.append(row)
    out: dict[str, Any] = {"rows": rows, "n_outer": len(rows)}
    if rows:
        ov = np.array([r["outer_value"] for r in rows])
        im = np.array([r["inner_mean"] for r in rows])
        out.update({
            "outer_mean": float(ov.mean()),
            "outer_min": float(ov.min()),
            "inner_mean": float(im.mean()),
            "optimism": float(im.mean() - ov.mean()),
            "mean_outer_percentile": float(np.mean([r["outer_percentile"] for r in rows])),
        })
        if baseline is not None:
            bo = np.array([r["baseline_outer"] for r in rows])
            d = ov - bo
            out.update({"baseline_outer_mean": float(bo.mean()), "delta_vs_baseline": float(d.mean()),
                        "delta_se": float(d.std(ddof=1) / math.sqrt(len(d))) if len(d) > 1 else math.nan,
                        "win_rate_vs_baseline": float((d > 0).mean())})
    return out


# ---------------------------------------------------------------- symbol selection penalty

def expected_max_normal(n: int) -> float:
    if n <= 1:
        return 0.0
    return _norm_ppf((n - 0.375) / (n + 0.25))


def jaccard(a: set, b: set) -> float:
    u = a | b
    return len(a & b) / len(u) if u else 1.0


def symbol_penalty(
    scores: np.ndarray,
    subsets: list[frozenset],
    symbols: Sequence[str],
    *,
    null_scores: Sequence[float] | None = None,
    full_score: float | None = None,
    nested_rows: list[dict] | None = None,
    nested_subsets: list[frozenset] | None = None,
    top_frac: float = 0.1,
) -> dict:
    """How much of the best universe's in-sample edge is selection luck.

    scores: selection score per trial (higher better); subsets: included symbols per trial.
    null_scores: scores of the chosen params with random universes of the same size
    (``mopt symnull``); gives the expected best-of-N luck under no symbol skill.
    """
    scores = np.asarray(scores, float)
    n = len(scores)
    distinct = len(set(subsets))
    b = int(np.argmax(scores))
    null = np.asarray(null_scores if null_scores is not None else [], float)
    e_max = float(null.mean() + null.std(ddof=1) * expected_max_normal(distinct)) if null.size > 1 else math.nan
    k = max(1, int(round(n * top_frac)))
    top = np.argsort(-scores)[:k]
    freq_top = {s: float(np.mean([s in subsets[i] for i in top])) for s in symbols}
    freq_all = {s: float(np.mean([s in x for x in subsets])) for s in symbols}
    out: dict[str, Any] = {
        "n_trials": n, "distinct_subsets": distinct,
        "best_score": float(scores[b]), "best_subset": sorted(subsets[b]),
        "null_n": int(null.size),
        "null_mean": float(null.mean()) if null.size else math.nan,
        "null_sd": float(null.std(ddof=1)) if null.size > 1 else math.nan,
        "expected_max_null": e_max,
        "luck_adjusted_edge": float(scores[b] - e_max) if math.isfinite(e_max) else math.nan,
        "inclusion_lift_top": {s: round(freq_top[s] - freq_all[s], 3) for s in symbols},
    }
    if full_score is not None:
        out["full_universe_score"] = float(full_score)
        out["is_gain_vs_full"] = float(scores[b] - full_score)
    if nested_rows:
        inner = np.array([r["inner_mean"] - r.get("baseline_inner_mean", 0.0) for r in nested_rows])
        outer = np.array([r["outer_value"] - r.get("baseline_outer", 0.0) for r in nested_rows])
        out["nested_inner_gain"] = float(inner.mean())
        out["nested_outer_gain"] = float(outer.mean())
        out["selection_penalty"] = float(inner.mean() - outer.mean())
    if nested_subsets and len(nested_subsets) > 1:
        pairs = [jaccard(set(a), set(b)) for a, b in itertools.combinations(nested_subsets, 2)]
        out["outer_pick_jaccard"] = float(np.mean(pairs))
    return out
