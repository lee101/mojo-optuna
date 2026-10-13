"""Study report: best/pareto, PBO, deflated Sharpe, nested walk-forward, symbol-selection penalty."""

from __future__ import annotations

import collections
import json
import math
from typing import Any

import numpy as np
from optuna.trial import TrialState

from .engine import Engine, Metric
from .space import UPREFIX
from .stats import agg, deflated_sharpe, effective_trials, nested_select, pbo, sharpe, symbol_penalty


def fold_matrix(e: Engine, metric: Metric, trials=None):
    trials = trials if trials is not None else e.trials()
    folds = sorted(e.folds)
    cols, keep = [], []
    for t in trials:
        fm = e.fold_metrics(t)
        if not set(folds) <= set(fm) or not all(metric.metric in fm[f] for f in folds):
            continue
        cols.append([metric.sign * float(fm[f][metric.metric]) for f in folds])
        keep.append(t)
    M = np.array(cols, float).T if cols else np.zeros((len(folds), 0))
    return folds, M, keep


def build(e: Engine, *, outer: list[int] | None = None, top: int = 5) -> dict[str, Any]:
    cfg = e.cfg
    allt = e.study.get_trials(deepcopy=False)
    states = collections.Counter(t.state.name for t in allt)
    pruned_by = collections.Counter((t.user_attrs.get("pruned_by", "?").split("@")[0]) for t in allt
                                    if t.state == TrialState.PRUNED)
    rep: dict[str, Any] = {"study": e.name, "states": dict(states), "pruned_by": dict(pruned_by),
                           "objectives": [o.name + ":" + o.direction for o in e.objectives],
                           "select": e.select.name, "folds": e.folds}
    rep["best"] = e.best()[: max(top, 1) if len(e.objectives) == 1 else 50]
    sel = e.select
    folds, M, trials = fold_matrix(e, sel)
    rep["n_full_trials"] = len(trials)
    if not trials:
        return rep
    sc = np.array([agg(M[:, j] * sel.sign, sel.agg) * sel.sign for j in range(M.shape[1])])
    ranked = np.argsort(-sc)
    rep["top"] = [{"trial": trials[j].number, "select": round(float(sc[j] * sel.sign), 6),
                   "params": e.space.describe(trials[j].params),
                   **({"n_universe": len(e.space.universe.members(trials[j].params))} if e.space.universe else {})}
                  for j in ranked[:top]]
    b = int(ranked[0])
    of: dict[str, Any] = {}
    if M.shape[0] >= 2 and M.shape[1] >= 2:
        of["pbo"] = pbo(M)
    srs = [sharpe(M[:, j]) for j in range(M.shape[1])]
    neff = effective_trials(M)
    of["dsr"] = deflated_sharpe(M[:, b], srs, n_trials=neff)
    of["dsr"]["n_trials_raw"] = len(trials)
    of["dsr"]["note"] = f"per-fold {sel.metric} of best-by-{sel.name}"
    ncfg = cfg.get("nested") or {}
    outer = outer or ncfg.get("outer") or folds[len(folds) // 2:]
    base_idx = next((j for j, t in enumerate(trials) if t.user_attrs.get("baseline")), None)

    def select_fn(rows: np.ndarray) -> np.ndarray:
        return np.array([agg(rows[:, j] * sel.sign, sel.agg) * sel.sign for j in range(rows.shape[1])])

    fms = [e.fold_metrics(t) for t in trials]

    def feasible(inn: list[int]) -> np.ndarray:
        if not e.constraints:
            return np.ones(len(trials), bool)
        return np.array([all(c <= 0 for c in e.constraint_values([fm[f] for f in inn])) for fm in fms])

    nest = nested_select(M, folds, outer, select_fn, mode=ncfg.get("mode", "before"),
                         purge=int(ncfg.get("purge", 0)), min_inner=int(ncfg.get("min_inner", 2)),
                         baseline=base_idx, feasible=feasible)
    for r in nest["rows"]:
        r["trial"] = trials[r.pop("trial_idx")].number
        r["inner"] = f"{r['inner'][0]}..{r['inner'][-1]}" if r["inner"] else ""
    of["nested_posthoc"] = nest
    uni = e.space.universe
    if uni is not None and uni.free:
        subsets = [frozenset(uni.members(t.params)) for t in trials]
        sn = e.study.user_attrs.get("symnull") or {}
        full = [j for j, t in enumerate(trials) if all(int(t.params.get(UPREFIX + s, 0)) for s in uni.free)]
        tn = {t.number: j for j, t in enumerate(trials)}
        rows = nest["rows"] if base_idx is not None else None
        of["symbol_selection"] = symbol_penalty(
            sc * sel.sign, subsets, uni.free, null_scores=sn.get("scores"),
            full_score=float(sc[full[0]] * sel.sign) if full else None,
            nested_rows=rows, nested_subsets=[subsets[tn[r["trial"]]] for r in nest["rows"]])
        if base_idx is None:
            of["symbol_selection"]["note"] = "add a baseline (e.g. full universe) for nested selection penalty"
    rep["overfit"] = of
    return rep


def _f(x: Any, nd: int = 4) -> str:
    if isinstance(x, float):
        return "nan" if math.isnan(x) else f"{x:.{nd}g}"
    return str(x)


def text(rep: dict[str, Any]) -> str:
    L = [f"study {rep['study']} states {rep['states']} pruned_by {rep.get('pruned_by', {})}",
         f"objectives {rep['objectives']} select {rep['select']} folds {len(rep['folds'])} full-trials {rep.get('n_full_trials', 0)}"]
    for b in rep.get("best", [])[:10]:
        L.append(f"best t{b['trial']} " + " ".join(f"{k}={_f(v)}" for k, v in b["values"].items())
                 + f" params={json.dumps(b['params'])}" + (f" U={b['universe']}" if "universe" in b else ""))
    for t in rep.get("top", []):
        L.append(f"top t{t['trial']} {rep['select']}={_f(t['select'])}" + (f" |U|={t['n_universe']}" if "n_universe" in t else ""))
    of = rep.get("overfit", {})
    if "pbo" in of:
        p = of["pbo"]
        L.append(f"PBO {_f(p['pbo'], 3)} (CSCV S={p['blocks']} combos={p['combos']} N={p['n_trials']}) "
                 f"oos_best={_f(p['oos_of_best_mean'])} is_best={_f(p['is_of_best_mean'])} slope={_f(p['degradation_slope'], 3)}")
    if "dsr" in of:
        d = of["dsr"]
        L.append(f"DSR {_f(d['dsr'], 3)} PSR {_f(d['psr'], 3)} SR={_f(d['sr'], 3)} SR0={_f(d['sr0'], 3)} T={d['T']} "
                 f"N_eff={_f(d['n_trials'], 3)}/{d['n_trials_raw']} ({d['note']})")
    n = of.get("nested_posthoc")
    if n and n.get("n_outer"):
        s = (f"nested(post-hoc) outer={n['n_outer']} outer_mean={_f(n['outer_mean'])} inner_mean={_f(n['inner_mean'])} "
             f"optimism={_f(n['optimism'])} pctile={_f(n['mean_outer_percentile'], 3)}")
        if "delta_vs_baseline" in n:
            s += f" vs_baseline={_f(n['delta_vs_baseline'])}±{_f(n['delta_se'], 3)} win={_f(n['win_rate_vs_baseline'], 3)}"
        L.append(s)
    u = of.get("symbol_selection")
    if u:
        s = f"symbols best={_f(u['best_score'])} distinct={u['distinct_subsets']}"
        if u.get("null_n"):
            s += (f" null(n={u['null_n']}) mean={_f(u['null_mean'])} sd={_f(u['null_sd'])} "
                  f"E[max null]={_f(u['expected_max_null'])} luck_adj={_f(u['luck_adjusted_edge'])}")
        if "is_gain_vs_full" in u:
            s += f" IS_gain_vs_full={_f(u['is_gain_vs_full'])}"
        if "selection_penalty" in u:
            s += f" nested_gain in={_f(u['nested_inner_gain'])} out={_f(u['nested_outer_gain'])} penalty={_f(u['selection_penalty'])}"
        if "outer_pick_jaccard" in u:
            s += f" pick_jaccard={_f(u['outer_pick_jaccard'], 3)}"
        L.append(s)
        lift = sorted(u["inclusion_lift_top"].items(), key=lambda kv: -kv[1])
        L.append("symbol lift(top10%) " + " ".join(f"{k}:{v:+.2f}" for k, v in lift[:8]) + " ... "
                 + " ".join(f"{k}:{v:+.2f}" for k, v in lift[-4:]))
    return "\n".join(L)
