"""Fold-aware ask/tell engine on top of an Optuna study (journal-file storage)."""

from __future__ import annotations

import json
import math
import os
import warnings
from dataclasses import dataclass
from typing import Any

import numpy as np
import optuna
from optuna.study import StudyDirection
from optuna.trial import TrialState

from .space import UPREFIX, Space
from .stats import agg

optuna.logging.set_verbosity(optuna.logging.WARNING)
warnings.filterwarnings("ignore", category=optuna.exceptions.ExperimentalWarning)
warnings.filterwarnings("ignore", message=".*constant_liar.*")
warnings.filterwarnings("ignore", message=".*multivariate.*")
warnings.filterwarnings("ignore", message=".*group.*")

FKEY = "f:"


@dataclass
class Metric:
    metric: str
    agg: str = "mean"
    direction: str = "maximize"

    @classmethod
    def parse(cls, spec: dict[str, Any] | str, default_dir: str = "maximize") -> "Metric":
        if isinstance(spec, str):
            return cls(spec, "mean", default_dir)
        return cls(spec["metric"], spec.get("agg", "mean"), spec.get("direction", default_dir))

    @property
    def name(self) -> str:
        return f"{self.metric}.{self.agg}"

    @property
    def sign(self) -> float:
        return 1.0 if self.direction == "maximize" else -1.0

    def value(self, per_fold: list[dict[str, Any]]) -> float:
        vals = [m[self.metric] for m in per_fold if self.metric in m and m[self.metric] is not None]
        if len(vals) != len(per_fold):
            raise KeyError(f"metric {self.metric!r} missing in some folds")
        return agg(vals, self.agg)


def spread_order(folds: list[int]) -> list[int]:
    """Farthest-point order (starting at the most recent fold) so early rungs span all regimes."""
    if len(folds) <= 2:
        return list(reversed(folds))
    chosen = [len(folds) - 1]
    while len(chosen) < len(folds):
        best, bd = None, -1.0
        for i in range(len(folds) - 1, -1, -1):
            if i in chosen:
                continue
            d = min(abs(i - j) for j in chosen)
            if d > bd:
                best, bd = i, d
        chosen.append(best)
    return [folds[i] for i in chosen]


def load_config(path_or_dict: str | dict[str, Any]) -> dict[str, Any]:
    if isinstance(path_or_dict, dict):
        cfg = dict(path_or_dict)
        cfg.setdefault("_dir", os.getcwd())
        return cfg
    with open(path_or_dict) as fh:
        cfg = json.load(fh)
    cfg["_dir"] = os.path.dirname(os.path.abspath(path_or_dict))
    return cfg


def resolve(cfg: dict[str, Any], p: str) -> str:
    p = os.path.expanduser(os.path.expandvars(p))
    return p if os.path.isabs(p) else os.path.join(cfg["_dir"], p)


def make_storage(cfg: dict[str, Any]):
    s = cfg.get("storage") or f"{cfg.get('name', 'study')}.journal"
    if s.startswith(("sqlite:", "postgresql:", "mysql:")):
        return s
    if s == ":memory:":
        return optuna.storages.InMemoryStorage()
    from optuna.storages import JournalStorage
    from optuna.storages.journal import JournalFileBackend

    path = resolve(cfg, s)
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    return JournalStorage(JournalFileBackend(path))


class Engine:
    def __init__(self, cfg: dict[str, Any] | str, *, study_name: str | None = None,
                 folds: list[int] | None = None, storage=None):
        self.cfg = load_config(cfg)
        cfg = self.cfg
        self.space = Space.from_config(cfg)
        self.dists = self.space.distributions()
        self.objectives = [Metric.parse(o) for o in cfg.get("objectives", [{"metric": "value"}])]
        self.constraints = cfg.get("constraints", [])
        self.select = Metric.parse(cfg["select"]) if cfg.get("select") else self.objectives[0]
        f = cfg.get("folds", [0])
        self.folds = list(folds) if folds is not None else (list(range(f)) if isinstance(f, int) else list(f))
        order = cfg.get("fold_order", "spread")
        self.order = spread_order(self.folds) if order == "spread" else list(self.folds)
        self.seeds = list(cfg.get("seeds", [None]))
        pr = cfg.get("pruner") or {}
        self.pruner = pr if pr.get("type", "none") != "none" or pr.get("kill") else None
        self.prune_metric = Metric.parse(pr["metric"]) if pr.get("metric") else self.objectives[0]
        self.name = study_name or cfg.get("name", "study")
        from .factory import build_sampler

        self.storage = storage if storage is not None else make_storage(cfg)
        self.sampler = build_sampler(cfg, self.space, len(self.objectives))
        self.study = optuna.create_study(
            study_name=self.name, storage=self.storage, load_if_exists=True,
            directions=[o.direction for o in self.objectives], sampler=self.sampler,
        )
        if self.study.user_attrs.get("fold_order") is None:
            self.study.set_user_attr("fold_order", self.order)
            self.study.set_user_attr("config", {k: v for k, v in cfg.items() if not k.startswith("_")})
            base = cfg.get("baseline")
            if base is not None:
                self.enqueue(base, baseline=True)

    # ------------------------------------------------------------------ params
    def to_params(self, spec: dict[str, Any]) -> dict[str, Any]:
        params = {k: v for k, v in spec.items() if k in self.dists}
        u = self.space.universe
        if u is not None:
            members = spec.get("universe", "all")
            members = set(u.free if members == "all" else members)
            params.update({UPREFIX + s: int(s in members) for s in u.free})
        return params

    def enqueue(self, spec: dict[str, Any], baseline: bool = False) -> None:
        attrs = {"baseline": True} if baseline else {}
        self.study.enqueue_trial(self.to_params(spec), user_attrs=attrs, skip_if_exists=True)

    def _payload(self, number: int, params: dict[str, Any]) -> dict[str, Any]:
        out = {"trial": number, "params": self.space.describe(params), "folds": self.order, "seeds": self.seeds}
        if self.space.universe is not None:
            out["universe"] = self.space.universe.members(params)
        return out

    def ask(self) -> dict[str, Any]:
        trial = self.study.ask()
        params = {}
        for name, d in self.dists.items():
            params[name] = trial._suggest(name, d)
        return self._payload(trial.number, params)

    # ------------------------------------------------------------------ tell
    def _trial(self, number: int):
        return self.study._storage.get_trial(self._tid(number))

    def _tid(self, number: int) -> int:
        return self.study._storage.get_trial_id_from_study_id_trial_number(self.study._study_id, number)

    def fold_metrics(self, t) -> dict[int, dict[str, Any]]:
        return {int(k[len(FKEY):]): v for k, v in t.user_attrs.items() if k.startswith(FKEY)}

    def tell(self, number: int, fold: int | None = None, metrics: dict[str, Any] | None = None,
             values: list[float] | None = None) -> dict[str, Any]:
        t = self._trial(number)
        if t.state != TrialState.RUNNING:
            return {"trial": number, "state": t.state.name, "pruned": t.state == TrialState.PRUNED, "complete": True}
        if values is not None:
            self.study.tell(number, values)
            return {"trial": number, "state": "COMPLETE", "pruned": False, "complete": True, "values": values}
        if fold is None or metrics is None:
            raise ValueError("tell needs fold+metrics or values")
        if int(fold) not in self.folds:
            return {"trial": number, "state": "RUNNING", "pruned": False, "complete": False, "ignored_fold": fold}
        st = self.study._storage
        st.set_trial_user_attr(t._trial_id, f"{FKEY}{int(fold)}", metrics)
        t = self._trial(number)
        fm = self.fold_metrics(t)
        done = [f for f in self.order if f in fm]
        per = [fm[f] for f in done]
        if set(done) >= set(self.folds):
            return self._finish(number, per)
        res = {"trial": number, "state": "RUNNING", "pruned": False, "complete": False, "folds_done": len(done)}
        try:
            v = self.prune_metric.value(per)
        except KeyError:
            return res
        if len(self.objectives) == 1:
            st.set_trial_intermediate_value(t._trial_id, len(done), v)
        why = self._should_prune(number, fm, done)
        if why:
            st.set_trial_user_attr(t._trial_id, "pruned_by", why)
            self.study.tell(number, state=TrialState.PRUNED)
            res.update(state="PRUNED", pruned=True, complete=True, reason=why)
        return res

    def _finish(self, number: int, per: list[dict[str, Any]]) -> dict[str, Any]:
        try:
            values = [o.value(per) for o in self.objectives]
        except KeyError as e:
            return self.fail(number, str(e))
        cons = self.constraint_values(per)
        tid = self._tid(number)
        if cons:
            self.study._storage.set_trial_user_attr(tid, "constraints", cons)
            self.study._storage.set_trial_system_attr(tid, "constraints", cons)
        bad = [v for v in values if not math.isfinite(v)]
        if bad:
            return self.fail(number, "non-finite objective")
        self.study.tell(number, values)
        return {"trial": number, "state": "COMPLETE", "pruned": False, "complete": True, "values": values,
                "constraints": cons}

    def constraint_values(self, per: list[dict[str, Any]]) -> list[float]:
        out = []
        for c in self.constraints:
            v = Metric.parse(c).value(per)
            if "le" in c:
                out.append(v - float(c["le"]))
            elif "ge" in c:
                out.append(float(c["ge"]) - v)
        return out

    def fail(self, number: int, error: str = "") -> dict[str, Any]:
        t = self._trial(number)
        if t.state == TrialState.RUNNING:
            self.study._storage.set_trial_user_attr(t._trial_id, "error", error[-2000:])
            self.study.tell(number, state=TrialState.FAIL)
        return {"trial": number, "state": "FAIL", "pruned": False, "complete": True, "error": error[-500:]}

    # ------------------------------------------------------------------ pruning
    def _should_prune(self, number: int, fm: dict[int, dict], done: list[int]) -> str | None:
        pr = self.pruner
        if not pr:
            return None
        per = [fm[f] for f in done]
        for k in pr.get("kill", []):
            v = Metric.parse(k).value(per) if all(k["metric"] in m for m in per) else None
            if v is None:
                continue
            if ("gt" in k and v > k["gt"]) or ("lt" in k and v < k["lt"]):
                return f"kill:{k['metric']}.{k.get('agg', 'mean')}={v:.4g}"
        kind = pr.get("type", "none")
        if kind == "none":
            return None
        s = len(done)
        eta = float(pr.get("eta", pr.get("reduction_factor", 3)))
        rmin = int(pr.get("min_folds", pr.get("min_resource", 1)))
        startup = int(pr.get("n_startup", 5))
        nf = len(self.folds)
        nbk = 1
        if kind == "hyperband":
            nbk = int(pr.get("brackets", max(1, int(math.floor(math.log(max(nf / rmin, 1), eta))) + 1)))
            rmin = int(rmin * eta ** (number % nbk))
        if s < rmin:
            return None
        if kind in ("asha", "sha", "hyperband"):
            rungs, r = set(), float(rmin)
            while r < nf:
                rungs.add(int(round(r)))
                r *= eta
            if s not in rungs:
                return None
        m = self.prune_metric
        mine = m.sign * m.value(per)
        fset = set(done)
        comp = []
        for t in self.study._get_trials(deepcopy=False, use_cache=False):
            if t.number == number:
                continue
            if t.number % nbk != number % nbk:
                continue
            ofm = self.fold_metrics(t)
            if not fset <= set(ofm):
                continue
            try:
                comp.append(m.sign * m.value([ofm[f] for f in done]))
            except KeyError:
                continue
        if len(comp) < max(startup, 1):
            return None
        comp_arr = np.array(comp)
        if kind == "median":
            return f"median@{s}" if mine < float(np.median(comp_arr)) else None
        if kind == "percentile":
            q = float(pr.get("percentile", 25))
            return f"p{q:g}@{s}" if mine < float(np.percentile(comp_arr, q)) else None
        n = len(comp) + 1
        keep = max(1, int(n // eta))
        rank = int((comp_arr > mine).sum())
        return f"{kind}@{s}:rank{rank + 1}/{n}" if rank >= keep else None

    # ------------------------------------------------------------------ views
    def trials(self, states=(TrialState.COMPLETE,)):
        return [t for t in self.study.get_trials(deepcopy=False) if t.state in states]

    def is_feasible(self, t) -> bool:
        return all(c <= 0 for c in t.user_attrs.get("constraints", []))

    def best(self) -> list[dict[str, Any]]:
        done = [t for t in self.trials() if self.is_feasible(t)] or self.trials()
        if not done:
            return []
        if len(self.objectives) == 1:
            s = self.objectives[0].sign
            ts = [max(done, key=lambda t: s * t.values[0])]
        else:
            from .subset import pareto_rank

            V = np.array([[-o.sign * v for o, v in zip(self.objectives, t.values)] for t in done])
            rk = pareto_rank(V)
            ts = [t for t, r in zip(done, rk) if r == 0]
        return [self.describe(t) for t in ts]

    def describe(self, t) -> dict[str, Any]:
        out = self._payload(t.number, t.params)
        out.pop("folds", None)
        out.pop("seeds", None)
        out["state"] = t.state.name
        out["values"] = {o.name: v for o, v in zip(self.objectives, t.values or [])}
        if t.user_attrs.get("constraints"):
            out["constraints"] = t.user_attrs["constraints"]
        if t.user_attrs.get("baseline"):
            out["baseline"] = True
        return out
