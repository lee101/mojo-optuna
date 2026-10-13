"""Symbol-universe (subset) search: cardinality repair, evolutionary and stepwise samplers."""

from __future__ import annotations

from typing import Any

import numpy as np
from optuna.distributions import BaseDistribution, CategoricalDistribution
from optuna.samplers import BaseSampler, RandomSampler
from optuna.trial import FrozenTrial, TrialState

from .gp import scalarize
from .space import UPREFIX, Universe, from_unit, to_unit


def _key(params: dict[str, Any]) -> tuple:
    return tuple(sorted((k, v if not isinstance(v, float) else round(v, 12)) for k, v in params.items()))


def pareto_rank(F: np.ndarray) -> np.ndarray:
    """Non-dominated sorting rank (0 = front) for minimisation rows."""
    n = len(F)
    rank = np.full(n, -1)
    dom = (F[:, None, :] <= F[None, :, :]).all(-1) & (F[:, None, :] < F[None, :, :]).any(-1)
    count = dom.sum(0)
    r, cur = 0, np.where(count == 0)[0]
    while len(cur):
        rank[cur] = r
        count = count - dom[cur].sum(0)
        count[rank >= 0] = -1
        cur = np.where(count == 0)[0]
        r += 1
    return rank


def fitness(study, trials: list[FrozenTrial], rng) -> np.ndarray:
    """Lower is better. Multi-objective: pareto rank + scalarised tie-break."""
    if not trials:
        return np.zeros(0)
    f = scalarize(study, trials, np.random.RandomState(0))
    if len(study.directions) == 1:
        return f
    V = np.array([t.values for t in trials], dtype=float)
    for j, d in enumerate(study.directions):
        if d.name == "MAXIMIZE":
            V[:, j] = -V[:, j]
    rk = pareto_rank(V).astype(float)
    return rk + (f - f.min()) / ((f.max() - f.min()) or 1.0) * 0.5


class UniverseRepair(BaseSampler):
    """Wraps any sampler; always returns universe bits jointly and enforces cardinality."""

    def __init__(self, base: BaseSampler, universe: Universe, seed: int | None = None):
        self.base, self.universe = base, universe
        self._rng = np.random.RandomState(seed)
        self._udists = {n: CategoricalDistribution([0, 1]) for n in universe.params}

    def reseed_rng(self) -> None:
        self._rng.seed()
        self.base.reseed_rng()

    def infer_relative_search_space(self, study, trial):
        space = dict(self.base.infer_relative_search_space(study, trial))
        space.update(self._udists)
        return space

    def sample_relative(self, study, trial, search_space) -> dict[str, Any]:
        base_space = self.base.infer_relative_search_space(study, trial)
        sub = {k: v for k, v in search_space.items() if k in base_space}
        out = dict(self.base.sample_relative(study, trial, sub)) if sub else {}
        names = self.universe.params
        if not names:
            return out
        if all(n in out for n in names):
            mask = self.universe.repair({n: out[n] for n in names}, self._rng)
        else:
            mask = self.universe.random_mask(self._rng)
        out.update(mask)
        return out

    def sample_independent(self, study, trial, param_name, param_distribution):
        return self.base.sample_independent(study, trial, param_name, param_distribution)

    def before_trial(self, study, trial):
        self.base.before_trial(study, trial)

    def after_trial(self, study, trial, state, values):
        self.base.after_trial(study, trial, state, values)


def _finished(study) -> list[FrozenTrial]:
    return [t for t in study._get_trials(deepcopy=False, use_cache=False) if t.state == TrialState.COMPLETE]


class SubsetEvoSampler(BaseSampler):
    """Steady-state GA over universe bits + numeric/categorical params.

    Binary tournament on (pareto rank, scalarised loss); uniform crossover on
    bits, blend crossover + gaussian mutation on numeric params in unit space;
    duplicates of evaluated/running configs are re-mutated.
    """

    def __init__(self, *, seed: int | None = None, population: int = 32, n_startup_trials: int = 20,
                 mutation: float | None = None, universe: Universe | None = None):
        self._rng = np.random.RandomState(seed)
        self._random = RandomSampler(seed=seed)
        self.population = population
        self.n_startup = n_startup_trials
        self.mutation = mutation
        self.universe = universe

    def reseed_rng(self) -> None:
        self._rng.seed()
        self._random.reseed_rng()

    def infer_relative_search_space(self, study, trial):
        from optuna.search_space import intersection_search_space

        space = intersection_search_space(study._get_trials(deepcopy=False, use_cache=True))
        return {k: v for k, v in sorted(space.items()) if not v.single()}

    def sample_relative(self, study, trial, search_space) -> dict[str, Any]:
        if not search_space:
            return {}
        done = [t for t in _finished(study) if all(k in t.params for k in search_space)]
        if len(done) < self.n_startup:
            return {}
        fit = fitness(study, done, self._rng)
        order = np.argsort(fit, kind="stable")[: self.population]
        pop = [done[i] for i in order]
        pf = fit[order]
        seen = {_key({k: t.params[k] for k in search_space}) for t in study._get_trials(deepcopy=False, use_cache=False)
                if all(k in t.params for k in search_space)}

        def tour():
            i, j = self._rng.randint(0, len(pop), 2)
            return pop[i] if pf[i] <= pf[j] else pop[j]

        names = list(search_space)
        pm = self.mutation or 1.0 / max(1, len(names))
        child: dict[str, Any] = {}
        for attempt in range(30):
            a, b = tour(), tour()
            boost = 1 + attempt / 5
            child = {}
            for k in names:
                d = search_space[k]
                pa, pb = a.params[k], b.params[k]
                if isinstance(d, CategoricalDistribution):
                    v = pa if self._rng.rand() < 0.5 else pb
                    if self._rng.rand() < pm * boost:
                        v = d.choices[self._rng.randint(len(d.choices))]
                    child[k] = v
                else:
                    ua, ub = to_unit(pa, d), to_unit(pb, d)
                    al = self._rng.uniform(-0.25, 1.25)
                    u = ua + al * (ub - ua)
                    if self._rng.rand() < max(pm, 0.2) * boost:
                        u += self._rng.randn() * 0.1 * boost
                    child[k] = from_unit(u, d)
            if self.universe:
                child.update(self.universe.repair({n: child[n] for n in self.universe.params if n in child}, self._rng))
            if _key(child) not in seen:
                break
        return child

    def sample_independent(self, study, trial, param_name, param_distribution):
        return self._random.sample_independent(study, trial, param_name, param_distribution)


class StepwiseSampler(BaseSampler):
    """Greedy forward selection / backward elimination over the universe.

    Numeric params are held at their config defaults. Rounds are reconstructed
    from history, so the sampler is stateless; when every candidate of the
    current round is in flight, or the search has converged, it falls back to
    the evolutionary sampler.
    """

    def __init__(self, *, universe: Universe, direction: str = "forward", seed: int | None = None,
                 defaults: dict[str, Any] | None = None, fallback: BaseSampler | None = None,
                 min_improvement: float = 0.0):
        if direction not in ("forward", "backward"):
            raise ValueError("direction must be forward or backward")
        self.universe, self.direction = universe, direction
        self._rng = np.random.RandomState(seed)
        self.defaults = defaults or {}
        self.fallback = fallback or SubsetEvoSampler(seed=seed, n_startup_trials=4, universe=universe)
        self.min_improvement = min_improvement
        self._random = RandomSampler(seed=seed)

    def reseed_rng(self) -> None:
        self._rng.seed()
        self.fallback.reseed_rng()

    def infer_relative_search_space(self, study, trial):
        return self.fallback.infer_relative_search_space(study, trial) | {
            n: CategoricalDistribution([0, 1]) for n in self.universe.params}

    def _default(self, k: str, d: BaseDistribution) -> Any:
        if k in self.defaults:
            return self.defaults[k]
        if isinstance(d, CategoricalDistribution):
            return d.choices[0]
        return from_unit(0.5, d)

    def plan(self, study) -> tuple[frozenset | None, str]:
        """Return (next set to evaluate or None, status)."""
        free = self.universe.free
        lo, hi = self.universe.kmin(), self.universe.kmax()
        trials = study._get_trials(deepcopy=False, use_cache=False)
        names = self.universe.params

        def members(t):
            return frozenset(n for n in names if int(t.params.get(n, 0)))

        done = [t for t in trials if t.state == TrialState.COMPLETE and all(n in t.params for n in names)]
        f = fitness(study, done, self._rng) if done else np.zeros(0)
        score: dict[frozenset, float] = {}
        for t, v in zip(done, f):
            s = members(t)
            score[s] = min(score.get(s, np.inf), float(v))
        for t in trials:
            if t.state in (TrialState.PRUNED, TrialState.FAIL) and all(n in t.params for n in names):
                score.setdefault(members(t), np.inf)
        busy = {members(t) for t in trials if t.state == TrialState.RUNNING and all(n in t.params for n in names)}
        allp = frozenset(names)
        if self.direction == "forward":
            base = frozenset()
            if lo == 0 and base not in score:
                return (None, "wait") if base in busy else (base, "round0")
            if lo > 0:
                srng = np.random.RandomState(7 + lo)
                seeds = list(dict.fromkeys(
                    frozenset(srng.choice(sorted(allp), lo, replace=False)) for _ in range(min(8, len(names)))))
                todo = [s for s in seeds if s not in score and s not in busy]
                if todo:
                    return todo[0], "round0"
                if any(s not in score for s in seeds):
                    return None, "wait"
                base = min(seeds, key=lambda s: score[s])
        else:
            base = allp
            if len(base) > hi:
                base = frozenset(self._rng.choice(sorted(allp), hi, replace=False))
            if base not in score:
                return (None, "wait") if base in busy else (base, "round0")
        rnd = 0
        while True:
            if self.direction == "forward":
                cands = [base | {n} for n in names if n not in base] if len(base) < hi else []
            else:
                cands = [base - {n} for n in names if n in base] if len(base) > lo else []
            if not cands:
                return None, "done"
            todo = [c for c in cands if c not in score and c not in busy]
            if todo:
                return todo[0], f"round{rnd + 1}"
            if any(c not in score for c in cands):
                return None, "wait"
            best = min(cands, key=lambda c: score[c])
            cur = score.get(base, np.inf)
            must = self.direction == "backward" and len(base) > hi
            if must or score[best] < cur - self.min_improvement:
                base = best
                rnd += 1
                continue
            return None, "done"

    def sample_relative(self, study, trial, search_space) -> dict[str, Any]:
        nxt, status = self.plan(study)
        study._storage.set_trial_system_attr(trial._trial_id, "stepwise", status)
        if nxt is None:
            space = {k: v for k, v in search_space.items()}
            out = self.fallback.sample_relative(study, trial, space)
            if not out or not all(n in out for n in self.universe.params):
                out = dict(out)
                out.update(self.universe.random_mask(self._rng))
            return out
        out = {}
        for k, d in search_space.items():
            if k.startswith(UPREFIX):
                out[k] = int(k in nxt)
            else:
                out[k] = self._default(k, d)
        return out

    def sample_independent(self, study, trial, param_name, param_distribution):
        if param_name.startswith(UPREFIX):
            return 0
        return self._default(param_name, param_distribution)
