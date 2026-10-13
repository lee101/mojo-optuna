"""Stateless CMA-ES with IPOP restarts.

The optimizer state is never pickled: every ask replays the update equations
from the trial history (solutions are stored as system attrs), so any number of
ask/tell processes sharing one storage see the same distribution.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
from optuna.distributions import BaseDistribution, FloatDistribution, IntDistribution
from optuna.samplers import BaseSampler, RandomSampler
from optuna.study import StudyDirection
from optuna.trial import FrozenTrial, TrialState

from .space import from_unit

KX, KGEN, KRESTART = "cma:x", "cma:gen", "cma:restart"


class _State:
    def __init__(self, n: int, lam: int, mean: np.ndarray, sigma: float):
        self.n, self.lam = n, lam
        mu = lam // 2
        w = math.log((lam + 1) / 2) - np.log(np.arange(1, mu + 1))
        self.w = w / w.sum()
        self.mu = mu
        self.mueff = 1.0 / float((self.w ** 2).sum())
        me = self.mueff
        self.cc = (4 + me / n) / (n + 4 + 2 * me / n)
        self.cs = (me + 2) / (n + me + 5)
        self.c1 = 2 / ((n + 1.3) ** 2 + me)
        self.cmu = min(1 - self.c1, 2 * (me - 2 + 1 / me) / ((n + 2) ** 2 + me))
        self.damps = 1 + 2 * max(0.0, math.sqrt((me - 1) / (n + 1)) - 1) + self.cs
        self.chin = math.sqrt(n) * (1 - 1 / (4 * n) + 1 / (21 * n * n))
        self.m = mean.copy()
        self.sigma = sigma
        self.C = np.eye(n)
        self.B = np.eye(n)
        self.D = np.ones(n)
        self.pc = np.zeros(n)
        self.ps = np.zeros(n)
        self.gen = 0
        self.best_hist: list[float] = []

    def update(self, xs: np.ndarray, fs: np.ndarray) -> None:
        n = self.n
        idx = np.argsort(fs, kind="stable")[: self.mu]
        sel = xs[idx]
        old = self.m
        self.m = self.w @ sel
        y = (self.m - old) / self.sigma
        cinv_half = self.B @ np.diag(1 / self.D) @ self.B.T
        self.ps = (1 - self.cs) * self.ps + math.sqrt(self.cs * (2 - self.cs) * self.mueff) * (cinv_half @ y)
        norm_ps = float(np.linalg.norm(self.ps))
        hsig = norm_ps / math.sqrt(1 - (1 - self.cs) ** (2 * (self.gen + 1))) / self.chin < 1.4 + 2 / (n + 1)
        self.pc = (1 - self.cc) * self.pc + hsig * math.sqrt(self.cc * (2 - self.cc) * self.mueff) * y
        art = (sel - old) / self.sigma
        rank_mu = (art.T * self.w) @ art
        self.C = (
            (1 - self.c1 - self.cmu) * self.C
            + self.c1 * (np.outer(self.pc, self.pc) + (1 - hsig) * self.cc * (2 - self.cc) * self.C)
            + self.cmu * rank_mu
        )
        self.sigma *= math.exp((self.cs / self.damps) * (norm_ps / self.chin - 1))
        self.sigma = min(self.sigma, 1.0)
        self.C = (self.C + self.C.T) / 2
        ev, B = np.linalg.eigh(self.C)
        ev = np.maximum(ev, 1e-20)
        self.B, self.D = B, np.sqrt(ev)
        self.gen += 1
        self.best_hist.append(float(np.min(fs)))

    def should_restart(self, tolx: float, tolfun: float) -> bool:
        if self.sigma * float(self.D.max()) < tolx:
            return True
        if float(self.D.max() / self.D.min()) ** 2 > 1e14:
            return True
        win = 10 + int(math.ceil(30 * self.n / self.lam))
        if len(self.best_hist) >= win:
            h = self.best_hist[-win:]
            if max(h) - min(h) < tolfun * (1 + abs(min(h))):
                return True
        return False

    def sample(self, rng: np.random.RandomState) -> np.ndarray:
        x = self.m
        for _ in range(16):
            z = rng.randn(self.n)
            x = self.m + self.sigma * (self.B @ (self.D * z))
            if np.all((x >= 0) & (x <= 1)):
                return x
        return np.clip(x, 0, 1)


class CmaEsIpopSampler(BaseSampler):
    """CMA-ES on the unit cube with IPOP restarts (popsize doubles per restart)."""

    def __init__(self, *, seed: int | None = None, popsize: int | None = None,
                 sigma0: float = 0.25, inc_popsize: float = 2.0, max_restarts: int = 9,
                 tolx: float = 1e-5, tolfun: float = 1e-10, n_startup_trials: int = 0,
                 x0: dict[str, Any] | None = None, penalty: float = 1e30):
        self._seed = 0 if seed is None else int(seed)
        self._rng = np.random.RandomState(seed)
        self._random = RandomSampler(seed=seed)
        self.popsize, self.sigma0, self.inc = popsize, sigma0, inc_popsize
        self.max_restarts, self.tolx, self.tolfun = max_restarts, tolx, tolfun
        self.n_startup = n_startup_trials
        self.x0 = x0
        self.penalty = penalty
        self._cache: dict[tuple, tuple] = {}

    def reseed_rng(self) -> None:
        self._rng.seed()
        self._random.reseed_rng()

    def infer_relative_search_space(self, study, trial) -> dict[str, BaseDistribution]:
        from optuna.search_space import intersection_search_space

        space = intersection_search_space(study._get_trials(deepcopy=False, use_cache=True))
        return {k: v for k, v in sorted(space.items())
                if isinstance(v, (FloatDistribution, IntDistribution)) and v.low < v.high}

    def _lam(self, n: int, restart: int) -> int:
        base = self.popsize or 4 + int(3 * math.log(n))
        return int(round(base * self.inc ** restart))

    def _fitness(self, study, t: FrozenTrial) -> float:
        if t.state != TrialState.COMPLETE or t.values is None:
            return self.penalty
        v = float(t.values[0])
        if study.directions[0] == StudyDirection.MAXIMIZE:
            v = -v
        cons = t.system_attrs.get("constraints") or t.user_attrs.get("constraints")
        if cons:
            viol = sum(max(0.0, float(c)) for c in cons)
            if viol > 0:
                return self.penalty / 2 + viol
        return v if math.isfinite(v) else self.penalty

    def _replay(self, study, names: list[str]) -> tuple[_State, int]:
        trials = [t for t in study._get_trials(deepcopy=False, use_cache=False)
                  if KGEN in t.system_attrs and t.system_attrs.get("cma:names") == names]
        key = tuple((t.number, t.state.value) for t in trials if t.state != TrialState.RUNNING)
        if key in self._cache:
            return self._cache[key]
        n = len(names)
        groups: dict[tuple[int, int], list[FrozenTrial]] = {}
        for t in trials:
            groups.setdefault((t.system_attrs[KRESTART], t.system_attrs[KGEN]), []).append(t)
        restart = 0
        state = self._new_state(n, 0, names)
        while True:
            members = groups.get((restart, state.gen), [])
            done = [t for t in members if t.state != TrialState.RUNNING]
            if len(done) < state.lam:
                break
            done.sort(key=lambda t: t.number)
            done = done[: state.lam]
            xs = np.array([t.system_attrs[KX] for t in done], dtype=float)
            fs = np.array([self._fitness(study, t) for t in done])
            state.update(xs, fs)
            if state.should_restart(self.tolx, self.tolfun) and restart < self.max_restarts:
                restart += 1
                state = self._new_state(n, restart, names)
        self._cache[key] = (state, restart)
        return state, restart

    def _new_state(self, n: int, restart: int, names: list[str]) -> _State:
        lam = self._lam(n, restart)
        if restart == 0:
            mean = np.full(n, 0.5)
            if self.x0:
                mean = np.array([float(self.x0.get(k, 0.5)) for k in names])
        else:
            mean = np.random.RandomState(self._seed * 1009 + restart).uniform(0, 1, n)
        return _State(n, lam, mean, self.sigma0 if restart == 0 else max(self.sigma0, 0.3))

    def sample_relative(self, study, trial, search_space) -> dict[str, Any]:
        if not search_space or len(study.directions) > 1:
            return {}
        done = [t for t in study._get_trials(deepcopy=False, use_cache=True)
                if t.state == TrialState.COMPLETE]
        if len(done) < self.n_startup:
            return {}
        names = list(search_space)
        state, restart = self._replay(study, names)
        x = state.sample(self._rng)
        st = study._storage
        st.set_trial_system_attr(trial._trial_id, KX, [float(v) for v in x])
        st.set_trial_system_attr(trial._trial_id, KGEN, state.gen)
        st.set_trial_system_attr(trial._trial_id, KRESTART, restart)
        st.set_trial_system_attr(trial._trial_id, "cma:names", names)
        return {k: from_unit(x[i], search_space[k]) for i, k in enumerate(names)}

    def sample_independent(self, study, trial, param_name, param_distribution):
        return self._random.sample_independent(study, trial, param_name, param_distribution)
