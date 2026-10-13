"""Small-budget Bayesian optimisation: Matern-5/2 ARD GP + expected improvement.

Multi-objective studies are scalarised per ask with a random augmented
Chebyshev weight (ParEGO). Running trials enter as constant liars (worst
observed value) so parallel workers spread out.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
from optuna.distributions import (
    BaseDistribution,
    CategoricalDistribution,
    FloatDistribution,
    IntDistribution,
)
from optuna.samplers import BaseSampler, RandomSampler
from optuna.study import StudyDirection
from optuna.trial import FrozenTrial, TrialState

from .space import from_unit, to_unit

SQRT5 = math.sqrt(5.0)


def _ndtr(z):
    from math import erf

    return 0.5 * (1 + np.vectorize(erf)(z / math.sqrt(2)))


def matern52_cross(a: np.ndarray, b: np.ndarray, ls: np.ndarray) -> np.ndarray:
    try:
        from .kernels import matern52_cross as mk

        return mk(a, b, ls)
    except Exception:
        return matern52_cross_np(a, b, ls)


def matern52_cross_np(a: np.ndarray, b: np.ndarray, ls: np.ndarray) -> np.ndarray:
    a, b = a / ls, b / ls
    d2 = (a * a).sum(1)[:, None] + (b * b).sum(1)[None, :] - 2 * a @ b.T
    r = np.sqrt(np.maximum(d2, 0.0))
    return (1 + SQRT5 * r + 5.0 / 3.0 * r * r) * np.exp(-SQRT5 * r)


class GP:
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X, self.y = X, y
        self.n, self.d = X.shape
        self.theta = np.concatenate([np.full(self.d, math.log(0.5)), [0.0, math.log(1e-3)]])

    def _nll(self, th: np.ndarray, grad: bool = True):
        X, y, n, d = self.X, self.y, self.n, self.d
        ls = np.exp(th[:d])
        s2, nz = math.exp(th[d]), math.exp(th[d + 1]) + 1e-8
        diff2 = ((X[:, None, :] - X[None, :, :]) / ls) ** 2
        r = np.sqrt(diff2.sum(-1))
        e = np.exp(-SQRT5 * r)
        k0 = (1 + SQRT5 * r + 5.0 / 3.0 * r * r) * e
        K = s2 * k0 + nz * np.eye(n)
        try:
            L = np.linalg.cholesky(K)
        except np.linalg.LinAlgError:
            return (1e10, np.zeros_like(th)) if grad else 1e10
        alpha = np.linalg.solve(L.T, np.linalg.solve(L, y))
        nll = 0.5 * y @ alpha + np.log(np.diag(L)).sum() + 0.5 * n * math.log(2 * math.pi)
        # priors: lengthscale ~ logN(log .5, 1), signal ~ logN(0, 1), noise ~ logN(-7, 2)
        nll += 0.5 * ((th[:d] - math.log(0.5)) ** 2).sum() + 0.5 * th[d] ** 2 + 0.125 * (th[d + 1] + 7) ** 2
        if not grad:
            return nll
        Kinv = np.linalg.solve(L.T, np.linalg.solve(L, np.eye(n)))
        A = np.outer(alpha, alpha) - Kinv
        g = np.empty_like(th)
        base = s2 * (5.0 / 3.0) * (1 + SQRT5 * r) * e
        for k in range(d):
            g[k] = -0.5 * np.sum(A * base * diff2[:, :, k])
        g[d] = -0.5 * np.sum(A * (s2 * k0))
        g[d + 1] = -0.5 * np.trace(A) * (nz - 1e-8)
        g[:d] += th[:d] - math.log(0.5)
        g[d] += th[d]
        g[d + 1] += 0.25 * (th[d + 1] + 7)
        return nll, g

    def fit(self, theta0: np.ndarray | None = None, rng: np.random.RandomState | None = None):
        starts = [self.theta if theta0 is None or len(theta0) != len(self.theta) else theta0, self.theta.copy()]
        if rng is not None:
            starts.append(self.theta + rng.randn(len(self.theta)) * 0.5)
        best = (math.inf, starts[0])
        bounds = [(math.log(1e-2), math.log(20.0))] * self.d + [(math.log(1e-2), math.log(1e2)), (math.log(1e-8), math.log(1.0))]
        try:
            from scipy.optimize import minimize
        except Exception:
            minimize = None
        for th0 in starts:
            th0 = np.clip(th0, [b[0] for b in bounds], [b[1] for b in bounds])
            if minimize is not None:
                res = minimize(lambda t: self._nll(t), th0, jac=True, method="L-BFGS-B",
                               bounds=bounds, options={"maxiter": 60})
                th, f = res.x, float(res.fun)
            else:
                th, f = th0, self._nll(th0, grad=False)
            if f < best[0]:
                best = (f, th)
        self.theta = best[1]
        d = self.d
        self.ls = np.exp(self.theta[:d])
        self.s2 = math.exp(self.theta[d])
        self.nz = math.exp(self.theta[d + 1]) + 1e-8
        K = self.s2 * matern52_cross_np(self.X, self.X, self.ls) + self.nz * np.eye(self.n)
        self.L = np.linalg.cholesky(K)
        self.alpha = np.linalg.solve(self.L.T, np.linalg.solve(self.L, self.y))
        return self

    def predict(self, Xc: np.ndarray):
        ks = self.s2 * matern52_cross(Xc, self.X, self.ls)
        mu = ks @ self.alpha
        v = np.linalg.solve(self.L, ks.T)
        var = np.maximum(self.s2 - (v * v).sum(0), 1e-12)
        return mu, np.sqrt(var)


def expected_improvement(mu, sd, best):
    z = (best - mu) / sd
    pdf = np.exp(-0.5 * z * z) / math.sqrt(2 * math.pi)
    from scipy.special import ndtr  # noqa: PLC0415

    return sd * (z * ndtr(z) + pdf)


def _ei(mu, sd, best):
    try:
        return expected_improvement(mu, sd, best)
    except Exception:
        z = (best - mu) / sd
        pdf = np.exp(-0.5 * z * z) / math.sqrt(2 * math.pi)
        return sd * (z * _ndtr(z) + pdf)


class _Encoder:
    def __init__(self, space: dict[str, BaseDistribution]):
        self.space = space
        self.slices: list[tuple[str, int, int]] = []
        d = 0
        for k, dist in space.items():
            w = len(dist.choices) if isinstance(dist, CategoricalDistribution) else 1
            self.slices.append((k, d, w))
            d += w
        self.d = d

    def encode(self, params: dict[str, Any]) -> np.ndarray:
        x = np.zeros(self.d)
        for k, s, w in self.slices:
            dist = self.space[k]
            if isinstance(dist, CategoricalDistribution):
                x[s + dist.to_internal_repr(params[k])] = 1.0
            else:
                x[s] = to_unit(params[k], dist)
        return x

    def random(self, rng, m: int) -> np.ndarray:
        X = rng.uniform(0, 1, (m, self.d))
        for k, s, w in self.slices:
            if w > 1 or isinstance(self.space[k], CategoricalDistribution):
                c = rng.randint(0, w, m)
                X[:, s:s + w] = 0
                X[np.arange(m), s + c] = 1
        return X

    def project(self, X: np.ndarray) -> np.ndarray:
        X = np.clip(X, 0, 1)
        for k, s, w in self.slices:
            if isinstance(self.space[k], CategoricalDistribution):
                c = X[:, s:s + w].argmax(1)
                X[:, s:s + w] = 0
                X[np.arange(len(X)), s + c] = 1
        return X

    def decode(self, x: np.ndarray) -> dict[str, Any]:
        out = {}
        for k, s, w in self.slices:
            dist = self.space[k]
            if isinstance(dist, CategoricalDistribution):
                out[k] = dist.choices[int(np.argmax(x[s:s + w]))]
            else:
                out[k] = from_unit(x[s], dist)
        return out


def scalarize(study, trials: list[FrozenTrial], rng, penalty_scale: float = 1.0) -> np.ndarray:
    """Return per-trial loss to minimise. Infeasible trials are pushed above the worst feasible."""
    V = np.array([t.values for t in trials], dtype=float)
    for j, d in enumerate(study.directions):
        if d == StudyDirection.MAXIMIZE:
            V[:, j] = -V[:, j]
    if V.shape[1] == 1:
        f = V[:, 0]
    else:
        lo, hi = V.min(0), V.max(0)
        Z = (V - lo) / np.where(hi > lo, hi - lo, 1.0)
        w = rng.dirichlet(np.ones(V.shape[1]))
        f = (Z * w).max(1) + 0.05 * (Z * w).sum(1)
    viol = np.array([sum(max(0.0, float(c)) for c in (t.system_attrs.get("constraints") or t.user_attrs.get("constraints") or [])) for t in trials])
    if np.any(viol > 0):
        feas = viol <= 0
        worst = f[feas].max() if feas.any() else f.max()
        span = (f.max() - f.min()) or 1.0
        f = np.where(feas, f, worst + span * (0.1 + viol / (viol.max() or 1.0)) * penalty_scale)
    return f


class GPEISampler(BaseSampler):
    def __init__(self, *, seed: int | None = None, n_startup_trials: int = 10,
                 n_candidates: int = 2048, max_history: int = 400, constant_liar: bool = True):
        self._rng = np.random.RandomState(seed)
        self._random = RandomSampler(seed=seed)
        self.n_startup = n_startup_trials
        self.n_candidates = n_candidates
        self.max_history = max_history
        self.constant_liar = constant_liar
        self._theta: np.ndarray | None = None

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
        trials = study._get_trials(deepcopy=False, use_cache=False)
        done = [t for t in trials if t.state == TrialState.COMPLETE and all(k in t.params for k in search_space)]
        if len(done) < self.n_startup:
            return {}
        if len(done) > self.max_history:
            f_all = scalarize(study, done, self._rng)
            keep = np.argsort(f_all)[: self.max_history // 2]
            rest = self._rng.choice(np.setdiff1d(np.arange(len(done)), keep), self.max_history - len(keep), replace=False)
            done = [done[i] for i in np.concatenate([keep, rest])]
        enc = _Encoder(search_space)
        X = np.array([enc.encode(t.params) for t in done])
        f = scalarize(study, done, self._rng)
        if self.constant_liar:
            run = [t for t in trials if t.state == TrialState.RUNNING and t.number != trial.number
                   and all(k in t.params for k in search_space)]
            if run:
                X = np.vstack([X, [enc.encode(t.params) for t in run]])
                f = np.concatenate([f, np.full(len(run), f.max())])
        mu_y, sd_y = f.mean(), f.std() or 1.0
        y = (f - mu_y) / sd_y
        gp = GP(X, y).fit(self._theta, self._rng)
        self._theta = gp.theta
        best = y[: len(done)].min()
        C = enc.random(self._rng, self.n_candidates)
        top = X[np.argsort(y)[:5]]
        loc = np.repeat(top, 128, axis=0) + self._rng.randn(len(top) * 128, enc.d) * 0.05
        C = np.vstack([C, enc.project(loc)])
        ei = _ei(*gp.predict(C), best)
        x = C[int(np.argmax(ei))]
        for scale in (0.02, 0.005):
            loc = enc.project(x + self._rng.randn(256, enc.d) * scale)
            e2 = _ei(*gp.predict(loc), best)
            if e2.max() > ei.max():
                x = loc[int(np.argmax(e2))]
                ei = e2
        return enc.decode(x)

    def sample_independent(self, study, trial, param_name, param_distribution):
        return self._random.sample_independent(study, trial, param_name, param_distribution)
