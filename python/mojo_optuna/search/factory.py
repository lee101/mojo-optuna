"""Sampler construction from the study config."""

from __future__ import annotations

from typing import Any

import optuna
from optuna.samplers import BaseSampler

from .space import Space, to_unit


def constraints_func(t) -> list[float]:
    return list(t.user_attrs.get("constraints", []))


SAMPLERS = ("tpe", "tpe-optuna", "random", "cmaes", "cmaes-optuna", "gp", "nsga2", "ga", "forward", "backward", "qmc")


def build_sampler(cfg: dict[str, Any], space: Space, n_objectives: int) -> BaseSampler:
    sc = dict(cfg.get("sampler") or {})
    kind = sc.pop("type", "tpe")
    seed = sc.pop("seed", cfg.get("seed"))
    uni = space.universe
    has_cons = bool(cfg.get("constraints"))
    s: BaseSampler
    if kind in ("tpe", "motpe", "tpe-optuna"):
        if kind == "tpe-optuna":
            cls = optuna.samplers.TPESampler
        else:
            try:
                from .._lib import lib

                lib()
                from ..samplers import TPESampler as cls
            except Exception:
                cls = optuna.samplers.TPESampler
        s = cls(
            seed=seed,
            multivariate=sc.pop("multivariate", True),
            group=sc.pop("group", False),
            constant_liar=sc.pop("constant_liar", True),
            n_startup_trials=sc.pop("n_startup", 10 if not uni else max(10, min(40, len(uni.free)))),
            n_ei_candidates=sc.pop("n_ei_candidates", 24),
            constraints_func=constraints_func if has_cons else None,
            **sc,
        )
    elif kind == "random":
        s = optuna.samplers.RandomSampler(seed=seed)
    elif kind == "qmc":
        s = optuna.samplers.QMCSampler(seed=seed, scramble=True, warn_independent_sampling=False)
    elif kind == "cmaes":
        from .cmaes import CmaEsIpopSampler

        x0 = sc.pop("x0", None)
        if x0:
            dists = space.numeric()
            x0 = {k: to_unit(v, dists[k]) for k, v in x0.items() if k in dists}
        s = CmaEsIpopSampler(seed=seed, x0=x0, n_startup_trials=sc.pop("n_startup", 0), **sc)
    elif kind == "cmaes-optuna":
        s = optuna.samplers.CmaEsSampler(seed=seed, warn_independent_sampling=False, **sc)
    elif kind == "gp":
        from .gp import GPEISampler

        s = GPEISampler(seed=seed, n_startup_trials=sc.pop("n_startup", 10), **sc)
    elif kind == "nsga2":
        s = optuna.samplers.NSGAIISampler(
            seed=seed, population_size=sc.pop("population", 32),
            constraints_func=constraints_func if has_cons else None, **sc)
    elif kind == "ga":
        from .subset import SubsetEvoSampler

        s = SubsetEvoSampler(seed=seed, population=sc.pop("population", 32),
                             n_startup_trials=sc.pop("n_startup", 20), universe=uni, **sc)
    elif kind in ("forward", "backward"):
        from .subset import StepwiseSampler

        if uni is None:
            raise ValueError("stepwise samplers need a universe")
        return StepwiseSampler(universe=uni, direction=kind, seed=seed, defaults=space.defaults(), **sc)
    else:
        raise ValueError(f"unknown sampler {kind!r}; choose from {SAMPLERS}")
    if uni is not None and uni.free:
        from .subset import UniverseRepair

        s = UniverseRepair(s, uni, seed=seed)
    return s
