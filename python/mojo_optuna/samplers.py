"""Samplers compatible with Optuna's public sampler protocol."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

import numpy as np
from optuna.distributions import CategoricalChoiceType
from optuna.samplers import BaseSampler
from optuna.samplers import TPESampler as _OptunaTPESampler
from optuna.samplers._tpe.sampler import _split_trials
from optuna.trial import FrozenTrial
from optuna.trial import TrialState

from ._lib import best_acquisition
from ._tpe import MojoParzenEstimator


class TPESampler(_OptunaTPESampler):
    """Tree-structured Parzen estimator accelerated by Mojo."""

    def __init__(
        self,
        *,
        consider_prior: bool | None = None,
        prior_weight: float | None = None,
        consider_magic_clip: bool | None = None,
        consider_endpoints: bool | None = None,
        n_startup_trials: int = 10,
        n_ei_candidates: int = 24,
        gamma: Callable[[int], int] | None = None,
        weights: Callable[[int], np.ndarray] | None = None,
        seed: int | None = None,
        multivariate: bool = False,
        group: bool = False,
        warn_independent_sampling: bool | None = None,
        constant_liar: bool = False,
        constraints_func: Callable[[FrozenTrial], Sequence[float]] | None = None,
        categorical_distance_func: (
            dict[str, Callable[[CategoricalChoiceType, CategoricalChoiceType], float]]
            | None
        ) = None,
    ) -> None:
        super().__init__(
            consider_prior=consider_prior,
            prior_weight=prior_weight,
            consider_magic_clip=consider_magic_clip,
            consider_endpoints=consider_endpoints,
            n_startup_trials=n_startup_trials,
            n_ei_candidates=n_ei_candidates,
            gamma=gamma,
            weights=weights,
            seed=seed,
            multivariate=multivariate,
            group=group,
            warn_independent_sampling=warn_independent_sampling,
            constant_liar=constant_liar,
            constraints_func=constraints_func,
            categorical_distance_func=categorical_distance_func,
        )
        self._parzen_estimator_cls = MojoParzenEstimator

    def _sample(self, study, trial, search_space) -> dict[str, Any]:
        states = (
            [TrialState.COMPLETE, TrialState.PRUNED, TrialState.RUNNING]
            if self._constant_liar
            else [TrialState.COMPLETE, TrialState.PRUNED]
        )
        trials = study._get_trials(
            deepcopy=False, states=states, use_cache=not self._constant_liar
        )
        if self._constant_liar:
            trials = [item for item in trials if trial.number != item.number]
        n_finished = sum(item.state != TrialState.RUNNING for item in trials)
        below, above = _split_trials(
            study,
            trials,
            self._gamma(n_finished),
            self._constraints_func is not None,
        )
        estimator_below = self._build_parzen_estimator(
            study, search_space, below, handle_below=True
        )
        estimator_above = self._build_parzen_estimator(
            study, search_space, above, handle_below=False
        )
        samples = estimator_below.sample(self._rng.rng, self._n_ei_candidates)
        acquisition = self._compute_acquisition_func(
            samples, estimator_below, estimator_above
        )
        selected = self._compare(samples, acquisition)
        for name, distribution in search_space.items():
            selected[name] = distribution.to_external_repr(selected[name])
        return selected

    @classmethod
    def _compare(
        cls, samples: dict[str, np.ndarray], acquisition_func_vals: np.ndarray
    ) -> dict[str, int | float]:
        sample_size = next(iter(samples.values())).size
        if sample_size == 0:
            raise ValueError(f"The size of `samples` must be positive, but got {sample_size}.")
        if sample_size != acquisition_func_vals.size:
            raise ValueError(
                "The sizes of `samples` and `acquisition_func_vals` must be same, but got "
                "(samples.size, acquisition_func_vals.size) = "
                f"({sample_size}, {acquisition_func_vals.size})."
            )
        zeros = np.zeros_like(acquisition_func_vals)
        best_index = best_acquisition(acquisition_func_vals, zeros)
        return {name: values[best_index].item() for name, values in samples.items()}


__all__ = ["BaseSampler", "TPESampler"]
