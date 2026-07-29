"""Pruners implementing Optuna's public ``BasePruner`` protocol."""

from __future__ import annotations

import binascii
from collections.abc import Container
import functools
import math

import numpy as np
import optuna
from optuna.pruners import BasePruner
from optuna.study import StudyDirection
from optuna.trial import TrialState


def _first_in_interval(
    step: int, intermediate_steps, n_warmup_steps: int, interval_steps: int
) -> bool:
    pruning_step = (
        (step - n_warmup_steps) // interval_steps * interval_steps + n_warmup_steps
    )
    previous = functools.reduce(
        lambda current, value: (
            value if value > current and value != step else current
        ),
        intermediate_steps,
        -1,
    )
    return previous < pruning_step


class NopPruner(BasePruner):
    def prune(self, study, trial) -> bool:
        return False


class PercentilePruner(BasePruner):
    def __init__(
        self,
        percentile: float,
        n_startup_trials: int = 5,
        n_warmup_steps: int = 0,
        interval_steps: int = 1,
        *,
        n_min_trials: int = 1,
    ) -> None:
        if not 0.0 <= percentile <= 100:
            raise ValueError(
                f"Percentile must be between 0 and 100 inclusive, but got {percentile=}."
            )
        if n_startup_trials < 0:
            raise ValueError(
                f"Number of startup trials cannot be negative, but got {n_startup_trials=}."
            )
        if n_warmup_steps < 0:
            raise ValueError(
                f"Number of warmup steps cannot be negative, but got {n_warmup_steps=}."
            )
        if interval_steps < 1:
            raise ValueError(
                f"Pruning interval steps must be at least 1, but got {interval_steps=}."
            )
        if n_min_trials < 1:
            raise ValueError(
                f"Number of trials for pruning must be at least 1, but got {n_min_trials=}."
            )
        self._percentile = percentile
        self._n_startup_trials = n_startup_trials
        self._n_warmup_steps = n_warmup_steps
        self._interval_steps = interval_steps
        self._n_min_trials = n_min_trials

    def prune(self, study, trial) -> bool:
        completed = study.get_trials(
            deepcopy=False, states=(TrialState.COMPLETE,)
        )
        if len(completed) == 0 or len(completed) < self._n_startup_trials:
            return False
        step = trial.last_step
        if step is None or step < self._n_warmup_steps:
            return False
        if not _first_in_interval(
            step,
            trial.intermediate_values.keys(),
            self._n_warmup_steps,
            self._interval_steps,
        ):
            return False

        current_values = np.asarray(
            list(trial.intermediate_values.values()), dtype=float
        )
        best = (
            np.nanmax(current_values)
            if study.direction == StudyDirection.MAXIMIZE
            else np.nanmin(current_values)
        )
        if math.isnan(float(best)):
            return True
        references = [
            completed_trial.intermediate_values[step]
            for completed_trial in completed
            if step in completed_trial.intermediate_values
        ]
        if len(references) < self._n_min_trials:
            return False
        percentile = (
            100 - self._percentile
            if study.direction == StudyDirection.MAXIMIZE
            else self._percentile
        )
        threshold = float(
            np.nanpercentile(np.asarray(references, dtype=float), percentile)
        )
        if math.isnan(threshold):
            return False
        if study.direction == StudyDirection.MAXIMIZE:
            return bool(best < threshold)
        return bool(best > threshold)


class MedianPruner(PercentilePruner):
    def __init__(
        self,
        n_startup_trials: int = 5,
        n_warmup_steps: int = 0,
        interval_steps: int = 1,
        *,
        n_min_trials: int = 1,
    ) -> None:
        super().__init__(
            50.0,
            n_startup_trials,
            n_warmup_steps,
            interval_steps,
            n_min_trials=n_min_trials,
        )


class ThresholdPruner(BasePruner):
    def __init__(
        self,
        lower: float | None = None,
        upper: float | None = None,
        n_warmup_steps: int = 0,
        interval_steps: int = 1,
    ) -> None:
        if lower is None and upper is None:
            raise TypeError("Either lower or upper must be specified.")
        lower = -math.inf if lower is None else float(lower)
        upper = math.inf if upper is None else float(upper)
        if lower > upper:
            raise ValueError("lower should be smaller than upper.")
        if n_warmup_steps < 0:
            raise ValueError(
                f"Number of warmup steps cannot be negative but got {n_warmup_steps}."
            )
        if interval_steps < 1:
            raise ValueError(
                f"Pruning interval steps must be at least 1 but got {interval_steps}."
            )
        self._lower = lower
        self._upper = upper
        self._n_warmup_steps = n_warmup_steps
        self._interval_steps = interval_steps

    def prune(self, study, trial) -> bool:
        step = trial.last_step
        if step is None or step < self._n_warmup_steps:
            return False
        if not _first_in_interval(
            step,
            trial.intermediate_values.keys(),
            self._n_warmup_steps,
            self._interval_steps,
        ):
            return False
        value = trial.intermediate_values[step]
        return bool(math.isnan(value) or value < self._lower or value > self._upper)


class PatientPruner(BasePruner):
    def __init__(
        self,
        wrapped_pruner: BasePruner | None,
        patience: int,
        min_delta: float = 0.0,
    ) -> None:
        if patience < 0:
            raise ValueError(f"patience cannot be negative but got {patience}.")
        if min_delta < 0:
            raise ValueError(f"min_delta cannot be negative but got {min_delta}.")
        self._wrapped_pruner = wrapped_pruner
        self._patience = patience
        self._min_delta = min_delta

    def prune(self, study, trial) -> bool:
        if trial.last_step is None:
            return False
        intermediate = trial.intermediate_values
        steps = np.sort(np.asarray(list(intermediate)))
        if steps.size <= self._patience + 1:
            return False
        before = np.asarray(
            [intermediate[int(step)] for step in steps[: -self._patience - 1]]
        )
        after = np.asarray(
            [intermediate[int(step)] for step in steps[-self._patience - 1 :]]
        )
        if study.direction == StudyDirection.MINIMIZE:
            stalled = np.nanmin(before) + self._min_delta < np.nanmin(after)
        else:
            stalled = np.nanmax(before) - self._min_delta > np.nanmax(after)
        if not stalled:
            return False
        return (
            True
            if self._wrapped_pruner is None
            else self._wrapped_pruner.prune(study, trial)
        )


def _rung_key(rung: int) -> str:
    return f"completed_rung_{rung}"


class SuccessiveHalvingPruner(BasePruner):
    def __init__(
        self,
        min_resource: str | int = "auto",
        reduction_factor: int = 4,
        min_early_stopping_rate: int = 0,
        bootstrap_count: int = 0,
    ) -> None:
        if isinstance(min_resource, str) and min_resource != "auto":
            raise ValueError("min_resource must be an integer >= 1 or 'auto'")
        if isinstance(min_resource, int) and min_resource < 1:
            raise ValueError("min_resource must be an integer >= 1 or 'auto'")
        if reduction_factor < 2:
            raise ValueError("reduction_factor must be >= 2")
        if min_early_stopping_rate < 0:
            raise ValueError("min_early_stopping_rate must be >= 0")
        if bootstrap_count < 0:
            raise ValueError("bootstrap_count must be >= 0")
        if bootstrap_count > 0 and min_resource == "auto":
            raise ValueError(
                "bootstrap_count > 0 and min_resource == 'auto' are mutually incompatible"
            )
        self._min_resource = min_resource if isinstance(min_resource, int) else None
        self._reduction_factor = reduction_factor
        self._min_early_stopping_rate = min_early_stopping_rate
        self._bootstrap_count = bootstrap_count

    def prune(self, study, trial) -> bool:
        step = trial.last_step
        if step is None:
            return False
        rung = 0
        while _rung_key(rung) in trial.system_attrs:
            rung += 1
        value = trial.intermediate_values[step]
        trials = None
        while True:
            if self._min_resource is None:
                trials = study.get_trials(deepcopy=False)
                completed_steps = [
                    item.last_step
                    for item in trials
                    if item.state == TrialState.COMPLETE and item.last_step is not None
                ]
                if not completed_steps:
                    return False
                self._min_resource = max(max(completed_steps) // 100, 1)
            promotion_step = self._min_resource * self._reduction_factor ** (
                self._min_early_stopping_rate + rung
            )
            if step < promotion_step:
                return False
            if math.isnan(value):
                return True
            if trials is None:
                trials = study.get_trials(deepcopy=False)
            key = _rung_key(rung)
            study._storage.set_trial_system_attr(trial._trial_id, key, value)
            competing = [
                item.system_attrs[key] for item in trials if key in item.system_attrs
            ]
            competing.append(value)
            if len(competing) <= self._bootstrap_count:
                return True
            promotable_index = len(competing) // self._reduction_factor - 1
            if promotable_index == -1:
                promotable_index = 0
            competing.sort()
            if study.direction == StudyDirection.MAXIMIZE:
                promotable = value >= competing[-(promotable_index + 1)]
            else:
                promotable = value <= competing[promotable_index]
            if not promotable:
                return True
            rung += 1


class HyperbandPruner(BasePruner):
    def __init__(
        self,
        min_resource: int = 1,
        max_resource: str | int = "auto",
        reduction_factor: int = 3,
        bootstrap_count: int = 0,
    ) -> None:
        if not isinstance(max_resource, int) and max_resource != "auto":
            raise ValueError("max_resource must be an integer or 'auto'")
        if bootstrap_count > 0 and max_resource == "auto":
            raise ValueError(
                "bootstrap_count > 0 and max_resource == 'auto' are mutually incompatible"
            )
        self._min_resource = min_resource
        self._max_resource = max_resource
        self._reduction_factor = reduction_factor
        self._bootstrap_count = bootstrap_count
        self._pruners: list[SuccessiveHalvingPruner] = []
        self._trial_allocation_budgets: list[int] = []
        self._total_trial_allocation_budget = 0
        self._n_brackets: int | None = None

    def _initialize(self, study) -> None:
        if self._max_resource == "auto":
            completed = study.get_trials(
                deepcopy=False, states=(TrialState.COMPLETE,)
            )
            steps = [
                trial.last_step for trial in completed if trial.last_step is not None
            ]
            if not steps:
                return
            self._max_resource = max(steps) + 1
        assert isinstance(self._max_resource, int)
        self._n_brackets = (
            math.floor(
                math.log(
                    self._max_resource / self._min_resource,
                    self._reduction_factor,
                )
            )
            + 1
        )
        for bracket_id in range(self._n_brackets):
            s = self._n_brackets - 1 - bracket_id
            budget = math.ceil(
                self._n_brackets * self._reduction_factor**s / (s + 1)
            )
            self._trial_allocation_budgets.append(budget)
            self._total_trial_allocation_budget += budget
            self._pruners.append(
                SuccessiveHalvingPruner(
                    self._min_resource,
                    self._reduction_factor,
                    bracket_id,
                    self._bootstrap_count,
                )
            )

    def _get_bracket_id(self, study, trial) -> int:
        if not self._pruners:
            return 0
        assert self._n_brackets is not None
        value = (
            binascii.crc32(f"{study.study_name}_{trial.number}".encode())
            % self._total_trial_allocation_budget
        )
        for bracket_id, budget in enumerate(self._trial_allocation_budgets):
            value -= budget
            if value < 0:
                return bracket_id
        raise RuntimeError("unreachable Hyperband bracket")

    def _bracket_study(self, study, bracket_id: int):
        parent = self

        class BracketStudy(optuna.study.Study):
            def __init__(self):
                super().__init__(
                    study_name=study.study_name,
                    storage=study._storage,
                    sampler=study.sampler,
                    pruner=parent,
                )

            def get_trials(
                self,
                deepcopy: bool = True,
                states: Container[TrialState] | None = None,
            ):
                trials = super()._get_trials(deepcopy=deepcopy, states=states)
                return [
                    trial
                    for trial in trials
                    if parent._get_bracket_id(self, trial) == bracket_id
                ]

        return BracketStudy()

    def prune(self, study, trial) -> bool:
        if not self._pruners:
            self._initialize(study)
            if not self._pruners:
                return False
        bracket_id = self._get_bracket_id(study, trial)
        return self._pruners[bracket_id].prune(
            self._bracket_study(study, bracket_id), trial
        )


__all__ = [
    "BasePruner",
    "HyperbandPruner",
    "MedianPruner",
    "NopPruner",
    "PatientPruner",
    "PercentilePruner",
    "SuccessiveHalvingPruner",
    "ThresholdPruner",
]
