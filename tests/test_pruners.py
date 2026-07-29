from __future__ import annotations

import math
import inspect

import optuna
import pytest

import mojo_optuna.pruners as mojo_pruners

optuna.logging.set_verbosity(optuna.logging.WARNING)


def run_pruning_study(pruner, *, name="parity-study", n_trials=30):
    study = optuna.create_study(
        direction="minimize", pruner=pruner, study_name=name
    )

    def objective(trial):
        for step in range(12):
            value = (
                (trial.number % 7) * 0.3
                + 1.0 / (step + 1)
                + ((trial.number * step) % 3) * 0.01
            )
            trial.report(value, step)
            if trial.should_prune():
                raise optuna.TrialPruned()
        return value

    study.optimize(objective, n_trials=n_trials)
    return (
        [trial.state for trial in study.trials],
        [trial.last_step for trial in study.trials],
    )


@pytest.mark.parametrize(
    ("upstream", "mojo"),
    [
        (
            optuna.pruners.MedianPruner(
                n_startup_trials=3, n_warmup_steps=2, interval_steps=2
            ),
            mojo_pruners.MedianPruner(
                n_startup_trials=3, n_warmup_steps=2, interval_steps=2
            ),
        ),
        (
            optuna.pruners.PercentilePruner(
                25, n_startup_trials=3, n_warmup_steps=2, interval_steps=2
            ),
            mojo_pruners.PercentilePruner(
                25, n_startup_trials=3, n_warmup_steps=2, interval_steps=2
            ),
        ),
        (
            optuna.pruners.ThresholdPruner(
                lower=0.2, upper=1.6, n_warmup_steps=1, interval_steps=2
            ),
            mojo_pruners.ThresholdPruner(
                lower=0.2, upper=1.6, n_warmup_steps=1, interval_steps=2
            ),
        ),
        (
            optuna.pruners.SuccessiveHalvingPruner(
                min_resource=1, reduction_factor=3
            ),
            mojo_pruners.SuccessiveHalvingPruner(
                min_resource=1, reduction_factor=3
            ),
        ),
        (
            optuna.pruners.HyperbandPruner(
                min_resource=1, max_resource=12, reduction_factor=3
            ),
            mojo_pruners.HyperbandPruner(
                min_resource=1, max_resource=12, reduction_factor=3
            ),
        ),
    ],
)
def test_pruner_decisions_match_upstream(upstream, mojo):
    assert run_pruning_study(mojo) == run_pruning_study(upstream)


def test_patient_pruner_matches_upstream():
    with pytest.warns(optuna.exceptions.ExperimentalWarning):
        upstream = optuna.pruners.PatientPruner(None, patience=2, min_delta=0.02)
    mojo = mojo_pruners.PatientPruner(None, patience=2, min_delta=0.02)
    assert run_pruning_study(mojo) == run_pruning_study(upstream)


def test_nop_pruner_completes_every_trial():
    states, steps = run_pruning_study(mojo_pruners.NopPruner(), n_trials=8)
    assert states == [optuna.trial.TrialState.COMPLETE] * 8
    assert steps == [11] * 8


def test_nan_threshold_behavior_matches_upstream():
    def decision(pruner):
        study = optuna.create_study(pruner=pruner)
        trial = study.ask()
        trial.report(math.nan, 0)
        return trial.should_prune()

    assert decision(mojo_pruners.ThresholdPruner(upper=1.0))
    assert decision(mojo_pruners.ThresholdPruner(upper=1.0)) == decision(
        optuna.pruners.ThresholdPruner(upper=1.0)
    )


@pytest.mark.parametrize(
    "constructor",
    [
        lambda: mojo_pruners.PercentilePruner(-1),
        lambda: mojo_pruners.PercentilePruner(50, interval_steps=0),
        lambda: mojo_pruners.SuccessiveHalvingPruner(reduction_factor=1),
        lambda: mojo_pruners.ThresholdPruner(),
        lambda: mojo_pruners.PatientPruner(None, patience=-1),
    ],
)
def test_invalid_pruner_parameters_raise(constructor):
    with pytest.raises((TypeError, ValueError)):
        constructor()


@pytest.mark.parametrize(
    "name",
    [
        "HyperbandPruner",
        "MedianPruner",
        "NopPruner",
        "PatientPruner",
        "PercentilePruner",
        "SuccessiveHalvingPruner",
        "ThresholdPruner",
    ],
)
def test_pruner_public_constructor_parameters_match_upstream(name):
    mojo_parameters = inspect.signature(getattr(mojo_pruners, name)).parameters
    upstream_parameters = inspect.signature(getattr(optuna.pruners, name)).parameters
    assert list(mojo_parameters) == list(upstream_parameters)
    assert [
        (parameter.kind, parameter.default) for parameter in mojo_parameters.values()
    ] == [
        (parameter.kind, parameter.default)
        for parameter in upstream_parameters.values()
    ]
