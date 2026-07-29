from __future__ import annotations

import inspect

import numpy as np
import optuna
import pytest
from optuna.distributions import (
    CategoricalDistribution,
    FloatDistribution,
    IntDistribution,
)
from optuna.samplers import BaseSampler
from optuna.samplers._tpe.parzen_estimator import (
    _ParzenEstimator,
    _ParzenEstimatorParameters,
)
from optuna.samplers._tpe.sampler import default_weights

from mojo_optuna import TPESampler
from mojo_optuna._lib import best_acquisition
from mojo_optuna._lib import addr
from mojo_optuna._tpe import MojoParzenEstimator

optuna.logging.set_verbosity(optuna.logging.WARNING)


def parameters(multivariate=False, distances=None):
    return _ParzenEstimatorParameters(
        prior_weight=1.0,
        consider_magic_clip=True,
        consider_endpoints=False,
        weights=default_weights,
        multivariate=multivariate,
        categorical_distance_func=distances or {},
    )


def mixed_problem():
    space = {
        "linear": FloatDistribution(-5, 5),
        "log": FloatDistribution(1e-3, 10, log=True),
        "integer": IntDistribution(1, 19, step=2),
        "log_integer": IntDistribution(1, 16, log=True),
        "category": CategoricalDistribution(["a", "b", "c"]),
    }
    observations = {
        "linear": np.array([-4.0, -2.0, 0.0, 3.0, 4.0]),
        "log": np.array([0.002, 0.01, 0.8, 3.0, 9.0]),
        "integer": np.array([1.0, 3.0, 7.0, 13.0, 19.0]),
        "log_integer": np.array([1.0, 2.0, 4.0, 8.0, 16.0]),
        "category": np.array([0.0, 1.0, 1.0, 2.0, 0.0]),
    }
    return observations, space


@pytest.mark.parametrize("multivariate", [False, True])
def test_mojo_log_pdf_matches_optuna_mixed_distributions(multivariate):
    observations, space = mixed_problem()
    reference = _ParzenEstimator(observations, space, parameters(multivariate))
    mojo = MojoParzenEstimator(observations, space, parameters(multivariate))
    samples = reference.sample(np.random.RandomState(13), 2000)
    assert mojo.log_pdf(samples) == pytest.approx(
        reference.log_pdf(samples), abs=3e-7
    )


def test_categorical_distance_distribution_matches_optuna():
    observations = {"choice": np.array([0.0, 1.0, 3.0, 3.0])}
    space = {"choice": CategoricalDistribution(["aa", "b", "cccc", "ddd"])}
    distance = {"choice": lambda a, b: abs(len(a) - len(b))}
    reference = _ParzenEstimator(observations, space, parameters(distances=distance))
    mojo = MojoParzenEstimator(observations, space, parameters(distances=distance))
    samples = {"choice": np.tile(np.arange(4.0), 100)}
    assert mojo.log_pdf(samples) == pytest.approx(
        reference.log_pdf(samples), abs=2e-7
    )


def test_empty_observations_prior_matches_optuna():
    observations = {"x": np.array([])}
    space = {"x": FloatDistribution(-2.0, 7.0)}
    reference = _ParzenEstimator(observations, space, parameters())
    mojo = MojoParzenEstimator(observations, space, parameters())
    samples = {"x": np.linspace(-2.0, 7.0, 501)}
    assert mojo.log_pdf(samples) == pytest.approx(
        reference.log_pdf(samples), abs=2e-7
    )


def test_simd_tail_matches_optuna():
    rng = np.random.default_rng(17)
    observations = {"x": rng.normal(size=10)}
    space = {"x": FloatDistribution(-4.0, 4.0)}
    reference = _ParzenEstimator(observations, space, parameters())
    mojo = MojoParzenEstimator(observations, space, parameters())
    samples = {"x": rng.uniform(-4.0, 4.0, size=9)}
    assert mojo.log_pdf(samples) == pytest.approx(
        reference.log_pdf(samples), abs=2e-7
    )


@pytest.mark.parametrize("observation_count", [7, 8, 9, 15, 16, 17])
def test_simd_boundaries_match_optuna(observation_count):
    rng = np.random.default_rng(observation_count)
    observations = {"x": rng.uniform(-4.0, 4.0, size=observation_count)}
    space = {"x": FloatDistribution(-4.0, 4.0)}
    reference = _ParzenEstimator(observations, space, parameters())
    mojo = MojoParzenEstimator(observations, space, parameters())
    samples = {"x": rng.uniform(-4.0, 4.0, size=13)}
    assert mojo.log_pdf(samples) == pytest.approx(reference.log_pdf(samples), abs=2e-7)


def test_empty_samples_do_not_cross_ffi():
    observations = {"x": np.array([0.0])}
    space = {"x": FloatDistribution(-1.0, 1.0)}
    mojo = MojoParzenEstimator(observations, space, parameters())
    result = mojo.log_pdf({"x": np.array([], dtype=np.float64)})
    assert result.dtype == np.float64
    assert result.shape == (0,)


def test_categorical_only_estimator_does_not_pass_empty_numeric_buffers():
    observations = {"x": np.array([0.0, 1.0])}
    space = {"x": CategoricalDistribution(("a", "b"))}
    reference = _ParzenEstimator(observations, space, parameters())
    mojo = MojoParzenEstimator(observations, space, parameters())
    samples = {"x": np.array([0.0, 1.0, 0.0])}
    assert mojo.log_pdf(samples) == pytest.approx(reference.log_pdf(samples), abs=2e-7)


def test_invalid_categorical_index_is_rejected_before_ffi():
    mojo = MojoParzenEstimator(
        {"x": np.array([0.0, 1.0])},
        {"x": CategoricalDistribution(("a", "b"))},
        parameters(),
    )
    with pytest.raises(ValueError, match="categorical samples"):
        mojo.log_pdf({"x": np.array([0.5])})


def test_addr_rejects_unsafe_buffers():
    with pytest.raises(ValueError, match="zero-length"):
        addr(np.empty(0, dtype=np.float64), np.float64)
    with pytest.raises(ValueError, match="contiguous"):
        addr(np.ones((2, 2), dtype=np.float64)[:, 0], np.float64)
    with pytest.raises(TypeError, match="expected float64"):
        addr(np.ones(2, dtype=np.int64), np.float64)


@pytest.mark.parametrize("candidate_count", [31, 160])
def test_discrete_serial_and_parallel_thresholds_match_optuna(candidate_count):
    rng = np.random.default_rng(candidate_count)
    observations = {"x": rng.integers(0, 201, size=1024).astype(float)}
    space = {"x": IntDistribution(0, 200)}
    reference = _ParzenEstimator(observations, space, parameters())
    mojo = MojoParzenEstimator(observations, space, parameters())
    samples = {
        "x": rng.integers(0, 201, size=candidate_count).astype(float)
    }
    assert mojo.log_pdf(samples) == pytest.approx(
        reference.log_pdf(samples), abs=3e-7
    )


def test_categorical_parallel_stream_path_matches_optuna():
    rng = np.random.default_rng(23)
    observations = {"x": rng.integers(0, 8, size=16_384).astype(float)}
    space = {"x": CategoricalDistribution(tuple(range(8)))}
    reference = _ParzenEstimator(observations, space, parameters())
    mojo = MojoParzenEstimator(observations, space, parameters())
    samples = {"x": rng.integers(0, 8, size=65).astype(float)}
    assert mojo.log_pdf(samples) == pytest.approx(
        reference.log_pdf(samples), abs=2e-7
    )


def test_gpu_request_matches_optuna_or_falls_back():
    rng = np.random.default_rng(29)
    observations = {"x": rng.integers(0, 201, size=1024).astype(float)}
    space = {"x": IntDistribution(0, 200)}
    reference = _ParzenEstimator(observations, space, parameters())
    mojo = MojoParzenEstimator(observations, space, parameters())
    samples = {"x": rng.integers(0, 201, size=160).astype(float)}
    assert mojo.log_pdf(samples, device="gpu") == pytest.approx(
        reference.log_pdf(samples), abs=3e-7
    )
    assert mojo._last_device in ("cpu", "gpu")


def _run_mixed_study(sampler, trials=60):
    study = optuna.create_study(sampler=sampler)
    values = []
    for _ in range(trials):
        trial = study.ask()
        x = trial.suggest_float("x", -5, 5)
        scale = trial.suggest_float("scale", 1e-3, 10, log=True)
        count = trial.suggest_int("count", 1, 19, step=2)
        mode = trial.suggest_categorical("mode", ["a", "b", "c"])
        objective = (
            (x - 1.2) ** 2
            + (np.log(scale) + 0.3) ** 2
            + abs(count - 7)
            + (0 if mode == "b" else 2)
        )
        study.tell(trial, objective)
        values.append((x, scale, count, mode))
    return study, values


def test_tpe_end_to_end_is_deterministic_and_uses_upstream_startup_sequence():
    upstream, upstream_values = _run_mixed_study(
        optuna.samplers.TPESampler(seed=11, n_startup_trials=5)
    )
    mojo, mojo_values = _run_mixed_study(
        TPESampler(seed=11, n_startup_trials=5)
    )
    repeated, repeated_values = _run_mixed_study(
        TPESampler(seed=11, n_startup_trials=5)
    )
    assert mojo_values[:5] == upstream_values[:5]
    assert mojo_values == repeated_values
    assert mojo.best_value == repeated.best_value
    assert np.isfinite(mojo.best_value)


def test_multivariate_tpe_sequence_matches_optuna():
    with pytest.warns(optuna.exceptions.ExperimentalWarning):
        upstream_sampler = optuna.samplers.TPESampler(
            seed=7, n_startup_trials=5, multivariate=True
        )
    with pytest.warns(optuna.exceptions.ExperimentalWarning):
        mojo_sampler = TPESampler(
            seed=7, n_startup_trials=5, multivariate=True
        )
    _, upstream_values = _run_mixed_study(upstream_sampler, trials=35)
    _, mojo_values = _run_mixed_study(mojo_sampler, trials=35)
    assert mojo_values == upstream_values


def test_custom_gamma_and_weights_sequence_matches_optuna():
    gamma = lambda n: min(3, max(1, n // 4))
    weights = lambda n: np.linspace(0.5, 1.0, n)
    _, upstream_values = _run_mixed_study(
        optuna.samplers.TPESampler(
            seed=19, n_startup_trials=5, gamma=gamma, weights=weights
        ),
        trials=35,
    )
    _, mojo_values = _run_mixed_study(
        TPESampler(seed=19, n_startup_trials=5, gamma=gamma, weights=weights),
        trials=35,
    )
    assert mojo_values == upstream_values


def _run_multiobjective_study(sampler):
    study = optuna.create_study(directions=("minimize", "maximize"), sampler=sampler)
    suggestions = []
    for _ in range(35):
        trial = study.ask()
        x = trial.suggest_float("x", -3.0, 3.0)
        category = trial.suggest_categorical("category", ("a", "b"))
        study.tell(trial, ((x - 0.5) ** 2, x + (category == "b")))
        suggestions.append((x, category))
    return suggestions


def test_multiobjective_sampling_is_deterministic():
    mojo = _run_multiobjective_study(TPESampler(seed=23, n_startup_trials=5))
    repeated = _run_multiobjective_study(TPESampler(seed=23, n_startup_trials=5))
    assert mojo == repeated
    assert len(mojo) == 35


def _run_grouped_study(sampler):
    study = optuna.create_study(sampler=sampler)
    suggestions = []
    for _ in range(30):
        trial = study.ask()
        branch = trial.suggest_categorical("branch", ("a", "b"))
        name = "a_value" if branch == "a" else "b_value"
        value = trial.suggest_float(name, -2.0, 2.0)
        study.tell(trial, (value - (0.5 if branch == "a" else -0.5)) ** 2)
        suggestions.append((branch, value))
    return suggestions


def test_grouped_multivariate_sampling_is_deterministic():
    with pytest.warns(optuna.exceptions.ExperimentalWarning):
        mojo_sampler = TPESampler(
            seed=29, n_startup_trials=5, multivariate=True, group=True
        )
    with pytest.warns(optuna.exceptions.ExperimentalWarning):
        repeated_sampler = TPESampler(
            seed=29, n_startup_trials=5, multivariate=True, group=True
        )
    assert _run_grouped_study(mojo_sampler) == _run_grouped_study(repeated_sampler)


def _run_constrained_study(sampler):
    study = optuna.create_study(sampler=sampler)

    def objective(trial):
        x = trial.suggest_float("x", -2.0, 2.0)
        return (x - 0.25) ** 2

    study.optimize(objective, n_trials=35)
    return [trial.params for trial in study.trials]


def test_constraints_sequence_matches_optuna():
    constraint = lambda trial: (trial.params["x"] - 1.0,)
    with pytest.warns(optuna.exceptions.ExperimentalWarning):
        upstream_sampler = optuna.samplers.TPESampler(
            seed=31, n_startup_trials=5, constraints_func=constraint
        )
    with pytest.warns(optuna.exceptions.ExperimentalWarning):
        mojo_sampler = TPESampler(
            seed=31, n_startup_trials=5, constraints_func=constraint
        )
    assert _run_constrained_study(mojo_sampler) == _run_constrained_study(
        upstream_sampler
    )


def _running_trial_suggestions(sampler):
    study = optuna.create_study(sampler=sampler)
    values = []
    for _ in range(15):
        trial = study.ask()
        values.append(trial.suggest_float("x", -2.0, 2.0))
        if trial.number % 3:
            study.tell(trial, values[-1] ** 2)
    return values


def test_constant_liar_sequence_matches_optuna():
    with pytest.warns(optuna.exceptions.ExperimentalWarning):
        upstream_sampler = optuna.samplers.TPESampler(
            seed=37, n_startup_trials=3, constant_liar=True
        )
    with pytest.warns(optuna.exceptions.ExperimentalWarning):
        mojo_sampler = TPESampler(seed=37, n_startup_trials=3, constant_liar=True)
    assert _running_trial_suggestions(mojo_sampler) == _running_trial_suggestions(
        upstream_sampler
    )


def test_sampler_public_signature_and_protocol_match():
    assert isinstance(TPESampler(seed=0), BaseSampler)
    assert inspect.signature(TPESampler) == inspect.signature(
        optuna.samplers.TPESampler
    )


def test_best_acquisition_uses_strict_first_maximum():
    below = np.array([1.0, 2.0, 2.0 + 5e-10, 1.5])
    above = np.zeros(4)
    assert best_acquisition(below, above) == 2

    below[1] = below[2]
    assert best_acquisition(below, above) == 1


def test_compare_rejects_inconsistent_candidate_count():
    with pytest.raises(ValueError):
        TPESampler._compare({"x": np.array([1.0, 2.0])}, np.array([1.0]))
