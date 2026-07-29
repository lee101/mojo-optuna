"""Benchmark Mojo TPE density evaluation against Optuna 4.9."""

from __future__ import annotations

import math
import os
import platform
import sys
import time

import numpy as np

sys.path.insert(
    0,
    os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "python"
    ),
)

from optuna.distributions import CategoricalDistribution, FloatDistribution, IntDistribution
from optuna.samplers._tpe.parzen_estimator import (
    _ParzenEstimator,
    _ParzenEstimatorParameters,
)
from optuna.samplers._tpe.sampler import default_weights

from mojo_optuna._tpe import MojoParzenEstimator


def timeit(function, repeat=5):
    best = math.inf
    for _ in range(repeat):
        start = time.perf_counter()
        function()
        best = min(best, time.perf_counter() - start)
    return best


PARAMETERS = _ParzenEstimatorParameters(
    prior_weight=1.0,
    consider_magic_clip=True,
    consider_endpoints=False,
    weights=default_weights,
    multivariate=False,
    categorical_distance_func={},
)


def estimators(
    observations, space, samples, *, multivariate=False, device="cpu"
):
    parameters = PARAMETERS._replace(multivariate=multivariate)
    upstream = _ParzenEstimator(observations, space, parameters)
    mojo = MojoParzenEstimator(observations, space, parameters)
    mojo.log_pdf(samples, device=device)
    return (
        lambda: mojo.log_pdf(samples, device=device),
        lambda: upstream.log_pdf(samples),
    )


def continuous_1d():
    rng = np.random.default_rng(1)
    observations = {"x": rng.normal(size=20_000)}
    space = {"x": FloatDistribution(-6.0, 6.0)}
    samples = {"x": rng.uniform(-6.0, 6.0, size=24)}
    return estimators(observations, space, samples)


def multivariate_8d():
    rng = np.random.default_rng(2)
    observations = {
        f"x{i}": rng.uniform(-5.0, 5.0, size=5_000) for i in range(8)
    }
    space = {f"x{i}": FloatDistribution(-5.0, 5.0) for i in range(8)}
    samples = {f"x{i}": rng.uniform(-5.0, 5.0, size=128) for i in range(8)}
    return estimators(observations, space, samples, multivariate=True)


def mixed_5d(device="cpu"):
    rng = np.random.default_rng(3)
    n = 5_000
    observations = {
        "a": rng.uniform(-5.0, 5.0, n),
        "b": np.exp(rng.uniform(np.log(1e-3), np.log(10.0), n)),
        "c": rng.integers(0, 100, n).astype(float),
        "d": rng.integers(1, 65, n).astype(float),
        "e": rng.integers(0, 5, n).astype(float),
    }
    space = {
        "a": FloatDistribution(-5.0, 5.0),
        "b": FloatDistribution(1e-3, 10.0, log=True),
        "c": IntDistribution(0, 99),
        "d": IntDistribution(1, 64, log=True),
        "e": CategoricalDistribution(["a", "b", "c", "d", "e"]),
    }
    samples = {
        "a": rng.uniform(-5.0, 5.0, 64),
        "b": np.exp(rng.uniform(np.log(1e-3), np.log(10.0), 64)),
        "c": rng.integers(0, 100, 64).astype(float),
        "d": rng.integers(1, 65, 64).astype(float),
        "e": rng.integers(0, 5, 64).astype(float),
    }
    return estimators(observations, space, samples, device=device)


def categorical():
    rng = np.random.default_rng(4)
    observations = {"x": rng.integers(0, 8, 20_000).astype(float)}
    space = {"x": CategoricalDistribution(tuple(range(8)))}
    samples = {"x": rng.integers(0, 8, 256).astype(float)}
    return estimators(observations, space, samples)


CASES = [
    ("1D continuous: 24 candidates x 20k kernels", continuous_1d),
    ("8D continuous: 128 candidates x 5k kernels", multivariate_8d),
    ("mixed 5D: 64 candidates x 5k kernels", mixed_5d),
    ("categorical: 256 candidates x 20k kernels", categorical),
]


def cpu_name():
    try:
        with open("/proc/cpuinfo", encoding="utf-8") as handle:
            for line in handle:
                if line.startswith("model name"):
                    return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or platform.machine()


def main():
    print(f"Machine: {cpu_name()}, {platform.system()} {platform.machine()}")
    print()
    print("| case | mojo-optuna | optuna 4.9 | speedup |")
    print("| --- | ---: | ---: | ---: |")
    for name, build in CASES:
        mojo, upstream = build()
        mojo_time = timeit(mojo)
        upstream_time = timeit(upstream)
        print(
            f"| {name} | {mojo_time * 1e3:.2f} ms | "
            f"{upstream_time * 1e3:.2f} ms | {upstream_time / mojo_time:.2f}x |"
        )
    if os.environ.get("MOJO_OPTUNA_BENCH_GPU") == "1":
        gpu, upstream = mixed_5d(device="gpu")
        gpu_time = timeit(gpu)
        upstream_time = timeit(upstream)
        print()
        print("| GPU case | mojo-optuna GPU | optuna 4.9 | speedup |")
        print("| --- | ---: | ---: | ---: |")
        print(
            "| mixed 5D: 64 candidates x 5k kernels | "
            f"{gpu_time * 1e3:.2f} ms | {upstream_time * 1e3:.2f} ms | "
            f"{upstream_time / gpu_time:.2f}x |"
        )


if __name__ == "__main__":
    main()
