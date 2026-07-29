"""Optuna's Parzen model with mixture scoring executed by Mojo."""

from __future__ import annotations

import os
import warnings

import numpy as np
from optuna.samplers._tpe.parzen_estimator import _ParzenEstimator
from optuna.samplers._tpe.probability_distributions import (
    _BatchedCategoricalDistributions,
    _BatchedDiscreteTruncLogNormDistributions,
    _BatchedDiscreteTruncNormDistributions,
    _BatchedTruncLogNormDistributions,
    _BatchedTruncNormDistributions,
)

from ._lib import addr, f64, i64, lib


class MojoParzenEstimator(_ParzenEstimator):
    """A parity-compatible estimator whose hot log-density loop is Mojo."""

    def _prepare_mojo(self):
        mixture = self._mixture_distribution
        kernels = len(mixture.weights)

        numerical_indices: list[int] = []
        numerical_distributions = []
        kinds: list[int] = []
        for index, distribution in enumerate(mixture.distributions):
            if isinstance(distribution, _BatchedCategoricalDistributions):
                continue
            numerical_indices.append(index)
            numerical_distributions.append(distribution)
            if isinstance(distribution, _BatchedTruncNormDistributions):
                kinds.append(0)
            elif isinstance(distribution, _BatchedTruncLogNormDistributions):
                kinds.append(1)
            elif isinstance(distribution, _BatchedDiscreteTruncNormDistributions):
                kinds.append(2)
            elif isinstance(
                distribution, _BatchedDiscreteTruncLogNormDistributions
            ):
                kinds.append(3)
            else:
                raise TypeError(f"unsupported TPE distribution: {type(distribution)!r}")

        dims = len(numerical_distributions)
        dimension_major = int(any(kind >= 2 for kind in kinds))
        if not dims:
            mus = np.empty((kernels, 0), dtype=np.float64)
            sigmas = np.empty((kernels, 0), dtype=np.float64)
        elif dimension_major:
            mus = f64(
                np.vstack(
                    [distribution.mu for distribution in numerical_distributions]
                )
            )
            sigmas = f64(
                np.vstack(
                    [distribution.sigma for distribution in numerical_distributions]
                )
            )
        else:
            mus = f64(
                np.column_stack(
                    [distribution.mu for distribution in numerical_distributions]
                )
            )
            sigmas = f64(
                np.column_stack(
                    [distribution.sigma for distribution in numerical_distributions]
                )
            )
        lows = f64([distribution.low for distribution in numerical_distributions])
        highs = f64([distribution.high for distribution in numerical_distributions])
        steps = f64(
            [getattr(distribution, "step", 0.0) for distribution in numerical_distributions]
        )
        kind_array = i64(kinds)
        normalizers = np.empty_like(mus)

        if dims:
            library = lib()
            library.mot_compute_normalizers(
                addr(mus, np.float64),
                addr(sigmas, np.float64),
                addr(lows, np.float64),
                addr(highs, np.float64),
                addr(steps, np.float64),
                addr(kind_array, np.int64),
                addr(normalizers, np.float64),
                kernels,
                dims,
                dimension_major,
            )

        categoricals = []
        for index, distribution in enumerate(mixture.distributions):
            if isinstance(distribution, _BatchedCategoricalDistributions):
                categoricals.append((index, f64(distribution.weights)))

        return (
            tuple(numerical_indices),
            kind_array,
            mus,
            sigmas,
            steps,
            normalizers,
            dimension_major,
            tuple(categoricals),
            f64(mixture.weights),
        )

    def log_pdf(
        self, samples_dict: dict[str, np.ndarray], device: str | None = None
    ) -> np.ndarray:
        device = device or os.environ.get("MOJO_OPTUNA_DEVICE", "cpu")
        if device not in ("cpu", "gpu"):
            raise ValueError("device must be 'cpu' or 'gpu'")
        samples = self._transform(samples_dict)
        mixture = self._mixture_distribution
        n = len(samples)
        kernels = len(mixture.weights)
        if n == 0:
            return np.empty(0, dtype=np.float64)
        if kernels == 0:
            raise RuntimeError("Optuna produced a mixture with no kernels")
        accum = np.zeros((n, kernels), dtype=np.float64)

        try:
            prepared = self._mojo_prepared
        except AttributeError:
            prepared = self._prepare_mojo()
            self._mojo_prepared = prepared

        (
            numerical_indices,
            kind_array,
            mus,
            sigmas,
            steps,
            normalizers,
            dimension_major,
            categoricals,
            weights,
        ) = prepared
        dims = len(numerical_indices)
        if not dims:
            x_numeric = np.empty((n, 0), dtype=np.float64)
        elif numerical_indices == tuple(range(samples.shape[1])):
            x_numeric = f64(samples)
        else:
            x_numeric = f64(samples[:, numerical_indices])

        library = lib()
        used_gpu = False
        if (
            device == "gpu"
            and dimension_major
            and n * kernels * max(dims, 1) >= 131_072
        ):
            used_gpu = bool(
                library.mot_score_numeric_gpu(
                    addr(x_numeric, np.float64),
                    addr(mus, np.float64),
                    addr(sigmas, np.float64),
                    addr(steps, np.float64),
                    addr(kind_array, np.int64),
                    addr(normalizers, np.float64),
                    addr(accum, np.float64),
                    n,
                    kernels,
                    dims,
                    dimension_major,
                )
            )
            if not used_gpu:
                warnings.warn(
                    "GPU scoring was unavailable or failed; falling back to CPU",
                    RuntimeWarning,
                    stacklevel=2,
                )
        if not used_gpu and dims:
            library.mot_score_numeric(
                addr(x_numeric, np.float64),
                addr(mus, np.float64),
                addr(sigmas, np.float64),
                addr(steps, np.float64),
                addr(kind_array, np.int64),
                addr(normalizers, np.float64),
                addr(accum, np.float64),
                n,
                kernels,
                dims,
                dimension_major,
            )
        self._last_device = "gpu" if used_gpu else "cpu"

        for index, probabilities in categoricals:
            values = f64(samples[:, index])
            choices = probabilities.shape[1]
            if (
                not np.all(np.isfinite(values))
                or not np.all(values == np.floor(values))
                or np.any(values < 0)
                or np.any(values >= choices)
            ):
                raise ValueError("categorical samples must contain valid integer indices")
            library.mot_score_categorical(
                addr(values, np.float64),
                addr(probabilities, np.float64),
                addr(accum, np.float64),
                n,
                kernels,
                choices,
            )

        result = np.empty(n, dtype=np.float64)
        library.mot_finish_log_pdf(
            addr(accum, np.float64),
            addr(weights, np.float64),
            addr(result, np.float64),
            n,
            kernels,
        )
        return result
