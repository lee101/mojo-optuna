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

    CHUNK_BYTES = 1 << 23
    GPU_MIN_WORK = 1 << 20

    def _prepare_mojo(self):
        mixture = self._mixture_distribution
        kernels = len(mixture.weights)
        library = lib()

        continuous: list[int] = []
        log_continuous: list[int] = []
        mus: list[np.ndarray] = []
        sigmas: list[np.ndarray] = []
        lows: list[float] = []
        highs: list[float] = []
        discrete: list[tuple] = []
        categoricals: list[tuple[int, np.ndarray]] = []

        for index, distribution in enumerate(mixture.distributions):
            if isinstance(distribution, _BatchedCategoricalDistributions):
                weights = np.ascontiguousarray(
                    np.log(distribution.weights.T), dtype=np.float64
                )
                categoricals.append((index, weights))
            elif isinstance(distribution, _BatchedTruncLogNormDistributions):
                continuous.append(index)
                log_continuous.append(len(continuous) - 1)
                mus.append(distribution.mu)
                sigmas.append(distribution.sigma)
                lows.append(np.log(distribution.low))
                highs.append(np.log(distribution.high))
            elif isinstance(distribution, _BatchedTruncNormDistributions):
                continuous.append(index)
                mus.append(distribution.mu)
                sigmas.append(distribution.sigma)
                lows.append(distribution.low)
                highs.append(distribution.high)
            elif isinstance(
                distribution,
                (
                    _BatchedDiscreteTruncNormDistributions,
                    _BatchedDiscreteTruncLogNormDistributions),
            ):
                discrete.append(
                    self._prepare_discrete(
                        library,
                        index,
                        distribution,
                        isinstance(
                            distribution, _BatchedDiscreteTruncLogNormDistributions
                        ),
                    )
                )
            else:
                raise TypeError(f"unsupported TPE distribution: {type(distribution)!r}")

        if continuous:
            mus_array = f64(np.vstack(mus))
            sigmas_array = f64(np.vstack(sigmas))
            inv_sigmas = f64(1.0 / sigmas_array)
            centers = f64(mus_array * inv_sigmas)
            normalizers = np.empty_like(mus_array)
            low_array = f64(lows)
            high_array = f64(highs)
            library.mot_compute_normalizers(
                addr(mus_array, np.float64),
                addr(sigmas_array, np.float64),
                addr(low_array, np.float64),
                addr(high_array, np.float64),
                addr(normalizers, np.float64),
                kernels,
                len(continuous),
            )
        else:
            inv_sigmas = f64(np.empty((0, kernels)))
            centers = inv_sigmas
            normalizers = inv_sigmas

        return (
            tuple(continuous),
            tuple(log_continuous),
            inv_sigmas,
            centers,
            normalizers,
            tuple(discrete),
            tuple(categoricals),
            f64(np.log(mixture.weights)),
        )

    @staticmethod
    def _prepare_discrete(library, index, distribution, log_scaled):
        pairs = np.ascontiguousarray(
            np.stack([distribution.mu, distribution.sigma], axis=1)
        )
        unique, col_offsets = np.unique(pairs, axis=0, return_inverse=True)
        col_offsets = i64(col_offsets)
        inv_sigma = f64(1.0 / unique[:, 1])
        center = f64(unique[:, 0] * inv_sigma)
        step = distribution.step
        low = distribution.low - 0.5 * step
        high = distribution.high + 0.5 * step
        if log_scaled:
            low = np.log(low)
            high = np.log(high)
        edges = f64(
            np.stack([(low - unique[:, 0]) * inv_sigma, (high - unique[:, 0]) * inv_sigma])
        )
        unique_count = unique.shape[0]
        normalizer = np.empty(unique_count, dtype=np.float64)
        library.mot_log_gauss_mass(
            addr(edges[0], np.float64),
            addr(edges[1], np.float64),
            addr(normalizer, np.float64),
            unique_count,
        )
        return (
            index,
            log_scaled,
            step,
            inv_sigma,
            center,
            col_offsets,
            f64(-normalizer),
        )

    @staticmethod
    def _continuous_samples(samples: np.ndarray, log_columns) -> np.ndarray:
        if not log_columns:
            return f64(samples)
        values = samples.copy()
        for column in log_columns:
            np.log(values[:, column], out=values[:, column])
        return values

    def log_pdf(
        self, samples_dict: dict[str, np.ndarray], device: str | None = None
    ) -> np.ndarray:
        device = device or os.environ.get("MOJO_OPTUNA_DEVICE", "cpu")
        if device not in ("cpu", "gpu"):
            raise ValueError("device must be 'cpu' or 'gpu'")
        samples = self._transform(samples_dict)
        kernels = len(self._mixture_distribution.weights)
        n = len(samples)
        if n == 0:
            return np.empty(0, dtype=np.float64)
        if kernels == 0:
            raise RuntimeError("Optuna produced a mixture with no kernels")

        try:
            prepared = self._mojo_prepared
        except AttributeError:
            prepared = self._prepare_mojo()
            self._mojo_prepared = prepared

        (
            continuous,
            log_columns,
            inv_sigmas,
            centers,
            normalizers,
            discrete,
            categoricals,
            log_weights,
        ) = prepared
        dims = len(continuous)
        library = lib()
        used_gpu = False
        if dims:
            x_numeric = self._continuous_samples(
                samples[:, list(continuous)], log_columns
            )

        tables = []
        for index, log_scaled, step, inv_sigma, center, cols, norms in discrete:
            values = f64(samples[:, index])
            unique, inverse = np.unique(values, return_inverse=True)
            if log_scaled:
                low = np.log(unique - 0.5 * step)
                high = np.log(unique + 0.5 * step)
            else:
                low = unique - 0.5 * step
                high = unique + 0.5 * step
            table = f64(low[:, None] * inv_sigma[None, :] - center[None, :])
            high_edges = f64(high[:, None] * inv_sigma[None, :] - center[None, :])
            library.mot_log_gauss_mass(
                addr(table, np.float64),
                addr(high_edges, np.float64),
                addr(table, np.float64),
                table.size,
            )
            table += norms[None, :]
            tables.append((table, i64(inverse * center.shape[0]), cols))

        choices = []
        for index, log_probabilities in categoricals:
            values = f64(samples[:, index])
            count_choices = log_probabilities.shape[0]
            if (
                not np.all(np.isfinite(values))
                or not np.all(values == np.floor(values))
                or np.any(values < 0)
                or np.any(values >= count_choices)
            ):
                raise ValueError("categorical samples must contain valid integer indices")
            choices.append((values, log_probabilities))

        rows_per_chunk = max(1, self.CHUNK_BYTES // (8 * kernels))
        accum = getattr(self, "_mojo_accum", None)
        if accum is None or accum.shape != (rows_per_chunk, kernels):
            accum = np.empty((rows_per_chunk, kernels), dtype=np.float64)
            self._mojo_accum = accum
        result = np.empty(n, dtype=np.float64)

        for start in range(0, n, rows_per_chunk):
            stop = min(start + rows_per_chunk, n)
            count = stop - start
            first = True
            if dims:
                chunk = x_numeric[start:stop]
                if device == "gpu" and count * kernels * dims >= self.GPU_MIN_WORK:
                    used_gpu = bool(
                        library.mot_score_numeric_gpu(
                            addr(chunk, np.float64),
                            addr(inv_sigmas, np.float64),
                            addr(centers, np.float64),
                            addr(normalizers, np.float64),
                            addr(accum, np.float64),
                            count,
                            kernels,
                            dims,
                        )
                    )
                    if not used_gpu:
                        warnings.warn(
                            "GPU scoring was unavailable or failed; falling back to CPU",
                            stacklevel=2,
                        )
                if not used_gpu:
                    library.mot_score_numeric(
                        addr(chunk, np.float64),
                        addr(inv_sigmas, np.float64),
                        addr(centers, np.float64),
                        addr(normalizers, np.float64),
                        addr(accum, np.float64),
                        count,
                        kernels,
                        dims,
                    )
                first = False
            for table, rows, cols in tables:
                library.mot_add_table(
                    addr(accum, np.float64),
                    addr(table, np.float64),
                    addr(rows[start:stop], np.int64),
                    addr(cols, np.int64),
                    0 if first else 1,
                    count,
                    kernels,
                )
                first = False
            for values, log_probabilities in choices:
                library.mot_score_categorical(
                    addr(values[start:stop], np.float64),
                    addr(log_probabilities, np.float64),
                    addr(accum, np.float64),
                    0 if first else 1,
                    count,
                    kernels,
                )
                first = False
            if first:
                accum.fill(0.0)
            library.mot_finish_log_pdf(
                addr(accum, np.float64),
                addr(log_weights, np.float64),
                addr(result[start:stop], np.float64),
                count,
                kernels,
            )
        self._last_device = "gpu" if used_gpu else "cpu"
        return result
