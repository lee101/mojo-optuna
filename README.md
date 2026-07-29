# mojo-optuna

Optuna's TPE sampler and a focused set of pruners, with the expensive
mixture-density scoring loop implemented in Mojo.

This is not a replacement for all of Optuna. It plugs into a normal
`optuna.study.Study`: Optuna continues to provide trials, distributions, storage,
and study orchestration, while `mojo_optuna.TPESampler` evaluates the Parzen
mixtures in a compiled shared library. The public sampler and pruner constructors
mirror Optuna 4.9, so the covered classes can be swapped at the import.

## Coverage

Covered and tested TPE behavior:

- float and integer parameters, including `log=True` and quantized `step`;
- categorical parameters and categorical distance functions;
- univariate, multivariate, grouped, single-objective, and multi-objective TPE;
- startup random sampling, custom gamma/weights, constant liar, and constraints.

The implemented pruners are `MedianPruner`, `PercentilePruner`,
`SuccessiveHalvingPruner`, `HyperbandPruner`, `ThresholdPruner`,
`PatientPruner`, and `NopPruner`.

The tests compare density results with upstream for every listed distribution
kind, exercise each listed TPE mode, and compare pruning decisions with upstream
for every listed pruner. This project does not port Optuna's other samplers,
`WilcoxonPruner`, visualization and dashboard features, storage implementations,
or integration packages. It also does not replace Optuna's trial, study,
distribution, or storage APIs. Candidate generation and study bookkeeping remain
in upstream Optuna.

## Install

```bash
pixi install
pixi run build
```

`pixi.lock` pins the tested Mojo toolchain and Optuna dependency set. The project
currently targets Linux x86-64 and builds a native shared library; it does not
ship a prebuilt wheel.

## Usage

```python
import optuna

from mojo_optuna import MedianPruner, TPESampler


def objective(trial):
    x = trial.suggest_float("x", -10.0, 10.0)
    scale = trial.suggest_float("scale", 1e-3, 10.0, log=True)
    for step in range(5):
        loss = (x - 2.0) ** 2 + scale / (step + 1)
        trial.report(loss, step)
        if trial.should_prune():
            raise optuna.TrialPruned()
    return loss


study = optuna.create_study(
    sampler=TPESampler(seed=7),
    pruner=MedianPruner(n_startup_trials=5),
)
study.optimize(objective, n_trials=50)
print(study.best_params)
```

Run the example in the Pixi environment so `python/mojo_optuna` is on
`PYTHONPATH`.

## Benchmarks

Measured with `pixi run bench` on this machine, an Intel Xeon E5-2697 v4 at 2.30 GHz,
Linux x86_64. Times are best of five warm runs and cover `log_pdf`, the hot
operation used to rank TPE candidates.

| case | mojo-optuna | optuna 4.9 | speedup |
| --- | ---: | ---: | ---: |
| 1D continuous: 24 candidates x 20k kernels | 6.97 ms | 37.22 ms | 5.34x |
| 8D continuous: 128 candidates x 5k kernels | 22.10 ms | 327.39 ms | 14.81x |
| mixed 5D: 64 candidates x 5k kernels | 28.21 ms | 68.31 ms | 2.42x |
| categorical: 256 candidates x 20k kernels | 45.10 ms | 307.24 ms | 6.81x |

Benchmark numbers are machine-specific; rerun `pixi run bench` rather than
copying this table to evaluate another system.

## How it works

`src/kernels.mojo` is one compilation unit built with
`mojo build --emit shared-lib` into `dist/libmojo-optuna.so`. The Python
estimator uses Optuna's exact Parzen component construction and truncated-normal
candidate generator, then passes contiguous NumPy `float64`/`int64` buffers as
integer addresses through `ctypes`.

Continuous mixture parameters use row-major `[component, dimension]` layout;
discrete mixtures use `[dimension, component]`. Scores use
`[candidate, component]`; Mojo adds numerical and categorical log densities,
performs a stable log-sum-exp with mixture weights, and chooses the highest
acquisition value. Python validates shapes, contiguity, categorical indices, and
non-empty pointers before each call. Python retains ownership of every array for
the full synchronous call, and no allocator-owned memory crosses the C ABI.

Continuous kernel rows, categorical scoring, and log-sum-exp use native-width
`float64` SIMD with scalar remainder loops. Discrete mixtures use a
dimension-major parameter layout so continuous dimensions remain contiguous,
and large independent candidate rows use an eight-worker CPU path. Prepared
mixture buffers and normalizers are cached on the estimator, while transformed
NumPy sample buffers cross the FFI boundary by address without an allocator
handoff.

Numerical parity tests compare all covered distribution types against Optuna's
estimator to absolute tolerance `3e-7`. Small approximation differences can
change a suggestion when acquisition values are nearly tied, so exact trial
sequences are not promised. Behavioral tests cover the sampler modes listed
above and assert identical upstream pruning outcomes.

## Development

```bash
pixi run build
pixi run test
pixi run bench
```

The test suite compares directly against the installed upstream Optuna package.

MIT licensed.
