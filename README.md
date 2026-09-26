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
Linux x86-64. Times are best of five warm runs and cover `log_pdf`, the hot
operation used to rank TPE candidates. The machine is shared, so upstream and
port columns drift together; the port column is stable to about ±5% across runs.

| case | mojo-optuna | optuna 4.9 | speedup |
| --- | ---: | ---: | ---: |
| 1D continuous: 24 candidates x 20k kernels | 2.96 ms | 29.06 ms | 9.81x |
| 8D continuous: 128 candidates x 5k kernels | 5.45 ms | 218.66 ms | 40.16x |
| mixed 5D: 64 candidates x 5k kernels | 5.63 ms | 65.53 ms | 11.64x |
| categorical: 256 candidates x 20k kernels | 28.17 ms | 217.78 ms | 7.73x |

What the same port measured before this optimization pass, same harness and
machine:

| case | before | after | port speedup gained |
| --- | ---: | ---: | ---: |
| 1D continuous: 24 candidates x 20k kernels | 6.69 ms | 2.96 ms | 2.3x |
| 8D continuous: 128 candidates x 5k kernels | 16.78 ms | 5.45 ms | 3.1x |
| mixed 5D: 64 candidates x 5k kernels | 124.46 ms | 5.63 ms | 22.1x |
| categorical: 256 candidates x 20k kernels | 114.94 ms | 28.17 ms | 4.1x |

The mixed 5D case was the only one that was slower than upstream before this
work; it is now 11.6x ahead.

With `MOJO_OPTUNA_BENCH_GPU=1 pixi run bench` the GPU row is reported too:

| GPU case | mojo-optuna GPU | optuna 4.9 | speedup |
| --- | ---: | ---: | ---: |
| mixed 5D: 64 candidates x 5k kernels | 6.20 ms | 64.97 ms | 10.49x |

That is parity with the CPU path on the same case, and the crossover data is in
the "Parallelism and GPU" section below.

Benchmark numbers are machine-specific; rerun `pixi run bench` rather than
copying this table to evaluate another system.

## How it works

`src/kernels.mojo` is one compilation unit built with
`mojo build --emit shared-lib` into `dist/libmojo-optuna.so`. The Python
estimator uses Optuna's exact Parzen component construction, then passes
contiguous NumPy `float64`/`int64` buffers as integer addresses through
`ctypes`. Python retains ownership of every array for the full synchronous call,
and no allocator-owned memory crosses the C ABI.

Mixture parameters always use row-major `[dimension, component]` layout, so the
component axis is contiguous and every scoring loop is a unit-stride vector loop.
Per component the estimator precomputes `1 / sigma`, `mu / sigma` and the
truncation normalizer once, so the hot loops contain no divisions. Scores use
`[candidate, component]`.

Three things carry most of the throughput:

- **Deduplicated discrete scoring.** Integer and quantized parameters are
  evaluated on a `[unique candidate value, unique (mu, sigma) component]` table
  rather than on every candidate/component pair, and the normalizer is folded
  into that table so the scatter is a single gather and add. For the mixed 5D
  benchmark this replaces 640 000 `log_gauss_mass` evaluations with about 10 000.
- **Fused continuous scoring.** One pass over each component block accumulates
  every dimension in a SIMD register, so the parameter arrays are read once and
  the score buffer is written once instead of once per dimension.
- **Row chunking.** `log_pdf` walks the candidates in blocks sized to stay
  cache-resident, so the `[candidate, component]` score buffer never has to make
  a round trip to main memory. The first scoring kernel for a chunk stores and
  the rest accumulate, so the buffer is never zeroed.

Mixture weights and categorical probabilities are logged once, at mixture
construction, and the categorical log-probability table is stored choice-major
so each candidate reads a contiguous row. Python validates shapes, contiguity,
categorical indices, and non-empty pointers before each call.

Numerical parity tests compare all covered distribution types against Optuna's
estimator to absolute tolerance `3e-7`. Small approximation differences can
change a suggestion when acquisition values are nearly tied, so exact trial
sequences are not promised. Behavioral tests cover the sampler modes listed
above and assert identical upstream pruning outcomes.

## Parallelism and GPU

The CPU work is serial. `parallelize` does not exist in the pinned toolchain (see
`MOJO_NOTES.md` section 4) and no working replacement was found in the `max`
package, so no multi-worker path is shipped. The earlier README claimed an
eight-worker path; that code did not exist and this claim is withdrawn. The
loops are bandwidth- and transcendental-bound rather than ALU-bound, so
splitting them across cores was not the missing lever — the wins came from
removing work, from cache residency, and from SIMD.

The GPU path does build and run: `DeviceContext` is present in this toolchain,
one launch covers the whole input, data moves once in and once out, and the
kernel matches the CPU results to about `5e-15`. It is opt-in through
`device="gpu"`. It is **not** the default and it is not a general win:

| scored elements (`candidates x components x dimensions`) | GPU | CPU | speedup |
| --- | ---: | ---: | ---: |
| 320 016 | 0.66 ms | 0.32 ms | 0.48x |
| 640 032 | 1.07 ms | 1.36 ms | 1.27x |
| 1 280 064 | 2.05 ms | 4.70 ms | 2.29x |
| 20 481 024 | 4.60 ms | 27.89 ms | 6.07x |

The GPU loses below roughly a million scored elements because host-to-device
copies and context setup dominate a kernel that the CPU finishes in a fraction of
a millisecond. `MojoParzenEstimator.GPU_MIN_WORK` is set to `1 << 20` from that
measurement, so a small study never pays for a GPU round trip. A real TPE study
carries a few hundred components and a few dozen candidates, which is well under
that threshold; the benchmark's 20k-component mixtures are synthetic. The
honest summary is that this library's default path is CPU-only, and the GPU
helps only for unusually large mixtures.

## Development

```bash
pixi run build
pixi run test
pixi run bench
```

The test suite compares directly against the installed upstream Optuna package.

MIT licensed.
