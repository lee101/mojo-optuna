# mojo-optuna

Two things in one repo:

1. **`mopt`**: a hyperparameter and symbol-universe search engine for trading
   research harnesses. It works through ask/tell, prunes across folds, and
   reports overfitting. Each objective is an external command (Go, Mojo,
   Python, or anything else) that prints JSON metrics per fold. mopt handles the
   search, the parallel workers, fold pruning, nested walk-forward and the
   overfitting statistics.
2. **Mojo-accelerated Optuna TPE**: a drop-in `TPESampler` whose Parzen
   mixture scoring runs in a compiled Mojo library, plus Optuna-compatible
   pruners. This was the original port and is documented further down.

## Install

```bash
pixi install && pixi run build          # Mojo kernels + env (optuna, numpy, scipy, cmaes)
bin/mopt --help                         # wrapper: pixi env if present, else python3

# or as a plain Python tool (numpy fallbacks if the Mojo lib is absent)
uv tool install --editable /path/to/mojo-optuna   # editable keeps dist/libmojo-optuna.so reachable
```

mopt only needs `optuna>=4.9`, `numpy` and `scipy`. The Mojo kernels are
optional accelerators: the TPE density, the GP cross-covariance, and CSCV. Each
has a parity-tested numpy fallback.

## mopt in 60 seconds

```bash
bin/mopt init > study.json        # annotated example config
# edit "command", "params", "universe", "folds", "objectives"
bin/mopt run study.json -n 200 -j 4 --threads 2
bin/mopt report study.json        # best/pareto + PBO + DSR + nested WF + symbol-selection penalty
bin/mopt nested study.json -n 100 # true nested walk-forward (fresh search per outer fold)
bin/mopt symnull study.json -n 30 # random-universe null for the chosen params
```

Each study lives in an Optuna journal file. `"storage"` is resolved relative to
the config, and `sqlite:///...` also works. Several `mopt run` processes, or
machines sharing a filesystem, can work on one study concurrently.

### Objective protocol

The command is a shell template. `{name}` placeholders are filled per trial,
and shell `${VAR:-default}` passes through untouched.

| placeholder | value |
| --- | --- |
| `{<param>}` | param value (fixed params too) |
| `{args}` | all params formatted with `arg_format` (default `--{name}={value}`; e.g. `-{name} {value}`, `--set {name}={value}`); a spec with `"arg": false` is skipped |
| `{universe}` / `{n_universe}` / `{universe_json}` | selected symbols joined by `universe.join` (default `,`) / count / JSON list |
| `{fold}` / `{folds}` | current fold (per_fold mode) / all folds in evaluation order, comma-joined |
| `{seed}` / `{seeds}` | per_fold mode repeats each fold for every entry of `"seeds"` and averages numeric metrics (`<k>@min`/`@max` kept) |
| `{trial}` `{study}` `{workdir}` `{params_json}` | trial number, study name, run dir, path of a JSON file with everything above |

The same values are exported as `MOPT_TRIAL`, `MOPT_FOLD`, `MOPT_SEED`,
`MOPT_PARAMS` (JSON), `MOPT_UNIVERSE`, `MOPT_FOLDS`, `MOPT_JSON` and
`MOPT_STUDY`. `--threads N` (or `"threads_per_job"`) sets `OMP_NUM_THREADS`,
`GOMAXPROCS`, `MKL/OPENBLAS/RAYON/POLARS/NUMBA` to N for each job. mopt's own
pixi python is removed from the child `PATH`, so `python3` in a command means
your normal python. Set `"inherit_python_env": true` to keep it.

Output: every stdout line that is a JSON object is a metrics row, for example
`{"fold": 3, "ret": 4.2, "mdd": 18.0}`. Set `"parse": {"type": "kv", "line":
"^fold (\\d+)", "fold_group": 1}` to scrape `key=value` text lines instead.

- `"mode": "per_fold"` runs the command once per fold (and seed), in
  `fold_order`. A fold pruner stops the trial between folds.
- `"mode": "stream"` runs the command once. It prints a row per fold as each
  fold finishes, and mopt kills the process group as soon as the trial is
  pruned. A row whose fold is not in the study's fold set is ignored, which is
  how `nested` reuses all-fold harnesses.

A non-zero exit, missing folds or a non-finite objective marks the trial
`FAIL`. The stderr tail is stored in the trial and in `runs/<study>/t*.err`.

### Config reference

```jsonc
{
  "name": "kucoin_risk", "storage": "studies/kucoin_risk.journal", "seed": 0,
  "command": "...", "mode": "stream", "cwd": "..", "timeout": 10800, "env": {}, "threads_per_job": 4,
  "folds": 17,                       // int n -> 0..n-1, or explicit list (e.g. years)
  "fold_order": "spread",            // farthest-point order from the newest fold, so early rungs span regimes; or "given"
  "params": {
    "vol-target": {"type": "float", "low": 0.3, "high": 0.6, "step": 0.05, "default": 0.45},
    "lb":         {"type": "int", "low": 24, "high": 720, "log": true},
    "mode":       {"type": "cat", "choices": ["a", "b"]},
    "data":       {"type": "fixed", "value": "/x", "arg": false}
  },
  "universe": {"symbols": ["BTC", "ETH", "..."], "always": ["BTC"], "never": [], "min": 8, "max": 24, "join": ","},
  "objectives": [{"metric": "ret", "agg": "mean", "direction": "maximize"},
                 {"metric": "mdd", "agg": "max",  "direction": "minimize"}],
  "constraints": [{"metric": "mdd", "agg": "max", "le": 35}],      // or "ge"
  "select": {"metric": "ret", "agg": "lcb", "direction": "maximize"}, // scalar used by nested/PBO/DSR/symnull
  "sampler": {"type": "tpe", "multivariate": true, "constant_liar": true, "n_startup": 20},
  "pruner": {"type": "asha", "min_folds": 2, "eta": 3, "n_startup": 6,
             "kill": [{"metric": "mdd", "agg": "max", "gt": 55}]},
  "nested": {"outer": [11, 12, 13, 14, 15, 16], "mode": "before", "purge": 0, "min_inner": 6},
  "baseline": {"vol-target": 0.45, "...": "...", "universe": "all"}   // enqueued as trial 0, used as comparator
}
```

Aggregations are `mean median min max sum std sharpe lcb (mean - 1 s.e.) first
last count qNN` (for example `q10`). `--set dotted.key=json` overrides any key
from the command line, for example `--set sampler.type='"gp"'`.

### Samplers (`sampler.type`)

| type | what | use for |
| --- | --- | --- |
| `tpe` (default) | Mojo TPE, multivariate, `constant_liar` for parallel workers; with 2+ objectives this is MOTPE | general, mixed spaces, universes |
| `gp` | Matern-5/2 ARD GP + EI. Hyperparameters by L-BFGS on the marginal likelihood with priors; running trials enter as constant liars; multi-objective via ParEGO random Chebyshev weights | small budgets (< ~150 trials) of slow harnesses |
| `cmaes` | CMA-ES on the unit cube with **IPOP restarts** (popsize x2 on tolx/tolfun/condition stall). Stateless: the state is replayed from the trial history, so any number of ask/tell processes agree | continuous risk knobs, 3-30 dims, 200+ trials |
| `nsga2` | Optuna NSGA-II with constraints | multi-objective with large budgets |
| `ga` | steady-state GA over universe bits + numeric params (tournament on Pareto rank, uniform crossover, blend + gaussian mutation, duplicate avoidance, cardinality repair) | symbol subsets with interactions |
| `forward` / `backward` | greedy stepwise symbol selection (numeric params held at `default`); falls back to `ga` when converged or when all candidates are in flight | cheap, interpretable universe pruning |
| `random`, `qmc`, `tpe-optuna`, `cmaes-optuna` | baselines | |

With a `universe`, every sampler is wrapped so the include/exclude bits are
sampled jointly and repaired to `min <= |U| <= max` (`always` and `never` are
not searched).

### Fold pruning (`pruner.type`)

The intermediate value after *k* folds is the prune metric (default: the first
objective) aggregated over those folds. It is compared with every other trial
**on the same fold set**, so it stays fair when folds arrive out of order. This
works for multi-objective studies too, because mopt implements it instead of
calling Optuna's single-objective pruners.

- `asha`: rungs at `min_folds * eta^i`. At a rung, the trial continues only if
  it is in the top `1/eta` of the trials that reached that rung (after
  `n_startup` competitors).
- `hyperband`: like ASHA, with brackets `trial % B` that each start at
  `min_folds * eta^b`.
- `median` / `percentile`: prune if the trial is below the median or the given
  percentile at any fold count >= `min_folds`.
- `kill`: hard stops, for example worst-fold monthly DD > 55, applied at every
  fold.

When a harness only prints its folds at the very end (kucoin, poloniex, binance
kfold), use `"type": "none"` with `kill` rules. Pruning would only throw away
work that is already done.

### Overfitting control (`mopt report`, `nested`, `symnull`)

- **PBO via CSCV** (Bailey et al. 2015). The `[fold, trial]` matrix of the
  select metric is split into S <= 16 contiguous blocks, and every S/2
  combination is used once as in-sample. PBO is P(the IS-best trial ranks below
  the OOS median), plus the mean OOS of the IS-best and the IS-to-OOS
  degradation slope. The CSCV loop is a Mojo kernel: 12 870 combos x 500 trials
  in 64 ms, against 690 ms in numpy.
- **Deflated Sharpe** (Bailey and Lopez de Prado 2014). This uses the per-fold
  select-metric series of the best trial, with skew and kurtosis. SR0 is the
  expected max Sharpe of N_eff trials, where N_eff = rho + (1 - rho) N comes
  from the mean pairwise correlation of trial fold vectors. The report also
  gives PSR.
- **Nested walk-forward**:
  - `report` does it post hoc. For each outer fold it picks the best trial on
    the inner folds (`before` = earlier folds minus `purge`, or `other` = all
    folds outside the purge band), applies the constraints on the inner folds
    as well, and scores that pick on the outer fold. It reports the outer mean,
    the optimism (inner - outer), the pick's percentile on the outer fold
    (0.5 = no skill), and the delta and win rate against `baseline`.
  - `mopt nested` is the strict version. It runs a fresh search per outer fold
    that only ever sees the inner folds, evaluates the winner (and the
    baseline) on the outer fold, and reports the same statistics plus the
    Jaccard stability of the universes it picked.
- **Selection-on-symbols penalty** (when a `universe` is searched):
  - The in-sample gain of the best universe over the full universe.
  - The nested inner gain against the outer gain. `selection_penalty` is the
    part of the gain that does not survive out of fold.
  - Per-symbol inclusion lift in the top 10% of trials.
  - The Jaccard stability of the outer-fold picks.
  - `mopt symnull` evaluates the chosen params on K random universes of the
    same size. With that, the report adds E[max of N_distinct null draws] and
    the luck-adjusted edge.

### Process interfaces (any language)

```bash
# stateless one-shot CLI (journal storage is shared across processes and hosts)
bin/mopt ask  study.json                                    # {"trial":7,"params":{...},"universe":[...],"folds":[...],"seeds":[null]}
bin/mopt tell study.json 7 --fold 3 --metrics '{"ret":1.2,"mdd":14}'   # -> {"pruned":false,"complete":false,...}
bin/mopt tell study.json 7 --values 1.2,14                  # or tell final objective values directly
bin/mopt fail study.json 7 --error "panic"
bin/mopt enqueue study.json '{"vol-target":0.45,"universe":"all"}' --baseline

# JSON lines over stdin/stdout (one long-lived process)
bin/mopt serve study.json
{"op":"ask"}                         -> {"trial":8,...}
{"op":"tell","trial":8,"fold":0,"metrics":{"ret":1.0,"mdd":9}}  -> {"pruned":false,...}
{"op":"best"} / {"op":"report"} / {"op":"fail","trial":8} / {"op":"enqueue","params":{...}}

# local HTTP (same ops)
bin/mopt http study.json --port 8765
curl -s -XPOST localhost:8765/ask -d '{}'
curl -s -XPOST localhost:8765/tell -d '{"trial":8,"fold":0,"metrics":{"ret":1}}'
curl -s localhost:8765/best
```

A harness that drives itself loops `ask`, runs its folds, and sends one `tell`
per fold. When the answer says `"pruned": true`, it stops that trial.

### Adapters (`adapters/`)

| adapter | harness | per-fold metrics | config |
| --- | --- | --- | --- |
| `kucoin_kfold.py` | bitbankkucoin `kfold` (Go GBDT). Universe via a shadow `-data` dir with a filtered `universe.json` | ret, dd, mdd (worst calendar month), rdd30, sharpe, trades... | `configs/kucoin.json` |
| `binancego_kfold.py` | binancego `scripts/kfold_eval.py` (honest sim, E14). Params override grid flags with `--set`; universe = `--pairs` | ret, dd, monthly_median, monthly_p10, trades | `configs/binancego.json` |
| `vmidx_wf.py` | predictionmojo VM-SPY (`research/r13` idx/idx2). One fold per calendar year | ret, spy, excess, dd, mdd, rdd21 | `configs/vmidx.json` |
| `poloniex_replay.py` | bitbankpoloniex hourly k-fold replay (`run_replay.sh`). Universe through cached `make_fixture.py --drop` fixtures; params override the replay config with `--cfg` | ret (fee mean), ret_14/ret_33, dd, liq, fills, mdd | `configs/poloniex.json` |

The configs need the harness location from the environment: `KC_REPO`,
`BGO_REPO` (plus optional `BGO_DATA`/`BGO_CACHE`), `PM_ROOT`, and
`BBPX_WT`/`BBPX_BIN`/`BBPX_PANEL`/`BBPX_SCORES`/`BBPX_BASE`. `ADAPTER_PY`
picks the python. Adjust `folds` and the param ranges to the harness revision
you run. Run heavy sweeps on a research box. On a machine that runs live bots,
keep it to `-j 2 --threads 2` or less.

## Benchmarks

Measured on a shared AMD Ryzen 9 3900X running other sweeps (load ~25 on 24
threads) with Python 3.13, Optuna 4.9 and cmaes 0.13. Each run used 10 seeds
and 4 processes, one thread each. The ms/trial column is sampler overhead only,
since the objectives cost nothing. Raw results are in `bench/results_search.json`.
To reproduce, run `pixi run bench-search -- --seeds 10 --procs 4`.

median regret after 50 trials (10 seeds; lower is better)

| sampler | sphere5 | rosen5 | rastrigin5 | ackley5 | styb5 | branin2 | hartmann6 | ms/trial |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| random | 7.16 | 85.1 | 46.4 | 17.2 | 61.4 | 0.884 | 1.74 | 0.5 |
| optuna-tpe | 3.59 | 33 | 34.6 | 14.7 | 34.8 | 0.426 | 0.771 | 8.8 |
| optuna-tpe-mv | 1.33 | 28.1 | 37.6 | 12.5 | 62.3 | 0.148 | 0.633 | 4.5 |
| mojo-tpe-mv | 1.33 | 28.1 | 37.6 | 12.5 | 62.3 | 0.148 | 0.633 | 3.6 |
| optuna-cma | 0.792 | 10.5 | 27.7 | 9.93 | 66.9 | 1.13 | 0.922 | 1.1 |
| optuna-cma-ipop | 0.792 | 10.5 | 27.7 | 9.93 | 66.9 | 1.13 | 0.922 | 1.0 |
| mopt-cma-ipop | 1.25 | 19.5 | 27.8 | 12.1 | 53.6 | 0.967 | 1.12 | 3.4 |
| mopt-gp-ei | 0.00133 | 13.3 | 32.6 | 5.34 | 35 | 2.76e-05 | 0.00331 | 36.9 |

median regret after 200 trials (10 seeds; lower is better)

| sampler | sphere5 | rosen5 | rastrigin5 | ackley5 | styb5 | branin2 | hartmann6 | ms/trial |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| random | 5.86 | 70.1 | 35.4 | 16.4 | 45.5 | 0.101 | 1.18 | 0.5 |
| optuna-tpe | 0.273 | 5.17 | 17 | 6.66 | 6.49 | 0.00482 | 0.11 | 8.8 |
| optuna-tpe-mv | 0.435 | 8.87 | 23 | 7.43 | 16.6 | 0.0359 | 0.173 | 4.5 |
| mojo-tpe-mv | 0.435 | 8.87 | 23 | 7.43 | 16.6 | 0.0359 | 0.173 | 3.6 |
| optuna-cma | 0.00945 | 3.3 | 16.9 | 2.59 | 29.4 | 8.35e-06 | 0.0755 | 1.1 |
| optuna-cma-ipop | 0.00945 | 3.3 | 16.9 | 2.59 | 29.4 | 8.35e-06 | 0.0755 | 1.0 |
| mopt-cma-ipop | 0.0094 | 3.91 | 16.9 | 2.62 | 16.4 | 4.93e-06 | 0.0522 | 3.4 |

parallel workers (4 asks before tells, hartmann6, 200 trials): optuna-tpe-mv 0.202, optuna-tpe-mv-cl 0.209, mojo-tpe-mv-cl 0.209

ZDT1 (6-d) hypervolume after 200 trials, higher is better: random 4.541 (0.4 ms/trial), optuna-nsga2 5.237 (0.5 ms/trial), optuna-motpe 5.779 (6.8 ms/trial), mojo-motpe 5.779 (5.8 ms/trial)

30-symbol universe (4-10 symbols, additive + pairwise terms), best score after 200 trials, higher is better: random 2.154, mojo-tpe-mv 3.231, ga 3.388, forward 3.402

What the numbers say:

- **Mojo TPE matches Optuna's TPE suggestion for suggestion.** The regret is
  identical on every function and seed, and so is the hypervolume. It costs
  20% less per trial (3.6 ms vs 4.5 ms; MOTPE 5.8 ms vs 6.8 ms). At study
  scale, Optuna's own bookkeeping dominates the time, not the density
  evaluation; the kernel-level gains are in the table further down.
- **GP-EI is the small-budget choice.** After 50 trials it has 100-1000x lower
  regret than TPE or CMA on smooth functions (sphere, branin, hartmann6) and
  the best ackley result. It costs ~37 ms per suggestion, which is negligible
  next to a backtest. It is weak on rastrigin and styblinski, which have many
  local minima.
- **CMA-ES with IPOP restarts.** At 200 trials it matches Optuna's CMA-ES on
  sphere, rastrigin, ackley and branin, and is better on multimodal styblinski
  (16.4 vs 29.4) and on hartmann6 (0.052 vs 0.076). Optuna's
  `restart_strategy="ipop"` never actually restarted within 200 trials (its
  rows are identical to plain CMA-ES); mopt's stall tests do restart.
  Optuna's CMA-ES is ahead at 50 trials and on rosenbrock.
- **Multivariate vs univariate TPE.** Univariate TPE wins at 200 trials on
  these separable test functions. Multivariate wins at 50 trials on 5 of 7.
  mopt defaults to multivariate because it is needed for joint universe repair
  and real harness parameters interact. Use `"multivariate": false` for
  independent knobs.
- **`constant_liar` did not help** on hartmann6 with 4 parallel asks (0.209 vs
  0.202). It is kept on by default because it stops duplicate suggestions when
  workers share a study.
- **Universe search.** Over 30 symbols with pairwise interactions, greedy
  forward (3.40) and the GA (3.39) beat multivariate TPE (3.23) and random
  (2.15) at 200 trials.
- **Multi-objective.** MOTPE beats NSGA-II on ZDT1 hypervolume at 200 trials
  (5.78 vs 5.24).

Search-engine Mojo kernels on the same machine
(`bench/bench_search_kernels.py`, single-threaded numpy, results asserted equal):

| kernel | mojo | numpy | speedup |
| --- | ---: | ---: | ---: |
| Matern-5/2 cross 2560x300x8 (GP-EI candidates) | 5.13 ms | 49.38 ms | 9.6x |
| Matern-5/2 cross 2560x400x30 | 15.87 ms | 72.24 ms | 4.6x |
| CSCV S=12 (924 combos) x 300 trials | 1.05 ms | 33.26 ms | 31.6x |
| CSCV S=16 (12870 combos) x 1000 trials | 54.98 ms | 611.16 ms | 11.1x |

Example run: `adapters/configs/vmidx.json`, 80 TPE trials x 21 yearly folds,
2 local jobs, about 2.5 s per trial. The report gave PBO 0.003, DSR 0.34
(N_eff 35 of 80), and a nested post-hoc pick at the 0.75 outer percentile,
+8.5 +/- 4.5 %/yr excess over the deployed baseline. The folds start in 2005,
so this does not cover the 1999-2004 G3 failure of the QQQ blend. That is a
reminder that the fold set defines what the statistics can see.

## Mojo TPE for Optuna

```python
import optuna
from mojo_optuna import MedianPruner, TPESampler

study = optuna.create_study(sampler=TPESampler(seed=7, multivariate=True, constant_liar=True),
                            pruner=MedianPruner(n_startup_trials=5))
```

This plugs into a normal `optuna.study.Study`. Optuna still provides trials,
distributions, storage and orchestration, and `mojo_optuna.TPESampler`
evaluates the Parzen mixtures in a compiled shared library. The constructors
mirror Optuna 4.9.

What is covered and tested:

- float and int params, including log and step, plus categorical params and
  categorical distances;
- univariate, multivariate, group, multi-objective, startup, gamma/weights,
  constant liar and constraints;
- the pruners `MedianPruner`, `PercentilePruner`, `SuccessiveHalvingPruner`,
  `HyperbandPruner`, `ThresholdPruner`, `PatientPruner` and `NopPruner`, whose
  decisions are asserted identical to upstream.

### TPE density kernel benchmark

Measured with `pixi run bench` on the same 3900X (shared, loaded). Times are
the best of five warm runs of `log_pdf`, the hot operation for ranking TPE
candidates.

| case | mojo-optuna | optuna 4.9 | speedup |
| --- | ---: | ---: | ---: |
| 1D continuous: 24 candidates x 20k kernels | 1.66 ms | 43.00 ms | 25.87x |
| 8D continuous: 128 candidates x 5k kernels | 8.93 ms | 287.03 ms | 32.13x |
| mixed 5D: 64 candidates x 5k kernels | 5.25 ms | 82.21 ms | 15.65x |
| categorical: 256 candidates x 20k kernels | 38.07 ms | 280.19 ms | 7.36x |

`MOJO_OPTUNA_BENCH_GPU=1 pixi run bench` adds the opt-in GPU row. It only pays
off above about 1M scored elements; see below.

### How it works

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

### Parallelism and GPU

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
pixi run build          # dist/libmojo-optuna.so (TPE density, GP covariance, CSCV kernels)
pixi run test           # TPE/pruner parity vs upstream Optuna + search engine, CLI, HTTP, adapters' toy
pixi run bench          # TPE density kernels
pixi run bench-search   # search quality vs Optuna (run on a research box, not next to live bots)
```

The pinned toolchain is `mojo ==1.2.0.dev2026092605`, and `pixi.lock` is
format v7, so it needs pixi >= 0.60. Dialect notes are in `MOJO_NOTES.md`.

MIT licensed.

