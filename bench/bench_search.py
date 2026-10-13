"""Search-quality benchmark: regret + sampler overhead vs Python Optuna.

python bench/bench_search.py --seeds 10 --procs 4 --out bench/results_search.json
"""

from __future__ import annotations

import argparse
import json
import math
import os
import platform
import sys
import time
import warnings
from concurrent.futures import ProcessPoolExecutor

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "python"))
warnings.filterwarnings("ignore")


def sphere(x):
    return float(np.sum(x ** 2))


def rosenbrock(x):
    return float(np.sum(100 * (x[1:] - x[:-1] ** 2) ** 2 + (1 - x[:-1]) ** 2))


def rastrigin(x):
    return float(10 * len(x) + np.sum(x ** 2 - 10 * np.cos(2 * np.pi * x)))


def ackley(x):
    d = len(x)
    return float(-20 * np.exp(-0.2 * np.sqrt(np.sum(x ** 2) / d)) - np.exp(np.sum(np.cos(2 * np.pi * x)) / d) + 20 + np.e)


def styblinski(x):
    return float(np.sum(x ** 4 - 16 * x ** 2 + 5 * x) / 2 + 39.16616570377142 * len(x))


def branin(x):
    x1, x2 = x
    a, b, c, r, s, t = 1, 5.1 / (4 * np.pi ** 2), 5 / np.pi, 6, 10, 1 / (8 * np.pi)
    return float(a * (x2 - b * x1 ** 2 + c * x1 - r) ** 2 + s * (1 - t) * np.cos(x1) + s - 0.397887357729738)


H6A = np.array([[10, 3, 17, 3.5, 1.7, 8], [0.05, 10, 17, 0.1, 8, 14], [3, 3.5, 1.7, 10, 17, 8], [17, 8, 0.05, 10, 0.1, 14]])
H6P = 1e-4 * np.array([[1312, 1696, 5569, 124, 8283, 5886], [2329, 4135, 8307, 3736, 1004, 9991],
                       [2348, 1451, 3522, 2883, 3047, 6650], [4047, 8828, 8732, 5743, 1091, 381]])
H6C = np.array([1.0, 1.2, 3.0, 3.2])


def hartmann6(x):
    return float(-np.sum(H6C * np.exp(-np.sum(H6A * (x - H6P) ** 2, axis=1))) + 3.32236801141551)


FUNCS = {
    "sphere5": (sphere, 5, -5.12, 5.12),
    "rosen5": (rosenbrock, 5, -2.048, 2.048),
    "rastrigin5": (rastrigin, 5, -5.12, 5.12),
    "ackley5": (ackley, 5, -32.768, 32.768),
    "styb5": (styblinski, 5, -5, 5),
    "branin2": (branin, 2, None, None),
    "hartmann6": (hartmann6, 6, 0, 1),
}


def make(kind, seed):
    import optuna

    from mojo_optuna.samplers import TPESampler as MojoTPE
    from mojo_optuna.search.cmaes import CmaEsIpopSampler
    from mojo_optuna.search.gp import GPEISampler

    if kind == "random":
        return optuna.samplers.RandomSampler(seed=seed)
    if kind == "optuna-tpe":
        return optuna.samplers.TPESampler(seed=seed)
    if kind == "optuna-tpe-mv":
        return optuna.samplers.TPESampler(seed=seed, multivariate=True)
    if kind == "mojo-tpe-mv":
        return MojoTPE(seed=seed, multivariate=True)
    if kind == "optuna-tpe-mv-cl":
        return optuna.samplers.TPESampler(seed=seed, multivariate=True, constant_liar=True)
    if kind == "mojo-tpe-mv-cl":
        return MojoTPE(seed=seed, multivariate=True, constant_liar=True)
    if kind == "optuna-cma":
        return optuna.samplers.CmaEsSampler(seed=seed, warn_independent_sampling=False)
    if kind == "optuna-cma-ipop":
        return optuna.samplers.CmaEsSampler(seed=seed, restart_strategy="ipop", warn_independent_sampling=False)
    if kind == "mopt-cma-ipop":
        return CmaEsIpopSampler(seed=seed)
    if kind == "mopt-gp-ei":
        return GPEISampler(seed=seed, n_startup_trials=10)
    if kind == "optuna-gp":
        return optuna.samplers.GPSampler(seed=seed)
    raise ValueError(kind)


def run_one(args):
    fname, kind, seed, budget, batch = args
    import optuna

    optuna.logging.set_verbosity(optuna.logging.ERROR)
    f, d, lo, hi = FUNCS[fname]
    bounds = [(-5, 10), (0, 15)] if fname == "branin2" else [(lo, hi)] * d
    study = optuna.create_study(sampler=make(kind, seed))
    best, curve = math.inf, []
    t0 = time.perf_counter()
    n = 0
    while n < budget:
        k = min(batch, budget - n)
        trials = [study.ask() for _ in range(k)]
        xs = [np.array([t.suggest_float(f"x{i}", *bounds[i]) for i in range(d)]) for t in trials]
        for t, x in zip(trials, xs):
            v = f(x)
            study.tell(t, v)
            best = min(best, v)
            curve.append(best)
        n += k
    secs = time.perf_counter() - t0
    return {"f": fname, "sampler": kind, "seed": seed, "budget": budget, "batch": batch,
            "regret": max(best, 0.0), "ms_per_trial": secs / budget * 1e3,
            "regret_at": {str(b): max(curve[b - 1], 0.0) for b in (25, 50, 100, 200, 400) if b <= len(curve)}}


# ------------------------------------------------------------ multi-objective (ZDT1)

def zdt1(x):
    f1 = x[0]
    g = 1 + 9 * np.mean(x[1:])
    return f1, g * (1 - math.sqrt(f1 / g))


def hv2(F, ref=(1.1, 6.0)):
    F = np.array([p for p in F if p[0] < ref[0] and p[1] < ref[1]])
    if len(F) == 0:
        return 0.0
    F = F[np.argsort(F[:, 0])]
    hv, prev = 0.0, ref[1]
    for a, b in F:
        if b < prev:
            hv += (ref[0] - a) * (prev - b)
            prev = b
    return hv


def run_mo(args):
    kind, seed, budget = args
    import optuna

    from mojo_optuna.samplers import TPESampler as MojoTPE

    optuna.logging.set_verbosity(optuna.logging.ERROR)
    s = {"random": optuna.samplers.RandomSampler(seed=seed),
         "optuna-nsga2": optuna.samplers.NSGAIISampler(seed=seed, population_size=20),
         "optuna-motpe": optuna.samplers.TPESampler(seed=seed, multivariate=True),
         "mojo-motpe": MojoTPE(seed=seed, multivariate=True)}[kind]
    st = optuna.create_study(directions=["minimize", "minimize"], sampler=s)
    t0 = time.perf_counter()
    st.optimize(lambda t: zdt1(np.array([t.suggest_float(f"x{i}", 0, 1) for i in range(6)])), n_trials=budget)
    secs = time.perf_counter() - t0
    return {"f": "zdt1-6", "sampler": kind, "seed": seed, "budget": budget,
            "hv": hv2([t.values for t in st.best_trials]), "ms_per_trial": secs / budget * 1e3}


# ------------------------------------------------------------ universe subset

def subset_problem(n=30, seed=0):
    r = np.random.RandomState(1000 + seed)
    w = r.randn(n) * 0.5
    w[:6] += 1.2
    J = np.triu(r.randn(n, n) * 0.15, 1)
    return w, J


def run_subset(args):
    kind, seed, budget = args
    import optuna

    from mojo_optuna.samplers import TPESampler as MojoTPE
    from mojo_optuna.search.space import Universe
    from mojo_optuna.search.subset import StepwiseSampler, SubsetEvoSampler, UniverseRepair

    optuna.logging.set_verbosity(optuna.logging.ERROR)
    n = 30
    w, J = subset_problem(n, seed)
    syms = [f"S{i:02d}" for i in range(n)]
    u = Universe.from_spec({"symbols": syms, "min": 4, "max": 10})

    def val(on):
        z = np.array([on.get("u:" + s, 0) for s in syms], float)
        return float(w @ z + z @ J @ z) / math.sqrt(max(1, z.sum()))

    # exhaustive optimum is infeasible (C(30,4..10)); use best of a long GA+forward run as reference
    base = {"random": optuna.samplers.RandomSampler(seed=seed),
            "mojo-tpe-mv": MojoTPE(seed=seed, multivariate=True, n_startup_trials=30),
            "ga": SubsetEvoSampler(seed=seed, n_startup_trials=30, universe=u),
            "forward": None}[kind]
    s = StepwiseSampler(universe=u, seed=seed) if kind == "forward" else UniverseRepair(base, u, seed=seed)
    st = optuna.create_study(direction="maximize", sampler=s)
    t0 = time.perf_counter()
    st.optimize(lambda t: val({"u:" + x: t.suggest_categorical("u:" + x, [0, 1]) for x in syms}), n_trials=budget)
    return {"f": "subset30", "sampler": kind, "seed": seed, "budget": budget, "best": st.best_value,
            "ms_per_trial": (time.perf_counter() - t0) / budget * 1e3}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--procs", type=int, default=4)
    ap.add_argument("--budget", type=int, default=200)
    ap.add_argument("--gp-budget", type=int, default=60)
    ap.add_argument("--out", default="bench/results_search.json")
    ap.add_argument("--funcs", default=",".join(FUNCS))
    ap.add_argument("--table", default="", help="only print the table for an existing results JSON")
    ap.add_argument("--samplers", default="random,optuna-tpe,optuna-tpe-mv,mojo-tpe-mv,optuna-cma,optuna-cma-ipop,mopt-cma-ipop,mopt-gp-ei")
    a = ap.parse_args()
    if a.table:
        print(table(json.load(open(a.table))))
        return
    jobs = []
    for fn in a.funcs.split(","):
        for k in a.samplers.split(","):
            b = a.gp_budget if "gp" in k else a.budget
            for s in range(a.seeds):
                jobs.append((fn, k, s, b, 1))
    for s in range(a.seeds):  # parallel workers: 4 asks before tells
        for k in ("optuna-tpe-mv", "optuna-tpe-mv-cl", "mojo-tpe-mv-cl"):
            jobs.append(("hartmann6", k, s, a.budget, 4))
    mo = [(k, s, a.budget) for k in ("random", "optuna-nsga2", "optuna-motpe", "mojo-motpe") for s in range(a.seeds)]
    sub = [(k, s, a.budget) for k in ("random", "mojo-tpe-mv", "ga", "forward") for s in range(a.seeds)]
    t0 = time.time()
    with ProcessPoolExecutor(a.procs) as ex:
        res = list(ex.map(run_one, jobs, chunksize=1))
        res_mo = list(ex.map(run_mo, mo, chunksize=1))
        res_sub = list(ex.map(run_subset, sub, chunksize=1))
    out = {"machine": platform.processor() or platform.machine(), "python": sys.version.split()[0],
           "secs": time.time() - t0, "single": res, "mo": res_mo, "subset": res_sub}
    import optuna

    out["optuna"] = optuna.__version__
    with open(a.out, "w") as fh:
        json.dump(out, fh)
    print(table(out))


def table(out) -> str:
    import collections

    L = []
    g = collections.defaultdict(list)
    for r in out["single"]:
        g[(r["f"], r["sampler"], r["batch"])].append(r)
    fs = list(dict.fromkeys(r["f"] for r in out["single"] if r["batch"] == 1))
    ks = list(dict.fromkeys(r["sampler"] for r in out["single"] if r["batch"] == 1))
    for at in ("50", "200"):
        L.append(f"median regret after {at} trials ({len({r['seed'] for r in out['single']})} seeds; lower is better)")
        L.append("")
        L.append("| sampler | " + " | ".join(fs) + " | ms/trial |")
        L.append("| --- |" + " ---: |" * (len(fs) + 1))
        for k in ks:
            cells, ms = [], []
            for f in fs:
                rs = [r for r in g.get((f, k, 1), []) if at in r["regret_at"]]
                cells.append(f"{np.median([r['regret_at'][at] for r in rs]):.3g}" if rs else "-")
                ms.extend(r["ms_per_trial"] for r in rs)
            if any(c != "-" for c in cells):
                L.append(f"| {k} | " + " | ".join(cells) + f" | {np.median(ms):.1f} |")
        L.append("")
    L.append("parallel workers (4 asks before tells, hartmann6, 200 trials): " + ", ".join(
        f"{k} {np.median([r['regret'] for r in rs]):.3g}" for (f, k, b), rs in g.items() if b == 4))
    m = collections.defaultdict(list)
    for r in out["mo"]:
        m[r["sampler"]].append(r)
    L.append("")
    L.append("ZDT1 (6-d) hypervolume after 200 trials, higher is better: " + ", ".join(
        f"{k} {np.median([r['hv'] for r in rs]):.3f} ({np.median([r['ms_per_trial'] for r in rs]):.1f} ms/trial)" for k, rs in m.items()))
    s = collections.defaultdict(list)
    for r in out["subset"]:
        s[r["sampler"]].append(r)
    L.append("")
    L.append("30-symbol universe (4-10 symbols, additive + pairwise terms), best score after 200 trials, higher is better: " + ", ".join(
        f"{k} {np.median([r['best'] for r in rs]):.3f}" for k, rs in s.items()))
    return "\n".join(L)


if __name__ == "__main__":
    main()
