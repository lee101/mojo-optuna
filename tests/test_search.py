from __future__ import annotations

import itertools
import json
import math
import os
import subprocess
import sys
import threading
import urllib.request

import numpy as np
import optuna
import pytest

from mojo_optuna.search import Engine
from mojo_optuna.search import stats
from mojo_optuna.search.cmaes import CmaEsIpopSampler
from mojo_optuna.search.gp import GPEISampler, matern52_cross_np
from mojo_optuna.search.space import Universe, from_unit, to_unit
from mojo_optuna.search.subset import StepwiseSampler, SubsetEvoSampler, UniverseRepair, pareto_rank

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TOY = os.path.join(ROOT, "examples", "toy_objective.py")
SYMS = ["BTC", "ETH", "SOL", "XRP", "DOGE", "ADA", "AVAX", "LINK"]
optuna.logging.set_verbosity(optuna.logging.WARNING)


def sphere(x):
    return float(sum((v - 0.3) ** 2 for v in x))


def rastrigin(x):
    return float(10 * len(x) + sum(v * v - 10 * math.cos(2 * math.pi * v) for v in x))


# ------------------------------------------------------------------ space / stats

def test_unit_roundtrip_and_universe_repair():
    d = optuna.distributions.FloatDistribution(1e-4, 1.0, log=True)
    assert from_unit(to_unit(0.01, d), d) == pytest.approx(0.01)
    di = optuna.distributions.IntDistribution(2, 20, step=2)
    assert from_unit(to_unit(10, di), di) == 10
    u = Universe.from_spec({"symbols": SYMS, "always": ["BTC"], "never": ["ADA"], "min": 3, "max": 4})
    rng = np.random.RandomState(0)
    assert len(u.free) == 6 and u.kmin() == 2 and u.kmax() == 3
    for _ in range(50):
        m = u.repair({n: int(rng.rand() < 0.5) for n in u.params}, rng)
        k = sum(m.values())
        assert 2 <= k <= 3
        mem = u.members(m)
        assert "BTC" in mem and "ADA" not in mem
        assert 2 <= sum(u.random_mask(rng).values()) <= 3


def test_aggregations():
    v = [1.0, -2.0, 3.0, 4.0]
    assert stats.agg(v, "mean") == 1.5
    assert stats.agg(v, "min") == -2.0
    assert stats.agg(v, "q50") == pytest.approx(2.0)
    assert stats.agg(v, "sharpe") == pytest.approx(np.mean(v) / np.std(v, ddof=1))
    assert stats.agg(v, "lcb") < 1.5


def test_pbo_detects_noise_and_skill():
    rng = np.random.RandomState(0)
    noise = rng.randn(12, 200)
    assert 0.35 < stats.pbo(noise)["pbo"] < 0.65
    skill = rng.randn(12, 200) * 0.2 + np.linspace(0, 3, 200)[None, :]
    assert stats.pbo(skill)["pbo"] < 0.05


def test_cscv_mojo_matches_numpy():
    kernels = pytest.importorskip("mojo_optuna.search.kernels")
    rng = np.random.RandomState(1)
    B = rng.randn(10, 37)
    masks = np.array([sum(1 << i for i in c) for c in itertools.combinations(range(10), 5)])
    a = kernels.cscv(B, masks)
    b = stats.cscv_np(B, masks)
    for x, y in zip(a, b):
        np.testing.assert_allclose(x, y, atol=1e-8)


def test_matern_mojo_matches_numpy():
    kernels = pytest.importorskip("mojo_optuna.search.kernels")
    rng = np.random.RandomState(2)
    a, b, ls = rng.rand(37, 5), rng.rand(11, 5), rng.rand(5) + 0.1
    np.testing.assert_allclose(kernels.matern52_cross(a, b, ls), matern52_cross_np(a, b, ls), atol=1e-10)


def test_deflated_sharpe_penalises_many_trials():
    rng = np.random.RandomState(3)
    r = rng.randn(60) * 0.02 + 0.006
    srs = rng.randn(500) * 0.15
    few = stats.deflated_sharpe(r, srs[:5], n_trials=5)
    many = stats.deflated_sharpe(r, srs, n_trials=500)
    assert many["sr0"] > few["sr0"]
    assert many["dsr"] < few["dsr"]
    assert 0 <= many["dsr"] <= 1


def test_nested_select_measures_optimism():
    rng = np.random.RandomState(4)
    M = rng.randn(10, 300)  # pure noise: in-sample best is pure luck
    out = stats.nested_select(M, list(range(10)), [5, 6, 7, 8, 9], lambda r: r.mean(0), baseline=0)
    assert out["n_outer"] == 5
    assert out["optimism"] > 0.3
    assert 0.2 < out["mean_outer_percentile"] < 0.8


def test_pareto_rank():
    F = np.array([[1, 1], [2, 2], [0, 3], [3, 0], [2, 2.5]])
    assert pareto_rank(F).tolist() == [0, 1, 0, 0, 2]


# ------------------------------------------------------------------ samplers

def _opt(sampler, f, d, n, lo=-5.0, hi=5.0):
    study = optuna.create_study(sampler=sampler)
    study.optimize(lambda t: f([t.suggest_float(f"x{i}", lo, hi) for i in range(d)]), n_trials=n)
    return study


def test_cmaes_ipop_converges_and_restarts():
    s = _opt(CmaEsIpopSampler(seed=0), sphere, 4, 300)
    assert s.best_value < 1e-3
    s2 = _opt(CmaEsIpopSampler(seed=1, popsize=6, tolfun=1e-4), rastrigin, 2, 600)
    restarts = {t.system_attrs.get("cma:restart") for t in s2.trials}
    assert max(r for r in restarts if r is not None) >= 1
    assert s2.best_value < 2.0


def test_cmaes_is_stateless_across_processes():
    st = optuna.storages.InMemoryStorage()
    a = optuna.create_study(storage=st, study_name="s", sampler=CmaEsIpopSampler(seed=0))
    for _ in range(60):
        b = optuna.load_study(storage=st, study_name="s", sampler=CmaEsIpopSampler(seed=len(a.trials)))
        t = b.ask()
        b.tell(t, sphere([t.suggest_float(f"x{i}", -5, 5) for i in range(3)]))
    names = ["x0", "x1", "x2"]
    s1, _ = CmaEsIpopSampler(seed=5)._replay(a, names)
    s2, _ = CmaEsIpopSampler(seed=9)._replay(a, names)
    assert s1.gen >= 7 and s1.gen == s2.gen
    np.testing.assert_allclose(s1.m, s2.m)
    assert s1.sigma == s2.sigma < 0.25
    assert a.best_value < 2.0


def test_gp_beats_random_on_branin():
    def branin(x):
        a, b, c, r, s, t = 1, 5.1 / (4 * math.pi ** 2), 5 / math.pi, 6, 10, 1 / (8 * math.pi)
        x1, x2 = x[0] * 7.5 + 2.5, x[1] * 7.5 + 7.5
        return a * (x2 - b * x1 * x1 + c * x1 - r) ** 2 + s * (1 - t) * math.cos(x1) + s

    gp = [_opt(GPEISampler(seed=k, n_startup_trials=8), branin, 2, 30, -1, 1).best_value for k in range(2)]
    rnd = [_opt(optuna.samplers.RandomSampler(seed=k), branin, 2, 30, -1, 1).best_value for k in range(2)]
    assert np.mean(gp) < 0.6 and np.mean(gp) < np.mean(rnd)


def _subset_obj(u, good):
    def f(t):
        on = [s for s in u.free if t.suggest_categorical("u:" + s, [0, 1])]
        return sum(1.0 if s in good else -0.4 for s in on)
    return f


def test_forward_selection_finds_additive_optimum():
    u = Universe.from_spec({"symbols": SYMS, "min": 0})
    good = {"ETH", "SOL", "LINK"}
    s = optuna.create_study(direction="maximize", sampler=StepwiseSampler(universe=u, seed=0))
    s.optimize(_subset_obj(u, good), n_trials=40)
    best = {k[2:] for k, v in s.best_params.items() if v}
    assert best == good


def test_backward_and_ga_subset():
    good = {"ETH", "SOL", "LINK", "BTC"}
    ub = Universe.from_spec({"symbols": SYMS, "min": 1})
    s = optuna.create_study(direction="maximize", sampler=StepwiseSampler(universe=ub, direction="backward", seed=0))
    s.optimize(_subset_obj(ub, good), n_trials=60)
    assert s.best_value == pytest.approx(4.0)
    u = Universe.from_spec({"symbols": SYMS, "min": 1, "max": 5})
    ga = UniverseRepair(SubsetEvoSampler(seed=0, n_startup_trials=10, universe=u), u, seed=0)
    s2 = optuna.create_study(direction="maximize", sampler=ga)
    s2.optimize(_subset_obj(u, good), n_trials=80)
    assert s2.best_value >= 3.0
    assert all(1 <= sum(t.params.values()) <= 5 for t in s2.trials)


# ------------------------------------------------------------------ engine

def _cfg(tmp_path, **kw):
    cfg = {
        "name": "t", "storage": str(tmp_path / "s.journal"), "seed": 0, "folds": 6, "_dir": str(tmp_path),
        "params": {"x": {"type": "float", "low": -3, "high": 3}},
        "objectives": [{"metric": "ret", "agg": "mean", "direction": "maximize"},
                       {"metric": "dd", "agg": "max", "direction": "minimize"}],
        "constraints": [{"metric": "dd", "agg": "max", "le": 30}],
    }
    cfg.update(kw)
    return cfg


def _metrics(x, f):
    return {"ret": 2 - (x - 1) ** 2 + 0.1 * f, "dd": 10 + 5 * abs(x)}


def test_engine_folds_constraints_pareto(tmp_path):
    e = Engine(_cfg(tmp_path, sampler={"type": "nsga2", "population": 8}))
    for _ in range(20):
        p = e.ask()
        for f in p["folds"]:
            r = e.tell(p["trial"], f, _metrics(p["params"]["x"], f))
        assert r["complete"] and r["state"] == "COMPLETE"
        assert len(r["constraints"]) == 1
    best = e.best()
    assert best and all(b["values"]["dd.max"] <= 30 for b in best)
    assert e.study.user_attrs["fold_order"][0] == 5


@pytest.mark.parametrize("kind", ["asha", "hyperband", "median"])
def test_engine_fold_pruners_and_kill_rule(tmp_path, kind):
    cfg = _cfg(tmp_path, objectives=[{"metric": "ret"}], constraints=[],
               pruner={"type": kind, "min_folds": 1, "eta": 2, "n_startup": 3,
                       "kill": [{"metric": "dd", "agg": "max", "gt": 24}]},
               sampler={"type": "random"})
    e = Engine(cfg)
    states = []
    for _ in range(30):
        p = e.ask()
        for f in p["folds"]:
            r = e.tell(p["trial"], f, _metrics(p["params"]["x"], f))
            if r["complete"]:
                break
        states.append(r.get("reason", r["state"]))
    assert any(str(s).startswith(kind) for s in states)
    assert any(str(s).startswith("kill") for s in states)
    assert any(s == "COMPLETE" for s in states)


@pytest.mark.parametrize("kind", ["tpe", "gp", "cmaes", "ga", "qmc"])
def test_engine_samplers_with_universe(tmp_path, kind):
    cfg = _cfg(tmp_path, objectives=[{"metric": "ret"}], constraints=[], sampler={"type": kind, "n_startup": 5},
               universe={"symbols": SYMS, "always": ["BTC"], "min": 2, "max": 4},
               baseline={"x": 0.0, "universe": "all"})
    e = Engine(cfg)
    for _ in range(14):
        p = e.ask()
        assert "BTC" in p["universe"]
        if p["trial"] > 0:
            assert 2 <= len(p["universe"]) <= 4
        for f in p["folds"]:
            m = _metrics(p["params"]["x"], f)
            m["ret"] += 0.3 * ("ETH" in p["universe"])
            e.tell(p["trial"], f, m)
    assert e.trials()[0].user_attrs.get("baseline")
    from mojo_optuna.search.report import build, text

    rep = build(e)
    assert "pbo" in rep["overfit"] and "symbol_selection" in rep["overfit"]
    assert "PBO" in text(rep)


# ------------------------------------------------------------------ CLI / runner

def _toy_cfg(tmp_path, **kw):
    cfg = {
        "name": "toy", "storage": "s.journal", "seed": 1, "cwd": ROOT, "timeout": 60,
        "command": f"{sys.executable} {TOY} --x {{x}} --lr {{lr}} --universe {{universe}} --fold {{fold}}",
        "folds": 4,
        "params": {"x": {"type": "float", "low": -3, "high": 3}, "lr": {"type": "float", "low": 1e-4, "high": 0.1, "log": True}},
        "universe": {"symbols": SYMS, "always": ["BTC"], "min": 2, "max": 4},
        "objectives": [{"metric": "ret"}, {"metric": "dd", "agg": "max", "direction": "minimize"}],
        "select": {"metric": "ret"},
        "sampler": {"type": "tpe", "n_startup": 4},
        "baseline": {"x": 0, "lr": 0.01, "universe": "all"},
    }
    cfg.update(kw)
    p = tmp_path / "cfg.json"
    p.write_text(json.dumps(cfg))
    return str(p)


def _mopt(*args, **kw):
    env = dict(os.environ, PYTHONPATH=os.path.join(ROOT, "python"))
    return subprocess.run([sys.executable, "-m", "mojo_optuna.search.cli", *args], capture_output=True,
                          text=True, env=env, timeout=600, **kw)


def test_cli_run_per_fold_and_report(tmp_path):
    cfg = _toy_cfg(tmp_path)
    r = _mopt("run", cfg, "-n", "10", "-j", "2")
    assert r.returncode == 0, r.stderr
    assert "PBO" in r.stdout and "nested(post-hoc)" in r.stdout
    rep = json.loads(_mopt("report", cfg, "--json").stdout)
    assert rep["states"].get("COMPLETE", 0) == 10
    rows = [json.loads(x) for x in _mopt("trials", cfg).stdout.splitlines()]
    assert len(rows[1]["folds"]) == 4


def test_cli_stream_mode_prunes_by_killing(tmp_path):
    cmd = (f"{sys.executable} {TOY} --x {{x}} --lr {{lr}} --universe {{universe}} --folds {{folds}}")
    cfg = _toy_cfg(tmp_path, command=cmd, mode="stream", folds=6, objectives=[{"metric": "ret"}],
                   pruner={"type": "median", "min_folds": 1, "n_startup": 2}, sampler={"type": "random"})
    r = _mopt("run", cfg, "-n", "12")
    assert r.returncode == 0, r.stderr
    rep = json.loads(_mopt("report", cfg, "--json").stdout)
    assert rep["states"].get("PRUNED", 0) >= 1 and rep["states"].get("COMPLETE", 0) >= 1


def test_cli_ask_tell_and_jsonl_serve(tmp_path):
    cfg = _toy_cfg(tmp_path)
    a = json.loads(_mopt("ask", cfg).stdout)
    assert a["trial"] == 0 and a["params"]["x"] == 0
    for f in a["folds"]:
        t = json.loads(_mopt("tell", cfg, str(a["trial"]), "--fold", str(f),
                             "--metrics", json.dumps({"ret": 1.0, "dd": 5})).stdout)
    assert t["state"] == "COMPLETE"
    reqs = [{"op": "ask"}, {"op": "tell", "trial": 1, "values": [0.5, 3.0]}, {"op": "best"}, {"op": "nope"}]
    r = _mopt("serve", cfg, input="\n".join(json.dumps(x) for x in reqs) + "\n")
    out = [json.loads(x) for x in r.stdout.splitlines()]
    assert out[0]["trial"] == 1 and out[1]["state"] == "COMPLETE"
    assert len(out[2]) >= 1 and "error" in out[3]


def test_http_server(tmp_path):
    from mojo_optuna.search import cli

    cfg = _toy_cfg(tmp_path)
    port = 18000 + os.getpid() % 1000
    th = threading.Thread(target=cli.main, args=(["http", cfg, "--port", str(port)],), daemon=True)
    th.start()

    def post(path, body):
        for _ in range(100):
            try:
                req = urllib.request.Request(f"http://127.0.0.1:{port}/{path}", json.dumps(body).encode(),
                                             {"Content-Type": "application/json"})
                return json.loads(urllib.request.urlopen(req, timeout=10).read())
            except (ConnectionRefusedError, urllib.error.URLError):
                import time

                time.sleep(0.1)
        raise RuntimeError("server did not start")

    a = post("ask", {})
    r = None
    for f in a["folds"]:
        r = post("tell", {"trial": a["trial"], "fold": f, "metrics": {"ret": 2.0, "dd": 4.0}})
    assert r["state"] == "COMPLETE"
    assert post("best", {})[0]["trial"] == a["trial"]


def test_cli_nested_and_symnull(tmp_path):
    cfg = _toy_cfg(tmp_path, folds=5, nested={"outer": [3, 4], "mode": "before", "min_inner": 2},
                   sampler={"type": "random"})
    r = _mopt("nested", cfg, "-n", "5")
    assert r.returncode == 0, r.stderr
    out = json.loads(r.stdout.splitlines()[-1])
    assert out["n_outer"] == 2 and "optimism" in out and "delta_vs_baseline" in out
    assert _mopt("run", cfg, "-n", "6").returncode == 0
    s = _mopt("symnull", cfg, "-n", "4")
    assert s.returncode == 0, s.stderr
    assert json.loads(s.stdout)["n"] == 4
    assert "null(n=4)" in _mopt("report", cfg).stdout
