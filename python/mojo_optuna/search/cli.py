"""mopt: process-level search CLI (run / ask / tell / serve / http / report / nested)."""

from __future__ import annotations

import argparse
import json
import sys
import threading
from typing import Any

import numpy as np

from .engine import Engine, load_config
from .stats import agg, inner_folds, jaccard

EXAMPLE = {
    "name": "example",
    "storage": "studies/example.journal",
    "seed": 0,
    "command": "python3 examples/toy_objective.py --x {x} --lr {lr} --mode {mode} --universe {universe} --fold {fold}",
    "mode": "per_fold",
    "parse": "json",
    "cwd": ".",
    "timeout": 600,
    "folds": 8,
    "fold_order": "spread",
    "params": {
        "x": {"type": "float", "low": -3, "high": 3, "default": 0},
        "lr": {"type": "float", "low": 1e-4, "high": 1e-1, "log": True},
        "mode": {"type": "cat", "choices": ["a", "b"]},
    },
    "universe": {"symbols": ["BTC", "ETH", "SOL", "XRP", "DOGE", "ADA"], "always": ["BTC"], "min": 2, "max": 4},
    "objectives": [{"metric": "ret", "agg": "mean", "direction": "maximize"},
                   {"metric": "dd", "agg": "max", "direction": "minimize"}],
    "constraints": [{"metric": "dd", "agg": "max", "le": 35}],
    "select": {"metric": "ret", "agg": "mean", "direction": "maximize"},
    "sampler": {"type": "tpe", "multivariate": True, "constant_liar": True, "n_startup": 16},
    "pruner": {"type": "asha", "min_folds": 2, "eta": 2, "n_startup": 6,
               "kill": [{"metric": "dd", "agg": "max", "gt": 60}]},
    "nested": {"outer": [4, 5, 6, 7], "mode": "before", "purge": 0, "min_inner": 2},
    "baseline": {"x": 0, "lr": 0.01, "mode": "a", "universe": "all"},
}


def _set(cfg: dict[str, Any], kv: str) -> None:
    k, v = kv.split("=", 1)
    try:
        val = json.loads(v)
    except json.JSONDecodeError:
        val = v
    cur = cfg
    parts = k.split(".")
    for p in parts[:-1]:
        cur = cur.setdefault(p, {})
    cur[parts[-1]] = val


def _cfg(a) -> dict[str, Any]:
    cfg = load_config(a.config)
    for kv in a.set or []:
        _set(cfg, kv)
    return cfg


def _engine(a, **kw) -> Engine:
    return Engine(_cfg(a), study_name=getattr(a, "study", None), **kw)


def _out(obj: Any) -> None:
    print(json.dumps(obj, default=_default), flush=True)


def _default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (set, frozenset)):
        return sorted(o)
    return str(o)


def op(e: Engine, req: dict[str, Any], lock: threading.Lock) -> Any:
    o = req.get("op")
    with lock:
        if o == "ask":
            n = int(req.get("n", 1))
            got = [e.ask() for _ in range(n)]
            return got[0] if n == 1 else got
        if o == "tell":
            return e.tell(int(req["trial"]), req.get("fold"), req.get("metrics"), req.get("values"))
        if o == "fail":
            return e.fail(int(req["trial"]), req.get("error", ""))
        if o == "enqueue":
            e.enqueue(req["params"], baseline=bool(req.get("baseline")))
            return {"ok": True}
        if o == "best":
            return e.best()
        if o == "report":
            from .report import build

            return build(e, outer=req.get("outer"))
        if o == "config":
            return {k: v for k, v in e.cfg.items() if not k.startswith("_")}
    raise ValueError(f"unknown op {o!r}")


def cmd_serve(a) -> None:
    e = _engine(a)
    lock = threading.Lock()
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            _out(op(e, json.loads(line), lock))
        except Exception as ex:  # noqa: BLE001
            _out({"error": f"{type(ex).__name__}: {ex}"})


def cmd_http(a) -> None:
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

    e = _engine(a)
    lock = threading.Lock()

    class H(BaseHTTPRequestHandler):
        def _send(self, code, obj):
            body = json.dumps(obj, default=_default).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            name = self.path.strip("/") or "best"
            try:
                self._send(200, op(e, {"op": name}, lock))
            except Exception as ex:  # noqa: BLE001
                self._send(400, {"error": str(ex)})

        def do_POST(self):
            n = int(self.headers.get("Content-Length", 0))
            try:
                req = json.loads(self.rfile.read(n) or b"{}")
                req.setdefault("op", self.path.strip("/"))
                self._send(200, op(e, req, lock))
            except Exception as ex:  # noqa: BLE001
                self._send(400, {"error": f"{type(ex).__name__}: {ex}"})

        def log_message(self, *args):
            pass

    srv = ThreadingHTTPServer((a.host, a.port), H)
    print(f"mopt http {e.name} on {a.host}:{a.port}", file=sys.stderr, flush=True)
    srv.serve_forever()


def cmd_run(a) -> None:
    from .report import build, text
    from .runner import Runner

    e = _engine(a)
    Runner(e, jobs=a.jobs, threads=a.threads).run(a.n_trials, a.timeout)
    print(text(build(e)), flush=True)


def _best_trial(e: Engine):
    sel = e.select
    done = [t for t in e.trials() if e.is_feasible(t)] or e.trials()
    best, bs = None, -np.inf
    for t in done:
        fm = e.fold_metrics(t)
        try:
            v = sel.sign * sel.value([fm[f] for f in e.folds])
        except KeyError:
            continue
        if v > bs:
            best, bs = t, v
    return best, bs * sel.sign


def cmd_nested(a) -> None:
    """True nested walk-forward: a fresh search per outer fold on its inner folds only."""
    from .runner import Runner

    cfg = _cfg(a)
    ncfg = cfg.get("nested") or {}
    base = Engine(cfg, study_name=a.study)
    folds = base.folds
    outer = [int(x) for x in a.outer.split(",")] if a.outer else ncfg.get("outer") or folds[len(folds) // 2:]
    rows = []
    for k in outer:
        inn = inner_folds(folds, k, ncfg.get("mode", "before"), int(ncfg.get("purge", 0)))
        if len(inn) < int(ncfg.get("min_inner", 2)):
            continue
        ei = Engine(cfg, study_name=f"{base.name}.nested.o{k}", folds=inn)
        left = a.n_trials - len([t for t in ei.study.get_trials(deepcopy=False) if t.state.is_finished()])
        if left > 0:
            Runner(ei, jobs=a.jobs, threads=a.threads).run(left, a.timeout)
        bt, inner_score = _best_trial(ei)
        if bt is None:
            continue
        eo = Engine({k2: v for k2, v in cfg.items() if k2 != "pruner"},
                    study_name=f"{base.name}.nested.o{k}.eval", folds=[k])
        spec = dict(eo.space.describe(bt.params))
        if eo.space.universe is not None:
            spec["universe"] = eo.space.universe.members(bt.params)
        eo.enqueue(spec)
        waiting = len([t for t in eo.study.get_trials(deepcopy=False) if t.state.name == "WAITING"])
        if waiting:
            Runner(eo, jobs=min(a.jobs, waiting), threads=a.threads).run(waiting)
        row = {"outer": k, "inner": f"{inn[0]}..{inn[-1]}", "trial": bt.number, "inner_score": inner_score}
        for t in eo.trials():
            v = e_val(eo, t)
            if t.user_attrs.get("baseline"):
                row["baseline_outer"] = v
            if t.params == bt.params:
                row["outer_value"] = v
        if eo.space.universe is not None:
            row["universe"] = eo.space.universe.members(bt.params)
        rows.append(row)
        print(json.dumps(row, default=_default), file=sys.stderr, flush=True)
    ov = [r["outer_value"] for r in rows if "outer_value" in r]
    summ: dict[str, Any] = {"study": base.name, "n_outer": len(rows), "rows": rows}
    if ov:
        summ.update(outer_mean=float(np.mean(ov)), outer_min=float(np.min(ov)),
                    inner_mean=float(np.mean([r["inner_score"] for r in rows])))
        summ["optimism"] = summ["inner_mean"] - summ["outer_mean"]
        bo = [(r["outer_value"], r["baseline_outer"]) for r in rows if "outer_value" in r and "baseline_outer" in r]
        if bo:
            d = np.array([x - y for x, y in bo])
            summ.update(delta_vs_baseline=float(d.mean()), win_rate=float((d > 0).mean()),
                        delta_se=float(d.std(ddof=1) / np.sqrt(len(d))) if len(d) > 1 else None)
        us = [set(r["universe"]) for r in rows if "universe" in r]
        if len(us) > 1:
            summ["pick_jaccard"] = float(np.mean([jaccard(x, y) for i, x in enumerate(us) for y in us[i + 1:]]))
    _out(summ)


def cmd_symnull(a) -> None:
    """Random-universe null: chosen params x K random universes of the same size."""
    from .runner import Runner

    cfg = _cfg(a)
    e = Engine(cfg, study_name=a.study)
    uni = e.space.universe
    if uni is None:
        raise SystemExit("symnull needs a universe")
    if a.trial is not None:
        bt = e._trial(a.trial)
    else:
        bt, _ = _best_trial(e)
    k = sum(int(bt.params.get("u:" + s, 0)) for s in uni.free)
    quiet = {k2: v for k2, v in cfg.items() if k2 not in ("pruner", "baseline")}
    en = Engine(quiet, study_name=f"{e.name}.symnull.t{bt.number}")
    rng = np.random.RandomState(a.seed)
    base = dict(e.space.describe(bt.params))
    have = len(en.study.get_trials(deepcopy=False))
    for _ in range(max(0, a.n - have)):
        pick = rng.choice(len(uni.free), k, replace=False)
        en.study.enqueue_trial(en.to_params({**base, "universe": [uni.free[i] for i in pick]}))
    waiting = len([t for t in en.study.get_trials(deepcopy=False) if t.state.name == "WAITING"])
    if waiting:
        Runner(en, jobs=a.jobs, threads=a.threads).run(waiting)
    scores = [e_val(en, t) for t in en.trials() if not t.user_attrs.get("baseline")]
    scores = [s for s in scores if s == s]
    own = e_val(e, bt)
    out = {"trial": bt.number, "k": k, "n": len(scores), "own": own, "scores": scores,
           "null_mean": float(np.mean(scores)) if scores else None,
           "own_percentile": float(np.mean([s < own for s in scores])) if scores else None}
    e.study.set_user_attr("symnull", out)
    _out({k2: v for k2, v in out.items() if k2 != "scores"})


def e_val(e: Engine, t) -> float:
    fm = e.fold_metrics(t)
    vals = [fm[f][e.select.metric] for f in e.folds if f in fm and e.select.metric in fm[f]]
    return agg(vals, e.select.agg) if vals else float("nan")


def cmd_trials(a) -> None:
    e = _engine(a)
    for t in e.study.get_trials(deepcopy=False):
        row = {"trial": t.number, "state": t.state.name, "values": t.values, "params": e.space.describe(t.params),
               "folds": {k: v for k, v in e.fold_metrics(t).items()}}
        if e.space.universe is not None and t.params:
            row["universe"] = e.space.universe.members(t.params)
        for k in ("pruned_by", "error", "constraints", "baseline"):
            if k in t.user_attrs:
                row[k] = t.user_attrs[k]
        _out(row)


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="mopt", description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)

    def add(name, fn, help_):
        p = sub.add_parser(name, help=help_)
        p.add_argument("config")
        p.add_argument("--study", default=None, help="override study name")
        p.add_argument("--set", action="append", help="override config: dotted.key=json")
        p.set_defaults(fn=fn)
        return p

    p = add("run", cmd_run, "run N trials of the external command")
    p.add_argument("-n", "--n-trials", type=int, default=50)
    p.add_argument("-j", "--jobs", type=int, default=1)
    p.add_argument("--threads", type=int, default=None, help="per-job thread env (OMP/GOMAXPROCS/...)")
    p.add_argument("--timeout", type=float, default=None)
    p = add("nested", cmd_nested, "true nested walk-forward (search per outer fold)")
    p.add_argument("-n", "--n-trials", type=int, default=50)
    p.add_argument("-j", "--jobs", type=int, default=1)
    p.add_argument("--threads", type=int, default=None)
    p.add_argument("--timeout", type=float, default=None)
    p.add_argument("--outer", default=None)
    p = add("ask", lambda a: _out(_engine(a).ask()), "ask one trial (JSON)")
    p = add("tell", lambda a: _out(_engine(a).tell(
        a.trial, a.fold, json.loads(a.metrics) if a.metrics else None,
        [float(x) for x in a.values.split(",")] if a.values else None)), "tell fold metrics or final values")
    p.add_argument("trial", type=int)
    p.add_argument("--fold", type=int)
    p.add_argument("--metrics")
    p.add_argument("--values")
    p = add("fail", lambda a: _out(_engine(a).fail(a.trial, a.error)), "mark a trial failed")
    p.add_argument("trial", type=int)
    p.add_argument("--error", default="")
    p = add("enqueue", lambda a: (_engine(a).enqueue(json.loads(a.params), a.baseline), _out({"ok": True})),
            "enqueue params (JSON; 'universe': [..] or 'all')")
    p.add_argument("params")
    p.add_argument("--baseline", action="store_true")
    add("serve", cmd_serve, "JSON-lines ask/tell over stdin/stdout")
    p = add("http", cmd_http, "local HTTP ask/tell (POST /ask /tell /fail /enqueue, GET /best /report)")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8765)
    p = add("report", None, "best/pareto + PBO + DSR + nested + symbol-selection penalty")
    p.add_argument("--json", action="store_true")
    p.add_argument("--outer", default=None)
    p.set_defaults(fn=cmd_report)
    add("trials", cmd_trials, "dump trials as JSON lines")
    p = add("symnull", cmd_symnull, "evaluate best params on K random same-size universes (selection-luck null)")
    p.add_argument("-n", type=int, default=30)
    p.add_argument("--trial", type=int, default=None)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("-j", "--jobs", type=int, default=1)
    p.add_argument("--threads", type=int, default=None)
    pi = sub.add_parser("init", help="print an example config")
    pi.set_defaults(fn=lambda a: print(json.dumps(EXAMPLE, indent=2)))
    a = ap.parse_args(argv)
    a.fn(a)


def cmd_report(a) -> None:
    from .report import build, text

    e = _engine(a)
    rep = build(e, outer=[int(x) for x in a.outer.split(",")] if a.outer else None)
    if a.json:
        _out(rep)
    else:
        print(text(rep))


if __name__ == "__main__":
    main()
