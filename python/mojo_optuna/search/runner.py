"""Run trials as external commands (one per fold, or one streaming all folds)."""

from __future__ import annotations

import json
import os
import re
import shlex
import signal
import subprocess
import sys
import threading
import time
from typing import Any, Callable

from .engine import Engine, resolve

THREAD_ENV = ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS",
              "GOMAXPROCS", "RAYON_NUM_THREADS", "POLARS_MAX_THREADS", "NUMBA_NUM_THREADS")
KV = re.compile(r"([A-Za-z_][\w.\-]*)=([-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?)%?")


PLACEHOLDER = re.compile(r"(?<!\$)\{([A-Za-z_][\w\-]*)\}")


class _Fmt(dict):
    def __missing__(self, k):
        raise KeyError(f"unknown placeholder {{{k}}} in command template")


def clean_env(env: dict) -> dict:
    """Drop mopt's own pixi/conda python from the child env so `python3` means the harness's python."""
    pre = sys.prefix
    if ".pixi" in pre or env.get("CONDA_PREFIX") == pre:
        env["PATH"] = os.pathsep.join(p for p in env.get("PATH", "").split(os.pathsep)
                                      if not p.startswith(pre))
        for k in [k for k in env if k.startswith("CONDA_")]:
            env.pop(k)
        here = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        pp = [p for p in env.get("PYTHONPATH", "").split(os.pathsep)
              if p and p != "python" and os.path.abspath(p) != here]
        if pp:
            env["PYTHONPATH"] = os.pathsep.join(pp)
        else:
            env.pop("PYTHONPATH", None)
    return env


def render(tpl: str, ctx: dict) -> str:
    """Substitute {name} placeholders; shell ${VAR} / ${VAR:-x} pass through untouched."""
    return PLACEHOLDER.sub(lambda m: str(ctx[m.group(1)]), tpl)


def parse_line(line: str, spec: Any) -> dict[str, Any] | None:
    line = line.strip()
    if not line:
        return None
    kind = spec.get("type", "json") if isinstance(spec, dict) else (spec or "json")
    if kind == "json":
        if not line.startswith("{"):
            return None
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            return None
        return obj if isinstance(obj, dict) else None
    if kind == "kv":
        m = re.search(spec.get("line", "."), line)
        if not m:
            return None
        out: dict[str, Any] = {k: float(v) for k, v in KV.findall(line)}
        g = spec.get("fold_group")
        if g is not None:
            out["fold"] = int(m.group(g))
        return out
    raise ValueError(f"unknown parse type {kind!r}")


def _mean_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if len(rows) == 1:
        return dict(rows[0])
    out: dict[str, Any] = {}
    keys = {k for r in rows for k in r}
    for k in keys:
        vals = [r[k] for r in rows if k in r]
        if all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in vals):
            out[k] = sum(vals) / len(vals)
            if k not in ("fold", "seed"):
                out[k + "@min"] = min(vals)
                out[k + "@max"] = max(vals)
        else:
            out[k] = vals[-1]
    out["n_seeds"] = len(rows)
    return out


class Runner:
    def __init__(self, engine: Engine, *, jobs: int = 1, threads: int | None = None,
                 log: Callable[[str], None] | None = None):
        self.e = engine
        self.cfg = engine.cfg
        self.jobs = max(1, jobs)
        self.threads = threads
        self.log = log or (lambda s: print(s, file=sys.stderr, flush=True))
        self.workdir = resolve(self.cfg, self.cfg.get("workdir", f"runs/{engine.name}"))
        os.makedirs(self.workdir, exist_ok=True)
        self.parse = self.cfg.get("parse", "json")
        self._lock = threading.Lock()
        self._stop = False

    # ------------------------------------------------------------- rendering
    def context(self, p: dict[str, Any], fold: Any = "", seed: Any = "") -> tuple[dict, dict]:
        cfg = self.cfg
        uni = self.e.space.universe
        members = p.get("universe", [])
        joiner = uni.join if uni else ","
        params = dict(p["params"])
        argfmt = cfg.get("arg_format", "--{name}={value}")
        specs = self.e.space.specs
        args = " ".join(argfmt.format(name=k, value=shlex.quote(str(v))) for k, v in params.items()
                        if specs.get(k, {}).get("arg", True))
        pj = os.path.join(self.workdir, f"t{p['trial']}.json")
        with open(pj, "w") as fh:
            json.dump({"trial": p["trial"], "params": params, "universe": members, "folds": p["folds"],
                       "seeds": p["seeds"], "fold": fold, "seed": seed}, fh)
        ctx = _Fmt(params)
        ctx.update(trial=p["trial"], fold=fold, seed="" if seed is None else seed,
                   folds=",".join(map(str, p["folds"])),
                   seeds=",".join(str(s) for s in p["seeds"] if s is not None),
                   universe=joiner.join(members), universe_json=shlex.quote(json.dumps(members)),
                   params_json=pj, args=args, study=self.e.name, workdir=self.workdir,
                   n_universe=len(members))
        env = clean_env(dict(os.environ)) if not cfg.get("inherit_python_env") else dict(os.environ)
        env.update({k: str(v) for k, v in (cfg.get("env") or {}).items()})
        env.update(MOPT_TRIAL=str(p["trial"]), MOPT_FOLD=str(fold), MOPT_SEED="" if seed is None else str(seed),
                   MOPT_PARAMS=json.dumps(params), MOPT_UNIVERSE=joiner.join(members), MOPT_JSON=pj,
                   MOPT_FOLDS=ctx["folds"], MOPT_STUDY=self.e.name)
        th = self.threads or cfg.get("threads_per_job")
        if th:
            env.update({k: str(th) for k in THREAD_ENV})
        return ctx, env

    def _popen(self, cmd: str, env: dict, logpath: str):
        cwd = resolve(self.cfg, self.cfg.get("cwd", "."))
        return subprocess.Popen(["bash", "-c", cmd], cwd=cwd, env=env, stdout=subprocess.PIPE,
                                stderr=open(logpath, "w"), text=True, bufsize=1, start_new_session=True)

    @staticmethod
    def _kill(proc) -> None:
        try:
            os.killpg(proc.pid, signal.SIGTERM)
            proc.wait(5)
        except Exception:
            try:
                os.killpg(proc.pid, signal.SIGKILL)
            except Exception:
                pass

    def _stream(self, cmd: str, env: dict, logpath: str, on_row: Callable[[dict], bool]) -> tuple[int, str]:
        """Run cmd; feed parsed rows to on_row; on_row returning True kills the process."""
        timeout = float(self.cfg.get("timeout", 0) or 0)
        proc = self._popen(cmd, env, logpath)
        timer = None
        if timeout:
            timer = threading.Timer(timeout, self._kill, [proc])
            timer.start()
        tail: list[str] = []
        stopped = False
        try:
            for line in proc.stdout:
                tail = (tail + [line])[-20:]
                row = parse_line(line, self.parse)
                if row is not None and on_row(row):
                    stopped = True
                    self._kill(proc)
                    break
            proc.wait()
        finally:
            if timer:
                timer.cancel()
        rc = 0 if stopped else proc.returncode
        try:
            err = open(logpath).read()[-1500:]
        except OSError:
            err = ""
        return rc, (err + "".join(tail)[-500:]).strip()

    # ------------------------------------------------------------- one trial
    def run_trial(self, p: dict[str, Any]) -> dict[str, Any]:
        mode = self.cfg.get("mode", "per_fold")
        tpl = self.cfg["command"]
        n = p["trial"]
        t0 = time.time()
        res: dict[str, Any] = {"trial": n, "state": "RUNNING"}
        try:
            if mode == "per_fold":
                for fold in p["folds"]:
                    rows = []
                    for seed in p["seeds"]:
                        ctx, env = self.context(p, fold, seed)
                        got: list[dict] = []
                        rc, err = self._stream(render(tpl, ctx), env,
                                               os.path.join(self.workdir, f"t{n}_f{fold}_s{seed}.err"),
                                               lambda r: got.append(r) or False)
                        mine = [r for r in got if r.get("fold", fold) in (fold, str(fold))]
                        if rc != 0 or not mine:
                            return self.e.fail(n, f"rc={rc} fold={fold} seed={seed}: {err}")
                        merged: dict[str, Any] = {}
                        for r in mine:
                            merged.update(r)
                        rows.append(merged)
                    res = self.e.tell(n, fold, _mean_metrics(rows))
                    if res["complete"]:
                        break
            elif mode == "stream":
                ctx, env = self.context(p)
                want = len([s for s in p["seeds"]]) if self.cfg.get("seed_rows") else 1
                buf: dict[Any, list] = {}
                state = {"res": res}

                def on_row(r):
                    if "fold" not in r:
                        return False
                    f = int(r["fold"])
                    buf.setdefault(f, []).append(r)
                    if len(buf[f]) >= want:
                        state["res"] = self.e.tell(n, f, _mean_metrics(buf[f]))
                        return state["res"]["complete"]
                    return False

                rc, err = self._stream(render(tpl, ctx), env, os.path.join(self.workdir, f"t{n}.err"), on_row)
                res = state["res"]
                if not res.get("complete"):
                    return self.e.fail(n, f"rc={rc} incomplete folds: {err}")
            else:
                raise ValueError(f"unknown mode {mode!r}")
        except Exception as ex:  # noqa: BLE001
            return self.e.fail(n, f"{type(ex).__name__}: {ex}")
        res["secs"] = round(time.time() - t0, 2)
        return res

    # ------------------------------------------------------------- loop
    def run(self, n_trials: int, timeout: float | None = None) -> list[dict[str, Any]]:
        done: list[dict[str, Any]] = []
        budget = {"left": n_trials}
        t_end = time.time() + timeout if timeout else None

        def worker():
            while True:
                with self._lock:
                    if budget["left"] <= 0 or self._stop or (t_end and time.time() > t_end):
                        return
                    budget["left"] -= 1
                    p = self.e.ask()
                r = self.run_trial(p)
                with self._lock:
                    done.append(r)
                    self.log(self.fmt(p, r))

        ths = [threading.Thread(target=worker, daemon=True) for _ in range(self.jobs)]
        for t in ths:
            t.start()
        try:
            for t in ths:
                while t.is_alive():
                    t.join(0.5)
        except KeyboardInterrupt:
            self._stop = True
            raise
        return done

    def fmt(self, p: dict[str, Any], r: dict[str, Any]) -> str:
        vals = r.get("values")
        v = " ".join(f"{o.name}={x:.4g}" for o, x in zip(self.e.objectives, vals)) if vals else ""
        extra = r.get("reason") or (r.get("error", "")[:120].replace("\n", " ") if r.get("error") else "")
        u = f" |U|={len(p['universe'])}" if "universe" in p else ""
        return f"t{p['trial']} {r.get('state')} {v}{u} {extra} {r.get('secs', '')}s".strip()
