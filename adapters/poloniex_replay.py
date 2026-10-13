"""bitbankpoloniex hourly k-fold replay -> mopt JSON lines (one per fold).

    python adapters/poloniex_replay.py --wt $BBPX_WT --bin $W/gobuild2/replay.test \
        --panel P --scores S --smooth ema:0.0178 --all ZEC,BTC,... --universe BTC,ETH,... \
        --base-config base.json --cfg MinHoldH=48 --cfg Reserve=0.4

Universe -> fixture via make_fixture.py --drop (all - universe), cached per
universe under --fixture-cache. --cfg KEY=JSON overrides the replay config.
Per fold emits ret (mean over fees), ret_<bps> per fee, dd, liq, fills, mdd
(worst calendar-month DD on the harsher fee's curve).
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import subprocess
import sys
import tempfile
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _common import emit, log, month_dd  # noqa: E402


def fixture(a, members: list[str]) -> str:
    if not a.universe:
        return a.fixture
    allsyms = [s for s in a.all.split(",") if s]
    drop = sorted(set(allsyms) - set(members))
    key = hashlib.sha1((a.panel + a.scores + a.smooth + ",".join(drop)).encode()).hexdigest()[:16]
    os.makedirs(a.fixture_cache, exist_ok=True)
    out = os.path.join(a.fixture_cache, f"fx_{key}.json")
    if not os.path.exists(out):
        tmp = out + f".{os.getpid()}.tmp"
        cmd = [sys.executable, "-I", os.path.join(a.wt, "research/hourly20261013/make_fixture.py"),
               "--panel", a.panel, "--scores", a.scores, "--out", tmp, "--smooth", a.smooth]
        if drop:
            cmd += ["--drop", *drop]
        subprocess.run(cmd, check=True, cwd=a.wt, stdout=sys.stderr)
        os.replace(tmp, out)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--wt", required=True)
    ap.add_argument("--bin", required=True)
    ap.add_argument("--fixture", default="", help="prebuilt fixture (no universe search)")
    ap.add_argument("--panel", default="")
    ap.add_argument("--scores", default="")
    ap.add_argument("--smooth", default="ema:0.0178")
    ap.add_argument("--all", default="", help="all symbols in the panel")
    ap.add_argument("--universe", default="")
    ap.add_argument("--fixture-cache", default=os.path.expanduser("~/.cache/mopt-bbp-fixtures"))
    ap.add_argument("--base-config", default="")
    ap.add_argument("--cfg", action="append", default=[])
    a = ap.parse_args()
    members = [s for s in a.universe.split(",") if s]
    fx = fixture(a, members)
    cfg = json.load(open(a.base_config)) if a.base_config else {}
    for kv in a.cfg:
        k, v = kv.split("=", 1)
        try:
            cfg[k] = json.loads(v)
        except json.JSONDecodeError:
            cfg[k] = v
    with tempfile.TemporaryDirectory(prefix="mopt-bbp-") as out:
        env = dict(os.environ, BIN=a.bin)
        p = subprocess.run(["bash", os.path.join(a.wt, "research/hourly20261013/run_replay.sh"), fx, out, json.dumps(cfg)],
                           cwd=a.wt, env=env, capture_output=True, text=True)
        res = os.path.join(out, "results.jsonl")
        if p.returncode != 0 or not os.path.exists(res):
            log(p.stderr[-3000:], open(os.path.join(out, "run.log")).read()[-2000:] if os.path.exists(os.path.join(out, "run.log")) else "")
            raise SystemExit(p.returncode or 1)
        by = defaultdict(dict)
        for line in open(res):
            r = json.loads(line)
            by[int(r["Fold"])][float(r["Fee"])] = r
        curves = defaultdict(lambda: ([], []))
        cp = os.path.join(out, "curves.csv")
        if os.path.exists(cp):
            for r in csv.DictReader(open(cp)):
                ts, e = curves[(int(r["fold"]), float(r["fee"]))]
                ts.append(int(r["ts"]))
                e.append(float(r["equity"]))
        for f in sorted(by):
            fees = sorted(by[f])
            row = {"fold": f}
            for fee in fees:
                rep = by[f][fee]["Report"]
                row[f"ret_{round(fee * 1e4)}"] = rep["ReturnPct"]
                row[f"dd_{round(fee * 1e4)}"] = rep["MaxDrawdownPct"]
            row["ret"] = sum(by[f][x]["Report"]["ReturnPct"] for x in fees) / len(fees)
            row["dd"] = max(by[f][x]["Report"]["MaxDrawdownPct"] for x in fees)
            row["liq"] = sum(by[f][x]["LiqReturnPct"] for x in fees) / len(fees)
            row["fills"] = by[f][fees[-1]]["Report"]["Fills"]
            if (f, fees[-1]) in curves:
                row["mdd"] = month_dd(*curves[(f, fees[-1])])
            emit(row)


if __name__ == "__main__":
    main()
