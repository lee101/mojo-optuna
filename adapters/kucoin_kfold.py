"""bitbankkucoin kfold -> mopt JSON lines (one per fold).

    python adapters/kucoin_kfold.py --repo /path/bitbankkucoin --data data --universe BTC,ETH,... -- \
        -start 2021-01-01 -model gbdt -bag 4 -vol-target 0.45 ...

Universe: the harness reads <data>/universe.json, so a shadow data dir is built
with symlinks to every input plus a filtered universe.json (entries by `base`).
Per fold emits ret, dd (fold max DD), mdd (worst calendar-month DD), rdd30
(worst rolling-30d DD), sharpe, trades; all in percent.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import subprocess
import sys
import tempfile
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _common import emit, log, max_dd, month_dd, roll_dd  # noqa: E402

FOLD = re.compile(r"^fold\s+(\d+)\s+(\S+)\.\.(\S+)\s+(.*)$")
KV = re.compile(r"(\w+)=([-+]?\d+(?:\.\d+)?)%?")


def shadow_data(src: str, universe: list[str], tmp: str) -> str:
    dst = os.path.join(tmp, "data")
    os.makedirs(dst)
    for name in os.listdir(src):
        if name != "universe.json":
            os.symlink(os.path.join(src, name), os.path.join(dst, name))
    u = json.load(open(os.path.join(src, "universe.json")))
    keep = set(universe)
    u["entries"] = [e for e in u["entries"] if e.get("base") in keep]
    missing = keep - {e["base"] for e in u["entries"]}
    if missing:
        raise SystemExit(f"unknown symbols {sorted(missing)}")
    json.dump(u, open(os.path.join(dst, "universe.json"), "w"))
    return dst


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True)
    ap.add_argument("--bin", default="bin/bitbankkucoin")
    ap.add_argument("--data", default="data")
    ap.add_argument("--universe", default="", help="comma list of bases; empty = data/universe.json as is")
    ap.add_argument("rest", nargs=argparse.REMAINDER)
    a = ap.parse_args()
    rest = [x for x in a.rest if x != "--"]
    data = a.data if os.path.isabs(a.data) else os.path.join(a.repo, a.data)
    with tempfile.TemporaryDirectory(prefix="mopt-kc-") as tmp:
        if a.universe:
            data = shadow_data(data, [s for s in a.universe.split(",") if s], tmp)
        curves = os.path.join(tmp, "curves.csv")
        cmd = [os.path.join(a.repo, a.bin), "kfold", "-data", data, "-curves-out", curves, *rest]
        log(" ".join(cmd))
        p = subprocess.run(cmd, cwd=a.repo, capture_output=True, text=True)
        if p.returncode != 0:
            log(p.stderr[-3000:])
            raise SystemExit(p.returncode)
        rows = {}
        for line in p.stdout.splitlines():
            m = FOLD.match(line.strip())
            if m:
                kv = {k: float(v) for k, v in KV.findall(m.group(4))}
                rows[int(m.group(1))] = {"fold": int(m.group(1)), "start": m.group(2), "end": m.group(3), **kv}
            elif line.startswith("KFOLD"):
                log(line)
        eq = defaultdict(lambda: ([], []))
        if os.path.exists(curves):
            with open(curves) as fh:
                for r in csv.DictReader(fh):
                    ts, e = eq[int(r["fold"])]
                    ts.append(int(r["ts"]))
                    e.append(float(r["equity"]))
        for f in sorted(rows):
            r = rows[f]
            if f in eq:
                ts, e = eq[f]
                r.update(mdd=month_dd(ts, e), rdd30=roll_dd(ts, e, 30), dd_curve=max_dd(e))
            emit(r)


if __name__ == "__main__":
    main()
