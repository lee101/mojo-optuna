"""binancego scripts/kfold_eval.py -> mopt JSON lines (one per fold).

    python adapters/binancego_kfold.py --repo /path/binancego --grid-file experiments/simhonest_20261011/grid_s4.txt \
        --pairs BTCUSDT,ETHUSDT --set edges=75 --set gross=2.0 -- --bin ./binancego_sim_honest --data D --cache C \
        --train-start 2025-06-01 --fold-start 2025-08-15 --fold-count 14 --bar-stride 4 --jobs 3 --dd 35

`--set name=value` overrides (or appends) `--name value` in the grid; exactly one
config must result. Per fold emits ret, dd, monthly_median, monthly_p10, trades.
"""

from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _common import emit, log  # noqa: E402


def merge_grid(base: str, sets: list[str]) -> str:
    toks = shlex.split(base)
    order: list[str] = []
    vals: dict[str, list[str]] = {}
    i = 0
    while i < len(toks):
        k = toks[i]
        if not k.startswith("--"):
            raise SystemExit(f"bad grid token {k!r}")
        j = i + 1
        v = []
        while j < len(toks) and not toks[j].startswith("--"):
            v.append(toks[j])
            j += 1
        order.append(k)
        vals[k] = v
        i = j
    for s in sets:
        k, v = s.split("=", 1)
        k = "--" + k.lstrip("-")
        if k not in vals:
            order.append(k)
        vals[k] = [v] if v != "" else []
    for k, v in vals.items():
        if any("," in x for x in v):
            raise SystemExit(f"{k} {v}: comma lists sweep inside kfold_eval; mopt needs a single config")
    return " ".join(shlex.join([k, *vals[k]]) for k in order)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True)
    ap.add_argument("--grid-file", required=True)
    ap.add_argument("--pairs", required=True)
    ap.add_argument("--set", action="append", default=[])
    ap.add_argument("--extra-grid", default="", help="flags appended verbatim (e.g. --warm-history)")
    ap.add_argument("rest", nargs=argparse.REMAINDER)
    a = ap.parse_args()
    rest = [x for x in a.rest if x != "--"]
    gf = a.grid_file if os.path.isabs(a.grid_file) else os.path.join(a.repo, a.grid_file)
    grid = merge_grid(open(gf).read(), a.set) + (" " + a.extra_grid if a.extra_grid else "")
    with tempfile.TemporaryDirectory(prefix="mopt-bgo-") as tmp:
        out = os.path.join(tmp, "out.json")
        cmd = [sys.executable, "scripts/kfold_eval.py", "--pairs", a.pairs, "--grid", grid, "--json-out", out,
               "--top", "5", *rest]
        log(shlex.join(cmd)[:400])
        p = subprocess.run(cmd, cwd=a.repo, capture_output=True, text=True)
        if p.returncode != 0 or not os.path.exists(out):
            log(p.stdout[-1500:], p.stderr[-3000:])
            raise SystemExit(p.returncode or 1)
        js = json.load(open(out))
        rows = js.get("top_all") or []
        if len(rows) != 1:
            raise SystemExit(f"expected exactly 1 config, got {len(rows)}")
        r = rows[0]
        folds = js.get("folds", [])
        for i, ret in enumerate(r["rets"]):
            emit({"fold": i, "ret": ret, "dd": r["dds"][i], "monthly_median": r["monthly_medians"][i],
                  "monthly_p10": r["monthly_p10s"][i], "trades": r["trades"][i],
                  "test_start": folds[i]["test_start"] if i < len(folds) else None})


if __name__ == "__main__":
    main()
