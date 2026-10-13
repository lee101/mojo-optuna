"""Synthetic fold objective for smoke tests: prints one JSON line per fold.

ret depends on params + a per-symbol edge that is mostly noise (so universe
selection overfits), dd grows with |x| and universe concentration.
"""

import argparse
import json
import math
import random

SYMS = ["BTC", "ETH", "SOL", "XRP", "DOGE", "ADA", "AVAX", "LINK", "DOT", "LTC", "TRX", "BCH"]
TRUE = {s: (0.6 if s in ("BTC", "ETH") else 0.0) for s in SYMS}

ap = argparse.ArgumentParser()
ap.add_argument("--x", type=float, default=0.0)
ap.add_argument("--lr", type=float, default=0.01)
ap.add_argument("--mode", default="a")
ap.add_argument("--universe", default="BTC")
ap.add_argument("--fold", default="")
ap.add_argument("--folds", default="")
a = ap.parse_args()
uni = [s for s in a.universe.split(",") if s]
folds = [int(f) for f in (a.folds.split(",") if a.folds else [a.fold])]
for f in folds:
    rng = random.Random(f * 7919 + 13)
    sym = {s: TRUE[s] + rng.gauss(0, 2.0) for s in SYMS}
    edge = sum(sym[s] for s in uni) / max(1, len(uni))
    core = 3 - (a.x - 1.2) ** 2 - 0.5 * (math.log10(a.lr) + 2.5) ** 2 + (0.5 if a.mode == "b" else 0)
    ret = core + edge + rng.gauss(0, 1.0)
    dd = 10 + 4 * abs(a.x) + 20 / max(1, len(uni)) + abs(rng.gauss(0, 5))
    print(json.dumps({"fold": f, "ret": round(ret, 6), "dd": round(dd, 6)}), flush=True)
