"""predictionmojo vmidx (VM-SPY vol-target) walk-forward -> mopt JSON lines, one fold per calendar year.

    python adapters/vmidx_wf.py --pm-root $PM_ROOT --tv 0.17 --cap 1.75 --q 0 --trend 1 \
        --years 2005-2025 --bps 5

Uses research/r13/idx.py (data load + simulate) and idx2.arm. Per year emits
ret, spy, excess (all %), dd (max DD within the year), mdd (worst calendar-month
DD), rdd21 (worst rolling-21-trading-day DD).
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _common import emit, max_dd  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pm-root", required=True)
    ap.add_argument("--tv", type=float, default=0.15)
    ap.add_argument("--cap", type=float, default=1.75)
    ap.add_argument("--q", type=float, default=0.0)
    ap.add_argument("--trend", type=int, default=1)
    ap.add_argument("--bps", type=float, default=5.0)
    ap.add_argument("--spread", type=float, default=0.025)
    ap.add_argument("--years", default="2005-2025")
    ap.add_argument("--folds", default="", help="comma list of years (overrides --years)")
    a = ap.parse_args()
    sys.path.insert(0, os.path.join(a.pm_root, "research", "r13"))
    import idx as I  # noqa: PLC0415
    import idx2  # noqa: PLC0415

    I.SPREAD = a.spread
    C, IRX = I.C, I.IRX
    r = I.simulate(C, idx2.arm(C, a.tv, a.cap, a.q, bool(a.trend)), IRX, a.bps)
    spy = I.simulate(C, I.arm_bh(C), IRX, a.bps)
    if a.folds:
        years = [int(y) for y in a.folds.split(",")]
    else:
        y0, y1 = (int(x) for x in a.years.split("-"))
        years = list(range(y0, y1 + 1))
    for y in years:
        m = r.index.year == y
        rr, ss = r[m].to_numpy(), spy[m].to_numpy()
        if rr.size == 0:
            continue
        eq = np.cumprod(1 + rr)
        mon = r[m].index.month
        mdd = max(max_dd(np.concatenate([[1.0], np.cumprod(1 + rr[mon == k])])) for k in np.unique(mon))
        roll = 0.0
        for i in range(len(eq)):
            lo = max(0, i - 20)
            pk = max(eq[lo:i + 1].max(), eq[lo - 1] if lo > 0 else 1.0)
            roll = max(roll, (pk - eq[i]) / pk * 100)
        ret = (eq[-1] - 1) * 100
        sret = (np.prod(1 + ss) - 1) * 100
        emit({"fold": y, "ret": ret, "spy": sret, "excess": ret - sret, "dd": max_dd(np.concatenate([[1.0], eq])),
              "mdd": mdd, "rdd21": roll, "days": int(rr.size)})


if __name__ == "__main__":
    main()
