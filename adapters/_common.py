"""Helpers shared by harness adapters: drawdown metrics and JSON-line emission."""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone

import numpy as np


def max_dd(eq) -> float:
    e = np.asarray(eq, float)
    if e.size == 0:
        return 0.0
    peak = np.maximum.accumulate(e)
    return float(((peak - e) / peak).max() * 100)


def month_dd(ts, eq) -> float:
    """Worst calendar-month peak-to-trough drawdown (%), peak reset each month."""
    ts, eq = np.asarray(ts), np.asarray(eq, float)
    if ts.size == 0:
        return 0.0
    keys = [(d.year, d.month) for d in (datetime.fromtimestamp(int(t), timezone.utc) for t in ts)]
    worst, start = 0.0, 0
    for i in range(1, len(keys) + 1):
        if i == len(keys) or keys[i] != keys[start]:
            seg = eq[max(0, start - 1):i]  # include prior month close as the opening peak
            worst = max(worst, max_dd(seg))
            start = i
    return worst


def roll_dd(ts, eq, days: float = 30.0) -> float:
    """Worst drawdown from any rolling `days` window high (%)."""
    ts, eq = np.asarray(ts, float), np.asarray(eq, float)
    worst, j = 0.0, 0
    from collections import deque

    dq: deque = deque()
    for i in range(len(eq)):
        while dq and eq[dq[-1]] <= eq[i]:
            dq.pop()
        dq.append(i)
        while ts[i] - ts[dq[0]] > days * 86400:
            dq.popleft()
        worst = max(worst, (eq[dq[0]] - eq[i]) / eq[dq[0]] * 100)
    return float(worst)


def emit(row: dict) -> None:
    print(json.dumps(row), flush=True)


def log(*a) -> None:
    print(*a, file=sys.stderr, flush=True)
