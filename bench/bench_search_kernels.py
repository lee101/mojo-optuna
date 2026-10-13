"""Mojo vs numpy for the search-engine kernels (GP cross-covariance, CSCV/PBO)."""

import itertools
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "python"))
from mojo_optuna.search import kernels, stats  # noqa: E402
from mojo_optuna.search.gp import matern52_cross_np  # noqa: E402


def best(f, n=5):
    t = []
    for _ in range(n):
        s = time.perf_counter()
        f()
        t.append(time.perf_counter() - s)
    return min(t) * 1e3


r = np.random.RandomState(0)
print("| kernel | mojo | numpy | speedup |\n| --- | ---: | ---: | ---: |")
for m, n, d in ((2560, 300, 8), (2560, 400, 30)):
    a, b, ls = r.rand(m, d), r.rand(n, d), r.rand(d) + 0.2
    assert np.abs(kernels.matern52_cross(a, b, ls) - matern52_cross_np(a, b, ls)).max() < 1e-9
    x, y = best(lambda: kernels.matern52_cross(a, b, ls)), best(lambda: matern52_cross_np(a, b, ls))
    print(f"| Matern-5/2 cross {m}x{n}x{d} (GP-EI candidates) | {x:.2f} ms | {y:.2f} ms | {y / x:.1f}x |")
for S, N in ((12, 300), (16, 1000)):
    B = r.randn(S, N)
    masks = np.array([sum(1 << i for i in c) for c in itertools.combinations(range(S), S // 2)])
    x, y = best(lambda: kernels.cscv(B, masks), 3), best(lambda: stats.cscv_np(B, masks), 1)
    print(f"| CSCV S={S} ({len(masks)} combos) x {N} trials | {x:.2f} ms | {y:.2f} ms | {y / x:.1f}x |")
