"""Mojo kernels for the search engine (GP cross-covariance, CSCV). Raise if the library is unavailable."""

from __future__ import annotations

import numpy as np

from .._lib import addr, f64, i64, lib


def matern52_cross(a: np.ndarray, b: np.ndarray, ls: np.ndarray) -> np.ndarray:
    a = f64(np.asarray(a) / ls)
    bt = f64((np.asarray(b) / ls).T)
    m, d = a.shape
    n = bt.shape[1]
    out = np.empty((m, n))
    if m == 0 or n == 0:
        return out
    lib().mot_matern52_cross(addr(a, np.float64), addr(bt, np.float64), m, n, d, addr(out, np.float64))
    return out


def cscv(B: np.ndarray, masks: np.ndarray):
    B = f64(B)
    s, n = B.shape
    masks = i64(masks)
    c = masks.size
    isv, total = np.empty(n), np.empty(n)
    lam, bis, boos = np.empty(c), np.empty(c), np.empty(c)
    lib().mot_cscv(addr(B, np.float64), s, n, addr(masks, np.int64), c, addr(isv, np.float64),
                   addr(total, np.float64), addr(lam, np.float64), addr(bis, np.float64), addr(boos, np.float64))
    return lam, bis, boos
