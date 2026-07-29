"""ctypes bindings for the Mojo TPE kernels."""

from __future__ import annotations

import ctypes
import os
import subprocess

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
LIB = os.environ.get("MOJO_OPTUNA_LIB") or os.path.join(
    ROOT, "dist", "libmojo-optuna.so"
)
SRC = os.path.join(ROOT, "src", "kernels.mojo")

I = ctypes.c_int64

_SIGNATURES = {
    "mot_score_numeric": ([I] * 11, None),
    "mot_score_numeric_gpu": ([I] * 11, I),
    "mot_compute_normalizers": ([I] * 10, None),
    "mot_score_categorical": ([I] * 6, None),
    "mot_finish_log_pdf": ([I] * 5, None),
    "mot_best_acquisition": ([I, I, I], I),
}


class BuildError(RuntimeError):
    pass


def build(force: bool = False) -> str:
    if (
        not force
        and os.path.exists(LIB)
        and (
            os.environ.get("MOJO_OPTUNA_LIB")
            or os.path.getmtime(LIB) >= os.path.getmtime(SRC)
        )
    ):
        return LIB
    if os.environ.get("MOJO_OPTUNA_LIB"):
        raise BuildError(f"MOJO_OPTUNA_LIB does not exist: {LIB}")
    proc = subprocess.run(
        ["bash", os.path.join(ROOT, "build", "build.sh")],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=1800,
    )
    if proc.returncode != 0 or not os.path.exists(LIB):
        raise BuildError((proc.stderr or proc.stdout).strip()[:4000])
    return LIB


_LIBRARY: ctypes.CDLL | None = None


def lib() -> ctypes.CDLL:
    global _LIBRARY
    if _LIBRARY is None:
        _LIBRARY = ctypes.CDLL(build())
        for name, (argtypes, restype) in _SIGNATURES.items():
            function = getattr(_LIBRARY, name)
            function.argtypes = argtypes
            function.restype = restype
    return _LIBRARY


def f64(values) -> np.ndarray:
    return np.ascontiguousarray(values, dtype=np.float64)


def i64(values) -> np.ndarray:
    return np.ascontiguousarray(values, dtype=np.int64)


def addr(values: np.ndarray, dtype) -> int:
    if not isinstance(values, np.ndarray):
        raise TypeError("FFI buffers must be NumPy arrays")
    expected_dtype = np.dtype(dtype)
    if values.dtype != expected_dtype:
        raise TypeError(
            f"FFI buffer has dtype {values.dtype}, expected {expected_dtype}"
        )
    if values.size == 0:
        raise ValueError("zero-length buffers must not cross the Mojo FFI")
    if not values.flags.c_contiguous:
        raise ValueError("FFI buffers must be C-contiguous")
    address = int(values.ctypes.data)
    if address == 0:
        raise ValueError("FFI buffers must have a non-null address")
    return address


def best_acquisition(below: np.ndarray, above: np.ndarray) -> int:
    below = f64(below)
    above = f64(above)
    if below.ndim != 1 or below.shape != above.shape or below.size == 0:
        raise ValueError("below and above must be non-empty one-dimensional arrays")
    return int(
        lib().mot_best_acquisition(
            addr(below, np.float64), addr(above, np.float64), int(below.size)
        )
    )
