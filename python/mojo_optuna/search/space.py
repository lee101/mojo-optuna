"""Search-space config: numeric/categorical params plus a symbol universe."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from optuna.distributions import (
    BaseDistribution,
    CategoricalDistribution,
    FloatDistribution,
    IntDistribution,
)

UPREFIX = "u:"


def distribution(spec: dict[str, Any]) -> BaseDistribution | None:
    kind = spec.get("type", "float")
    if kind in ("fixed", "const"):
        return None
    if kind == "float":
        return FloatDistribution(
            float(spec["low"]), float(spec["high"]), log=bool(spec.get("log", False)),
            step=spec.get("step"),
        )
    if kind == "int":
        return IntDistribution(
            int(spec["low"]), int(spec["high"]), log=bool(spec.get("log", False)),
            step=int(spec.get("step", 1)),
        )
    if kind in ("cat", "categorical"):
        return CategoricalDistribution(list(spec["choices"]))
    raise ValueError(f"unknown param type {kind!r}")


@dataclass
class Universe:
    symbols: list[str]
    always: list[str] = field(default_factory=list)
    never: list[str] = field(default_factory=list)
    min: int = 1
    max: int | None = None
    join: str = ","

    @classmethod
    def from_spec(cls, spec: dict[str, Any] | None) -> "Universe | None":
        if not spec:
            return None
        u = cls(
            symbols=list(spec["symbols"]),
            always=list(spec.get("always", [])),
            never=list(spec.get("never", [])),
            min=int(spec.get("min", 1)),
            max=spec.get("max"),
            join=spec.get("join", ","),
        )
        if u.max is None:
            u.max = len(u.free) + len(u.always)
        u.max = int(u.max)
        if u.min > u.max or u.max < len(u.always):
            raise ValueError("universe cardinality bounds are infeasible")
        return u

    @property
    def free(self) -> list[str]:
        fixed = set(self.always) | set(self.never)
        return [s for s in self.symbols if s not in fixed]

    @property
    def params(self) -> list[str]:
        return [UPREFIX + s for s in self.free]

    def kmin(self) -> int:
        return max(0, self.min - len(self.always))

    def kmax(self) -> int:
        return min(len(self.free), self.max - len(self.always))

    def members(self, params: dict[str, Any]) -> list[str]:
        on = {s for s in self.free if int(params.get(UPREFIX + s, 0))}
        return [s for s in self.symbols if s in on or s in self.always]

    def random_mask(self, rng: np.random.RandomState) -> dict[str, int]:
        free = self.free
        k = int(rng.randint(self.kmin(), self.kmax() + 1))
        on = set(rng.choice(len(free), size=k, replace=False).tolist()) if k else set()
        return {UPREFIX + s: int(i in on) for i, s in enumerate(free)}

    def repair(self, mask: dict[str, int], rng: np.random.RandomState,
               score: dict[str, float] | None = None) -> dict[str, int]:
        mask = dict(mask)
        names = self.params
        on = [n for n in names if int(mask.get(n, 0))]
        off = [n for n in names if not int(mask.get(n, 0))]
        lo, hi = self.kmin(), self.kmax()

        def order(xs, reverse):
            if score:
                return sorted(xs, key=lambda n: score.get(n, 0.0) + 1e-9 * rng.rand(), reverse=reverse)
            xs = list(xs)
            rng.shuffle(xs)
            return xs

        if len(on) > hi:
            for n in order(on, False)[: len(on) - hi]:
                mask[n] = 0
        elif len(on) < lo:
            for n in order(off, True)[: lo - len(on)]:
                mask[n] = 1
        for n in names:
            mask[n] = int(mask.get(n, 0))
        return mask


@dataclass
class Space:
    specs: dict[str, dict[str, Any]]
    universe: Universe | None

    @classmethod
    def from_config(cls, cfg: dict[str, Any]) -> "Space":
        return cls(dict(cfg.get("params", {})), Universe.from_spec(cfg.get("universe")))

    def distributions(self) -> dict[str, BaseDistribution]:
        out: dict[str, BaseDistribution] = {}
        for name, spec in self.specs.items():
            d = distribution(spec)
            if d is not None:
                out[name] = d
        if self.universe:
            for n in self.universe.params:
                out[n] = CategoricalDistribution([0, 1])
        return out

    def numeric(self) -> dict[str, BaseDistribution]:
        return {k: v for k, v in self.distributions().items() if not k.startswith(UPREFIX)}

    def fixed(self) -> dict[str, Any]:
        return {k: s["value"] for k, s in self.specs.items() if s.get("type") in ("fixed", "const")}

    def defaults(self) -> dict[str, Any]:
        out = {}
        for k, s in self.specs.items():
            if "default" in s:
                out[k] = s["default"]
        return out

    def describe(self, params: dict[str, Any]) -> dict[str, Any]:
        out = {k: v for k, v in params.items() if not k.startswith(UPREFIX)}
        out.update(self.fixed())
        return out


def to_unit(value: Any, d: BaseDistribution) -> float:
    if isinstance(d, (FloatDistribution, IntDistribution)):
        lo, hi, v = float(d.low), float(d.high), float(value)
        if d.log:
            lo, hi, v = math.log(lo), math.log(hi), math.log(v)
        return 0.0 if hi == lo else (v - lo) / (hi - lo)
    raise TypeError("categorical has no unit transform")


def from_unit(u: float, d: BaseDistribution) -> Any:
    u = min(1.0, max(0.0, float(u)))
    lo, hi = float(d.low), float(d.high)
    if d.log:
        v = math.exp(math.log(lo) + u * (math.log(hi) - math.log(lo)))
    else:
        v = lo + u * (hi - lo)
    if isinstance(d, IntDistribution):
        step = d.step
        v = int(round((v - d.low) / step)) * step + d.low
        return int(min(d.high, max(d.low, v)))
    if d.step is not None:
        v = round((v - lo) / d.step) * d.step + lo
    return float(min(hi, max(lo, v)))
