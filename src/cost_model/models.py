"""Core data structures for the small tensor cost model."""

from __future__ import annotations

from dataclasses import dataclass
from math import prod


DTYPE_BYTES: dict[str, int] = {
    "bool": 1,
    "bf16": 2,
    "f16": 2,
    "f32": 4,
    "f64": 8,
    "i8": 1,
    "i32": 4,
    "i64": 8,
    "u8": 1,
}


@dataclass(frozen=True)
class Tensor:
    """A statically shaped tensor value in the graph."""

    name: str
    shape: tuple[int, ...]
    dtype: str

    @property
    def elements(self) -> int:
        return prod(self.shape) if self.shape else 1

    @property
    def nbytes(self) -> int:
        return self.elements * DTYPE_BYTES[self.dtype]


@dataclass(frozen=True)
class Operation:
    """A single-output operation in source order."""

    name: str
    kind: str
    inputs: tuple[str, ...]
    output: str


@dataclass(frozen=True)
class Target:
    """Idealized hardware ceilings used by the roofline lower-bound model."""

    peak_compute_gflops: float
    memory_bandwidth_gbps: float


@dataclass(frozen=True)
class Graph:
    """A validated graph with shapes inferred for every value."""

    name: str
    inputs: dict[str, Tensor]
    operations: tuple[Operation, ...]
    outputs: tuple[str, ...]
    values: dict[str, Tensor]
    target: Target | None = None
