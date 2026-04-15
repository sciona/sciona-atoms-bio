from __future__ import annotations

from sciona.ghost.abstract import AbstractArray, AbstractScalar, AbstractDistribution, AbstractSignal


def witness_apccoreevaluation(x: AbstractArray) -> AbstractArray:
    """Shape-and-type check for apc core evaluation. Returns output metadata without running the real computation."""
    result = AbstractArray(
        shape=x.shape,
        dtype="float64",
    )
    return result
