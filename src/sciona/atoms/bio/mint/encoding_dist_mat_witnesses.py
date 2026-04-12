from __future__ import annotations

from ageoa.ghost.abstract import AbstractArray, AbstractScalar, AbstractDistribution, AbstractSignal


def witness_encodedistancematrix(mat_list: AbstractArray, max_cdr3: AbstractScalar, max_epi: AbstractScalar) -> AbstractArray:
    """Shape-and-type check for encode distance matrix. Returns output metadata without running the real computation."""
    result = AbstractArray(
        shape=mat_list.shape,
        dtype="float64",
    )
    return result
