from __future__ import annotations

from sciona.ghost.abstract import AbstractArray, AbstractScalar


def witness_encodedistancematrix(
    mat_list: AbstractArray,
    max_cdr3: AbstractScalar,
    max_epi: AbstractScalar,
) -> tuple[AbstractArray, AbstractArray]:
    """Distance-matrix encoding returns encoded values plus a boolean mask."""
    n_items = mat_list.shape[0] if mat_list.shape else 0
    cdr3 = int(max_cdr3.max_val or max_cdr3.min_val or 20)
    epi = int(max_epi.max_val or max_epi.min_val or 12)
    return (
        AbstractArray(shape=(n_items, cdr3, epi), dtype="float32"),
        AbstractArray(shape=(n_items, cdr3, epi), dtype="bool"),
    )
