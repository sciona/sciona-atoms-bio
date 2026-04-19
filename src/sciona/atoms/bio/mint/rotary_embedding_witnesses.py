from __future__ import annotations

from sciona.ghost.abstract import AbstractArray


def witness_rotaryembedding(q: AbstractArray, k: AbstractArray) -> tuple:
    """Shape-and-type check for opaque boundary: rotary embedding."""
    q_shape = tuple(q.shape)
    k_shape = tuple(k.shape)
    q_out = AbstractArray(shape=q_shape, dtype=q.dtype)
    k_out = AbstractArray(shape=k_shape, dtype=k.dtype)
    return q_out, k_out
