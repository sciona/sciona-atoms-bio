from __future__ import annotations

from ageoa.ghost.abstract import AbstractArray, AbstractScalar, AbstractDistribution, AbstractSignal


def witness_rotaryembedding(q: AbstractArray, k: AbstractArray) -> tuple:
    """Shape-and-type check for opaque boundary: rotary embedding. Returns output metadata without running the real computation."""
    q_shape = tuple(q.shape)
    k_shape = tuple(k.shape)
    q_out = AbstractArray(shape=q_shape, dtype="float32")
    k_out = AbstractArray(shape=k_shape, dtype="float32")
    return q_out, k_out

from ageoa.ghost.abstract import AbstractArray, AbstractScalar, AbstractDistribution, AbstractSignal


def witness_rotaryembedding(q: AbstractArray, k: AbstractArray) -> AbstractArray:
    """Shape-and-type check for opaque boundary: rotary embedding. Returns output metadata without running the real computation."""
    # RotaryEmbedding applies in-place sinusoidal rotation to q and k.
    # Shape is fully preserved: no projection, pooling, or sequence change.
    # Typical layout: B=batch, H=num_heads, N=seq_len, D=head_dim.
    #
    # Under vmap the outermost batch dim B is stripped by the transform;
    # q and k then arrive as (H, N, D). The witness stays correct because
    # it mirrors q.shape and k.shape symbolically regardless of rank.
    #
    # Precondition: q.shape == k.shape (standard RoPE contract).
    q_out = AbstractArray(q.shape, dtype='float32')
    k_out = AbstractArray(k.shape, dtype='float32')
    return (q_out, k_out)
