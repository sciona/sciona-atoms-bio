from __future__ import annotations

import numpy as np
import icontract
from sciona.ghost.registry import register_atom

from .rotary_embedding_witnesses import witness_rotaryembedding


@register_atom(witness_rotaryembedding)
@icontract.require(lambda q: q.ndim >= 2, "q must be at least 2-D")
@icontract.require(lambda k: k.ndim >= 2, "k must be at least 2-D")
@icontract.require(lambda q: isinstance(q, np.ndarray), "q must be np.ndarray")
@icontract.require(lambda k: isinstance(k, np.ndarray), "k must be np.ndarray")
@icontract.ensure(lambda result: isinstance(result, tuple) and len(result) == 2, "result must be a 2-tuple")
def rotaryembedding(q: np.ndarray, k: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Apply Rotary Positional Embedding (RoPE) to query and key tensors.

    Args:
        q: Query tensor with shape (..., seq_len, head_dim).
        k: Key tensor with shape (..., seq_len, head_dim).

    Returns:
        Tuple of rotated (query, key) tensors with same shapes as inputs.
    """
    # RoPE: rotate query and key by position-dependent angles
    seq_len = q.shape[-2]
    head_dim = q.shape[-1]
    # Compute frequency bands
    freqs = 1.0 / (10000 ** (np.arange(0, head_dim, 2, dtype=np.float64) / head_dim))
    positions = np.arange(seq_len, dtype=np.float64)
    angles = np.outer(positions, freqs)  # (seq_len, head_dim/2)
    cos_a = np.cos(angles)
    sin_a = np.sin(angles)
    # Rotate pairs of dimensions
    def _rotate(x):
        x1 = x[..., 0::2]
        x2 = x[..., 1::2]
        rotated = np.empty_like(x)
        rotated[..., 0::2] = x1 * cos_a - x2 * sin_a
        rotated[..., 1::2] = x1 * sin_a + x2 * cos_a
        return rotated
    return (_rotate(q), _rotate(k))

from typing import TYPE_CHECKING, Tuple

if TYPE_CHECKING:
    import torch

import icontract

from sciona.ghost.registry import register_atom
from .rotary_embedding_witnesses import witness_rotaryembedding
@register_atom(witness_rotaryembedding)
@icontract.require(lambda q: q is not None, "q cannot be None")
@icontract.require(lambda k: k is not None, "k cannot be None")
@icontract.ensure(lambda result: result is not None, "RotaryEmbedding output must not be None")
def rotaryembedding(q: torch.Tensor, k: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Applies Rotary Positional Embedding (RoPE) to query and key tensors in a neural network attention layer. RoPE encodes token position by rotating the query (q) and key (k) vectors in pairs of dimensions, so that the dot-product attention score naturally decays with distance between tokens.

    Args:
        q: Query tensor from the attention layer.
        k: Key tensor from the attention layer.

    Returns:
        Position-encoded query and key tensors.
    """
    import torch
    seq_len = q.shape[-2]
    head_dim = q.shape[-1]
    freqs = 1.0 / (10000 ** (torch.arange(0, head_dim, 2, dtype=torch.float32, device=q.device) / head_dim))
    positions = torch.arange(seq_len, dtype=torch.float32, device=q.device)
    angles = torch.outer(positions, freqs)
    cos_a = torch.cos(angles)
    sin_a = torch.sin(angles)
    def _rotate(x):
        x1 = x[..., 0::2]
        x2 = x[..., 1::2]
        rotated = torch.empty_like(x)
        rotated[..., 0::2] = x1 * cos_a - x2 * sin_a
        rotated[..., 1::2] = x1 * sin_a + x2 * cos_a
        return rotated
    return (_rotate(q), _rotate(k))
