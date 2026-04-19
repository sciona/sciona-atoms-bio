from __future__ import annotations

from typing import TYPE_CHECKING

import icontract
import numpy as np

from sciona.ghost.registry import register_atom

from .rotary_embedding_witnesses import witness_rotaryembedding

if TYPE_CHECKING:
    import torch


def _numpy_rotate_half(x: np.ndarray) -> np.ndarray:
    x1, x2 = np.array_split(x, 2, axis=-1)
    return np.concatenate((-x2, x1), axis=-1)


def _numpy_cos_sin_tables(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    seq_len = x.shape[-2]
    dim = x.shape[-1]
    inv_freq = 1.0 / (10000 ** (np.arange(0, dim, 2, dtype=np.float64) / dim))
    positions = np.arange(seq_len, dtype=np.float64)
    freqs = np.einsum("i,j->ij", positions, inv_freq)
    emb = np.concatenate((freqs, freqs), axis=-1).astype(x.dtype, copy=False)
    cos = np.cos(emb)[None, :, :]
    sin = np.sin(emb)[None, :, :]
    return cos, sin


@register_atom(witness_rotaryembedding)
@icontract.require(lambda q: isinstance(q, np.ndarray), "q must be np.ndarray")
@icontract.require(lambda k: isinstance(k, np.ndarray), "k must be np.ndarray")
@icontract.require(lambda q: q.ndim >= 2, "q must be at least 2-D")
@icontract.require(lambda k: k.ndim >= 2, "k must be at least 2-D")
@icontract.require(lambda q, k: q.shape[-1] == k.shape[-1], "q/k dims must match")
@icontract.require(lambda q: q.shape[-1] % 2 == 0, "head dim must be even")
@icontract.ensure(lambda result: isinstance(result, tuple) and len(result) == 2, "result must be a 2-tuple")
def rotaryembedding_numpy(q: np.ndarray, k: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Apply MINT/RoFormer rotary positional embedding to NumPy query/key arrays."""
    cos, sin = _numpy_cos_sin_tables(k)
    return (q * cos) + (_numpy_rotate_half(q) * sin), (k * cos) + (
        _numpy_rotate_half(k) * sin
    )


@register_atom(witness_rotaryembedding)
@icontract.require(lambda q: q is not None, "q cannot be None")
@icontract.require(lambda k: k is not None, "k cannot be None")
@icontract.require(lambda q, k: q.shape[-1] == k.shape[-1], "q/k dims must match")
@icontract.require(lambda q: q.shape[-1] % 2 == 0, "head dim must be even")
@icontract.ensure(lambda result: result is not None, "RotaryEmbedding output must not be None")
def rotaryembedding_torch(q: torch.Tensor, k: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply MINT/RoFormer rotary positional embedding to torch query/key tensors."""
    import torch

    seq_len = k.shape[-2]
    dim = k.shape[-1]
    inv_freq = 1.0 / (
        10000 ** (torch.arange(0, dim, 2, dtype=torch.float32, device=k.device) / dim)
    )
    positions = torch.arange(seq_len, dtype=inv_freq.dtype, device=k.device)
    freqs = torch.einsum("i,j->ij", positions, inv_freq)
    emb = torch.cat((freqs, freqs), dim=-1).to(device=k.device, dtype=k.dtype)
    cos = emb.cos()[None, :, :]
    sin = emb.sin()[None, :, :]

    def rotate_half(x: torch.Tensor) -> torch.Tensor:
        x1, x2 = x.chunk(2, dim=-1)
        return torch.cat((-x2, x1), dim=-1)

    return (q * cos) + (rotate_half(q) * sin), (k * cos) + (rotate_half(k) * sin)


# Backward-compatible alias for the previous generated wrapper name.
rotaryembedding = rotaryembedding_torch
