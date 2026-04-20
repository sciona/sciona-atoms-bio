from __future__ import annotations

import math
from typing import TYPE_CHECKING

import icontract

from sciona.ghost.registry import register_atom

from .axial_attention_witnesses import witness_rowselfattention

if TYPE_CHECKING:
    import torch


def _projection_weight_shape_is_valid(weight: object, embed_dim: int) -> bool:
    return hasattr(weight, "shape") and tuple(weight.shape) == (embed_dim, embed_dim)


def _optional_bias_shape_is_valid(bias: object | None, embed_dim: int) -> bool:
    return bias is None or (hasattr(bias, "shape") and tuple(bias.shape) == (embed_dim,))


def _padding_mask_shape_is_valid(mask: object | None, x: torch.Tensor) -> bool:
    return mask is None or (
        hasattr(mask, "shape") and tuple(mask.shape) == (x.size(2), x.size(0), x.size(1))
    )


def _compute_attention_weights(
    x: torch.Tensor,
    *,
    q_proj_weight: torch.Tensor,
    k_proj_weight: torch.Tensor,
    q_proj_bias: torch.Tensor | None,
    k_proj_bias: torch.Tensor | None,
    num_heads: int,
    scaling: float,
    self_attn_mask: torch.Tensor | None,
    self_attn_padding_mask: torch.Tensor | None,
) -> torch.Tensor:
    import torch
    import torch.nn.functional as F

    num_rows, num_cols, batch_size, embed_dim = x.size()
    head_dim = embed_dim // num_heads
    q = F.linear(x, q_proj_weight, q_proj_bias).view(
        num_rows, num_cols, batch_size, num_heads, head_dim
    )
    k = F.linear(x, k_proj_weight, k_proj_bias).view(
        num_rows, num_cols, batch_size, num_heads, head_dim
    )
    q = q * scaling
    if self_attn_padding_mask is not None:
        row_padding = self_attn_padding_mask.permute(1, 2, 0).unsqueeze(3).unsqueeze(4)
        q = q * (1 - row_padding.to(dtype=q.dtype, device=q.device))

    attn_weights = torch.einsum("rinhd,rjnhd->hnij", q, k)

    if self_attn_mask is not None:
        raise NotImplementedError("MINT RowSelfAttention does not implement self_attn_mask")

    if self_attn_padding_mask is not None:
        key_padding = self_attn_padding_mask[:, 0].bool().unsqueeze(0).unsqueeze(2)
        attn_weights = attn_weights.masked_fill(key_padding, -10000)

    return attn_weights


def _compute_attention_update(
    x: torch.Tensor,
    attn_probs: torch.Tensor,
    *,
    v_proj_weight: torch.Tensor,
    out_proj_weight: torch.Tensor,
    v_proj_bias: torch.Tensor | None,
    out_proj_bias: torch.Tensor | None,
    num_heads: int,
) -> torch.Tensor:
    import torch
    import torch.nn.functional as F

    num_rows, num_cols, batch_size, embed_dim = x.size()
    head_dim = embed_dim // num_heads
    v = F.linear(x, v_proj_weight, v_proj_bias).view(
        num_rows, num_cols, batch_size, num_heads, head_dim
    )
    context = torch.einsum("hnij,rjnhd->rinhd", attn_probs, v)
    context = context.contiguous().view(num_rows, num_cols, batch_size, embed_dim)
    return F.linear(context, out_proj_weight, out_proj_bias)


def _row_self_attention_core(
    x: torch.Tensor,
    *,
    q_proj_weight: torch.Tensor,
    k_proj_weight: torch.Tensor,
    v_proj_weight: torch.Tensor,
    out_proj_weight: torch.Tensor,
    num_heads: int,
    q_proj_bias: torch.Tensor | None,
    k_proj_bias: torch.Tensor | None,
    v_proj_bias: torch.Tensor | None,
    out_proj_bias: torch.Tensor | None,
    self_attn_mask: torch.Tensor | None,
    self_attn_padding_mask: torch.Tensor | None,
    dropout_p: float,
    max_tokens_per_msa: int,
    training: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    import torch
    import torch.nn.functional as F

    num_rows, num_cols, _batch_size, embed_dim = x.size()
    head_dim = embed_dim // num_heads
    scaling = (head_dim**-0.5) / math.sqrt(num_rows)

    if (num_rows * num_cols > max_tokens_per_msa) and not torch.is_grad_enabled():
        max_rows = max(1, max_tokens_per_msa // num_cols)
        attns: torch.Tensor | int = 0
        for start in range(0, num_rows, max_rows):
            padding_chunk = (
                self_attn_padding_mask[:, start : start + max_rows]
                if self_attn_padding_mask is not None
                else None
            )
            attns = attns + _compute_attention_weights(
                x[start : start + max_rows],
                q_proj_weight=q_proj_weight,
                k_proj_weight=k_proj_weight,
                q_proj_bias=q_proj_bias,
                k_proj_bias=k_proj_bias,
                num_heads=num_heads,
                scaling=scaling,
                self_attn_mask=self_attn_mask,
                self_attn_padding_mask=padding_chunk,
            )
        attn_probs = F.dropout(torch.softmax(attns, dim=-1), p=dropout_p, training=training)
        outputs = [
            _compute_attention_update(
                x[start : start + max_rows],
                attn_probs,
                v_proj_weight=v_proj_weight,
                out_proj_weight=out_proj_weight,
                v_proj_bias=v_proj_bias,
                out_proj_bias=out_proj_bias,
                num_heads=num_heads,
            )
            for start in range(0, num_rows, max_rows)
        ]
        return torch.cat(outputs, dim=0), attn_probs

    attn_weights = _compute_attention_weights(
        x,
        q_proj_weight=q_proj_weight,
        k_proj_weight=k_proj_weight,
        q_proj_bias=q_proj_bias,
        k_proj_bias=k_proj_bias,
        num_heads=num_heads,
        scaling=scaling,
        self_attn_mask=self_attn_mask,
        self_attn_padding_mask=self_attn_padding_mask,
    )
    attn_probs = F.dropout(torch.softmax(attn_weights, dim=-1), p=dropout_p, training=training)
    output = _compute_attention_update(
        x,
        attn_probs,
        v_proj_weight=v_proj_weight,
        out_proj_weight=out_proj_weight,
        v_proj_bias=v_proj_bias,
        out_proj_bias=out_proj_bias,
        num_heads=num_heads,
    )
    return output, attn_probs


@register_atom(witness_rowselfattention)
@icontract.require(lambda x: x is not None, "x cannot be None")
@icontract.require(lambda x: x.dim() == 4, "x must be a 4D MSA tensor")
@icontract.require(lambda num_heads: num_heads > 0, "num_heads must be positive")
@icontract.require(
    lambda x, num_heads: x.size(-1) % num_heads == 0,
    "embed_dim must be divisible by num_heads",
)
@icontract.require(lambda dropout_p: 0.0 <= dropout_p < 1.0, "dropout_p must be in [0, 1)")
@icontract.require(lambda max_tokens_per_msa: max_tokens_per_msa > 0, "max_tokens_per_msa must be positive")
@icontract.require(
    lambda x, q_proj_weight, k_proj_weight, v_proj_weight, out_proj_weight: all(
        _projection_weight_shape_is_valid(weight, x.size(-1))
        for weight in (q_proj_weight, k_proj_weight, v_proj_weight, out_proj_weight)
    ),
    "projection weights must be shaped (embed_dim, embed_dim)",
)
@icontract.require(
    lambda x, q_proj_bias, k_proj_bias, v_proj_bias, out_proj_bias: all(
        _optional_bias_shape_is_valid(bias, x.size(-1))
        for bias in (q_proj_bias, k_proj_bias, v_proj_bias, out_proj_bias)
    ),
    "projection biases must be None or shaped (embed_dim,)",
)
@icontract.require(
    lambda x, self_attn_padding_mask: _padding_mask_shape_is_valid(self_attn_padding_mask, x),
    "padding mask must be shaped (batch_size, num_rows, num_cols)",
)
@icontract.ensure(lambda result, x: result[0].shape == x.shape, "output must preserve x shape")
@icontract.ensure(
    lambda result, x, num_heads: result[1].shape
    == (num_heads, x.size(2), x.size(1), x.size(1)),
    "attention probabilities must be shaped (num_heads, batch_size, num_cols, num_cols)",
)
def row_self_attention(
    x: torch.Tensor,
    q_proj_weight: torch.Tensor,
    k_proj_weight: torch.Tensor,
    v_proj_weight: torch.Tensor,
    out_proj_weight: torch.Tensor,
    num_heads: int,
    q_proj_bias: torch.Tensor | None = None,
    k_proj_bias: torch.Tensor | None = None,
    v_proj_bias: torch.Tensor | None = None,
    out_proj_bias: torch.Tensor | None = None,
    self_attn_mask: torch.Tensor | None = None,
    self_attn_padding_mask: torch.Tensor | None = None,
    dropout_p: float = 0.0,
    max_tokens_per_msa: int = 2**16,
    training: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute MINT row attention for an alignment tensor using supplied projection weights."""
    return _row_self_attention_core(
        x,
        q_proj_weight=q_proj_weight,
        k_proj_weight=k_proj_weight,
        v_proj_weight=v_proj_weight,
        out_proj_weight=out_proj_weight,
        num_heads=num_heads,
        q_proj_bias=q_proj_bias,
        k_proj_bias=k_proj_bias,
        v_proj_bias=v_proj_bias,
        out_proj_bias=out_proj_bias,
        self_attn_mask=self_attn_mask,
        self_attn_padding_mask=self_attn_padding_mask,
        dropout_p=dropout_p,
        max_tokens_per_msa=max_tokens_per_msa,
        training=training,
    )


@register_atom(witness_rowselfattention)
@icontract.require(lambda x: x is not None, "x cannot be None")
@icontract.require(lambda x: x.dim() == 4, "x must be a 4D MSA tensor")
@icontract.require(lambda num_heads: num_heads > 0, "num_heads must be positive")
@icontract.require(
    lambda x, num_heads: x.size(-1) % num_heads == 0,
    "embed_dim must be divisible by num_heads",
)
@icontract.require(lambda dropout_p: 0.0 <= dropout_p < 1.0, "dropout_p must be in [0, 1)")
@icontract.require(lambda max_tokens_per_msa: max_tokens_per_msa > 0, "max_tokens_per_msa must be positive")
@icontract.require(
    lambda x, q_proj_weight, k_proj_weight, v_proj_weight, out_proj_weight: all(
        _projection_weight_shape_is_valid(weight, x.size(-1))
        for weight in (q_proj_weight, k_proj_weight, v_proj_weight, out_proj_weight)
    ),
    "projection weights must be shaped (embed_dim, embed_dim)",
)
@icontract.require(
    lambda x, q_proj_bias, k_proj_bias, v_proj_bias, out_proj_bias: all(
        _optional_bias_shape_is_valid(bias, x.size(-1))
        for bias in (q_proj_bias, k_proj_bias, v_proj_bias, out_proj_bias)
    ),
    "projection biases must be None or shaped (embed_dim,)",
)
@icontract.require(
    lambda x, self_attn_padding_mask: _padding_mask_shape_is_valid(self_attn_padding_mask, x),
    "padding mask must be shaped (batch_size, num_rows, num_cols)",
)
@icontract.ensure(lambda result, x: result[0].shape == x.shape, "output must preserve x shape")
@icontract.ensure(
    lambda result, x, num_heads: result[1].shape
    == (num_heads, x.size(2), x.size(1), x.size(1)),
    "attention probabilities must be shaped (num_heads, batch_size, num_cols, num_cols)",
)
def rowselfattention(
    x: torch.Tensor,
    q_proj_weight: torch.Tensor,
    k_proj_weight: torch.Tensor,
    v_proj_weight: torch.Tensor,
    out_proj_weight: torch.Tensor,
    num_heads: int,
    q_proj_bias: torch.Tensor | None = None,
    k_proj_bias: torch.Tensor | None = None,
    v_proj_bias: torch.Tensor | None = None,
    out_proj_bias: torch.Tensor | None = None,
    self_attn_mask: torch.Tensor | None = None,
    self_attn_padding_mask: torch.Tensor | None = None,
    dropout_p: float = 0.0,
    max_tokens_per_msa: int = 2**16,
    training: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compatibility entrypoint for the same MINT row attention calculation."""
    return _row_self_attention_core(
        x,
        q_proj_weight=q_proj_weight,
        k_proj_weight=k_proj_weight,
        v_proj_weight=v_proj_weight,
        out_proj_weight=out_proj_weight,
        num_heads=num_heads,
        q_proj_bias=q_proj_bias,
        k_proj_bias=k_proj_bias,
        v_proj_bias=v_proj_bias,
        out_proj_bias=out_proj_bias,
        self_attn_mask=self_attn_mask,
        self_attn_padding_mask=self_attn_padding_mask,
        dropout_p=dropout_p,
        max_tokens_per_msa=max_tokens_per_msa,
        training=training,
    )
