from __future__ import annotations

from sciona.ghost.abstract import AbstractArray


def witness_rowselfattention(
    x: AbstractArray,
    q_proj_weight: AbstractArray,
    k_proj_weight: AbstractArray,
    v_proj_weight: AbstractArray,
    out_proj_weight: AbstractArray,
    num_heads: int,
    q_proj_bias: AbstractArray | None = None,
    k_proj_bias: AbstractArray | None = None,
    v_proj_bias: AbstractArray | None = None,
    out_proj_bias: AbstractArray | None = None,
    self_attn_mask: AbstractArray | None = None,
    self_attn_padding_mask: AbstractArray | None = None,
    dropout_p: float = 0.0,
    max_tokens_per_msa: int = 2**16,
    training: bool = False,
) -> tuple[AbstractArray, AbstractArray]:
    """Preserve MINT RowSelfAttention output and column-attention shapes."""
    num_rows, num_cols, batch_size, embed_dim = tuple(x.shape)
    output = AbstractArray(shape=(num_rows, num_cols, batch_size, embed_dim), dtype=x.dtype)
    attn_probs = AbstractArray(
        shape=(num_heads, batch_size, num_cols, num_cols),
        dtype=x.dtype,
    )
    return output, attn_probs


witness_row_self_attention = witness_rowselfattention
