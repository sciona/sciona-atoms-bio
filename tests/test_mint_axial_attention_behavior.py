from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest
import torch

from sciona.atoms.bio.mint.axial_attention import row_self_attention, rowselfattention


UPSTREAM_SOURCE = Path(
    "/Users/conrad/personal/ageo-atoms/third_party/mint/mint/axial_attention.py"
)


def _upstream_row_self_attention_class() -> type:
    spec = importlib.util.spec_from_file_location("mint_upstream_axial_attention", UPSTREAM_SOURCE)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.RowSelfAttention


def _projection_kwargs(module: torch.nn.Module) -> dict:
    return {
        "q_proj_weight": module.q_proj.weight,
        "k_proj_weight": module.k_proj.weight,
        "v_proj_weight": module.v_proj.weight,
        "out_proj_weight": module.out_proj.weight,
        "num_heads": module.num_heads,
        "q_proj_bias": module.q_proj.bias,
        "k_proj_bias": module.k_proj.bias,
        "v_proj_bias": module.v_proj.bias,
        "out_proj_bias": module.out_proj.bias,
        "dropout_p": module.dropout,
        "max_tokens_per_msa": module.max_tokens_per_msa,
        "training": module.training,
    }


def _module(embed_dim: int = 4, num_heads: int = 2, max_tokens_per_msa: int = 2**16) -> torch.nn.Module:
    torch.manual_seed(13)
    module = _upstream_row_self_attention_class()(
        embed_dim=embed_dim,
        num_heads=num_heads,
        dropout=0.0,
        max_tokens_per_msa=max_tokens_per_msa,
    )
    module.eval()
    return module


def test_row_self_attention_matches_upstream_projection_and_attention_shape() -> None:
    module = _module()
    x = torch.randn(3, 2, 2, 4)

    expected_output, expected_attn = module(x)
    actual_output, actual_attn = row_self_attention(x, **_projection_kwargs(module))

    torch.testing.assert_close(actual_output, expected_output)
    torch.testing.assert_close(actual_attn, expected_attn)
    assert actual_output.shape == x.shape
    assert actual_attn.shape == (2, 2, 2, 2)


def test_row_self_attention_matches_upstream_padding_mask_semantics() -> None:
    module = _module()
    x = torch.randn(3, 4, 2, 4)
    padding_mask = torch.zeros(2, 3, 4, dtype=torch.bool)
    padding_mask[0, 1, 3] = True
    padding_mask[1, 2, 1] = True

    expected_output, expected_attn = module(x, self_attn_padding_mask=padding_mask)
    actual_output, actual_attn = row_self_attention(
        x,
        self_attn_padding_mask=padding_mask,
        **_projection_kwargs(module),
    )

    torch.testing.assert_close(actual_output, expected_output)
    torch.testing.assert_close(actual_attn, expected_attn)


def test_row_self_attention_matches_upstream_no_grad_batched_forward() -> None:
    module = _module(max_tokens_per_msa=6)
    x = torch.randn(5, 3, 2, 4)
    padding_mask = torch.zeros(2, 5, 3, dtype=torch.bool)
    padding_mask[0, 4, 2] = True

    with torch.no_grad():
        expected_output, expected_attn = module(x, self_attn_padding_mask=padding_mask)
        actual_output, actual_attn = row_self_attention(
            x,
            self_attn_padding_mask=padding_mask,
            **_projection_kwargs(module),
        )

    torch.testing.assert_close(actual_output, expected_output)
    torch.testing.assert_close(actual_attn, expected_attn)


def test_lowercase_rowselfattention_uses_same_source_aligned_contract() -> None:
    module = _module()
    x = torch.randn(2, 3, 2, 4)

    expected_output, expected_attn = row_self_attention(x, **_projection_kwargs(module))
    actual_output, actual_attn = rowselfattention(x, **_projection_kwargs(module))

    torch.testing.assert_close(actual_output, expected_output)
    torch.testing.assert_close(actual_attn, expected_attn)


def test_self_attn_mask_preserves_upstream_not_implemented_behavior() -> None:
    module = _module()
    x = torch.randn(2, 3, 1, 4)
    mask = torch.zeros(3, 3)

    with pytest.raises(NotImplementedError):
        row_self_attention(x, self_attn_mask=mask, **_projection_kwargs(module))
