from __future__ import annotations

import numpy as np
import pytest
import torch

from sciona.atoms.bio.mint.apc_module import apccoreevaluation
from sciona.atoms.bio.mint.encoding_dist_mat import encodedistancematrix
from sciona.atoms.bio.mint.incremental_attention import (
    FairseqIncrementalState,
    enable_incremental_state_configuration,
)
from sciona.atoms.bio.mint.rotary_embedding import (
    rotaryembedding_numpy,
    rotaryembedding_torch,
)


def test_pubrev019_apc_matches_average_product_correction() -> None:
    x = np.array([[1.0, 2.0], [3.0, 4.0]])

    result = apccoreevaluation(x)

    row_sum = x.sum(axis=-1, keepdims=True)
    col_sum = x.sum(axis=-2, keepdims=True)
    total = x.sum(axis=(-1, -2), keepdims=True)
    np.testing.assert_allclose(result, x - (row_sum * col_sum / total))


def test_pubrev019_encoding_dist_mat_matches_mint_offsets_and_mask() -> None:
    mat = np.arange(24, dtype=np.float32).reshape(3, 8)

    encoding, mask = encodedistancematrix([mat], max_cdr3=5, max_epi=12)

    assert encoding.dtype == np.float32
    assert mask.dtype == bool
    assert encoding.shape == (1, 5, 12)
    assert mask.shape == (1, 5, 12)
    np.testing.assert_array_equal(encoding[0, 1:4, 2:10], mat)
    assert mask[0, 1:4, 2:10].all()
    assert not mask[0, 0].any()
    assert not mask[0, :, :2].any()


def test_pubrev019_encoding_dist_mat_rejects_oversized_inputs() -> None:
    with pytest.raises(ValueError, match="exceeds"):
        encodedistancematrix([np.zeros((21, 12), dtype=np.float32)])


def test_pubrev019_incremental_state_uses_external_uuid_scoped_state() -> None:
    class DemoAttention(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()

    decorated = enable_incremental_state_configuration(DemoAttention)
    instance = decorated()
    state: dict = {}

    assert decorated is DemoAttention
    assert issubclass(decorated, FairseqIncrementalState)
    assert instance.get_incremental_state(state, "attn_state") is None
    assert instance.set_incremental_state(state, "attn_state", {"prev_key": "value"}) is state
    assert instance.get_incremental_state(state, "attn_state") == {"prev_key": "value"}
    assert next(iter(state)).endswith(".attn_state")


def test_pubrev019_rotary_embedding_numpy_and_torch_match_rotate_half_semantics() -> None:
    q_np = np.arange(8, dtype=np.float32).reshape(1, 2, 4)
    k_np = q_np + 10
    q_torch = torch.from_numpy(q_np.copy())
    k_torch = torch.from_numpy(k_np.copy())

    q_np_out, k_np_out = rotaryembedding_numpy(q_np, k_np)
    q_torch_out, k_torch_out = rotaryembedding_torch(q_torch, k_torch)

    np.testing.assert_allclose(q_np_out, q_torch_out.numpy(), rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(k_np_out, k_torch_out.numpy(), rtol=1e-6, atol=1e-6)

    # Position zero has cos=1 and sin=0, so only later positions rotate.
    np.testing.assert_array_equal(q_np_out[:, 0, :], q_np[:, 0, :])
    assert not np.array_equal(q_np_out[:, 1, :], q_np[:, 1, :])
