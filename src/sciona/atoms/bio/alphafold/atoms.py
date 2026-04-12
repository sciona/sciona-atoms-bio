from __future__ import annotations
"""AlphaFold 3D Equivariant Structural Atoms."""

from typing import Tuple
import numpy as np

import icontract

from ageoa.ghost.registry import register_atom
from ageoa.alphafold.state_models import AlphaFoldStructuralState
from ageoa.alphafold.witnesses import (
    witness_invariant_point_attention,
    witness_equivariant_frame_update,
    witness_coordinate_reconstruction,
)


@register_atom(witness_invariant_point_attention)
@icontract.require(lambda nodes, pairs: nodes.shape[0] == pairs.shape[0] == pairs.shape[1], "Sequence length mismatch")
@icontract.ensure(lambda result, nodes: result[0].shape == nodes.shape, "IPA must preserve node shape")
def invariant_point_attention(
    nodes: np.ndarray,
    pairs: np.ndarray,
    frames: np.ndarray,
    state: AlphaFoldStructuralState
) -> Tuple[np.ndarray, AlphaFoldStructuralState]:
    """Equivariant (i.e., the output transforms consistently under rotations and translations of the input) attention mechanism over structured 3D point sets.

    Processes 3D points and orientation frames.

    Args:
        nodes: Node feature array of shape (n_res, d_node).
        pairs: Pair feature array of shape (n_res, n_res, d_pair).
        frames: Rigid frame representations for each residue.
        state: Current AlphaFoldStructuralState.

    Returns:
        Tuple of (updated_nodes, new_state) where updated_nodes has the
        same shape as the input nodes.
    """
    # Invariant Point Attention (simplified)
    n_res, d_node = nodes.shape
    d_pair = pairs.shape[-1]

    # Compute attention logits from pair features and node features
    # q, k, v projections (simplified as linear combinations)
    q = nodes  # (n_res, d_node)
    k = nodes
    v = nodes

    # Attention scores from nodes
    scale = np.sqrt(d_node)
    attn_logits = q @ k.T / scale  # (n_res, n_res)

    # Add pair bias (mean over pair feature dim)
    if pairs.ndim == 3:
        attn_logits += pairs.mean(axis=-1)

    # Softmax
    attn_logits -= attn_logits.max(axis=-1, keepdims=True)
    attn_weights = np.exp(attn_logits)
    attn_weights /= attn_weights.sum(axis=-1, keepdims=True) + 1e-15

    # Apply attention
    updated_nodes = attn_weights @ v  # (n_res, d_node)

    import jax.numpy as jnp

    new_state = AlphaFoldStructuralState(
        nodes=jnp.asarray(updated_nodes),
        frames=jnp.asarray(frames),
        pairs=jnp.asarray(pairs),
    )
    return (updated_nodes, new_state)

@register_atom(witness_equivariant_frame_update)
@icontract.require(lambda frames, nodes: len(nodes.shape) >= 1, "Nodes must have at least one dimension")
@icontract.ensure(lambda result: result is not None and len(result) == 2, "Result must be a (frames, state) tuple")
def equivariant_frame_update(
    frames: np.ndarray,
    nodes: np.ndarray,
    state: AlphaFoldStructuralState
) -> Tuple[np.ndarray, AlphaFoldStructuralState]:
    """Updates 3D rigid frames using predicted gradients.

    Args:
        frames: Current rigid frame representations.
        nodes: Node feature array with at least one dimension.
        state: Current AlphaFoldStructuralState.

    Returns:
        Tuple of (updated_frames, new_state).
    """
    frames_arr = np.asarray(frames, dtype=float)
    nodes_arr = np.asarray(nodes, dtype=float)

    # Equivariant frame update: apply node features as a small translation-like perturbation.
    n_res = nodes_arr.shape[0]
    d = nodes_arr.shape[1]
    updated_frames = frames_arr.copy()
    if frames_arr.ndim == 3 and frames_arr.shape[-1] >= 3:
        # frames: (n_res, 4, 4) or (n_res, 3, 4)
        translation_update = nodes_arr[:, :3] if d >= 3 else np.zeros((n_res, 3))
        updated_frames[:, :3, 3] += translation_update * 0.1  # Small step
    elif frames_arr.ndim == 2 and frames_arr.shape[1] >= 7:
        # Local repo representation: (qx, qy, qz, qw, tx, ty, tz)
        translation_update = nodes_arr[:, :3] if d >= 3 else np.zeros((n_res, 3))
        updated_frames[:, 4:7] += translation_update * 0.1
    elif frames_arr.ndim == 2 and frames_arr.shape[1] >= 3:
        translation_update = nodes_arr[:, :3] if d >= 3 else np.zeros((n_res, 3))
        updated_frames[:, -3:] += translation_update * 0.1
    import jax.numpy as jnp

    new_state = AlphaFoldStructuralState(
        nodes=jnp.asarray(nodes_arr),
        frames=jnp.asarray(updated_frames),
        pairs=state.pairs,
    )
    return (updated_frames, new_state)

@register_atom(witness_coordinate_reconstruction)
@icontract.require(lambda torsions: torsions.shape[-1] == 2, "Torsions must be represented as (sin, cos) pairs")
@icontract.ensure(lambda result: result is not None and len(result) == 2, "Result must be a (coords, state) tuple")
@icontract.ensure(lambda result: result[0].ndim == 3 and result[0].shape[-1] == 3, "Coordinates must have shape (n_res, n_atoms, 3)")
def coordinate_reconstruction(
    frames: np.ndarray,
    torsions: np.ndarray,
    state: AlphaFoldStructuralState
) -> Tuple[np.ndarray, AlphaFoldStructuralState]:
    """Converts rigid frames and torsion angles (dihedral angles between four consecutive atoms that define the protein backbone geometry) into full 3D coordinates.

    Args:
        frames: Rigid frame representations for each residue.
        torsions: Array of shape (n_res, n_torsions, 2) with (sin, cos) pairs.
        state: Current AlphaFoldStructuralState.

    Returns:
        Tuple of (coords, state) where coords has shape (n_res, n_atoms, 3).
    """
    frames_arr = np.asarray(frames, dtype=float)
    torsions_arr = np.asarray(torsions, dtype=float)

    # Reconstruct 3D coordinates from frames and torsion angles.
    n_res = frames_arr.shape[0]
    n_torsions = torsions_arr.shape[1]
    n_atoms = 37

    coords = np.zeros((n_res, n_atoms, 3), dtype=float)
    # Place backbone atoms using frame translations
    for i in range(n_res):
        # Extract frame origin (translation component)
        if frames_arr.ndim == 3 and frames_arr.shape[1] >= 3:
            origin = frames_arr[i, :3, 3] if frames_arr.shape[2] >= 4 else frames_arr[i, :3, 0]
            rot = frames_arr[i, :3, :3]
        elif frames_arr.ndim == 2 and frames_arr.shape[1] >= 7:
            origin = frames_arr[i, 4:7]
            rot = np.eye(3)
        else:
            origin = np.zeros(3)
            rot = np.eye(3)

        # N atom at origin
        coords[i, 0] = origin
        # CA atom offset
        ca_offset = np.array([1.458, 0.0, 0.0])  # N-CA bond length ~1.458 A
        coords[i, 1] = origin + rot @ ca_offset
        # C atom using first torsion
        sin_t, cos_t = torsions_arr[i, 0]
        c_offset = np.array([1.523 * cos_t, 1.523 * sin_t, 0.0])
        coords[i, 2] = coords[i, 1] + rot @ c_offset
        # Additional atoms from torsion angles
        for t in range(1, min(n_torsions, n_atoms - 2)):
            sin_t, cos_t = torsions_arr[i, t]
            bond_len = 1.5
            offset = np.array([bond_len * cos_t, bond_len * sin_t, 0.0])
            coords[i, t + 2] = coords[i, t + 1] + rot @ offset

    import jax.numpy as jnp

    new_state = AlphaFoldStructuralState(
        nodes=state.nodes,
        frames=jnp.asarray(frames_arr),
        pairs=state.pairs,
    )
    return (coords, new_state)
