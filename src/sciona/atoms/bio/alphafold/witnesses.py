from __future__ import annotations
from ageoa.ghost.abstract import AbstractArray, AbstractScalar, AbstractDistribution, AbstractSignal


def witness_invariant_point_attention(nodes: AbstractArray, pairs: AbstractArray, frames: AbstractArray, state: AbstractSignal) -> tuple[AbstractArray, AbstractSignal]:
    """Ghost witness for Invariant Point Attention.

    Transforms (N, C_n) + (N, N, C_p) + (N, 7) -> (N, C_n).
    """
    # Verify shapes
    n_res = nodes.shape[0]
    if pairs.shape[0] != n_res or pairs.shape[1] != n_res:
        raise ValueError(f"Pairs shape {pairs.shape} incompatible with nodes {nodes.shape}")

    # IPA preserves node dimensionality
    return AbstractArray(shape=nodes.shape, dtype="float32"), state

def witness_equivariant_frame_update(frames: AbstractArray, nodes: AbstractArray, state: AbstractSignal) -> tuple[AbstractArray, AbstractSignal]:
    """Ghost witness for Frame Update.

    Updates (N, 7) frames using (N, C_n) features.
    """
    return AbstractArray(shape=frames.shape, dtype="float32"), state

def witness_coordinate_reconstruction(frames: AbstractArray, torsions: AbstractArray, state: AbstractSignal) -> tuple[AbstractArray, AbstractSignal]:
    """Ghost witness for Coordinate Reconstruction.

    Transforms (N, 7) frames + (N, 7, 2) torsions -> (N, 37, 3) Cartesian coordinates.
    """
    n_res = frames.shape[0]
    # AlphaFold predicts (N_res, 37, 3) coordinates
    return AbstractArray(shape=(n_res, 37, 3), dtype="float32"), state
