from __future__ import annotations

from sciona.ghost.abstract import AbstractArray, AbstractScalar


def witness_pair_distance_compatibility_check(
    L_feature_min_max: AbstractArray,
    R_features_distance: AbstractArray,
    interaction_distance: AbstractArray,
) -> AbstractScalar:
    """Shape-and-type check for pair distance compatibility check. Returns output metadata without running the real computation."""
    _ = (L_feature_min_max, R_features_distance, interaction_distance)
    return AbstractScalar(dtype="bool")


def witness_weighted_interaction_edge_derivation(
    L_feature_pair: AbstractArray,
    R_feature_pair: AbstractArray,
) -> AbstractArray:
    """Return metadata for the list of possible interaction-edge pairings."""
    _ = (L_feature_pair, R_feature_pair)
    return AbstractArray(shape=("E", 2, 2), dtype="object")


def witness_networkx_weighted_graph_materialization(
    edges: AbstractArray,
    nodes: AbstractArray,
) -> AbstractArray:
    """Shape-and-type check for network x weighted graph materialization. Returns output metadata without running the real computation."""
    _ = (edges, nodes)
    return AbstractArray(shape=(1,), dtype="float64")
