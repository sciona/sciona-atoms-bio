from __future__ import annotations
"""Auto-generated atom wrappers following the ageoa pattern."""

from collections.abc import Hashable, Sequence
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    import networkx as nx

import icontract
from ageoa.ghost.registry import register_atom
from .witnesses import witness_networkx_weighted_graph_materialization, witness_pair_distance_compatibility_check, witness_weighted_interaction_edge_derivation

# Witness functions should be imported from the generated witnesses module

Feature = Hashable
FeaturePair = tuple[Feature, Feature]
InteractionNode = tuple[Feature, Feature]
InteractionEdge = tuple[InteractionNode, InteractionNode]

@register_atom(witness_pair_distance_compatibility_check)
@icontract.require(lambda interaction_distance: isinstance(interaction_distance, (float, int, np.number)), "interaction_distance must be numeric")
@icontract.ensure(lambda result: result is not None, "Pair Distance Compatibility Check output must not be None")
def pair_distance_compatibility_check(
    L_feature_min_max: Sequence[float] | np.ndarray,
    R_features_distance: float | Sequence[float] | np.ndarray,
    interaction_distance: float,
) -> bool:
    """Evaluates whether a candidate left/right feature pair satisfies interaction-distance constraints.

    Args:
        L_feature_min_max: Expected to encode min/max distance envelope for left feature context.
        R_features_distance: Distance values for right-side feature candidates.
        interaction_distance: Non-negative interaction threshold/target distance.

    Returns:
        True when pair satisfies distance constraints.
    """
    l_min_max = np.asarray(L_feature_min_max, dtype=float)
    r_dist = np.asarray(R_features_distance, dtype=float)
    if l_min_max.size == 0 or r_dist.size == 0:
        return False
    # Treat the left feature distances as an interval and accept when any
    # right-side distance falls within the interaction-distance-expanded band.
    l_min = float(l_min_max.min())
    l_max = float(l_min_max.max())
    lower = l_min - float(interaction_distance)
    upper = l_max + float(interaction_distance)
    return bool(np.any((r_dist >= lower) & (r_dist <= upper)))

@register_atom(witness_weighted_interaction_edge_derivation)
@icontract.require(lambda L_feature_pair: L_feature_pair is not None, "L_feature_pair cannot be None")
@icontract.require(lambda R_feature_pair: R_feature_pair is not None, "R_feature_pair cannot be None")
@icontract.ensure(lambda result: result is not None, "Weighted Interaction Edge Derivation output must not be None")
def weighted_interaction_edge_derivation(
    L_feature_pair: FeaturePair,
    R_feature_pair: FeaturePair,
) -> list[InteractionEdge]:
    """Return the two physically possible interaction-edge layouts for a ligand/receptor feature pair."""
    return [
        ((L_feature_pair[0], R_feature_pair[1]), (L_feature_pair[1], R_feature_pair[0])),
        ((L_feature_pair[1], R_feature_pair[1]), (L_feature_pair[0], R_feature_pair[0])),
    ]

@register_atom(witness_networkx_weighted_graph_materialization)
@icontract.require(lambda edges: edges is not None, "edges cannot be None")
@icontract.require(lambda nodes: nodes is not None, "nodes cannot be None")
@icontract.ensure(lambda result: result is not None, "NetworkX Weighted Graph Materialization output must not be None")
# Invalid pseudo-signature removed; keep the typed definition below.
def networkx_weighted_graph_materialization(edges: list[tuple[object, object, float]], nodes: list[object] | set[object]) -> nx.Graph | nx.DiGraph:
    """
    Args:
        nodes: Node IDs should be hashable.

    Returns:
        Contains all provided nodes and weighted edges.
    """
    import networkx as nx
    G = nx.Graph()
    G.add_nodes_from(nodes)
    for u, v, w in edges:
        G.add_edge(u, v, weight=w)
    return G
