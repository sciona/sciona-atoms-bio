from __future__ import annotations

"""Auto-generated atom wrappers following the sciona pattern."""


import numpy as np
import icontract
from sciona.ghost.registry import register_atom
from .greedy_subgraph_witnesses import witness_greedy_maximum_subgraph


@register_atom(witness_greedy_maximum_subgraph)
@icontract.require(lambda adjacency: adjacency.ndim >= 1, "adjacency must have at least one dimension")
@icontract.require(lambda scores: scores.ndim >= 1, "scores must have at least one dimension")
@icontract.require(lambda adjacency: adjacency is not None, "adjacency cannot be None")
@icontract.require(lambda adjacency: isinstance(adjacency, np.ndarray), "adjacency must be np.ndarray")
@icontract.require(lambda scores: scores is not None, "scores cannot be None")
@icontract.require(lambda scores: isinstance(scores, np.ndarray), "scores must be np.ndarray")
@icontract.ensure(lambda result: isinstance(result, np.ndarray), "result must be np.ndarray")
@icontract.ensure(lambda result: result is not None, "result must not be None")
def greedy_maximum_subgraph(adjacency: np.ndarray, scores: np.ndarray) -> np.ndarray:
    """Greedily selects a maximum-weight subgraph from a molecular interaction graph by iteratively adding the highest-scoring connected node.

    Args:
        adjacency: Adjacency matrix of the molecular interaction graph, shape (n_nodes, n_nodes)
        scores: Node affinity scores used to guide the greedy selection, shape (n_nodes,)

    Returns:
        Boolean mask of selected subgraph nodes, shape (n_nodes,)
    """
    n = adjacency.shape[0]
    selected = np.zeros(n, dtype=bool)
    # Sort nodes by score descending
    order = np.argsort(-scores)
    for idx in order:
        # Check if adding this node conflicts with any already-selected node
        conflicts = adjacency[idx] & selected
        if not np.any(conflicts):
            selected[idx] = True
    return selected
