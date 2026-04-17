from __future__ import annotations
from typing import List

"""Auto-generated atom wrappers following the sciona pattern."""


import numpy as np

import icontract
from sciona.ghost.registry import register_atom
from .witnesses import witness_calculate_weight, witness_is_independent_set, witness_load_graphs_from_folder, witness_to_qubo

# Witness functions should be imported from the generated witnesses module

@register_atom(witness_load_graphs_from_folder)
@icontract.require(lambda folder_path: folder_path is not None, "folder_path cannot be None")
@icontract.ensure(lambda result: result is not None, "Load Graphs From Folder output must not be None")
def load_graphs_from_folder(folder_path: str) -> list[np.ndarray]:
    """Load graphs from folder.

    Args:
        folder_path (str): Description.

    Returns:
        list[np.ndarray]: Description.
    """
    import os
    graphs = []
    if os.path.isdir(folder_path):
        for fname in sorted(os.listdir(folder_path)):
            fpath = os.path.join(folder_path, fname)
            if fname.endswith('.npy'):
                graphs.append(np.load(fpath))
            elif fname.endswith('.csv'):
                graphs.append(np.loadtxt(fpath, delimiter=','))
    return graphs

@register_atom(witness_is_independent_set)
@icontract.require(lambda graph: graph is not None, "graph cannot be None")
@icontract.require(lambda subset: subset is not None, "subset cannot be None")
@icontract.ensure(lambda result: result is not None, "Is Independent Set output must not be None")
def is_independent_set(graph: np.ndarray, subset: list[int]) -> bool:
    """Checks if the given subset of nodes is an independent set in the graph.

:param graph: The input graph.
:param subset: The list of nodes to check.
:return: True if the subset is an independent set, False otherwise.

    Args:
        graph: Input data.
        subset: Input data.

    Returns:
        Result data.
    """
    for i in range(len(subset)):
        for j in range(i + 1, len(subset)):
            if graph[subset[i], subset[j]] != 0:
                return False
    return True

@register_atom(witness_calculate_weight)
@icontract.require(lambda graph: graph is not None, "graph cannot be None")
@icontract.require(lambda node_list: node_list is not None, "node_list cannot be None")
@icontract.ensure(lambda result: result is not None, "Calculate Weight output must not be None")
def calculate_weight(graph: np.ndarray, node_list: list[int]) -> float:
    """Calculates the total weight of a given list of nodes in the graph.

:param node_list: List of nodes.
:return: Total weight of the nodes.

    Args:
        graph: Input data.
        node_list: Input data.

    Returns:
        Result data.
    """
    # Node weights are stored on the diagonal of the adjacency matrix
    return float(sum(graph[n, n] for n in node_list))

@register_atom(witness_to_qubo)
@icontract.require(lambda graph: graph is not None, "graph cannot be None")
@icontract.require(lambda penalty: penalty is not None, "penalty cannot be None")
@icontract.ensure(lambda result: result is not None, "To Qubo output must not be None")
def to_qubo(graph: np.ndarray, penalty: float) -> np.ndarray:
    """Convert a Maximum Weight Independent Set (MWIS) problem to a Quadratic Unconstrained Binary Optimization (QUBO) matrix. Encodes node weights as diagonal entries and edge penalties as off-diagonal entries so the problem can be solved by a binary optimizer.

    Args:
        graph: adjacency matrix of the input graph
        penalty: edge penalty factor; must be >= 2 * max(node weight)

    Returns:
        QUBO matrix encoding the MWIS objective
    """
    max_weight = float(np.max(np.diag(graph))) if graph.size else 0.0
    if penalty < 2.0 * max_weight:
        raise ValueError("penalty must be at least 2 * max(node weight)")

    n = graph.shape[0]
    Q = np.zeros((n, n), dtype=float)
    # Diagonal: negative node weights (we minimize, so negate for maximization)
    for i in range(n):
        Q[i, i] = -graph[i, i]
    # Off-diagonal: penalty for edges
    for i in range(n):
        for j in range(i + 1, n):
            if graph[i, j] != 0:
                Q[i, j] = penalty
                Q[j, i] = penalty
    return Q
