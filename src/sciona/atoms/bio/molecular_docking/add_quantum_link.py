from __future__ import annotations

from typing import Any
Graph: Any = Any
Node: Any = Any

"""Auto-generated atom wrappers following the sciona pattern."""


import numpy as np

import icontract
from ageoa.ghost.registry import register_atom
from .add_quantum_link_witnesses import witness_addquantumlink

# Witness functions should be imported from the generated witnesses module

@register_atom(witness_addquantumlink)
@icontract.require(lambda G: G is not None, "G cannot be None")
@icontract.require(lambda node_A: node_A is not None, "node_A cannot be None")
@icontract.require(lambda node_B: node_B is not None, "node_B cannot be None")
@icontract.require(lambda chain_size: chain_size is not None, "chain_size cannot be None")
@icontract.ensure(lambda result: result is not None, "AddQuantumLink output must not be None")
def addquantumlink(G: Graph, node_A: Node, node_B: Node, chain_size: int) -> Graph:
    """Adds a specialized 'quantum link' between two nodes in a graph, potentially creating a chain of intermediate nodes based on chain_size.

    Args:
        G: The input graph to be modified.
        node_A: The first node to connect.
        node_B: The second node to connect.
        chain_size: Parameter defining the size or structure of the link.

    Returns:
        The graph with the added quantum link.
    """
    import networkx as nx
    H = G.copy()
    if chain_size <= 1:
        H.add_edge(node_A, node_B)
    else:
        chain_nodes = [f"_qlink_{node_A}_{node_B}_{i}" for i in range(chain_size - 1)]
        path = [node_A] + chain_nodes + [node_B]
        for u, v in zip(path[:-1], path[1:]):
            H.add_edge(u, v)
    return H
