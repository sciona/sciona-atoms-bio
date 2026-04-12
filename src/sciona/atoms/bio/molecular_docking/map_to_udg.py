from __future__ import annotations

from typing import TYPE_CHECKING

import icontract

if TYPE_CHECKING:
    import networkx as nx
from ageoa.ghost.registry import register_atom
from .map_to_udg_witnesses import witness_graphtoudgmapping

@register_atom(witness_graphtoudgmapping)  # type: ignore[untyped-decorator]
@icontract.require(lambda G: G is not None, "Input graph G cannot be None")
@icontract.ensure(lambda result: result is not None, "GraphToUDGMapping output must not be None")
def graphtoudgmapping(G: nx.Graph) -> nx.Graph:
    """Map a graph to a UDG mapping.

    Args:
        G: Must be a valid graph object accepted by map_to_UDG.

    Returns:
        New mapped graph output; no hidden state mutation.
    """
    import networkx as nx
    import numpy as np
    # Use spectral layout to embed graph nodes in 2D, then scale so that
    # edges correspond to pairs within unit distance.
    if G.number_of_nodes() == 0:
        return G.copy()

    pos = nx.spectral_layout(G, dim=2)
    if len(pos) < 2:
        pos = {n: np.zeros(2) for n in G.nodes()}

    # Scale positions so that the maximum edge length equals 1.0
    max_edge_dist = 0.0
    for u, v in G.edges():
        d = np.linalg.norm(np.array(pos[u]) - np.array(pos[v]))
        if d > max_edge_dist:
            max_edge_dist = d
    if max_edge_dist > 0:
        scale = 1.0 / max_edge_dist
        pos = {n: np.array(c) * scale for n, c in pos.items()}

    H = nx.Graph()
    for n in G.nodes(data=True):
        H.add_node(n[0], **n[1], pos=pos[n[0]])
    # Add edges for all pairs within unit distance
    nodes = list(H.nodes())
    for i in range(len(nodes)):
        for j in range(i + 1, len(nodes)):
            d = np.linalg.norm(np.array(pos[nodes[i]]) - np.array(pos[nodes[j]]))
            if d <= 1.0 + 1e-9:
                H.add_edge(nodes[i], nodes[j])
    return H
