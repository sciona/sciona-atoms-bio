from __future__ import annotations

from typing import TYPE_CHECKING

import icontract

if TYPE_CHECKING:
    import networkx as nx
from ageoa.ghost.registry import register_atom
from .build_complementary_witnesses import witness_constructcomplementarygraph

@register_atom(witness_constructcomplementarygraph)  # type: ignore[untyped-decorator]
@icontract.require(lambda graph: graph is not None, "graph cannot be None")
@icontract.ensure(lambda result: result is not None, "ConstructComplementaryGraph output must not be None")
def constructcomplementarygraph(graph: nx.Graph) -> nx.Graph:
    """Builds the complementary graph by deriving the inverse edge set relative to the input graph's node set.

    Args:
        graph: Input data.

    Returns:
        Result data.
    """
    import networkx as nx
    return nx.complement(graph)
