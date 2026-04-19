from __future__ import annotations

import networkx as nx
import numpy as np

from sciona.atoms.bio.molecular_docking.add_quantum_link import addquantumlink
from sciona.atoms.bio.molecular_docking.build_complementary import (
    constructcomplementarygraph,
)
from sciona.atoms.bio.molecular_docking.build_interaction_graph import (
    networkx_weighted_graph_materialization,
    pair_distance_compatibility_check,
    weighted_interaction_edge_derivation,
)
from sciona.atoms.bio.molecular_docking.greedy_mapping_d12 import init_problem_context


def test_pubrev005_add_quantum_link_materializes_chain_without_mutating_input() -> None:
    graph = nx.Graph()
    graph.add_nodes_from(["A", "B"])

    linked = addquantumlink(graph, "A", "B", chain_size=3)

    assert not graph.has_edge("A", "B")
    assert set(map(frozenset, linked.edges())) == {
        frozenset(("A", "_qlink_A_B_0")),
        frozenset(("_qlink_A_B_0", "_qlink_A_B_1")),
        frozenset(("_qlink_A_B_1", "B")),
    }


def test_pubrev005_construct_complementary_graph_matches_networkx_complement() -> None:
    graph = nx.Graph()
    graph.add_nodes_from([1, 2, 3])
    graph.add_edge(1, 2)

    complement = constructcomplementarygraph(graph)

    assert set(complement.nodes()) == {1, 2, 3}
    assert set(map(frozenset, complement.edges())) == {
        frozenset((1, 3)),
        frozenset((2, 3)),
    }


def test_pubrev005_pair_distance_compatibility_uses_expanded_interval() -> None:
    assert pair_distance_compatibility_check([2.0, 4.0], np.array([4.25]), 0.5)
    assert not pair_distance_compatibility_check([2.0, 4.0], np.array([4.75]), 0.5)
    assert not pair_distance_compatibility_check([], np.array([3.0]), 0.5)


def test_pubrev005_weighted_interaction_edge_derivation_returns_layouts() -> None:
    layouts = weighted_interaction_edge_derivation(("L1", "L2"), ("R1", "R2"))

    assert layouts == [
        (("L1", "R2"), ("L2", "R1")),
        (("L2", "R2"), ("L1", "R1")),
    ]


def test_pubrev005_networkx_weighted_graph_materialization_preserves_weights() -> None:
    graph = networkx_weighted_graph_materialization(
        [("a", "b", 2.5)],
        {"a", "b", "isolated"},
    )

    assert set(graph.nodes()) == {"a", "b", "isolated"}
    assert graph["a"]["b"]["weight"] == 2.5
    assert graph.degree("isolated") == 0


def test_pubrev005_init_problem_context_captures_lattice_context() -> None:
    class LatticeInstance:
        lattice = ("site-0", "site-1")

    graph = nx.path_graph(["n0", "n1"])
    context = init_problem_context(graph, LatticeInstance(), ("prior",), seed=17)

    assert context == {
        "graph": graph,
        "lattice": ("site-0", "site-1"),
        "lattice_instance": context["lattice_instance"],
        "previously_generated_subgraphs": ["prior"],
        "seed": 17,
    }
    assert isinstance(context["previously_generated_subgraphs"], list)
