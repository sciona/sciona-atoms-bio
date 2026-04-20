from __future__ import annotations

import networkx as nx
import numpy as np
import pytest

from sciona.atoms.bio.molecular_docking.greedy_mapping_d12 import (
    construct_mapping_state_via_greedy_expansion,
    init_problem_context,
    orchestrate_generation_and_validate,
)
from sciona.atoms.bio.molecular_docking.greedy_subgraph import greedy_maximum_subgraph
from sciona.atoms.bio.molecular_docking.map_to_udg import graphtoudgmapping


class LatticeInstance:
    def __init__(self, lattice: nx.Graph) -> None:
        self.lattice = lattice
        self.avg_degree = int(sum(dict(lattice.degree()).values()) / lattice.number_of_nodes())


def _path_context() -> dict[str, object]:
    graph = nx.path_graph(["A", "B", "C"])
    lattice = nx.path_graph([0, 1, 2, 3])
    return init_problem_context(graph, LatticeInstance(lattice), (), seed=11)


def test_d12_greedy_expansion_uses_lattice_center_frontier_without_mutating_input() -> None:
    context = _path_context()
    input_state: dict[str, object] = {"mapping": {}, "unmapping": {}, "unexpanded_nodes": set()}

    state, scored_nodes = construct_mapping_state_via_greedy_expansion(
        context,
        starting_node="B",
        mapping_state_in=input_state,
        considered_nodes=["A", "C"],
        remove_invalid_placement_nodes=True,
        rank_nodes=False,
    )

    assert input_state == {"mapping": {}, "unmapping": {}, "unexpanded_nodes": set()}
    assert state["mapping"] == {"B": 2, "A": 1, "C": 3}
    assert state["unmapping"] == {2: "B", 1: "A", 3: "C"}
    assert state["unexpanded_nodes"] == {"B", "A", "C"}
    assert [item["node"] for item in scored_nodes] == ["A", "C"]


def test_d12_orchestrator_drives_generation_and_detects_lattice_extra_edges() -> None:
    context = _path_context()

    state, is_valid = orchestrate_generation_and_validate(
        context,
        starting_node="B",
        remove_invalid_placement_nodes=True,
        rank_nodes=False,
        mapping_state={},
    )

    assert is_valid is True
    assert state["mapping"] == {"B": 2, "A": 1, "C": 3}

    graph = nx.path_graph(["A", "B", "C"])
    lattice = nx.complete_graph([0, 1, 2])
    invalid_context = init_problem_context(graph, LatticeInstance(lattice), (), seed=0)
    _, invalid = orchestrate_generation_and_validate(
        invalid_context,
        starting_node="B",
        remove_invalid_placement_nodes=True,
        rank_nodes=False,
        mapping_state={
            "mapping": {"B": 1, "A": 0, "C": 2},
            "unmapping": {1: "B", 0: "A", 2: "C"},
            "unexpanded_nodes": set(),
        },
    )

    assert invalid is False


def test_greedy_maximum_subgraph_returns_source_aligned_independent_set_mask() -> None:
    adjacency = np.array(
        [
            [0, 1, 0],
            [1, 0, 1],
            [0, 1, 0],
        ],
        dtype=int,
    )
    scores = np.array([4.0, 5.0, 4.0])

    selected = greedy_maximum_subgraph(adjacency, scores)

    assert selected.tolist() == [True, False, True]
    selected_indices = np.flatnonzero(selected)
    assert not np.any(adjacency[np.ix_(selected_indices, selected_indices)])


def test_graph_to_udg_certifies_embedding_without_adding_edges() -> None:
    graph = nx.Graph()
    graph.add_node("A", pos=(0.0, 0.0))
    graph.add_node("B", pos=(2.0, 0.0))
    graph.add_node("C", pos=(4.0, 0.0))
    graph.add_edges_from([("A", "B"), ("B", "C")])

    mapped = graphtoudgmapping(graph)

    assert set(map(frozenset, mapped.edges())) == {frozenset(("A", "B")), frozenset(("B", "C"))}
    assert mapped.graph["udg_certified"] is True
    assert mapped.graph["udg_radius"] == 1.0
    assert np.linalg.norm(mapped.nodes["A"]["pos"] - mapped.nodes["B"]["pos"]) == pytest.approx(1.0)
    assert np.linalg.norm(mapped.nodes["A"]["pos"] - mapped.nodes["C"]["pos"]) > 1.0


def test_graph_to_udg_rejects_positions_that_create_missing_unit_disk_edges() -> None:
    graph = nx.Graph()
    graph.add_node("A", pos=(0.0, 0.0))
    graph.add_node("B", pos=(0.5, 0.0))

    with pytest.raises(ValueError, match="would add a unit-disk edge"):
        graphtoudgmapping(graph)
