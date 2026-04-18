from __future__ import annotations

from sciona.atoms.bio.molecular_docking.greedy_mapping.atoms import (
    assemblestaticmappingcontext,
    initializefrontierfromstartnode,
    rungreedymappingpipeline,
    scoreandextendgreedycandidates,
    validatecurrentmapping,
)


class SimpleGraph:
    def __init__(self, adjacency: dict[str, set[str]]) -> None:
        self.adj = {node: set(neighbors) for node, neighbors in adjacency.items()}

    def neighbors(self, node: str):
        return iter(self.adj.get(node, ()))

    def edges(self) -> list[tuple[str, str]]:
        edges: list[tuple[str, str]] = []
        seen: set[tuple[str, str]] = set()
        for left, neighbors in self.adj.items():
            for right in neighbors:
                edge = tuple(sorted((left, right)))
                if edge not in seen:
                    seen.add(edge)
                    edges.append((left, right))
        return edges

    def subgraph(self, nodes: set[str]) -> "SimpleGraph":
        subset = set(nodes)
        return SimpleGraph(
            {
                node: {neighbor for neighbor in self.adj.get(node, set()) if neighbor in subset}
                for node in subset
            }
        )

    def copy(self) -> "SimpleGraph":
        return SimpleGraph(self.adj)


class SimpleLattice:
    def __init__(self, adjacency: dict[str, set[str]]) -> None:
        self.adj = {node: set(neighbors) for node, neighbors in adjacency.items()}

    def nodes(self) -> list[str]:
        return list(self.adj)

    def has_edge(self, left: str, right: str) -> bool:
        return right in self.adj.get(left, set())


class LatticeInstance:
    def __init__(self, lattice: SimpleLattice) -> None:
        self.lattice = lattice


def _build_context():
    graph = SimpleGraph(
        {
            "A": {"B", "C"},
            "B": {"A"},
            "C": {"A"},
            "D": set(),
        }
    )
    lattice = SimpleLattice(
        {
            "L0": {"L1"},
            "L1": {"L0", "L2"},
            "L2": {"L1"},
        }
    )
    context = assemblestaticmappingcontext(
        graph=graph,
        lattice_instance=LatticeInstance(lattice),
        previously_generated_subgraphs=({"nodes": ["legacy"]},),
        seed=7,
    )
    return graph, lattice, context


def test_assemblestaticmappingcontext_preserves_explicit_inputs() -> None:
    graph, lattice, context = _build_context()

    assert context["graph"] is graph
    assert context["lattice"] is lattice
    assert context["previously_generated_subgraphs"] == [{"nodes": ["legacy"]}]
    assert context["seed"] == 7


def test_initializefrontierfromstartnode_assigns_first_open_lattice_node() -> None:
    _, _, context = _build_context()
    mapping: dict[str, str] = {}
    unmapping: dict[str, str] = {}
    unexpanded_nodes: set[str] = set()

    state = initializefrontierfromstartnode(context, "A", mapping, unmapping, unexpanded_nodes)

    assert state["mapping"] == {"A": "L0"}
    assert state["unmapping"] == {"L0": "A"}
    assert state["unexpanded_nodes"] == {"A"}
    assert mapping == {}
    assert unmapping == {}
    assert unexpanded_nodes == set()


def test_scoreandextendgreedycandidates_scores_neighbors_and_prunes_invalid_nodes() -> None:
    _, _, context = _build_context()

    state, candidate_scores = scoreandextendgreedycandidates(
        mapping_context=context,
        considered_nodes=["B", "C", "D"],
        unexpanded_nodes={"A"},
        free_lattice_neighbors={"L1", "L2"},
        mapping={"A": "L0"},
        unmapping={"L0": "A"},
        remove_invalid_placement_nodes=True,
        rank_nodes=True,
    )

    assert candidate_scores == {"B": 1.0, "C": 1.0, "D": 0.0}
    assert set(state["mapping"]) == {"A", "B", "C"}
    assert "D" not in state["mapping"]
    assert state["unexpanded_nodes"] == {"A", "B", "C"}
    assert len(set(state["mapping"].values())) == 3


def test_validatecurrentmapping_checks_inverse_consistency_and_lattice_edges() -> None:
    _, _, context = _build_context()

    assert validatecurrentmapping(
        context,
        mapping={"A": "L0", "B": "L1"},
        unmapping={"L0": "A", "L1": "B"},
    ) is True
    assert validatecurrentmapping(
        context,
        mapping={"A": "L0", "B": "L1"},
        unmapping={"L0": "A", "L1": "C"},
    ) is False
    assert validatecurrentmapping(
        context,
        mapping={"A": "L0", "C": "L2"},
        unmapping={"L0": "A", "L2": "C"},
    ) is False


def test_rungreedymappingpipeline_prefers_valid_extension_and_falls_back_on_invalid_state() -> None:
    _, _, context = _build_context()
    initialized_state = initializefrontierfromstartnode(context, "A", {}, {}, set())
    valid_extended_state, _ = scoreandextendgreedycandidates(
        mapping_context=context,
        considered_nodes=["B"],
        unexpanded_nodes=initialized_state["unexpanded_nodes"],
        free_lattice_neighbors={"L1"},
        mapping=initialized_state["mapping"],
        unmapping=initialized_state["unmapping"],
        remove_invalid_placement_nodes=True,
        rank_nodes=True,
    )

    generated_subgraph, final_state = rungreedymappingpipeline(
        mapping_context=context,
        starting_node="A",
        remove_invalid_placement_nodes=True,
        rank_nodes=True,
        initialized_mapping_state=initialized_state,
        extended_mapping_state=valid_extended_state,
        mapping_is_valid=True,
    )

    assert final_state == valid_extended_state
    assert set(generated_subgraph.adj) == {"A", "B"}

    fallback_subgraph, fallback_state = rungreedymappingpipeline(
        mapping_context=context,
        starting_node="A",
        remove_invalid_placement_nodes=True,
        rank_nodes=True,
        initialized_mapping_state=initialized_state,
        extended_mapping_state={"mapping": {"A": "L0", "C": "L2"}},
        mapping_is_valid=False,
    )

    assert fallback_state == initialized_state
    assert set(fallback_subgraph.adj) == {"A"}
