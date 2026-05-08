from __future__ import annotations
"""Auto-generated atom wrappers following the sciona pattern."""
from typing import Any, Callable, Dict, List, Set, Collection, cast
Map = Dict


import numpy as np

import icontract
from sciona.ghost.registry import register_atom
from .witnesses import witness_assemblestaticmappingcontext, witness_initializefrontierfromstartnode, witness_rungreedymappingpipeline, witness_scoreandextendgreedycandidates, witness_validatecurrentmapping

# Generated type placeholders.
Graph = Any
LatticeInstance = Any
Subgraph = Any
MappingContext = Any
NodeId = Any
GraphNode = Any
LatticeNode = Any
MappingState = Any

@register_atom(witness_assemblestaticmappingcontext)
@icontract.require(lambda graph: graph is not None, "graph cannot be None")
@icontract.require(lambda lattice_instance: lattice_instance is not None, "lattice_instance cannot be None")
@icontract.require(lambda previously_generated_subgraphs: previously_generated_subgraphs is not None, "previously_generated_subgraphs cannot be None")
@icontract.require(lambda seed: seed is not None, "seed cannot be None")
@icontract.ensure(lambda result: result is not None, "AssembleStaticMappingContext output must not be None")
def assemble_static_mapping_context(graph: Graph, lattice_instance: LatticeInstance, previously_generated_subgraphs: Collection[Subgraph], seed: int|None) -> MappingContext:
    """Construct immutable algorithm context from constructor inputs so all later stages consume explicit state instead of hidden class fields.

    Args:
        graph: must be a valid source graph
        lattice_instance: must expose lattice topology
        previously_generated_subgraphs: used for reuse/avoidance scoring
        seed: deterministic tie-breaking if provided

    Returns:
        contains graph, lattice, lattice_instance, previously_generated_subgraphs, seed as immutable fields
    """
    lattice = getattr(lattice_instance, 'lattice', lattice_instance)
    return {
        'graph': graph,
        'lattice': lattice,
        'lattice_instance': lattice_instance,
        'previously_generated_subgraphs': list(previously_generated_subgraphs),
        'seed': seed,
    }

@register_atom(witness_initializefrontierfromstartnode)
@icontract.require(lambda mapping_context: mapping_context is not None, "mapping_context cannot be None")
@icontract.require(lambda starting_node: starting_node is not None, "starting_node cannot be None")
@icontract.require(lambda mapping: mapping is not None, "mapping cannot be None")
@icontract.require(lambda unmapping: unmapping is not None, "unmapping cannot be None")
@icontract.require(lambda unexpanded_nodes: unexpanded_nodes is not None, "unexpanded_nodes cannot be None")
@icontract.ensure(lambda result: result is not None, "InitializeFrontierFromStartNode output must not be None")
def initialize_frontier_from_start_node(mapping_context: MappingContext, starting_node: NodeId, mapping: Map[GraphNode,LatticeNode], unmapping: Map[LatticeNode,GraphNode], unexpanded_nodes: Set[GraphNode]) -> MappingState:
    """Create initial mapping/unmapping frontier state from a selected starting node.

    Args:
        mapping_context: requires lattice in context
        starting_node: must exist in graph
        mapping: input state, may be empty
        unmapping: inverse-consistent with mapping
        unexpanded_nodes: frontier nodes pending expansion

    Returns:
        returns new mapping/unmapping/frontier state; no hidden mutation
    """
    new_mapping = dict(mapping)
    new_unmapping = dict(unmapping)
    new_unexpanded = set(unexpanded_nodes)

    # Assign starting node to first available lattice node
    lattice = mapping_context.get('lattice', mapping_context.get('lattice_instance'))
    used_lattice_nodes = set(new_mapping.values())
    if hasattr(lattice, 'nodes'):
        available = [n for n in lattice.nodes() if n not in used_lattice_nodes]
    elif hasattr(lattice, '__iter__'):
        available = [n for n in lattice if n not in used_lattice_nodes]
    else:
        available = [i for i in range(1000) if i not in used_lattice_nodes]

    if available and starting_node not in new_mapping:
        lattice_node = available[0]
        new_mapping[starting_node] = lattice_node
        new_unmapping[lattice_node] = starting_node
        new_unexpanded.add(starting_node)

    return {
        'mapping': new_mapping,
        'unmapping': new_unmapping,
        'unexpanded_nodes': new_unexpanded,
    }

@register_atom(witness_scoreandextendgreedycandidates)
@icontract.require(lambda mapping_context: mapping_context is not None, "mapping_context cannot be None")
@icontract.require(lambda considered_nodes: considered_nodes is not None, "considered_nodes cannot be None")
@icontract.require(lambda unexpanded_nodes: unexpanded_nodes is not None, "unexpanded_nodes cannot be None")
@icontract.require(lambda free_lattice_neighbors: free_lattice_neighbors is not None, "free_lattice_neighbors cannot be None")
@icontract.require(lambda mapping: mapping is not None, "mapping cannot be None")
@icontract.require(lambda unmapping: unmapping is not None, "unmapping cannot be None")
@icontract.require(lambda remove_invalid_placement_nodes: remove_invalid_placement_nodes is not None, "remove_invalid_placement_nodes cannot be None")
@icontract.require(lambda rank_nodes: rank_nodes is not None, "rank_nodes cannot be None")
@icontract.ensure(lambda result: all(r is not None for r in result), "ScoreAndExtendGreedyCandidates all outputs must not be None")
def score_and_extend_greedy_candidates(mapping_context: MappingContext, considered_nodes: List[GraphNode], unexpanded_nodes: Set[GraphNode], free_lattice_neighbors: Set[LatticeNode], mapping: Map[GraphNode,LatticeNode], unmapping: Map[LatticeNode,GraphNode], remove_invalid_placement_nodes: bool = True, rank_nodes: bool = True) -> tuple[MappingState, Map[GraphNode,float]]:
    """Score candidate placements and greedily extend the mapping frontier using ranking and validity filtering flags.

    Args:
        mapping_context: uses graph, lattice, lattice_instance, previously_generated_subgraphs
        considered_nodes: nodes currently considered for placement
        unexpanded_nodes: current frontier
        free_lattice_neighbors: available placement sites
        mapping: current partial mapping
        unmapping: inverse map
        remove_invalid_placement_nodes: if true, prune invalid placements early
        rank_nodes: if true, apply scoring-based ordering

    Returns:
        extended_mapping_state: new mapping/unmapping/frontier after greedy extension
        candidate_scores: deterministic from inputs
    """
    graph = mapping_context.get('graph')
    new_mapping = dict(mapping)
    new_unmapping = dict(unmapping)
    new_unexpanded = set(unexpanded_nodes)
    candidate_scores = {}

    # Score each candidate node based on neighbor connectivity
    for node in considered_nodes:
        if node in new_mapping:
            continue
        # Score: count how many of node's neighbors are already mapped
        if hasattr(graph, 'neighbors'):
            neighbors = list(graph.neighbors(node))
        elif hasattr(graph, 'adj'):
            neighbors = list(graph.adj.get(node, []))
        else:
            neighbors = []
        score = sum(1 for nb in neighbors if nb in new_mapping)
        candidate_scores[node] = float(score)

    # Filter invalid placements if requested
    valid_candidates = list(candidate_scores.keys())
    if remove_invalid_placement_nodes:
        valid_candidates = [n for n in valid_candidates if candidate_scores[n] > 0 or len(new_mapping) == 0]

    # Sort by score if ranking requested
    if rank_nodes:
        valid_candidates.sort(key=lambda n: -candidate_scores.get(n, 0))

    # Greedily extend: assign candidates to free lattice neighbors
    free_sites = list(free_lattice_neighbors)
    for node in valid_candidates:
        if not free_sites:
            break
        if node in new_mapping:
            continue
        site = free_sites.pop(0)
        new_mapping[node] = site
        new_unmapping[site] = node
        new_unexpanded.add(node)

    extended_state = {
        'mapping': new_mapping,
        'unmapping': new_unmapping,
        'unexpanded_nodes': new_unexpanded,
    }
    return (extended_state, candidate_scores)

@register_atom(witness_validatecurrentmapping)
@icontract.require(lambda mapping_context: mapping_context is not None, "mapping_context cannot be None")
@icontract.require(lambda mapping: mapping is not None, "mapping cannot be None")
@icontract.require(lambda unmapping: unmapping is not None, "unmapping cannot be None")
@icontract.ensure(lambda result: result is not None, "ValidateCurrentMapping output must not be None")
def validate_current_mapping(mapping_context: MappingContext, mapping: Map[GraphNode,LatticeNode], unmapping: Map[LatticeNode,GraphNode]) -> bool:
    """Evaluate whether the current mapping/unmapping pair satisfies graph-lattice consistency constraints.

    Args:
        mapping_context: uses graph and lattice constraints
        mapping: candidate mapping
        unmapping: must correspond to mapping

    Returns:
        true iff all structural constraints hold
    """
    # Check bijectivity: mapping and unmapping must be consistent inverses
    for g_node, l_node in mapping.items():
        if l_node not in unmapping or unmapping[l_node] != g_node:
            return False
    for l_node, g_node in unmapping.items():
        if g_node not in mapping or mapping[g_node] != l_node:
            return False

    # Check edge preservation: every edge in the graph between mapped nodes
    # must correspond to an edge (adjacency) in the lattice
    graph = mapping_context.get('graph')
    lattice = mapping_context.get('lattice', mapping_context.get('lattice_instance'))

    if hasattr(graph, 'edges'):
        for u, v in graph.edges():
            if u in mapping and v in mapping:
                lu, lv = mapping[u], mapping[v]
                if hasattr(lattice, 'has_edge'):
                    if not lattice.has_edge(lu, lv):
                        return False
    return True

@register_atom(witness_rungreedymappingpipeline)
@icontract.require(lambda mapping_context: mapping_context is not None, "mapping_context cannot be None")
@icontract.require(lambda starting_node: starting_node is not None, "starting_node cannot be None")
@icontract.require(lambda remove_invalid_placement_nodes: remove_invalid_placement_nodes is not None, "remove_invalid_placement_nodes cannot be None")
@icontract.require(lambda rank_nodes: rank_nodes is not None, "rank_nodes cannot be None")
@icontract.require(lambda initialized_mapping_state: initialized_mapping_state is not None, "initialized_mapping_state cannot be None")
@icontract.require(lambda extended_mapping_state: extended_mapping_state is not None, "extended_mapping_state cannot be None")
@icontract.require(lambda mapping_is_valid: mapping_is_valid is not None, "mapping_is_valid cannot be None")
@icontract.ensure(lambda result: all(r is not None for r in result), "RunGreedyMappingPipeline all outputs must not be None")
def run_greedy_mapping_pipeline(mapping_context: MappingContext, starting_node: NodeId, remove_invalid_placement_nodes: bool, rank_nodes: bool, initialized_mapping_state: MappingState, extended_mapping_state: MappingState, mapping_is_valid: bool) -> tuple[Subgraph, MappingState]:
    """Orchestrate initialization, greedy extension, and validity checking to produce a greedy Unit Disk (UD) subgraph rooted at the starting node.

    Args:
        mapping_context: immutable shared context
        starting_node: seed node for generation
        remove_invalid_placement_nodes: forwarded to extension stage
        rank_nodes: forwarded to extension stage
        initialized_mapping_state: from initialization atom
        extended_mapping_state: from greedy extension atom
        mapping_is_valid: from validation atom

    Returns:
        generated_subgraph: greedy-generated UD subgraph
        final_mapping_state: final immutable mapping/unmapping/frontier snapshot
    """
    final_state = extended_mapping_state if mapping_is_valid else initialized_mapping_state
    final_mapping = final_state.get('mapping', {}) if isinstance(final_state, dict) else final_state

    # Build the generated subgraph from the final mapping
    graph = mapping_context.get('graph')
    mapped_nodes = set(final_mapping.keys()) if isinstance(final_mapping, dict) else set()

    if hasattr(graph, 'subgraph'):
        generated_subgraph = graph.subgraph(mapped_nodes).copy()
    else:
        generated_subgraph = {'nodes': list(mapped_nodes), 'mapping': dict(final_mapping) if isinstance(final_mapping, dict) else {}}

    return (generated_subgraph, final_state)
