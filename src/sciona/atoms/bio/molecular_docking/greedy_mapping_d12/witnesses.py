from __future__ import annotations
from sciona.ghost.abstract import AbstractArray, AbstractDistribution, AbstractScalar, AbstractSignal


def witness_init_problem_context(graph: AbstractArray, lattice_instance: AbstractArray, previously_generated_subgraphs: AbstractArray, seed: AbstractArray) -> AbstractArray:
    """Shape-and-type check for init problem context. Returns output metadata without running the real computation."""
    result = AbstractArray(
        shape=graph.shape,
        dtype="float64",)
    
    return result

def witness_construct_mapping_state_via_greedy_expansion(problem_context: AbstractArray, starting_node: AbstractArray, mapping_state_in: AbstractArray, considered_nodes: AbstractArray, remove_invalid_placement_nodes: AbstractArray, rank_nodes: AbstractArray) -> AbstractArray:
    """Shape-and-type check for construct mapping state via greedy expansion. Returns output metadata without running the real computation."""
    result = AbstractArray(
        shape=problem_context.shape,
        dtype="float64",)
    
    return result

def witness_orchestrate_generation_and_validate(problem_context: AbstractArray, starting_node: AbstractArray, remove_invalid_placement_nodes: AbstractArray, rank_nodes: AbstractArray, mapping_state: AbstractArray) -> AbstractArray:
    """Shape-and-type check for orchestrate generation and validate. Returns output metadata without running the real computation."""
    result = AbstractArray(
        shape=problem_context.shape,
        dtype="float64",)
    
    return result