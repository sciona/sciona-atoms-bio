from __future__ import annotations
from sciona.ghost.abstract import AbstractArray, AbstractDistribution, AbstractMCMCTrace, AbstractRNGState, AbstractScalar, AbstractSignal


def witness_quantumsolverorchestrator(graph: AbstractArray, coordinates_layout: AbstractArray, num_sol: AbstractArray, display_info: AbstractArray) -> AbstractArray:
    """Shape-and-type check for quantum solver orchestrator. Returns output metadata without running the real computation."""
    result = AbstractArray(
        shape=graph.shape,
        dtype="float64",)
    
    return result

def witness_interactionboundscomputer(register_coord: AbstractArray, graph: AbstractArray) -> AbstractArray:
    """Shape-and-type check for interaction bounds computer. Returns output metadata without running the real computation."""
    result = AbstractArray(
        shape=register_coord.shape,
        dtype="float64",)
    
    return result

def witness_adiabaticpulseassembler(register: AbstractArray, parameters: AbstractArray) -> AbstractArray:
    """Shape-and-type check for adiabatic pulse assembler. Returns output metadata without running the real computation."""
    result = AbstractArray(
        shape=register.shape,
        dtype="float64",)
    
    return result

def witness_quantumcircuitsampler(trace: AbstractMCMCTrace, target: AbstractDistribution, rng: AbstractRNGState) -> tuple[AbstractMCMCTrace, AbstractRNGState]:
    """Shape-and-type check for mcmc sampler: quantum circuit sampler. Returns output metadata without running the real computation."""
    if trace.param_dims != target.event_shape:
        raise ValueError(
            f"param_dims {trace.param_dims} vs event_shape {target.event_shape}"
        )
        
    return trace.step(accepted=True), rng.advance(n_draws=1)

def witness_quantumsolutionextractor(count_dist: AbstractArray, register: AbstractArray, num_solutions: AbstractArray) -> AbstractArray:
    """Shape-and-type check for quantum solution extractor. Returns output metadata without running the real computation."""
    result = AbstractArray(
        shape=count_dist.shape,
        dtype="float64",)
    
    return result