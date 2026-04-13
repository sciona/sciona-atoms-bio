from __future__ import annotations
from ageoa.ghost.abstract import AbstractArray, AbstractDistribution, AbstractMCMCTrace, AbstractRNGState, AbstractScalar, AbstractSignal

def witness_quantumproblemdefinition(family: object, event_shape: object, *args, **kwargs) -> AbstractDistribution:
    """Shape-and-type check for prior init: quantum problem definition. Returns output metadata without running the real computation."""
    return AbstractDistribution(
        family=family,
        event_shape=event_shape,)


def witness_adiabaticquantumsampler(trace: AbstractMCMCTrace, target: AbstractDistribution, rng: AbstractRNGState) -> tuple[AbstractMCMCTrace, AbstractRNGState]:
    """Shape-and-type check for mcmc sampler: adiabatic quantum sampler. Returns output metadata without running the real computation."""
    if trace.param_dims != target.event_shape:
        raise ValueError(
            f"param_dims {trace.param_dims} vs event_shape {target.event_shape}"
        )
    return trace.step(accepted=True), rng.advance(n_draws=1)

def witness_solutionextraction(measurement_counts: AbstractArray, final_register: AbstractArray, num_solutions: AbstractArray) -> AbstractArray:
    """Shape-and-type check for solution extraction. Returns output metadata without running the real computation."""
    result = AbstractArray(
        shape=measurement_counts.shape,
        dtype="float64",)

    return result
