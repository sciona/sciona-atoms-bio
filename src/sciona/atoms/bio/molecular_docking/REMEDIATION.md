# Molecular Docking Pubrev-005 Remediation

Batch: `pubrev-005`
Repo owner: `sciona-atoms-bio`
Scope: `wave_1_audit_completion`

The original audit held classical and quantum rows that were callable but
over-broad, placeholder-like, or missing source-aligned backend evidence.

## Remediated Classical Atoms

- `sciona.atoms.bio.molecular_docking.greedy_mapping_d12.construct_mapping_state_via_greedy_expansion`
- `sciona.atoms.bio.molecular_docking.greedy_mapping_d12.orchestrate_generation_and_validate`
- `sciona.atoms.bio.molecular_docking.greedy_subgraph.greedy_maximum_subgraph`
- `sciona.atoms.bio.molecular_docking.map_to_udg.graphtoudgmapping`

These rows were advanced in a classical remediation follow-up. The D12 mapping
atoms now use center-biased lattice seeding, frontier-neighbor placement,
explicit feasibility checks, deterministic tie-breaking, and immutable state
copies. `greedy_maximum_subgraph` has been narrowed to the source-aligned
greedy weighted independent-set contract. `graphtoudgmapping` now certifies an
edge-preserving unit-disk embedding instead of adding spectral-layout edges.

## Remediated Quantum Solver Atoms

- `sciona.atoms.bio.molecular_docking.quantum_solver.adiabaticquantumsampler`
- `sciona.atoms.bio.molecular_docking.quantum_solver.quantumproblemdefinition`
- `sciona.atoms.bio.molecular_docking.quantum_solver.solutionextraction`

These rows now delegate to the optional Pulser/emulator backend lane documented
in `docs/quantum_optional_dependencies.md`. Base imports remain lightweight;
runtime execution of the quantum backend requires installing the `quantum`
optional dependency group.

# Molecular Docking Pubrev-028 Remediation

Batch: `pubrev-028`
Repo owner: `sciona-atoms-bio`
Scope: `sciona.atoms.bio.molecular_docking.quantum_solver_d12`

## Remediated Quantum Solver D12 Atoms

- `sciona.atoms.bio.molecular_docking.quantum_solver_d12.adiabaticpulseassembler`
- `sciona.atoms.bio.molecular_docking.quantum_solver_d12.interactionboundscomputer`
- `sciona.atoms.bio.molecular_docking.quantum_solver_d12.quantumcircuitsampler`
- `sciona.atoms.bio.molecular_docking.quantum_solver_d12.quantumsolutionextractor`
- `sciona.atoms.bio.molecular_docking.quantum_solver_d12.quantumsolverorchestrator`

These rows now build Pulser registers, derive interaction bounds from register
geometry, assemble Rydberg/DMM adiabatic sequences, execute the requested
emulator backend, and decode measured bitstrings through the shared optional
backend helper. Focused tests run against the installed Pulser state-vector
backend and skip cleanly when the optional dependency group is absent.
