# Molecular Docking Pubrev-005 Holds

Batch: `pubrev-005`
Repo owner: `sciona-atoms-bio`
Scope: `wave_1_audit_completion`

The following atoms were not advanced to `ready_for_manifest_merge` in this
audit. They are callable, but the implementation is a placeholder, an
over-broad classical stand-in, or drifts from the published atom semantics.

## Held Atoms

- `sciona.atoms.bio.molecular_docking.greedy_mapping_d12.construct_mapping_state_via_greedy_expansion`
- `sciona.atoms.bio.molecular_docking.greedy_mapping_d12.orchestrate_generation_and_validate`
- `sciona.atoms.bio.molecular_docking.greedy_subgraph.greedy_maximum_subgraph`
- `sciona.atoms.bio.molecular_docking.map_to_udg.graphtoudgmapping`
- `sciona.atoms.bio.molecular_docking.quantum_solver.adiabaticquantumsampler`
- `sciona.atoms.bio.molecular_docking.quantum_solver.quantumproblemdefinition`
- `sciona.atoms.bio.molecular_docking.quantum_solver.solutionextraction`

## Required Remediation

- For `greedy_mapping_d12.construct_mapping_state_via_greedy_expansion`, replace
  the simplified first-free-site placement with the intended D12/lattice
  placement semantics, including explicit feasibility checks, deterministic
  seed usage, and preservation of the advertised immutable mapping-state
  contract.
- For `greedy_mapping_d12.orchestrate_generation_and_validate`, implement the
  advertised orchestration path. The current wrapper validates a supplied
  mapping state but does not drive iterative generation from `starting_node` or
  invoke expansion stages.
- For `greedy_subgraph.greedy_maximum_subgraph`, reconcile the atom contract
  with the implementation. The current code greedily selects a high-score
  independent set by treating adjacency as conflicts; it does not select a
  connected maximum-weight subgraph as described.
- For `map_to_udg.graphtoudgmapping`, return a semantically explicit UDG
  embedding/mapping and prove edge preservation. The current spectral-layout
  heuristic can add edges and does not establish that the input graph was
  faithfully mapped to a unit-disk graph representation.
- For `quantum_solver.*`, replace the classical placeholder path with the
  intended quantum/adiabatic solver semantics or rename and re-document the
  atoms as classical approximations. The current sampler uses deterministic
  classical greedy/SA logic and the problem definition does not build the
  advertised Hamiltonian, pulse, or backend-specific simulation objects.

# Molecular Docking Pubrev-028 Holds

Batch: `pubrev-028`
Repo owner: `sciona-atoms-bio`
Scope: `sciona.atoms.bio.molecular_docking.quantum_solver_d12`

The following atoms were directly audited against the current implementation,
`matches.json`, `cdg.json`, and references evidence. None were advanced to
`ready_for_manifest_merge`; the issues are semantic drift and insufficient
source-alignment evidence, not just the previously recorded uncertainty gap.

## Held Atoms

- `sciona.atoms.bio.molecular_docking.quantum_solver_d12.adiabaticpulseassembler`
- `sciona.atoms.bio.molecular_docking.quantum_solver_d12.interactionboundscomputer`
- `sciona.atoms.bio.molecular_docking.quantum_solver_d12.quantumcircuitsampler`
- `sciona.atoms.bio.molecular_docking.quantum_solver_d12.quantumsolutionextractor`
- `sciona.atoms.bio.molecular_docking.quantum_solver_d12.quantumsolverorchestrator`

## Required Remediation

- For `quantum_solver_d12.adiabaticpulseassembler`, replace the dict placeholder
  with the source-aligned Pulser sequence construction path: detuning map,
  amplitude/detuning waveforms, pulse object, Rydberg/DMM channel declarations,
  DMM detuning waveform, and locked sequence.
- For `quantum_solver_d12.interactionboundscomputer`, reconcile the atom
  contract with the source evidence. The current function returns raw
  `u_min`/`u_max` values and clamps them, while the evidence describes deriving
  pulse parameter bounds such as `detuning_maximum` and `amplitude_maximum`
  from graph/register geometry.
- For `quantum_solver_d12.quantumcircuitsampler`, replace the deterministic
  classical greedy/simulated-annealing stand-in with the advertised backend
  execution path for qutip, tensor-network/MPS, or state-vector sampling,
  including permutation restoration where required.
- For `quantum_solver_d12.quantumsolutionextractor`, provide source-aligned
  solution extraction evidence and tests. The current ranking helper decodes
  bitstrings into lists, but the evidence describes node-set solution objects
  tied to Pulser register/qubit mapping.
- For `quantum_solver_d12.quantumsolverorchestrator`, implement the end-to-end
  neutral-atom MWIS pipeline described by the evidence: node/weight extraction,
  register coordinate assembly, bandwidth optimization, pulse parameter
  packaging, adiabatic backend sampling, and solution extraction. The current
  function bypasses those stages and directly runs a classical greedy/SA loop.
