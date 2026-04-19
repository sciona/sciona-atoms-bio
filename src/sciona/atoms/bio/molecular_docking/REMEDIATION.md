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
