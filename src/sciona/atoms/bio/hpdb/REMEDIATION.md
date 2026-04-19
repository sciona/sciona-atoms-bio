# HPDB Pubrev-064 Holds

Batch: `pubrev-064`
Repo owner: `sciona-atoms-bio`
Scope: `sciona.atoms.bio.hpdb.__remainder__`

The following atoms were directly audited against the current implementation,
`references.json`, `cdg.json`, and `matches.json`. Neither row was advanced to
`ready_for_manifest_merge`; both implementations are deferred no-op iterators
that return no data while the public names and descriptions promise Protein
Data Bank traversal.

## Held Atoms

- `sciona.atoms.bio.hpdb.iterate_pdb_atoms`
- `sciona.atoms.bio.hpdb.iterate_pdb_residues`

## Required Remediation

- For `iterate_pdb_atoms`, replace the `iter([])` stub with source-aligned PDB
  atom traversal over an explicit PDB structure, file, parser, repository
  handle, or dataset input. Define and test exact element-filter behavior.
- For `iterate_pdb_residues`, replace the `iter([])` stub with source-aligned
  residue traversal and residue-level element filtering over an explicit PDB
  source input.
- Reconcile the parameter metadata before publication. The functions annotate
  `element` as `Optional[str]`, but the icontract precondition rejects `None`
  and the CDG currently marks the parameter as required while describing the
  filter as optional.
- Reconcile the descriptions before publication. Current docstrings/CDG text
  promise iteration over PDB structures, but no PDB source can be supplied and
  no atom or residue objects are produced.
