# Molecular Docking Quantum Optional Dependencies

The `molecular_docking.quantum_solver` and `molecular_docking.quantum_solver_d12`
atoms build Pulser neutral-atom sequences derived from
`third_party/Molecular-Docking/src/solver/quantum_solver_molecular.py`.

Install the optional quantum stack before running the Pulser-backed atoms:

```bash
/Users/conrad/personal/sciona-matcher/.venv/bin/python -m pip install -e '.[quantum]'
```

The optional group pins the stack validated for this lane:

- `pulser==1.7.2`
- `pulser-simulation==1.7.2`
- `emu-sv==2.7.2`
- `emu-mps==2.7.2`
- `qutip==5.2.3`

Without these packages, provider imports still work, but calls that construct or
sample a quantum sequence raise `MissingQuantumOptionalDependency`. Tests that
exercise the quantum backend use `pytest.importorskip(...)` so base environments
skip cleanly.
