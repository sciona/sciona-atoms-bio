import importlib


def test_molecular_docking_greedy_mapping_import_smoke() -> None:
    assert importlib.import_module("sciona.atoms.bio.molecular_docking.greedy_mapping") is not None
    assert importlib.import_module("sciona.probes.bio.molecular_docking_greedy_mapping") is not None
