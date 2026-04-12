import importlib


def test_molecular_docking_map_to_udg_import_smoke() -> None:
    assert importlib.import_module("sciona.atoms.bio.molecular_docking.map_to_udg") is not None
    assert importlib.import_module("sciona.probes.bio.molecular_docking_map_to_udg") is not None
