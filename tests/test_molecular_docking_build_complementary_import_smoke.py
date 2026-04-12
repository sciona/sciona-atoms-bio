import importlib


def test_molecular_docking_build_complementary_import_smoke() -> None:
    assert importlib.import_module("sciona.atoms.bio.molecular_docking.build_complementary") is not None
    assert importlib.import_module("sciona.probes.bio.molecular_docking_build_complementary") is not None
