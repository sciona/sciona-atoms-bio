import importlib


def test_mint_apc_module_import_smoke() -> None:
    assert importlib.import_module("sciona.atoms.bio.mint.apc_module") is not None
    assert importlib.import_module("sciona.probes.bio.mint_apc_module") is not None
