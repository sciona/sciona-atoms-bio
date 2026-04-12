import importlib


def test_mint_encoding_dist_mat_import_smoke() -> None:
    assert importlib.import_module("sciona.atoms.bio.mint.encoding_dist_mat") is not None
    assert importlib.import_module("sciona.probes.bio.mint_encoding_dist_mat") is not None
