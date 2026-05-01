from __future__ import annotations

import importlib
import inspect
import json
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
ROOT = REPO_ROOT / "src" / "sciona" / "atoms" / "bio" / "chemical_descriptors"
BUNDLE = REPO_ROOT / "data" / "review_bundles" / "chemical_descriptors.review_bundle.json"

CHEMICAL_DESCRIPTOR_FQDNS = {
    "sciona.atoms.bio.chemical_descriptors.smiles_to_mol",
    "sciona.atoms.bio.chemical_descriptors.compute_descriptors",
    "sciona.atoms.bio.chemical_descriptors.morgan_fingerprint",
    "sciona.atoms.bio.chemical_descriptors.maccs_keys",
}


def _bundle() -> dict:
    return json.loads(BUNDLE.read_text())


def _rows_by_fqdn() -> dict[str, dict]:
    rows: dict[str, dict] = {}
    for row in _bundle()["rows"]:
        atom_name = row["atom_name"]
        assert atom_name not in rows
        rows[atom_name] = row
    return rows


def test_chemical_descriptors_review_bundle_is_catalog_ready_with_limits() -> None:
    bundle = _bundle()

    assert bundle["schema_version"] == "1.0"
    assert bundle["bundle_id"] == "sciona.atoms.review_bundle.bio.chemical_descriptors.v1"
    assert bundle["provider_repo"] == "sciona-atoms-bio"
    assert bundle["family"] == "bio.chemical_descriptors"
    assert bundle["review_status"] == "reviewed"
    assert bundle["review_semantic_verdict"] == "pass_with_limits"
    assert bundle["trust_readiness"] == "catalog_ready"
    assert bundle["blocking_findings"] == []
    assert bundle["required_actions"] == []

    for source in bundle["authoritative_sources"]:
        assert (REPO_ROOT / source["path"]).exists()


def test_chemical_descriptors_review_bundle_rows_point_to_live_functions() -> None:
    rows = _rows_by_fqdn()

    assert set(rows) == CHEMICAL_DESCRIPTOR_FQDNS

    for idx, fqdn in enumerate(
        [
            "sciona.atoms.bio.chemical_descriptors.smiles_to_mol",
            "sciona.atoms.bio.chemical_descriptors.compute_descriptors",
            "sciona.atoms.bio.chemical_descriptors.morgan_fingerprint",
            "sciona.atoms.bio.chemical_descriptors.maccs_keys",
        ]
    ):
        row = rows[fqdn]
        assert row["atom_key"] == fqdn
        assert row["review_status"] == "reviewed"
        assert row["review_semantic_verdict"] == "pass_with_limits"
        assert row["trust_readiness"] == "catalog_ready"
        assert row["review_record_path"] == (
            f"data/review_bundles/chemical_descriptors.review_bundle.json#rows[{idx}]"
        )
        assert row["has_references"] is True
        assert row["references_status"] == "pass"
        assert row["blocking_findings"] == []
        assert row["required_actions"] == []
        assert row["source_paths"]

        source_rel, _, line_text = row["source_path"].partition(":")
        assert line_text
        source_file = REPO_ROOT / "src" / source_rel
        assert source_file == ROOT / "atoms.py"
        assert source_file.exists()

        module_name, function_name = fqdn.rsplit(".", 1)
        function = getattr(importlib.import_module(module_name), function_name)
        source_text, _ = inspect.getsourcelines(function)
        assert f"def {function_name}(" in "".join(source_text)
        assert source_file.read_text().splitlines()[int(line_text) - 1].lstrip().startswith(
            f"def {function_name}("
        )

