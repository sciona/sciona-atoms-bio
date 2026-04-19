from __future__ import annotations

import importlib
import inspect
import json
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
ROOT = REPO_ROOT / "src" / "sciona" / "atoms" / "bio" / "hpdb"
BUNDLE = ROOT / "review_bundle.json"
REMEDIATION = ROOT / "REMEDIATION.md"

HELD_PUBREV064_FQDNS = {
    "sciona.atoms.bio.hpdb.iterate_pdb_atoms",
    "sciona.atoms.bio.hpdb.iterate_pdb_residues",
}


def _expected_atom_keys() -> set[str]:
    data = json.loads((ROOT / "references.json").read_text())
    return set(data.get("atoms", {}).keys())


def _bundle() -> dict:
    return json.loads(BUNDLE.read_text())


def _rows_by_base_fqdn() -> dict[str, dict]:
    rows: dict[str, dict] = {}
    for row in _bundle()["rows"]:
        base_fqdn = row["atom_fqdn"].split("@", 1)[0]
        assert base_fqdn not in rows
        rows[base_fqdn] = row
    return rows


def test_hpdb_review_bundle_pubrev064_covers_reference_rows_as_held() -> None:
    bundle = _bundle()
    expected_keys = _expected_atom_keys()

    assert bundle["schema_version"] == "1.0"
    assert bundle["bundle_id"] == "bio.hpdb.family_batch.review.v1"
    assert bundle["provider_repo"] == "sciona-atoms-bio"
    assert bundle["family"] == "bio.hpdb"
    assert bundle["review_status"] == "reviewed"
    assert bundle["semantic_verdict"] == "held_requires_remediation"
    assert bundle["trust_readiness"] == "blocked_on_semantic_remediation"
    assert bundle["limitations"]
    assert bundle["required_actions"]
    assert {row["atom_fqdn"] for row in bundle["rows"]} == expected_keys

    for rel in bundle["authoritative_sources"]:
        assert (REPO_ROOT / rel).exists()


def test_hpdb_review_bundle_pubrev064_rows_are_not_manifest_ready() -> None:
    rows = _rows_by_base_fqdn()
    remediation = REMEDIATION.read_text()

    assert set(rows) == HELD_PUBREV064_FQDNS

    for idx, fqdn in enumerate(sorted(HELD_PUBREV064_FQDNS)):
        row = rows[fqdn]
        assert fqdn in remediation
        assert row["atom_fqdn"] == f"{fqdn}@{row['source_path']}"
        assert row["review_status"] == "reviewed"
        assert row["semantic_verdict"] == "held_requires_remediation"
        assert row["developer_semantic_verdict"]
        assert row["trust_readiness"] == "blocked_on_semantic_remediation"
        assert row["trust_readiness"] != "ready_for_manifest_merge"
        assert row["limitations"]
        assert row["required_actions"]
        assert row["review_record_path"] == (
            f"src/sciona/atoms/bio/hpdb/review_bundle.json#rows[{idx}]"
        )
        assert row["audit_batch"] == "pubrev-064"
        assert row["audit_scope"] == "sciona.atoms.bio.hpdb.__remainder__"

        source_rel, _, line_text = row["source_path"].partition(":")
        assert line_text
        assert (REPO_ROOT / "src" / source_rel).exists()
        for rel in row["authoritative_sources"]:
            assert (REPO_ROOT / rel).exists()

        module_name, function_name = fqdn.rsplit(".", 1)
        function = getattr(importlib.import_module(module_name), function_name)
        source_text, _ = inspect.getsourcelines(function)
        assert f"def {function_name}(" in "".join(source_text)
        source_lines = (REPO_ROOT / "src" / source_rel).read_text().splitlines()
        assert source_lines[int(line_text) - 1].lstrip().startswith(
            f"def {function_name}("
        )


def test_hpdb_pubrev064_records_parameter_and_description_blockers() -> None:
    text = json.dumps(_bundle())
    remediation = REMEDIATION.read_text()

    for required_text in [
        "Optional[str]",
        "rejects None",
        "PDB structure",
        "no PDB source",
        "descriptions",
    ]:
        assert required_text in text or required_text in remediation
