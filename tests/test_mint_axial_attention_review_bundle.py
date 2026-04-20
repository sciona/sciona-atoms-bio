from __future__ import annotations

import importlib
import inspect
import json
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
BUNDLE = REPO_ROOT / "data" / "review_bundles" / "bio_mint_axial_attention.review_bundle.json"
MANIFEST = REPO_ROOT / "data" / "audit_manifest.json"

TARGET_FQDNS = {
    "sciona.atoms.bio.mint.axial_attention.row_self_attention",
    "sciona.atoms.bio.mint.axial_attention.rowselfattention",
}


def _bundle() -> dict:
    return json.loads(BUNDLE.read_text())


def test_mint_axial_attention_review_bundle_is_catalog_ready_with_limits() -> None:
    bundle = _bundle()
    assert bundle["schema_version"] == "1.0"
    assert bundle["provider_repo"] == "sciona-atoms-bio"
    assert bundle["family_batch"] == "bio_mint_axial_attention"
    assert bundle["review_status"] == "reviewed"
    assert bundle["review_semantic_verdict"] == "pass_with_limits"
    assert bundle["trust_readiness"] == "catalog_ready"
    assert bundle["blocking_findings"] == []
    assert bundle["required_actions"] == []

    for source in bundle["authoritative_sources"]:
        assert (REPO_ROOT / source["path"]).exists()

    rows = {row["atom_name"]: row for row in bundle["rows"]}
    assert set(rows) == TARGET_FQDNS
    for fqdn, row in rows.items():
        assert row["atom_key"] == fqdn
        assert row["review_status"] == "reviewed"
        assert row["review_semantic_verdict"] == "pass_with_limits"
        assert row["trust_readiness"] == "catalog_ready"
        assert row["has_references"] is True
        assert row["references_status"] == "pass"
        assert row["blocking_findings"] == []
        assert row["required_actions"] == []
        assert isinstance(row["risk_score"], int)
        assert isinstance(row["acceptability_score"], int)
        assert row["acceptability_band"] == "acceptable_with_limits_candidate"
        assert row["parity_coverage_level"] == "parity_or_usage_equivalent"
        assert row["parity_case_count"] >= 5

        module_name, function_name = fqdn.rsplit(".", 1)
        function = getattr(importlib.import_module(module_name), function_name)
        source_text, _ = inspect.getsourcelines(function)
        assert f"def {function_name}(" in "".join(source_text)


def test_mint_axial_attention_audit_manifest_contains_merged_rows() -> None:
    manifest = json.loads(MANIFEST.read_text())
    entries = {entry["atom_name"]: entry for entry in manifest["atoms"]}
    missing = TARGET_FQDNS - set(entries)
    assert missing == set()

    for fqdn in TARGET_FQDNS:
        entry = entries[fqdn]
        assert entry["review_status"] == "approved"
        assert entry["review_semantic_verdict"] == "pass_with_limits"
        assert entry["review_developer_semantics_verdict"] == "pass_with_limits"
        assert entry["trust_readiness"] == "reviewed_with_limits"
        assert entry["references_status"] == "pass"
        assert entry["parity_test_status"] == "pass"
