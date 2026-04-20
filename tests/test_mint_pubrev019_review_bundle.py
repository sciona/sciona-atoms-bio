from __future__ import annotations

import importlib
import inspect
import json
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
ROOT = REPO_ROOT / "src" / "sciona" / "atoms" / "bio" / "mint"
BUNDLE = ROOT / "review_bundle.json"
REFERENCES = ROOT / "references.json"

PUBLISHABLE_PUBREV019_FQDNS = {
    "sciona.atoms.bio.mint.apc_module.apccoreevaluation",
    "sciona.atoms.bio.mint.axial_attention.row_self_attention",
    "sciona.atoms.bio.mint.axial_attention.rowselfattention",
    "sciona.atoms.bio.mint.encoding_dist_mat.encodedistancematrix",
    "sciona.atoms.bio.mint.incremental_attention.enable_incremental_state_configuration",
    "sciona.atoms.bio.mint.rotary_embedding.rotaryembedding_numpy",
    "sciona.atoms.bio.mint.rotary_embedding.rotaryembedding_torch",
}

HELD_PUBREV019_FQDNS: set[str] = set()


def _bundle() -> dict:
    return json.loads(BUNDLE.read_text())


def _rows_by_base_fqdn() -> dict[str, dict]:
    rows: dict[str, dict] = {}
    for row in _bundle()["rows"]:
        base_fqdn = row["atom_fqdn"].split("@", 1)[0]
        assert base_fqdn not in rows
        rows[base_fqdn] = row
    return rows


def test_mint_pubrev019_review_bundle_covers_references_and_sources() -> None:
    bundle = _bundle()
    reference_keys = set(json.loads(REFERENCES.read_text())["atoms"])

    assert bundle["schema_version"] == "1.0"
    assert bundle["bundle_id"] == "bio.mint.pubrev019.review.v1"
    assert bundle["provider_repo"] == "sciona-atoms-bio"
    assert bundle["family"] == "bio.mint"
    assert bundle["review_status"] == "reviewed"
    assert bundle["semantic_verdict"] == "publishable_candidate_after_axial_attention_remediation"
    assert bundle["trust_readiness"] == "ready_for_manifest_merge"
    assert {row["atom_fqdn"] for row in bundle["rows"]} == reference_keys
    assert set(_rows_by_base_fqdn()) == PUBLISHABLE_PUBREV019_FQDNS | HELD_PUBREV019_FQDNS

    for rel in bundle["authoritative_sources"]:
        assert (REPO_ROOT / rel).exists()


def test_mint_pubrev019_publishable_rows_are_manifest_ready() -> None:
    rows = _rows_by_base_fqdn()

    for fqdn in PUBLISHABLE_PUBREV019_FQDNS:
        row = rows[fqdn]
        assert row["atom_fqdn"] == f"{fqdn}@{row['source_path']}"
        assert row["review_status"] == "reviewed"
        assert row["semantic_verdict"] == "publishable_candidate"
        assert row["trust_readiness"] == "ready_for_manifest_merge"
        assert row["audit_batch"] == "pubrev-019"
        assert row["audit_scope"] == "bio.mint.remainder"
        assert row["description"]
        assert row["parameters"]
        assert row["io_specs"]
        assert row["uncertainty"] in {"low", "medium"}
        assert row["provenance"]

        source_rel, _, line_text = row["source_path"].partition(":")
        assert line_text
        source_file = REPO_ROOT / "src" / source_rel
        assert source_file.exists()
        for rel in row["authoritative_sources"]:
            assert (REPO_ROOT / rel).exists()

        module_name, function_name = fqdn.rsplit(".", 1)
        function = getattr(importlib.import_module(module_name), function_name)
        source_text, _ = inspect.getsourcelines(function)
        assert f"def {function_name}(" in "".join(source_text)
        source_lines = source_file.read_text().splitlines()
        assert source_lines[int(line_text) - 1].lstrip().startswith(
            f"def {function_name}("
        )


def test_mint_pubrev019_axial_attention_rows_are_remediated_and_ready() -> None:
    rows = _rows_by_base_fqdn()
    bundle_text = json.dumps(_bundle())

    for fqdn in {
        "sciona.atoms.bio.mint.axial_attention.row_self_attention",
        "sciona.atoms.bio.mint.axial_attention.rowselfattention",
    }:
        row = rows[fqdn]
        assert row["review_status"] == "reviewed"
        assert row["semantic_verdict"] == "publishable_candidate"
        assert row["trust_readiness"] == "ready_for_manifest_merge"
        assert row["limitations"]
        assert row["required_actions"] == []
        assert "RowSelfAttention" in json.dumps(row)
        assert "q/k/v/out" in json.dumps(row) or "projection" in json.dumps(row)
        assert fqdn in bundle_text

    assert "source_aligned" in bundle_text
    assert "4D MSA" in bundle_text
