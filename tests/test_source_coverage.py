"""
Tests that all source_ids used in raw data are present in the source trust registry,
and that all AVeriTeC URLs resolve to a known registry entry.
"""

import json
import collections
from pathlib import Path
from urllib.parse import urlparse

import pytest

from src.epistemic.registry import (
    load_source_trust_registry,
    resolve_source_id,
    get_source_trust,
    verify_generic_fallback_coverage,
)

ROOT = Path(__file__).parent.parent
REGISTRY_PATH = ROOT / "data/registry/source_trust_registry.jsonl"
RAW_AI2THOR   = ROOT / "data/raw/ai2thor/claims_all.jsonl"
RAW_SYNTHETIC = ROOT / "data/raw/synthetic/synthetic_current.jsonl"
AVERITEC_SPLITS = [
    ROOT / "data/raw/averitec/train.json",
    ROOT / "data/raw/averitec/dev.json",
    ROOT / "data/raw/averitec/test.json",
]


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def registry() -> dict:
    return load_source_trust_registry(str(REGISTRY_PATH))


@pytest.fixture(scope="module")
def explicit_source_ids(registry) -> dict[str, set]:
    """Collect every explicit source_id from JSONL data files → {source_id: {filename}}."""
    found: dict[str, set] = collections.defaultdict(set)
    for path in (RAW_AI2THOR, RAW_SYNTHETIC):
        if not path.exists():
            continue
        for line in path.read_text("utf-8").splitlines():
            if not line.strip():
                continue
            for ev in json.loads(line).get("evidence", []):
                sid = ev.get("source_id")
                if sid:
                    found[sid].add(path.name)
    return dict(found)


@pytest.fixture(scope="module")
def averitec_resolved_ids(registry) -> set[str]:
    """Resolve every AVeriTeC source_url to a registry source_id."""
    resolved: set[str] = set()
    for split_path in AVERITEC_SPLITS:
        if not split_path.exists():
            continue
        data = json.loads(split_path.read_text("utf-8"))
        for rec in data:
            for q in rec.get("questions", []):
                for a in q.get("answers", []):
                    url = a.get("source_url", "")
                    if not url:
                        continue
                    parsed = urlparse(url if "://" in url else "https://" + url)
                    domain = parsed.netloc or parsed.path.split("/")[0]
                    domain = domain.lower().removeprefix("www.")
                    if not domain:
                        continue
                    modality = "pdf" if url.lower().endswith(".pdf") else "web_text"
                    resolved.add(resolve_source_id(domain, modality, registry))
    return resolved


# ── Tests ─────────────────────────────────────────────────────────────────────

class TestExplicitSourceIds:
    """Explicit source_ids written into JSONL evidence records."""

    def test_all_explicit_ids_in_registry(self, explicit_source_ids, registry):
        missing = {sid for sid in explicit_source_ids if sid not in registry}
        assert not missing, (
            f"source_ids found in data but missing from registry: {sorted(missing)}"
        )

    def test_explicit_ids_have_positive_trust(self, explicit_source_ids, registry):
        for sid in explicit_source_ids:
            st = get_source_trust(sid, registry)
            assert st > 0, f"source_id '{sid}' has non-positive source_trust={st}"


class TestAveritecResolution:
    """Every AVeriTeC URL resolves to a registry entry (no silent DEFAULT fallback)."""

    def test_all_averitec_resolved_ids_in_registry(self, averitec_resolved_ids, registry):
        missing = averitec_resolved_ids - set(registry.keys())
        assert not missing, (
            f"AVeriTeC URLs resolved to source_ids missing from registry: {sorted(missing)}"
        )

    def test_averitec_resolves_at_least_one_entry(self, averitec_resolved_ids):
        assert len(averitec_resolved_ids) > 0, (
            "No AVeriTeC URLs were resolved — check that source_url fields exist in raw data"
        )

    def test_averitec_resolved_ids_have_positive_trust(self, averitec_resolved_ids, registry):
        for sid in averitec_resolved_ids:
            st = get_source_trust(sid, registry)
            assert st > 0, f"Resolved source_id '{sid}' has non-positive source_trust={st}"


class TestGenericFallbackCoverage:
    """Every (source_type, modality) pair must have a '{source_type}_{modality}' registry entry."""

    def test_registry_loads_without_error(self):
        # load_source_trust_registry raises ValueError if any generic entry is missing —
        # a clean load proves coverage is complete.
        load_source_trust_registry(str(REGISTRY_PATH))

    def test_generic_entries_have_positive_trust(self, registry):
        for st, mod in sorted({(e["source_type"], e["modality"]) for e in registry.values()}):
            sid = f"{st}_{mod}"
            entry = registry.get(sid)
            assert entry is not None, f"Generic entry '{sid}' not found"
            trust = entry["source_trust"]
            assert trust > 0, f"Generic entry '{sid}' has non-positive trust={trust}"
            assert trust <= 1.0, f"Generic entry '{sid}' has trust={trust} > 1.0"

    def test_generic_entries_have_correct_source_type(self, registry):
        for st, mod in sorted({(e["source_type"], e["modality"]) for e in registry.values()}):
            sid = f"{st}_{mod}"
            entry = registry[sid]
            assert entry["source_type"] == st, (
                f"'{sid}' source_type={entry['source_type']!r}, expected {st!r}"
            )
            assert entry["modality"] == mod, (
                f"'{sid}' modality={entry['modality']!r}, expected {mod!r}"
            )

    def test_generic_entries_have_required_fields(self, registry):
        required_fields = {"source_id", "source_type", "modality", "source_trust",
                           "prior_trust", "default_evidence_types"}
        for st, mod in sorted({(e["source_type"], e["modality"]) for e in registry.values()}):
            sid = f"{st}_{mod}"
            entry = registry[sid]
            missing_fields = required_fields - entry.keys()
            assert not missing_fields, f"'{sid}' missing fields: {missing_fields}"

    def test_trust_ordering_higher_than_unknown_for_trusted_types(self, registry):
        """High-trust source types must have higher trust than their unknown counterpart."""
        trusted_types = {"government", "scientific_paper", "fact_checker", "knowledge_graph"}
        unknown_web = registry["unknown_web_text"]["source_trust"]
        unknown_pdf = registry["unknown_pdf"]["source_trust"]

        for st, mod in sorted({(e["source_type"], e["modality"]) for e in registry.values()}):
            if st not in trusted_types:
                continue
            sid = f"{st}_{mod}"
            trust = registry[sid]["source_trust"]
            ref = unknown_pdf if mod == "pdf" else unknown_web
            assert trust > ref, (
                f"'{sid}' trust={trust} should be > unknown fallback ({ref}) "
                f"for trusted source_type '{st}'"
            )

    def test_load_registry_raises_on_missing_entry(self, tmp_path):
        """load_source_trust_registry raises ValueError if a generic entry is absent."""
        incomplete = tmp_path / "registry.jsonl"
        incomplete.write_text('{"source_id": "dummy", "source_type": "unknown", "modality": "unknown", "source_trust": 0.3}\n')
        with pytest.raises(ValueError, match="missing"):
            load_source_trust_registry(incomplete)
