"""Unit tests for the trainer-side PROV-JSON emitter (training.prov)."""

from __future__ import annotations

import json
from typing import Any

import pytest

from training.prov import (
    AgentFacts,
    ArtifactFacts,
    FeedFacts,
    ManifestFacts,
    ProvDocument,
    ProvenanceError,
    build_trainer_run,
)

# A fixed set of run facts reused across tests. Every value is a plausible
# recorded fact; the point is that the SAME facts always yield the SAME bytes.
_AGENT = AgentFacts(code_sha="abc123def456", dirty=False, producer="ci")
_FEED = FeedFacts(
    version="20260807-H-rockaways-extension-removed",
    sha256="f" * 64,
    start="2026-08-07",
    end="2026-09-04",
)
_MANIFEST = ManifestFacts(
    blake3="a" * 64, n_alert_keys=4032, n_vehicle_keys=4032, manifest_version=2
)
_ARTIFACTS = [
    ArtifactFacts(
        "params", "state/params/v1788229972.json", derived_from_manifest=True
    ),
    ArtifactFacts(
        "scheduled_headway",
        "state/scheduled_headway/v1788229972.json",
        derived_from_feed=True,
    ),
    ArtifactFacts(
        "segment_params",
        "state/segment_params/v1788229972.json",
        derived_from_feed=True,
        derived_from_manifest=True,
    ),
    ArtifactFacts("service_baseline", "state/service_baseline/v1788229972.json"),
]


def _doc(**overrides: Any) -> ProvDocument:
    kwargs: dict[str, Any] = {
        "trained_at": 1788229972,
        "started_at": 1788229900,
        "agent": _AGENT,
        "manifest": _MANIFEST,
        "artifacts": _ARTIFACTS,
        "feed": _FEED,
    }
    kwargs.update(overrides)
    return build_trainer_run(**kwargs)


# --- byte-stability -------------------------------------------------------


def test_to_json_is_byte_stable_across_builds() -> None:
    assert _doc().to_json() == _doc().to_json()


def test_to_json_is_canonical_sorted_and_compact() -> None:
    raw = _doc().to_json()
    # No incidental whitespace, keys sorted: a re-dump with the same options is
    # a no-op on the parsed structure.
    assert raw == json.dumps(json.loads(raw), sort_keys=True, separators=(",", ":"))
    assert ", " not in raw
    assert ": " not in raw


def test_a_changed_fact_changes_the_bytes() -> None:
    other_manifest = ManifestFacts(
        blake3="b" * 64, n_alert_keys=1, n_vehicle_keys=1, manifest_version=2
    )
    assert _doc().to_json() != _doc(manifest=other_manifest).to_json()


# --- PROV-JSON structural shape -------------------------------------------


def test_document_has_prov_json_maps() -> None:
    doc = _doc().to_dict()
    for section in ("entity", "activity", "agent", "used", "wasGeneratedBy"):
        assert isinstance(doc[section], dict), section
    # Elements and relations are keyed maps, not lists.
    assert doc["entity"], "expected at least one entity"
    assert doc["activity"], "expected the trainer-run activity"
    assert doc["agent"], "expected the trainer software agent"


def test_relation_records_use_prov_prefixed_keys() -> None:
    doc = _doc().to_dict()
    for rec in doc["used"].values():
        assert set(rec) == {"prov:activity", "prov:entity"}
    for rec in doc["wasGeneratedBy"].values():
        assert set(rec) == {"prov:entity", "prov:activity"}
    for rec in doc["wasAssociatedWith"].values():
        assert set(rec) == {"prov:activity", "prov:agent"}
    for rec in doc["wasDerivedFrom"].values():
        assert set(rec) == {"prov:generatedEntity", "prov:usedEntity"}


def test_activity_carries_start_and_end_datetimes() -> None:
    (activity,) = _doc().to_dict()["activity"].values()
    assert activity["prov:startedAtTime"]["type"] == "xsd:dateTime"
    assert activity["prov:endedAtTime"]["type"] == "xsd:dateTime"
    assert activity["prov:startedAtTime"]["$"] == "2026-09-01T02:31:40+00:00"
    assert activity["prov:endedAtTime"]["$"] == "2026-09-01T02:32:52+00:00"


def test_agent_is_software_agent_grounded_by_code_sha() -> None:
    (agent,) = _doc().to_dict()["agent"].values()
    assert agent["prov:type"] == "prov:SoftwareAgent"
    assert agent["mmly:codeSha"] == _AGENT.code_sha


def test_every_relation_endpoint_resolves_to_a_declared_node() -> None:
    doc = _doc().to_dict()
    declared = set(doc["entity"]) | set(doc["activity"]) | set(doc["agent"])
    for section in ("used", "wasGeneratedBy", "wasAssociatedWith", "wasDerivedFrom"):
        for rec in doc[section].values():
            for value in rec.values():
                assert value in declared, f"{section} references undeclared {value!r}"


def test_each_artifact_was_generated_by_the_run() -> None:
    doc = _doc().to_dict()
    (run_id,) = doc["activity"]
    generated_entities = {r["prov:entity"] for r in doc["wasGeneratedBy"].values()}
    for art in _ARTIFACTS:
        assert f"mmly:{art.bucket_key}" in generated_entities
    assert all(r["prov:activity"] == run_id for r in doc["wasGeneratedBy"].values())


# --- grounding rule / leakage ---------------------------------------------


def test_no_feed_entity_or_edges_without_a_digest() -> None:
    """A run whose GTFS fetch failed carries no digest; the feed entity and every
    feed-derived edge must be ABSENT, not emitted ungrounded."""
    grounded = _doc().to_dict()
    ungrounded = _doc(feed=None).to_dict()

    feed_ids = [e for e in grounded["entity"] if "gtfs/" in e]
    assert feed_ids, "sanity: the grounded run should carry a feed entity"
    for feed_id in feed_ids:
        assert feed_id not in ungrounded["entity"]

    # No used-edge and no derivation may point at a feed that was never named.
    grounded_feed = feed_ids[0]
    assert any(r["prov:entity"] == grounded_feed for r in grounded["used"].values())
    assert all(r["prov:entity"] != grounded_feed for r in ungrounded["used"].values())
    derived_targets = {
        r["prov:usedEntity"] for r in ungrounded.get("wasDerivedFrom", {}).values()
    }
    assert grounded_feed not in derived_targets


def test_registering_an_entity_without_a_grounding_fact_is_refused() -> None:
    doc = ProvDocument()
    with pytest.raises(ProvenanceError):
        doc.entity("mmly:orphan", ground={}, attributes={"prov:type": "mmly:Thing"})


def test_relating_an_unregistered_node_is_refused() -> None:
    """The structural leakage guard: an edge cannot name a node that was never
    added (i.e. a node with no grounding fact behind it)."""
    doc = ProvDocument()
    run = doc.activity("mmly:run/1", started_at=1, ended_at=2)
    with pytest.raises(ProvenanceError):
        doc.used(run, "mmly:gtfs/never-registered")
    with pytest.raises(ProvenanceError):
        doc.was_generated_by("mmly:artifact/never-registered", run)
    # And no partial edge leaked into the document.
    assert doc.to_dict()["used"] == {}
    assert doc.to_dict()["wasGeneratedBy"] == {}


def test_dirty_flag_rides_along_only_when_known() -> None:
    known = build_trainer_run(
        trained_at=2,
        started_at=1,
        agent=AgentFacts(code_sha="s", dirty=True, producer="local"),
        manifest=_MANIFEST,
        artifacts=[],
    ).to_dict()
    (agent_known,) = known["agent"].values()
    assert agent_known["mmly:dirty"] == {"$": True, "type": "xsd:boolean"}

    unknown = build_trainer_run(
        trained_at=2,
        started_at=1,
        agent=AgentFacts(code_sha="s", dirty=None, producer="local"),
        manifest=_MANIFEST,
        artifacts=[],
    ).to_dict()
    (agent_unknown,) = unknown["agent"].values()
    assert "mmly:dirty" not in agent_unknown
