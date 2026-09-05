"""W3C PROV-JSON emitter for trainer-run provenance.

Every trainer run publishes a bundle of artifacts (params.json,
scheduled_headway.json, segment_params.json, the service-baseline sidecar) whose
ad-hoc `provenance`/`feed_version`/`training_corpus` blocks already answer, one
artifact at a time, "which code and which inputs produced this". This module
serializes that same lineage once, in a standard vocabulary — W3C PROV, in its
PROV-JSON encoding (https://www.w3.org/submissions/prov-json/) — so a consumer
can answer "what produced this, from what inputs" with a single grammar instead
of one bespoke reader per artifact.

The mapping this module emits:
  - prov:Entity      — the GTFS static feed (named by feed_version + a content
                       digest computed at fetch), the archive input manifest
                       (named by its BLAKE3 fingerprint), and each published
                       artifact (named by its immutable bucket key).
  - prov:Activity    — the trainer run itself, with startedAtTime/endedAtTime.
  - prov:SoftwareAgent — the trainer build, named by its code_sha (dirty flagged).
  - used              — run -> each input entity.
  - wasGeneratedBy    — each artifact entity -> run.
  - wasDerivedFrom    — an artifact -> the input entity it is built from (e.g.
                        scheduled_headway from the GTFS feed).
  - wasAssociatedWith — run -> agent.

GROUNDING RULE (the epic's hard constraint): this emitter emits ONLY relations
grounded in a recorded fact — a sha, a content digest, a BLAKE3 manifest hash, a
bucket key, or a timestamp. It is structurally impossible to emit an aspirational
edge: every entity/agent must be registered with its grounding fact before any
relation can reference it, and a relation whose endpoint was never registered
raises rather than emitting a dangling edge. So a run that could not fetch the
GTFS feed (no digest) simply carries no feed entity and no feed-derived edges,
instead of an ungrounded `used` claim.

Pure: no I/O, no clock, no network. Given fixed inputs the document serializes
byte-for-byte identically (canonical `to_json()` sorts keys and drops
whitespace), so the emitter is unit-testable against a golden fixture and the
published document is reproducible.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

# Our own namespace for run-local identifiers and non-PROV attributes. The
# authority is nominal — it names the vocabulary, it is never dereferenced.
MMLY_NS = "https://momentarily.nyc/prov#"


class ProvenanceError(ValueError):
    """Raised on an attempt to emit an ungrounded or dangling PROV relation."""


def _xsd_datetime(epoch_seconds: int) -> dict[str, str]:
    """A PROV-JSON typed literal for an epoch second, as xsd:dateTime in UTC."""
    return {
        "$": datetime.fromtimestamp(epoch_seconds, UTC).isoformat(),
        "type": "xsd:dateTime",
    }


@dataclass
class ProvDocument:
    """An accumulating PROV-JSON document.

    Elements (entity/activity/agent) are registered with their grounding facts;
    relations reference elements by id and refuse to name an unregistered one.
    """

    _entity: dict[str, dict[str, Any]] = field(
        default_factory=dict[str, dict[str, Any]]
    )
    _activity: dict[str, dict[str, Any]] = field(
        default_factory=dict[str, dict[str, Any]]
    )
    _agent: dict[str, dict[str, Any]] = field(default_factory=dict[str, dict[str, Any]])
    _used: dict[str, dict[str, Any]] = field(default_factory=dict[str, dict[str, Any]])
    _generated: dict[str, dict[str, Any]] = field(
        default_factory=dict[str, dict[str, Any]]
    )
    _derived: dict[str, dict[str, Any]] = field(
        default_factory=dict[str, dict[str, Any]]
    )
    _associated: dict[str, dict[str, Any]] = field(
        default_factory=dict[str, dict[str, Any]]
    )

    def _known(self, node_id: str) -> bool:
        return (
            node_id in self._entity
            or node_id in self._activity
            or node_id in self._agent
        )

    def _require(self, *node_ids: str) -> None:
        for node_id in node_ids:
            if not self._known(node_id):
                raise ProvenanceError(
                    f"cannot relate unregistered node {node_id!r}: an entity, "
                    "activity, or agent must be added — with its grounding fact — "
                    "before a relation may reference it"
                )

    # --- elements ---------------------------------------------------------

    def entity(
        self, entity_id: str, *, ground: dict[str, Any], attributes: dict[str, Any]
    ) -> str:
        """Register an entity. `ground` is its non-empty recorded facts (a digest,
        a BLAKE3 hash, a bucket key); an empty `ground` is refused, which is what
        keeps an unnamed input out of the graph."""
        if not ground:
            raise ProvenanceError(
                f"entity {entity_id!r} has no grounding fact — an entity must be "
                "named by a recorded digest/hash/key or it does not get emitted"
            )
        self._entity[entity_id] = {**attributes, **ground}
        return entity_id

    def activity(
        self,
        activity_id: str,
        *,
        started_at: int,
        ended_at: int,
        attributes: dict[str, Any] | None = None,
    ) -> str:
        """Register an activity grounded by its start/end timestamps."""
        record: dict[str, Any] = {
            "prov:startedAtTime": _xsd_datetime(started_at),
            "prov:endedAtTime": _xsd_datetime(ended_at),
        }
        if attributes:
            record.update(attributes)
        self._activity[activity_id] = record
        return activity_id

    def agent(
        self, agent_id: str, *, ground: dict[str, Any], attributes: dict[str, Any]
    ) -> str:
        """Register an agent. `ground` is its recorded identity (the code_sha)."""
        if not ground:
            raise ProvenanceError(
                f"agent {agent_id!r} has no grounding fact — an agent must be "
                "named by its recorded code_sha"
            )
        self._agent[agent_id] = {**attributes, **ground}
        return agent_id

    # --- relations --------------------------------------------------------

    def used(self, activity_id: str, entity_id: str) -> None:
        """activity used entity. Both must already be registered."""
        self._require(activity_id, entity_id)
        rel_id = f"_:u{len(self._used) + 1}"
        self._used[rel_id] = {"prov:activity": activity_id, "prov:entity": entity_id}

    def was_generated_by(self, entity_id: str, activity_id: str) -> None:
        """entity wasGeneratedBy activity. Both must already be registered."""
        self._require(entity_id, activity_id)
        rel_id = f"_:g{len(self._generated) + 1}"
        self._generated[rel_id] = {
            "prov:entity": entity_id,
            "prov:activity": activity_id,
        }

    def was_derived_from(self, generated_id: str, used_id: str) -> None:
        """generated wasDerivedFrom used. Both must already be registered."""
        self._require(generated_id, used_id)
        rel_id = f"_:d{len(self._derived) + 1}"
        self._derived[rel_id] = {
            "prov:generatedEntity": generated_id,
            "prov:usedEntity": used_id,
        }

    def was_associated_with(self, activity_id: str, agent_id: str) -> None:
        """activity wasAssociatedWith agent. Both must already be registered."""
        self._require(activity_id, agent_id)
        rel_id = f"_:a{len(self._associated) + 1}"
        self._associated[rel_id] = {
            "prov:activity": activity_id,
            "prov:agent": agent_id,
        }

    # --- serialization ----------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        """The PROV-JSON document as a plain dict. Only non-empty relation maps
        are included, so a document with no derivations omits `wasDerivedFrom`
        rather than carrying an empty object."""
        doc: dict[str, Any] = {
            "prefix": {"mmly": MMLY_NS, "xsd": "http://www.w3.org/2001/XMLSchema#"},
            "entity": self._entity,
            "activity": self._activity,
            "agent": self._agent,
            "used": self._used,
            "wasGeneratedBy": self._generated,
            "wasAssociatedWith": self._associated,
        }
        if self._derived:
            doc["wasDerivedFrom"] = self._derived
        return doc

    def to_json(self) -> str:
        """Canonical, byte-stable serialization: sorted keys, no incidental
        whitespace. Identical inputs produce identical bytes."""
        return json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))


# --- trainer-run assembler ------------------------------------------------
#
# Pure facts -> a trainer-run PROV document. Every argument is a recorded fact
# the trainer already holds; nothing here fetches or computes. The grounding
# rule lives in the shape of these dataclasses: a fact absent from the input is
# a node and edge absent from the output.


@dataclass(frozen=True)
class AgentFacts:
    """The trainer build that ran, from training.provenance.code_provenance."""

    code_sha: str
    dirty: bool | None
    producer: str


@dataclass(frozen=True)
class FeedFacts:
    """The GTFS static feed a run read, named by version AND a content digest
    computed over the fetched bytes at fetch time."""

    version: str
    sha256: str
    start: str | None = None  # feed_start_date, ISO
    end: str | None = None  # feed_end_date, ISO


@dataclass(frozen=True)
class ManifestFacts:
    """The archive input manifest a run fed on, named by its BLAKE3 fingerprint
    over the fetched object keys (alerts + vehicles)."""

    blake3: str
    n_alert_keys: int
    n_vehicle_keys: int
    manifest_version: int


@dataclass(frozen=True)
class ArtifactFacts:
    """One published artifact, named by its immutable versioned bucket key."""

    name: str  # short label, e.g. "params"
    bucket_key: str  # e.g. "state/params/v1788229972.json"
    derived_from_feed: bool = False  # built from the GTFS feed timetable
    derived_from_manifest: bool = False  # built from the archive manifest


def _entity_id(key: str) -> str:
    return f"mmly:{key}"


def build_trainer_run(
    *,
    trained_at: int,
    started_at: int,
    agent: AgentFacts,
    manifest: ManifestFacts,
    artifacts: list[ArtifactFacts],
    feed: FeedFacts | None = None,
) -> ProvDocument:
    """Assemble the PROV document for one trainer run.

    `feed` is optional: a run whose GTFS fetch failed carries no digest, so it
    gets no feed entity and no feed-derived edges — the grounding rule, applied
    by omission rather than by emitting an unnamed input.
    """
    doc = ProvDocument()

    run_id = f"mmly:run/{trained_at}"
    doc.activity(
        run_id,
        started_at=started_at,
        ended_at=trained_at,
        attributes={"prov:type": "mmly:TrainerRun"},
    )

    dirty_attr: dict[str, Any] = {}
    if agent.dirty is not None:
        dirty_attr["mmly:dirty"] = {"$": agent.dirty, "type": "xsd:boolean"}
    agent_id = f"mmly:trainer/{agent.code_sha}"
    doc.agent(
        agent_id,
        ground={"mmly:codeSha": agent.code_sha},
        attributes={
            "prov:type": "prov:SoftwareAgent",
            "mmly:producer": agent.producer,
            **dirty_attr,
        },
    )
    doc.was_associated_with(run_id, agent_id)

    manifest_id = _entity_id(f"manifest/{manifest.blake3}")
    doc.entity(
        manifest_id,
        ground={"mmly:blake3": manifest.blake3},
        attributes={
            "prov:type": "mmly:InputManifest",
            "mmly:manifestVersion": manifest.manifest_version,
            "mmly:nAlertKeys": manifest.n_alert_keys,
            "mmly:nVehicleKeys": manifest.n_vehicle_keys,
        },
    )
    doc.used(run_id, manifest_id)

    feed_id: str | None = None
    if feed is not None:
        feed_attrs: dict[str, Any] = {
            "prov:type": "mmly:GtfsStaticFeed",
            "mmly:feedVersion": feed.version,
        }
        if feed.start is not None:
            feed_attrs["mmly:feedStartDate"] = feed.start
        if feed.end is not None:
            feed_attrs["mmly:feedEndDate"] = feed.end
        feed_id = _entity_id(f"gtfs/{feed.sha256}")
        doc.entity(
            feed_id,
            ground={"mmly:sha256": feed.sha256},
            attributes=feed_attrs,
        )
        doc.used(run_id, feed_id)

    for art in artifacts:
        art_id = _entity_id(art.bucket_key)
        doc.entity(
            art_id,
            ground={"mmly:bucketKey": art.bucket_key},
            attributes={"prov:type": "mmly:PublishedArtifact", "mmly:name": art.name},
        )
        doc.was_generated_by(art_id, run_id)
        if art.derived_from_manifest:
            doc.was_derived_from(art_id, manifest_id)
        if art.derived_from_feed and feed_id is not None:
            doc.was_derived_from(art_id, feed_id)

    return doc
