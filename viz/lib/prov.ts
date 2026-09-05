// Reader for one trainer run's W3C PROV-JSON document (the sidecar the trainer
// publishes per run — see training/prov.py, which is the sole producer). The
// snapshot points at the public copy through provenance.prov_ref; the models
// page fetches that URL and hands the parsed body here.
//
// This module is PURE: raw document in, a flat derivation chain out, no fetch
// and no clock. It exists so the parse is unit-testable against a document the
// emitter builds, rather than a hand-written one.
//
// FACTS-ONLY, mirroring the emitter's grounding rule: we surface exactly the
// nodes and edges the document carries. A run whose GTFS fetch failed publishes
// no feed entity, so `feed` is null and no feed node is drawn — never a
// fabricated "unknown feed". A document we cannot make sense of at all (not an
// object, or no trainer-run activity with grounded timestamps) parses to null,
// which the caller renders as "provenance unavailable" without touching the
// rest of the page.

export interface ProvRun {
  // The model version stamp, read off the run activity's id (mmly:run/<n>) —
  // the same integer a params key and the snapshot's params.trained_at carry.
  trainedAt: number;
  // Epoch seconds, from the activity's start/end xsd:dateTime literals.
  startedAt: number;
  endedAt: number;
}

export interface ProvAgent {
  codeSha: string;
  // null when the document did not record a dirty flag (the emitter omits it
  // when the trainer build couldn't tell) — distinct from a recorded `false`.
  dirty: boolean | null;
  producer: string | null;
}

export interface ProvFeed {
  version: string;
  sha256: string;
  start: string | null;
  end: string | null;
}

export interface ProvManifest {
  blake3: string;
  nAlertKeys: number;
  nVehicleKeys: number;
  manifestVersion: number;
}

export interface ProvArtifact {
  name: string;
  bucketKey: string;
  // Reconstructed from the document's wasDerivedFrom edges: this artifact was
  // built from the GTFS feed and/or the input manifest.
  derivedFromFeed: boolean;
  derivedFromManifest: boolean;
}

// The whole derivation chain, flattened for rendering: GTFS feed -> trainer run
// (its software agent) -> input manifest -> published artifacts. The snapshot
// the reader is looking at is appended by the caller, which holds it — it is
// not part of the run's own document.
export interface ProvChain {
  run: ProvRun;
  agent: ProvAgent | null;
  feed: ProvFeed | null;
  manifest: ProvManifest | null;
  artifacts: ProvArtifact[];
}

// A PROV-JSON element map (entity/activity/agent) as plain records, dropping any
// non-object member. A missing or wrong-shaped section is silence, not a parse
// failure, so it returns an empty map.
function recordSection(doc: Record<string, unknown>, key: string): Record<string, Record<string, unknown>> {
  const section = doc[key];
  if (!section || typeof section !== "object" || Array.isArray(section)) return {};
  const out: Record<string, Record<string, unknown>> = {};
  for (const [id, node] of Object.entries(section as Record<string, unknown>)) {
    if (node && typeof node === "object" && !Array.isArray(node)) {
      out[id] = node as Record<string, unknown>;
    }
  }
  return out;
}

// PROV-JSON carries typed values as { "$": value, "type": "xsd:…" } and plain
// values bare; unwrap both to the underlying value before a type check.
function litValue(v: unknown): unknown {
  if (v && typeof v === "object" && !Array.isArray(v) && "$" in v) return v.$;
  return v;
}

function str(v: unknown): string | null {
  const inner = litValue(v);
  return typeof inner === "string" ? inner : null;
}

function num(v: unknown): number | null {
  const inner = litValue(v);
  return typeof inner === "number" && Number.isFinite(inner) ? inner : null;
}

function bool(v: unknown): boolean | null {
  const inner = litValue(v);
  return typeof inner === "boolean" ? inner : null;
}

// An xsd:dateTime literal (or bare ISO string) -> epoch seconds, or null when
// it does not parse. Seconds, to match the epoch stamps the run is keyed by.
function epochSeconds(v: unknown): number | null {
  const s = str(v);
  if (s === null) return null;
  const ms = Date.parse(s);
  return Number.isNaN(ms) ? null : Math.round(ms / 1000);
}

// Parse a trainer-run PROV-JSON document into a flat derivation chain, or null
// when the document is unusable. Unusable means: not an object, or no
// mmly:TrainerRun activity carrying an integer version and two parseable
// timestamps — without the run there is no chain to anchor. Everything else is
// optional and rendered only if present.
export function parseProvChain(raw: unknown): ProvChain | null {
  if (!raw || typeof raw !== "object" || Array.isArray(raw)) return null;
  const doc = raw as Record<string, unknown>;

  const activities = recordSection(doc, "activity");
  const entities = recordSection(doc, "entity");
  const agents = recordSection(doc, "agent");

  // The run: the single mmly:TrainerRun activity. Its id encodes trained_at.
  let run: ProvRun | null = null;
  for (const [id, node] of Object.entries(activities)) {
    if (str(node["prov:type"]) !== "mmly:TrainerRun") continue;
    const m = /run\/(\d+)$/.exec(id);
    const trainedAt = m ? Number(m[1]) : NaN;
    const startedAt = epochSeconds(node["prov:startedAtTime"]);
    const endedAt = epochSeconds(node["prov:endedAtTime"]);
    if (!Number.isFinite(trainedAt) || startedAt === null || endedAt === null) {
      return null;
    }
    run = { trainedAt, startedAt, endedAt };
    break;
  }
  if (run === null) return null;

  // The software agent: read the first one grounded by a code sha.
  let agent: ProvAgent | null = null;
  for (const node of Object.values(agents)) {
    const codeSha = str(node["mmly:codeSha"]);
    if (codeSha === null) continue;
    agent = { codeSha, dirty: bool(node["mmly:dirty"]), producer: str(node["mmly:producer"]) };
    break;
  }

  // Inputs and artifacts, keyed by entity id so the derivation edges can point
  // back at them.
  let feed: ProvFeed | null = null;
  let feedId: string | null = null;
  let manifest: ProvManifest | null = null;
  let manifestId: string | null = null;
  const artifacts: Array<{ id: string; art: ProvArtifact }> = [];

  for (const [id, node] of Object.entries(entities)) {
    switch (str(node["prov:type"])) {
      case "mmly:GtfsStaticFeed": {
        const version = str(node["mmly:feedVersion"]);
        const sha256 = str(node["mmly:sha256"]);
        // A feed entity is grounded by its digest; without one it is not a real
        // feed node and we drop it rather than invent a version.
        if (version !== null && sha256 !== null) {
          feed = {
            version,
            sha256,
            start: str(node["mmly:feedStartDate"]),
            end: str(node["mmly:feedEndDate"]),
          };
          feedId = id;
        }
        break;
      }
      case "mmly:InputManifest": {
        const blake3 = str(node["mmly:blake3"]);
        const nAlertKeys = num(node["mmly:nAlertKeys"]);
        const nVehicleKeys = num(node["mmly:nVehicleKeys"]);
        const manifestVersion = num(node["mmly:manifestVersion"]);
        if (blake3 !== null && nAlertKeys !== null && nVehicleKeys !== null && manifestVersion !== null) {
          manifest = { blake3, nAlertKeys, nVehicleKeys, manifestVersion };
          manifestId = id;
        }
        break;
      }
      case "mmly:PublishedArtifact": {
        const bucketKey = str(node["mmly:bucketKey"]);
        if (bucketKey === null) break;
        artifacts.push({
          id,
          art: {
            name: str(node["mmly:name"]) ?? bucketKey,
            bucketKey,
            derivedFromFeed: false,
            derivedFromManifest: false,
          },
        });
        break;
      }
      default:
        break;
    }
  }

  // wasDerivedFrom edges: mark each artifact's real input(s). Only edges whose
  // used-endpoint is the feed or manifest we actually parsed count — a dangling
  // edge (the emitter cannot produce one) is ignored, never a guessed input.
  const derived = recordSection(doc, "wasDerivedFrom");
  for (const edge of Object.values(derived)) {
    const gen = str(edge["prov:generatedEntity"]);
    const used = str(edge["prov:usedEntity"]);
    if (gen === null || used === null) continue;
    const hit = artifacts.find((a) => a.id === gen);
    if (!hit) continue;
    if (feedId !== null && used === feedId) hit.art.derivedFromFeed = true;
    if (manifestId !== null && used === manifestId) hit.art.derivedFromManifest = true;
  }

  return { run, agent, feed, manifest, artifacts: artifacts.map((a) => a.art) };
}

// The three states a prov_ref fetch can land in, mirroring lib/feed.ts's
// TrainsFeed. There is deliberately no "loading" here — the caller owns that,
// before it has a URL to fetch. And there is no "absent": a snapshot with no
// prov_ref never calls this, because there is no document to point at.
export type ProvChainState =
  | { state: "unavailable"; reason: string }
  | { state: "malformed" }
  | { state: "ready"; chain: ProvChain };

// Fetch and parse a run's PROV-JSON document from its public URL
// (snapshot.provenance.prov_ref). A transport failure or a non-200 is
// `unavailable` (the network could not deliver the document); a body that is
// not JSON, or is JSON the reader cannot anchor to a run, is `malformed` (the
// document arrived but says nothing usable). The two are kept distinct so the
// page can label which kind of gap it hit. No cache option: the versioned
// document is immutable and long-cached upstream, so the browser's own cache
// policy is correct.
export async function fetchProvChain(url: string): Promise<ProvChainState> {
  let body: string;
  try {
    const res = await fetch(url);
    if (!res.ok) return { state: "unavailable", reason: `document returned ${res.status}` };
    body = await res.text();
  } catch (e) {
    return { state: "unavailable", reason: `not reachable (${(e as Error).message})` };
  }
  let raw: unknown;
  try {
    raw = JSON.parse(body);
  } catch {
    return { state: "malformed" };
  }
  const chain = parseProvChain(raw);
  return chain === null ? { state: "malformed" } : { state: "ready", chain };
}
