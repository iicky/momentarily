// The PROV-JSON reader (lib/prov.ts). The fixtures are NOT hand-written: they
// are emitted by training/prov.py's build_trainer_run (see
// scripts/gen_prov_fixtures.py), so this test pins the reader against the exact
// bytes the real producer ships. The two degraded-document cases (fetch failed,
// malformed body) are constructed here because they are the absence or
// corruption of a document, which no producer emits.

import { test } from "node:test";
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { parseProvChain } from "../lib/prov.ts";

function fixture(name: string): unknown {
  const path = fileURLToPath(new URL(`./fixtures/${name}`, import.meta.url));
  return JSON.parse(readFileSync(path, "utf8"));
}

const FULL = fixture("prov_full.json");
const FEEDLESS = fixture("prov_feedless.json");

test("a full run parses into feed, run, agent, manifest, and artifacts", () => {
  const chain = parseProvChain(FULL);
  assert.ok(chain, "expected a chain");

  assert.equal(chain.run.trainedAt, 1788229972);
  assert.equal(chain.run.startedAt, 1788229900);
  assert.equal(chain.run.endedAt, 1788229972);

  assert.deepEqual(chain.agent, {
    codeSha: "9f3c1ab7d240e5c8",
    dirty: true,
    producer: "train_em",
  });

  assert.deepEqual(chain.feed, {
    version: "20260807-H-rockaways-extension-removed",
    sha256: "e3" + "b".repeat(62),
    start: "2026-08-07",
    end: "2026-09-04",
  });

  assert.deepEqual(chain.manifest, {
    blake3: "7c" + "0".repeat(62),
    nAlertKeys: 4032,
    nVehicleKeys: 4032,
    manifestVersion: 2,
  });

  assert.equal(chain.artifacts.length, 4);
});

test("wasDerivedFrom edges are reconstructed per artifact", () => {
  const chain = parseProvChain(FULL);
  assert.ok(chain);
  const by = (name: string) => chain.artifacts.find((a) => a.name === name);

  const params = by("params");
  assert.ok(params);
  assert.equal(params.derivedFromManifest, true);
  assert.equal(params.derivedFromFeed, false);

  const headway = by("scheduled_headway");
  assert.ok(headway);
  assert.equal(headway.derivedFromFeed, true);
  assert.equal(headway.derivedFromManifest, false);

  const segment = by("segment_params");
  assert.ok(segment);
  assert.equal(segment.derivedFromFeed, true);
  assert.equal(segment.derivedFromManifest, true);

  const baseline = by("service_baseline");
  assert.ok(baseline);
  assert.equal(baseline.derivedFromFeed, false);
  assert.equal(baseline.derivedFromManifest, false);
});

test("a feedless run yields no feed node and no feed-derived edges", () => {
  const chain = parseProvChain(FEEDLESS);
  assert.ok(chain);
  assert.equal(chain.feed, null);
  assert.ok(chain.manifest);
  // The one artifact was derived from the manifest only; there is no feed to be
  // derived from, so the reader must never flip derivedFromFeed on.
  assert.ok(chain.artifacts.every((a) => a.derivedFromFeed === false));
});

test("a missing dirty flag reads as null, not false", () => {
  const chain = parseProvChain(FEEDLESS);
  assert.ok(chain);
  assert.ok(chain.agent);
  assert.equal(chain.agent.dirty, null);
});

test("a malformed body with no trainer-run activity parses to null", () => {
  assert.equal(parseProvChain({ entity: {}, activity: {}, agent: {} }), null);
  assert.equal(parseProvChain("not a document"), null);
  assert.equal(parseProvChain(null), null);
  assert.equal(parseProvChain([1, 2, 3]), null);
});

test("a trainer-run activity with an unparseable timestamp parses to null", () => {
  const broken = {
    activity: {
      "mmly:run/1788229972": {
        "prov:type": "mmly:TrainerRun",
        "prov:startedAtTime": { $: "not-a-date", type: "xsd:dateTime" },
        "prov:endedAtTime": { $: "2026-09-01T02:32:52+00:00", type: "xsd:dateTime" },
      },
    },
  };
  assert.equal(parseProvChain(broken), null);
});
