// The ingest-hardening contracts: every network
// read is time-bounded, the snapshot shape is validated before it is trusted,
// and a snapshot-supplied prov_ref is refused unless it is an https URL on the
// feed's own host under the public v1/prov/ prefix.

import { test, afterEach } from "node:test";
import assert from "node:assert/strict";
import {
  fetchSnapshot,
  SnapshotShapeError,
  isSnapshotShape,
  isAllowedProvRef,
} from "../lib/feed.ts";
import { fetchProvChain } from "../lib/prov.ts";

const realFetch = globalThis.fetch;
afterEach(() => {
  globalThis.fetch = realFetch;
});

// A well-formed snapshot carries every top-level field a render path reads.
function fullSnapshot(): Record<string, unknown> {
  return {
    generated_at: 1788229972,
    provenance: {},
    system: {},
    route_status: {},
    station_status: {},
    stations: {},
  };
}

test("a fetch that never resolves rejects within the timeout", async () => {
  // The mock honours the abort signal the way a real transport does — it hangs
  // until aborted — so this asserts fetchSnapshot's own timeout wiring fires.
  // Promise.withResolvers is ES2024; this package targets ES2022, so the
  // executor form is the available way to reject on abort.
  globalThis.fetch = ((_url: string, opts?: { signal?: AbortSignal }) =>
    new Promise<Response>((_, reject) => {
      opts?.signal?.addEventListener("abort", () => reject(opts.signal!.reason));
    })) as typeof fetch;

  const start = Date.now();
  await assert.rejects(fetchSnapshot(40));
  assert.ok(Date.now() - start < 2_000, "should reject well inside the poll interval");
});

test("a snapshot missing route_status is refused instead of trusted", async () => {
  const partial = fullSnapshot();
  delete partial.route_status;
  assert.equal(isSnapshotShape(partial), false);
  assert.equal(isSnapshotShape(fullSnapshot()), true);

  globalThis.fetch = (async () =>
    new Response(JSON.stringify(partial), { status: 200 })) as typeof fetch;

  // Rejecting with the distinct shape error is what lets the page hold last-good
  // data behind the degraded banner rather than render the partial body.
  await assert.rejects(fetchSnapshot(), (e) => e instanceof SnapshotShapeError);
});

test("a prov_ref on a foreign host is refused without fetching", async () => {
  let called = false;
  globalThis.fetch = (async () => {
    called = true;
    return new Response("{}");
  }) as typeof fetch;

  const res = await fetchProvChain("https://evil.example.com/v1/prov/run-1.json");
  assert.equal(res.state, "unavailable");
  assert.match(res.state === "unavailable" ? res.reason : "", /refused/);
  assert.equal(called, false, "a refused ref must never reach the network");
});

test("a PROV body over the cap is refused mid-stream, not buffered whole", async () => {
  // Emits > 1MB (the cap) in one chunk with no content-length, the case a
  // spoofing server would use; the streamed read must stop and refuse rather
  // than decode the whole body.
  const oversize = new Uint8Array(1_000_001);
  globalThis.fetch = (async () =>
    new Response(
      new ReadableStream({
        start(c) {
          c.enqueue(oversize);
          c.close();
        },
      }),
    )) as typeof fetch;

  const res = await fetchProvChain("https://feed.momentarily.nyc/v1/prov/run-1.json");
  assert.equal(res.state, "unavailable");
  assert.match(res.state === "unavailable" ? res.reason : "", /too large/);
});

test("isAllowedProvRef enforces https, the feed host, and the v1/prov/ prefix", () => {
  assert.equal(isAllowedProvRef("https://feed.momentarily.nyc/v1/prov/run-1.json"), true);
  assert.equal(isAllowedProvRef("http://feed.momentarily.nyc/v1/prov/run-1.json"), false);
  assert.equal(isAllowedProvRef("https://evil.example.com/v1/prov/run-1.json"), false);
  assert.equal(isAllowedProvRef("https://feed.momentarily.nyc/v1/snapshot.json"), false);
  assert.equal(isAllowedProvRef("not a url"), false);
});
