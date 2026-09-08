import fs from "node:fs";
import path from "node:path";
import type { Page, Route } from "@playwright/test";

// Playwright transpiles specs to CommonJS (viz has no "type":"module"), so
// __dirname resolves here — import.meta does not.
const FIXTURES = path.join(__dirname, "fixtures");
// Captured once from the live public feed (`curl https://feed.momentarily.nyc/
// v1/{snapshot,trains}.json`); segment_flow trimmed to keep the snapshot under
// 500 KB. Read once at module load, not per request.
const snapshot = fs.readFileSync(path.join(FIXTURES, "snapshot.json"), "utf8");
const trains = fs.readFileSync(path.join(FIXTURES, "trains.json"), "utf8");

const json = (route: Route, body: string) =>
  route.fulfill({ status: 200, contentType: "application/json", body });

// A 1x1 transparent PNG. Served in place of the station page's external
// Wikimedia photo so the offline run neither reaches the network nor logs a
// failed-resource console error — the "no console.error" gate stays unfiltered.
const STUB_PNG = Buffer.from(
  "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk+M8AAAMBAQDJ/pLvAAAAAElFTkSuQmCC",
  "base64",
);

export interface FeedMock {
  /** URLs the page tried to reach off-box — must stay empty for the run to be
   *  offline. Populated only by requests the handler aborts. */
  externalHits: string[];
}

/**
 * Route every request the page makes. The public feed and the three same-origin
 * API routes are served from committed fixtures (so the local server never
 * reaches data.ny.gov or R2, and no secrets are needed); same-origin app assets
 * pass through to `next start`; anything else is aborted and recorded. That last
 * clause is the disconnect check: a real page load must leave `externalHits`
 * empty.
 */
export async function installFeedMock(page: Page): Promise<FeedMock> {
  const mock: FeedMock = { externalHits: [] };
  await page.route("**/*", (route) => {
    const url = new URL(route.request().url());
    const p = url.pathname;

    if (p.endsWith("/v1/snapshot.json")) return json(route, snapshot);
    if (p.endsWith("/v1/trains.json")) return json(route, trains);
    // The station page's coordinate source (server-side hits data.ny.gov) — an
    // empty set renders the page without its coordinate block, which is enough
    // for a smoke pass and keeps the run offline.
    if (p === "/api/stations") return json(route, '{"stations":{}}');
    // The Models grading + movement recompute both read R2 server-side. Report
    // "not configured" so the page renders its credential notice deterministically
    // rather than the local server blocking on a credentialed fetch.
    if (p.startsWith("/api/grading")) return json(route, '{"configured":false}');
    if (p.startsWith("/api/movement")) return json(route, '{"configured":false}');

    if (url.hostname === "localhost" || url.hostname === "127.0.0.1") {
      return route.continue();
    }

    // Offline: nothing off-box may reach the network. An external image the
    // page references by design (the station page's Wikimedia photo) is served
    // a 1x1 stub, so the run stays offline without turning a real failed
    // resource into an ignored console error. A *data* dependency
    // (document/xhr/fetch) reaching here is an un-mocked leak the offline
    // assertion must catch; everything else is aborted.
    const type = route.request().resourceType();
    if (type === "image") {
      return route.fulfill({ status: 200, contentType: "image/png", body: STUB_PNG });
    }
    if (type === "document" || type === "xhr" || type === "fetch") {
      mock.externalHits.push(route.request().url());
    }
    return route.abort();
  });
  return mock;
}
