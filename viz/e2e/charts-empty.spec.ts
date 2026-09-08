import { test, expect } from "@playwright/test";
import type { Page, Route } from "@playwright/test";
import { installFeedMock } from "./mock";

// charts.tsx does the repo's heaviest array/record indexing (per-line transition
// matrices, recovery curves, PIT histograms). nzm1.17 turned on
// noUncheckedIndexedAccess + exactOptionalPropertyTypes and replaced the
// resulting holes with real guards. This proves those guards render — not throw —
// on the two shapes that actually reach them at runtime: a window with no graded
// data, and a params matrix that is ragged (a line missing a whole from-state
// row). A guard that threw here would trip the page's ChartErrorBoundary and
// surface as a console error, which every assertion below forbids.
//
// The static /models view is served from /api/grading; the real feed mock
// (e2e/mock.ts) answers it "not configured", so each case installs the mock for
// the rest of the page, then overrides just /api/grading with a hand-built
// GradingResponse. Registered after the mock so this handler wins.

const WINDOW = { days: 3, from: "2026-09-01", to: "2026-09-04" };
const COUNTS = {
  predictionFiles: 0,
  predictionRecords: 0,
  transitionFiles: 0,
  transitionRecords: 0,
  alertFiles: 0,
  alertVersions: 0,
  alertsCapped: false,
  pointsCapped: false,
};
const STATES = ["normal", "disrupted", "suspended"];

// A fully-populated, internally-consistent recovery report: aligned grid/curves
// (RecoveryDistCurve rejects a misaligned series) and a 10-bin PIT histogram
// (RecoveryScoreCard's verdict validates the bin count). n < 8 incidents lands
// the verdict on "Inconclusive" without needing a hand-tuned PIT shape.
const RECOVERY_DIST = {
  n: 3,
  meanCrps: 10,
  oracleBaselineCrps: 12,
  oracleSkill: 0.1,
  causalBaselineCrps: null,
  causalSkill: null,
  meanPit: 0.5,
  perTick: {
    n: 3,
    meanCrps: 10,
    oracleBaselineCrps: 12,
    oracleSkill: 0.1,
    causalBaselineCrps: null,
    causalSkill: null,
    meanPit: 0.5,
  },
  perRegime: {
    n: 3,
    meanCrps: 10,
    oracleBaselineCrps: 12,
    oracleSkill: 0.1,
    causalBaselineCrps: null,
    causalSkill: null,
    meanPit: 0.5,
  },
  pit: [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
  grid: [0, 60, 120, 180, 240],
  predictedCurve: [0, 0.4, 0.7, 0.9, 1],
  empiricalCurve: [0, 0.5, 0.75, 0.92, 1],
  horizons: [
    { h: 30, predicted: 0.3, observed: 0.35 },
    { h: 60, predicted: 0.5, observed: 0.55 },
    { h: 120, predicted: 0.8, observed: 0.82 },
  ],
};

function gradingResponse(extra: Record<string, unknown>): Record<string, unknown> {
  return {
    configured: true,
    source: "calibration",
    window: WINDOW,
    counts: COUNTS,
    routes: [],
    states: STATES,
    reliability: [],
    recovery: undefined,
    resumeChurn: undefined,
    adherence: undefined,
    detectionLatency: undefined,
    timelines: [],
    heatmap: [],
    paramsTrainedAt: null,
    paramsSelfLoopCap: null,
    generatedAt: 1_756_000_000,
    ...extra,
  };
}

async function serveGrading(page: Page, body: Record<string, unknown>): Promise<void> {
  await page.route("**/api/grading**", (route: Route) =>
    route.fulfill({
      status: 200,
      contentType: "application/json",
      body: JSON.stringify(body),
    }),
  );
}

function watchConsole(page: Page): { errors: string[] } {
  const errors: string[] = [];
  page.on("console", (msg) => {
    if (msg.type() === "error") errors.push(msg.text());
  });
  page.on("pageerror", (err) => errors.push(String(err)));
  return { errors };
}

test("models renders the empty window without tripping a chart guard", async ({
  page,
}, testInfo) => {
  test.skip(testInfo.project.name !== "desktop", "desktop only");
  const mock = await installFeedMock(page);
  const { errors } = watchConsole(page);
  await serveGrading(page, gradingResponse({}));

  await page.goto("/models", { waitUntil: "networkidle" });
  await expect(page.locator(".controls")).toBeVisible();
  // The recovery section always renders in the configured aggregate view.
  await expect(
    page.getByRole("heading", {
      name: "How good are the recovery time estimates?",
    }),
  ).toBeVisible();
  // No params → the transition heatmap must show its empty affordance, not crash.
  await expect(page.locator(".chart-empty")).toContainText(
    "No trained params available yet.",
  );

  expect(errors, errors.join("\n")).toEqual([]);
  expect(mock.externalHits, mock.externalHits.join("\n")).toEqual([]);
});

test("models renders a ragged params matrix and a recovery curve", async ({
  page,
}, testInfo) => {
  test.skip(testInfo.project.name !== "desktop", "desktop only");
  const mock = await installFeedMock(page);
  const { errors } = watchConsole(page);
  // A line whose transition matrix is missing whole from-state rows and columns:
  // the exact hole noUncheckedIndexedAccess exposed. The heatmap must read the
  // absent cells as zero, not index into undefined.
  await serveGrading(
    page,
    gradingResponse({
      recoveryDist: RECOVERY_DIST,
      heatmap: [{ route: "6", transition: [[0.9, 0.1]] }],
    }),
  );

  await page.goto("/models", { waitUntil: "networkidle" });
  await expect(page.locator(".controls")).toBeVisible();
  // recoveryDist.n > 0 → the guarded curve component renders its SVG.
  await expect(
    page.locator(".chart-title").filter({ hasText: "How honest are the odds?" }),
  ).toBeVisible();
  // The ragged matrix renders as a populated heatmap (not the empty affordance).
  await expect(page.locator(".small-multiples svg").first()).toBeVisible();
  await expect(page.locator(".chart-empty")).toHaveCount(0);

  expect(errors, errors.join("\n")).toEqual([]);
  expect(mock.externalHits, mock.externalHits.join("\n")).toEqual([]);
});
