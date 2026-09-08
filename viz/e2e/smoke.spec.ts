import { test, expect } from "@playwright/test";
import { installFeedMock } from "./mock";

// Screenshot + gate pass across the five views at both viewports. Each route
// must load against the mocked feed, reach a key landmark, emit no console
// error and no hydration warning, and touch nothing off-box.
const ROUTES = [
  { slug: "about", path: "/about", landmark: ".grp" },
  { slug: "home", path: "/", landmark: ".grid" },
  { slug: "lines", path: "/lines", landmark: ".line-grid" },
  { slug: "map", path: "/map", landmark: "svg.diagram" },
  { slug: "models", path: "/models", landmark: ".controls" },
  // Stop 635 (14 St-Union Sq) is present in the fixture snapshot's stations.
  { slug: "station-635", path: "/stations/635", landmark: ".line-head h2" },
] as const;

for (const route of ROUTES) {
  test(`${route.slug} loads clean`, async ({ page }, testInfo) => {
    const mock = await installFeedMock(page);

    const consoleErrors: string[] = [];
    const hydrationWarnings: string[] = [];
    page.on("console", (msg) => {
      const text = msg.text();
      if (msg.type() === "error") consoleErrors.push(text);
      // React reports a hydration mismatch as an error whose text names it; keep
      // a dedicated bucket so a failure says "hydration" rather than a generic
      // console error.
      if (/hydrat|did not match|did not expect server html/i.test(text)) {
        hydrationWarnings.push(text);
      }
    });
    page.on("pageerror", (err) => consoleErrors.push(String(err)));

    await page.goto(route.path, { waitUntil: "networkidle" });
    await expect(page.locator(route.landmark).first()).toBeVisible();

    const viewport = testInfo.project.name; // "desktop" | "mobile"
    await page.screenshot({
      path: `e2e/__screenshots__/${route.slug}-${viewport}.png`,
      fullPage: true,
    });

    expect(hydrationWarnings, hydrationWarnings.join("\n")).toEqual([]);
    expect(consoleErrors, consoleErrors.join("\n")).toEqual([]);
    // Offline: the page reached nothing off-box.
    expect(mock.externalHits, mock.externalHits.join("\n")).toEqual([]);
  });
}

// The global nav grew to seven items; at a phone width the single row used to
// overflow and clip the last link. It now wraps onto a second row, so the nav
// must never scroll horizontally and every item must stay fully reachable with
// a 40px-tall tap target.
test("nav wraps without clipping at 390px", async ({ page }) => {
  await installFeedMock(page);
  await page.setViewportSize({ width: 390, height: 844 });
  await page.goto("/", { waitUntil: "networkidle" });
  const nav = page.locator("nav.nav");
  await expect(nav).toBeVisible();
  const overflow = await nav.evaluate((el) => el.scrollWidth - el.clientWidth);
  expect(overflow, `nav overflows its box by ${overflow}px`).toBeLessThanOrEqual(0);

  // The last item (About) is the one that used to clip: it must be visible,
  // sit inside the nav's box, and give a >=40px tap target.
  const last = nav.locator("a").last();
  await expect(last).toHaveText("About");
  await expect(last).toBeVisible();
  const fits = await last.evaluate((el) => {
    const a = el.getBoundingClientRect();
    const box = el.closest("nav.nav")!.getBoundingClientRect();
    return { h: a.height, within: a.right <= box.right + 0.5 && a.left >= box.left - 0.5 };
  });
  expect(fits.within, "last nav item overflows the nav box").toBe(true);
  expect(fits.h, `last nav item is only ${fits.h}px tall`).toBeGreaterThanOrEqual(40);
});

// The 2026-09-07 review's front-door findings, asserted against the mocked feed
// (0 lines disrupted, 26 lines with advisories, train-position freshness live):
// the banner leads with the disrupted count, alert volume is demoted, no HMM
// jargon reaches the page, and the freshness strip carries the train dot.
test("front door reads in rider language", async ({ page }, testInfo) => {
  test.skip(testInfo.project.name !== "desktop", "desktop only");
  await installFeedMock(page);
  await page.goto("/", { waitUntil: "networkidle" });
  await expect(page.locator(".grid").first()).toBeVisible();

  await expect(page.locator(".banner-lead .label")).toHaveText(
    "No lines disrupted or suspended",
  );
  await expect(page.locator(".banner-advisories")).toContainText(
    "26 lines have advisories",
  );
  // No model jargon on the front door — "HMM" stays scoped to /models.
  await expect(page.locator("body")).not.toContainText("HMM");
  // The freshness strip gained the train-position dot bound to vehicle_positions.
  await expect(page.locator(".freshness")).toContainText("Train positions");
});
