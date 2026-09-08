import { test, expect } from "@playwright/test";
import { installFeedMock } from "./mock";

// Screenshot + gate pass across the five views at both viewports. Each route
// must load against the mocked feed, reach a key landmark, emit no console
// error and no hydration warning, and touch nothing off-box.
const ROUTES = [
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
