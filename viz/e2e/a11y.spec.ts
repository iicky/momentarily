import { test, expect } from "@playwright/test";
import AxeBuilder from "@axe-core/playwright";
import type { Result } from "axe-core";
import { installFeedMock } from "./mock";

// Grounds the 2026-09-07 review's keyboard/focus findings with a real axe scan
// on the Status page (closed and with the line drawer open) and on /about. The
// contrast baseline (nzm1.24) is now fixed, so this gates: any serious/critical
// violation fails the run rather than being printed and ignored.
function gate(label: string, violations: Result[]): void {
  const hits = violations.filter(
    (v) => v.impact === "serious" || v.impact === "critical",
  );
  const lines = hits.map(
    (v) =>
      `  [${v.impact}] ${v.id} (${v.nodes.length}) — ${v.help}\n` +
      v.nodes.map((n) => `      ${n.target.join(" ")}`).join("\n"),
  );
  console.log(
    `axe ${label}: ${hits.length} serious/critical violation(s)` +
      (lines.length ? `\n${lines.join("\n")}` : ""),
  );
  expect(hits, `axe ${label}: serious/critical violations\n${lines.join("\n")}`).toEqual([]);
}

// One run is enough for a static a11y gate; the desktop viewport carries it.
test("axe gate + keyboard: status page, open drawer, and /about", async ({
  page,
}, testInfo) => {
  test.skip(testInfo.project.name !== "desktop", "desktop only");
  await installFeedMock(page);
  await page.goto("/", { waitUntil: "networkidle" });
  await expect(page.locator(".grid").first()).toBeVisible();

  gate("/", (await new AxeBuilder({ page }).analyze()).violations);

  // Keyboard path the review flagged: Tab must reach the first status card, Enter
  // opens the drawer, Escape closes it. Tab from the header controls until focus
  // lands on the first card rather than assuming a fixed number of stops.
  const firstCard = page.locator(".grid .card").first();
  let reached = false;
  for (let i = 0; i < 25; i++) {
    await page.keyboard.press("Tab");
    if (await firstCard.evaluate((el) => el === document.activeElement)) {
      reached = true;
      break;
    }
  }
  expect(reached, "Tab never reached the first status card").toBe(true);

  await page.keyboard.press("Enter");
  await expect(page.locator("aside.drawer")).toBeVisible();

  // axe with the drawer open, reached by keyboard — the state the review cared about.
  gate("/ (drawer open)", (await new AxeBuilder({ page }).analyze()).violations);

  await page.keyboard.press("Escape");
  await expect(page.locator("aside.drawer")).toHaveCount(0);

  // Tokens are global, so /about (no route bullets, all body/muted text) is the
  // second surface the review named; gate it too.
  await page.goto("/about", { waitUntil: "networkidle" });
  gate("/about", (await new AxeBuilder({ page }).analyze()).violations);
});
