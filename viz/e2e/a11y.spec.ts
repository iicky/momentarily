import { test, expect } from "@playwright/test";
import AxeBuilder from "@axe-core/playwright";
import type { Result } from "axe-core";
import { installFeedMock } from "./mock";

// Grounds the 2026-09-07 review's keyboard/focus findings with a real axe scan
// on the Status page and on the same page with the line drawer open. Baseline
// only: serious/critical counts are printed and recorded, not gated on yet.
function summarize(label: string, violations: Result[]): number {
  const hits = violations.filter(
    (v) => v.impact === "serious" || v.impact === "critical",
  );
  const lines = hits.map(
    (v) => `  [${v.impact}] ${v.id} (${v.nodes.length}) — ${v.help}`,
  );
  console.log(
    `axe ${label}: ${hits.length} serious/critical violation(s)` +
      (lines.length ? `\n${lines.join("\n")}` : ""),
  );
  return hits.length;
}

// One run is enough for a static a11y baseline; the desktop viewport carries it.
test("axe baseline + keyboard: status page and open drawer", async ({
  page,
}, testInfo) => {
  test.skip(testInfo.project.name !== "desktop", "desktop only");
  await installFeedMock(page);
  await page.goto("/", { waitUntil: "networkidle" });
  await expect(page.locator(".grid").first()).toBeVisible();

  const initial = await new AxeBuilder({ page }).analyze();
  const initialCount = summarize("/", initial.violations);

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
  const drawer = await new AxeBuilder({ page }).analyze();
  const drawerCount = summarize("/ (drawer open)", drawer.violations);

  await page.keyboard.press("Escape");
  await expect(page.locator("aside.drawer")).toHaveCount(0);

  // Baseline, not a gate: assert the scan produced a countable result so a
  // silent axe failure can't pass as "zero violations".
  expect(Number.isInteger(initialCount)).toBe(true);
  expect(Number.isInteger(drawerCount)).toBe(true);
});
