import { defineConfig } from "@playwright/test";

// Local-only visual + smoke pass for the dashboard. Not wired into the publish
// path; the value is (a) a runnable screenshot pass for reviewers and (b) the
// console-error / hydration / a11y gates. See viz/README.md ("End-to-end").
//
// The feed is mocked in every spec (e2e/mock.ts), so the run needs no network
// and no secrets: the two viewports below are the only knobs. Served by a
// production build (`next build && next start`) rather than `next dev` — dev
// compiles each route on first navigation, which is both slower and a source of
// dev-only overlay warnings that would trip the no-console-error gate.
const PORT = 39207;

export default defineConfig({
  testDir: "./e2e",
  outputDir: "./e2e/.artifacts",
  fullyParallel: true,
  forbidOnly: !!process.env.CI,
  retries: 0,
  reporter: [["list"]],
  timeout: 30_000,
  expect: { timeout: 10_000 },
  use: {
    baseURL: `http://localhost:${PORT}`,
    // Fail the action rather than silently waiting on a request the offline
    // mock has aborted.
    actionTimeout: 10_000,
  },
  projects: [
    { name: "desktop", use: { viewport: { width: 1280, height: 800 } } },
    { name: "mobile", use: { viewport: { width: 390, height: 844 } } },
  ],
  webServer: {
    command: "npm run build && npm run start:e2e",
    url: `http://localhost:${PORT}`,
    reuseExistingServer: !process.env.CI,
    timeout: 120_000,
    stdout: "ignore",
    stderr: "pipe",
  },
});
