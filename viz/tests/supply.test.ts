import { test } from "node:test";
import assert from "node:assert/strict";
import {
  fmtMinutes,
  fmtProb,
  isRunningHigh,
  supplyBand,
  supplyBars,
  SUPPLY_DEGRADE_RATIO,
  SUPPLY_RECOVER_RATIO,
} from "../lib/feed.ts";
import type { RouteStatus } from "../lib/types.ts";

function route(
  service_condition: string,
  service_ratio: number | null,
  service_low_ratio: number | null = null,
  service_high_ratio: number | null = null,
): RouteStatus {
  // Only the supply fields drive supplyBand/isRunningHigh; the rest are type
  // completeness.
  return {
    route_id: "L",
    condition: "normal",
    condition_source: "movement",
    service_condition,
    service_ratio,
    service_low_ratio,
    service_high_ratio,
    category: "none",
    primary_alert_type: null,
    label: "Good Service",
    alerts: [],
    by_direction: {
      northbound: { alerts: [], primary_alert_type: null },
      southbound: { alerts: [], primary_alert_type: null },
    },
    inference: null,
  } as unknown as RouteStatus;
}

test("supplyBand follows the published regime, not the raw ratio", () => {
  // The regime is debounced and hysteretic, so a degraded route can read back
  // above the degrade floor and must still show as low.
  assert.equal(supplyBand(route("degraded", 0.62)), "low");
  // ...and a route still published normal can read below the floor mid-debounce.
  assert.equal(supplyBand(route("normal", 0.38)), "low");
});

test("supplyBand marks the hysteresis band between degrade and recover", () => {
  assert.equal(supplyBand(route("normal", SUPPLY_DEGRADE_RATIO)), "thin");
  assert.equal(supplyBand(route("normal", SUPPLY_RECOVER_RATIO - 0.01)), "thin");
  assert.equal(supplyBand(route("normal", SUPPLY_RECOVER_RATIO)), "normal");
});

test("supplyBand tops out at normal — over-norm is not a band", () => {
  // `ratio > 1` drives the over-norm marker at the render site, so the band must
  // not carry a competing second notion of "more trains than usual".
  assert.equal(supplyBand(route("normal", 1.0)), "normal");
  assert.equal(supplyBand(route("normal", 1.25)), "normal");
  assert.equal(supplyBand(route("normal", 2.17)), "normal");
});

test("supplyBand is unknown without a judgeable reading", () => {
  assert.equal(supplyBand(route("unknown", null)), "unknown");
  assert.equal(supplyBand(route("normal", null)), "unknown");
  assert.equal(supplyBars(supplyBand(route("unknown", null))), 0);
});

test("supplyBars lights one bar per severity step", () => {
  assert.equal(supplyBars("low"), 1);
  assert.equal(supplyBars("thin"), 2);
  assert.equal(supplyBars("normal"), 3);
});

test("fmtMinutes never prints a 60-minute remainder", () => {
  assert.equal(fmtMinutes(1379.7), "23h");
  assert.equal(fmtMinutes(1380), "23h");
  assert.equal(fmtMinutes(1381), "23h 1m");
  assert.equal(fmtMinutes(0), "—");
});

test("fmtProb never prints a certainty, and never fakes one either", () => {
  // The saturated posterior: exactly 1.0 with the other states at ~1e-17.
  assert.equal(fmtProb(1), ">99.9%");
  assert.equal(fmtProb(2.7e-17), "<0.1%");
  // The bounds are strict — these values ARE 99.9% and 0.1%, so claiming
  // "more than" or "less than" would misreport a number we can render exactly.
  assert.equal(fmtProb(0.999), "99.9%");
  assert.equal(fmtProb(0.001), "0.1%");
  // Just past the bounds, where one decimal would round to a certainty.
  assert.equal(fmtProb(0.9991), ">99.9%");
  assert.equal(fmtProb(0.0009), "<0.1%");
  // The horizon forecast that used to round up to a flat "100%".
  assert.equal(fmtProb(0.9953), "99.5%");
  assert.equal(fmtProb(0.5), "50.0%");
});

test("isRunningHigh compares against the route's own cell, never a global cutoff", () => {
  assert.equal(isRunningHigh(route("normal", 1.6, 0.7, 1.5)), true);
  // At or below the cell's own high ratio is not notable, just its usual top end.
  assert.equal(isRunningHigh(route("normal", 1.5, 0.7, 1.5)), false);
  assert.equal(isRunningHigh(route("normal", 1.4, 0.7, 1.5)), false);
  // No per-cell high ratio (no quantiles for this cell): never fall back to a
  // global multiple, no matter how large the ratio.
  assert.equal(isRunningHigh(route("normal", 5.0, 0.7, null)), false);
  // Unjudgeable route: no ratio to compare in the first place.
  assert.equal(isRunningHigh(route("unknown", null, 0.7, 1.5)), false);
});

test("isRunningHigh regression: an ordinary second mode is not a spike", () => {
  // Route 2, weekend hour 23: 16 trains sits exactly at this cell's own p90
  // (service_ratio == service_high_ratio) — an ordinary second mode, not
  // notable, even though a single global 1.25x cutoff would have flagged it.
  assert.equal(isRunningHigh(route("normal", 1.78, 0.7, 1.78)), false);
  // Route 1: 18 trains is above this cell's own (tighter) p90 — a genuinely
  // rare reading, correctly flagged even though its ratio is lower.
  assert.equal(isRunningHigh(route("normal", 1.64, 0.7, 1.55)), true);
});
