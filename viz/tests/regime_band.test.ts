import { test } from "node:test";
import assert from "node:assert/strict";
import { regimeBands, chooseBucketSec, bucketRuns } from "../lib/regime_band.ts";
import type { PredictionRecord } from "../lib/types.ts";

function pred(p: Partial<PredictionRecord>): PredictionRecord {
  return {
    ts: 0,
    route: "L",
    condition: "normal",
    p_normal: 1,
    p_disrupted: 0,
    p_suspended: 0,
    regime_entered_at: 0,
    recovery_minutes: 0,
    recovery_minutes_low: 0,
    recovery_minutes_high: 0,
    recovery_indeterminate: false,
    p_normal_in_30min: 1,
    p_normal_in_60min: 1,
    p_normal_in_120min: 1,
    primary_alert_type: null,
    params_version: 1,
    ...p,
  };
}

const MIN = 60;

test("chooseBucketSec keeps the grid under the target bucket count", () => {
  // 3 days at 240 buckets wants ~18min; the next step at-or-above is 30min.
  assert.equal(chooseBucketSec(3 * 24 * 3600), 1800);
  // A one-hour span fits in 60s buckets.
  assert.equal(chooseBucketSec(3600), 60);
  for (const span of [600, 86_400, 21 * 86_400]) {
    assert.ok(
      span / chooseBucketSec(span) <= 240,
      `span ${span} exceeded the bucket budget`,
    );
  }
});

test("buckets average the probabilities of every tick that lands in them", () => {
  // The grid is epoch-aligned, so a 60s bucket here spans [960, 1020): ts 1000
  // and 1010 share it, and 1030 would not.
  const bands = regimeBands([
    pred({ ts: 1000, p_normal: 1, p_disrupted: 0, p_suspended: 0 }),
    pred({ ts: 1010, p_normal: 0, p_disrupted: 1, p_suspended: 0 }),
    pred({ ts: 1000 + 30 * MIN, p_normal: 0, p_disrupted: 0, p_suspended: 1 }),
  ]);

  assert.equal(bands.bucketSec, 60);
  const s = bands.series[0];
  if (s === undefined) throw new Error("expected series[0]");
  assert.equal(s.route, "L");
  const bucket = s.buckets[0];
  if (bucket === undefined) throw new Error("expected buckets[0]");
  assert.equal(bucket.t, 960);
  assert.equal(bucket.n, 2);
  assert.equal(bucket.pNormal, 0.5);
  assert.equal(bucket.pDisrupted, 0.5);
  assert.equal(bucket.pSuspended, 0);
  assert.equal(s.n, 3);
});

test("every bucket sums to 1 even when the source tick does not", () => {
  // A tick published at 0.98 total must not weigh less than a clean one.
  const bands = regimeBands([
    pred({ ts: 0, p_normal: 0.49, p_disrupted: 0.49, p_suspended: 0 }),
  ]);
  const s = bands.series[0];
  if (s === undefined) throw new Error("expected series[0]");
  const b = s.buckets[0];
  if (b === undefined) throw new Error("expected buckets[0]");
  assert.ok(Math.abs(b.pNormal + b.pDisrupted + b.pSuspended - 1) < 1e-12);
  assert.ok(Math.abs(b.pNormal - 0.5) < 1e-12);
});

test("degenerate ticks are dropped rather than poisoning a bucket", () => {
  const bands = regimeBands([
    pred({ ts: 0, p_normal: 0, p_disrupted: 0, p_suspended: 0 }),
    pred({ ts: 10, p_normal: Number.NaN, p_disrupted: 1, p_suspended: 0 }),
    pred({ ts: 20, p_normal: 0, p_disrupted: 1, p_suspended: 0 }),
  ]);
  const s = bands.series[0];
  if (s === undefined) throw new Error("expected series[0]");
  const b = s.buckets[0];
  if (b === undefined) throw new Error("expected buckets[0]");
  assert.equal(b.n, 1);
  assert.equal(b.pDisrupted, 1);
});

test("a hole in the archive leaves a gap, not an interpolated bucket", () => {
  // Ticks every minute for 10min, then nothing for an hour, then 10 more.
  const records: PredictionRecord[] = [];
  for (let i = 0; i < 10; i++) records.push(pred({ ts: i * MIN }));
  for (let i = 0; i < 10; i++)
    records.push(pred({ ts: 70 * MIN + i * MIN, p_normal: 0, p_suspended: 1 }));

  const bands = regimeBands(records);
  const s0 = bands.series[0];
  if (s0 === undefined) throw new Error("expected series[0]");
  const ts = s0.buckets.map((b) => b.t);
  // Contiguous buckets would number (t1-t0)/bucketSec; only observed ones exist.
  assert.equal(ts.length, 20);
  assert.equal((bands.t1 - bands.t0) / bands.bucketSec, 80);
  // The gap is a genuine discontinuity in the emitted grid.
  const steps = ts.slice(1).map((t, i) => {
    const prev = ts[i];
    if (prev === undefined) throw new Error("expected previous tick");
    return t - prev;
  });
  assert.equal(steps.filter((d) => d !== bands.bucketSec).length, 1);
  // Last bucket of the first run is minute 9, first of the second is minute 70.
  assert.equal(Math.max(...steps), 61 * MIN);
});

test("bucketRuns splits the fill at archive gaps so none is drawn across them", () => {
  // Same shape the renderer sees: 10 minutes of ticks, an hour of nothing, 10
  // more. Drawing one shape over this would slide a flat edge across the hole.
  const records: PredictionRecord[] = [];
  for (let i = 0; i < 10; i++) records.push(pred({ ts: i * MIN }));
  for (let i = 0; i < 10; i++)
    records.push(pred({ ts: 70 * MIN + i * MIN, p_normal: 0, p_suspended: 1 }));

  const bands = regimeBands(records);
  const s = bands.series[0];
  if (s === undefined) throw new Error("expected series[0]");
  const runs = bucketRuns(s.buckets, bands.bucketSec);
  assert.equal(runs.length, 2);
  assert.deepEqual(
    runs.map((r) => r.length),
    [10, 10],
  );
  // Each run is internally contiguous — that's what lets it be one shape.
  for (const run of runs)
    for (let i = 1; i < run.length; i++) {
      const cur = run[i];
      const prev = run[i - 1];
      if (cur === undefined || prev === undefined) {
        throw new Error("expected contiguous buckets");
      }
      assert.equal(cur.t - prev.t, bands.bucketSec);
    }
  // No bucket is lost to the split.
  assert.equal(
    runs.reduce((a, r) => a + r.length, 0),
    s.buckets.length,
  );
});

test("bucketRuns keeps contiguous buckets as a single shape", () => {
  const records = Array.from({ length: 12 }, (_, i) => pred({ ts: i * MIN }));
  const bands = regimeBands(records);
  const s = bands.series[0];
  if (s === undefined) throw new Error("expected series[0]");
  const runs = bucketRuns(s.buckets, bands.bucketSec);
  assert.equal(runs.length, 1);
  const run0 = runs[0];
  if (run0 === undefined) throw new Error("expected runs[0]");
  assert.equal(run0.length, 12);
});

test("bucketRuns handles an empty series without inventing a run", () => {
  assert.deepEqual(bucketRuns([], 60), []);
});

test("series rank by expected time away from normal, not by tick count", () => {
  const records: PredictionRecord[] = [];
  // "A": sampled 4x per bucket but only mildly unsure.
  for (let i = 0; i < 40; i++)
    records.push(
      pred({
        ts: i * 15,
        route: "A",
        p_normal: 0.9,
        p_disrupted: 0.1,
        p_suspended: 0,
      }),
    );
  // "B": sampled once per bucket and fully suspended — fewer ticks, worse line.
  for (let i = 0; i < 10; i++)
    records.push(
      pred({ ts: i * MIN, route: "B", p_normal: 0, p_disrupted: 0, p_suspended: 1 }),
    );

  const bands = regimeBands(records);
  assert.deepEqual(
    bands.series.map((s) => s.route),
    ["B", "A"],
  );
  const [first, second] = bands.series;
  if (first === undefined || second === undefined) throw new Error("expected two series");
  assert.ok(first.nonNormalMin > second.nonNormalMin);
  // 10 buckets fully non-normal at 1 minute each.
  assert.ok(Math.abs(first.nonNormalMin - 10) < 1e-9);
});

test("flat-normal lines are dropped only when something else was disrupted", () => {
  const quiet = Array.from({ length: 5 }, (_, i) => pred({ ts: i * MIN, route: "Q" }));
  const loud = Array.from({ length: 5 }, (_, i) =>
    pred({ ts: i * MIN, route: "D", p_normal: 0, p_suspended: 1 }),
  );

  // Nothing disrupted anywhere: a flat band is the honest answer, not an
  // empty panel.
  const allQuiet = regimeBands(quiet);
  assert.deepEqual(
    allQuiet.series.map((s) => s.route),
    ["Q"],
  );
  const quietSeries = allQuiet.series[0];
  if (quietSeries === undefined) throw new Error("expected series[0]");
  assert.equal(quietSeries.nonNormalMin, 0);

  // With a disrupted line present, the flat one is noise and gets dropped.
  const mixed = regimeBands([...quiet, ...loud]);
  assert.deepEqual(
    mixed.series.map((s) => s.route),
    ["D"],
  );
});

test("the row cap reports what it dropped", () => {
  const records: PredictionRecord[] = [];
  for (let r = 0; r < 20; r++)
    for (let i = 0; i < 5; i++)
      records.push(
        pred({
          ts: i * MIN,
          route: `R${r}`,
          p_normal: 1 - (r + 1) / 40,
          p_disrupted: (r + 1) / 40,
          p_suspended: 0,
        }),
      );

  const bands = regimeBands(records, { limit: 6 });
  assert.equal(bands.series.length, 6);
  assert.equal(bands.truncated, 14);
  // Worst line first.
  const worst = bands.series[0];
  if (worst === undefined) throw new Error("expected series[0]");
  assert.equal(worst.route, "R19");
});

test("no predictions yields an empty, renderable result", () => {
  const bands = regimeBands([]);
  assert.deepEqual(bands.series, []);
  assert.equal(bands.truncated, 0);
});
