import { test } from "node:test";
import assert from "node:assert/strict";
import { buildTimelines } from "../lib/calibration.ts";
import {
  adherence,
  parseAlertVersion,
  resumeChurn,
  type AlertVersion,
} from "../lib/schedule.ts";
import type { PredictionRecord, TransitionRecord } from "../lib/types.ts";

function ver(o: Partial<AlertVersion>): AlertVersion {
  return {
    id: "lmm:planned_work:1",
    route: "A",
    alertType: "Planned - Part Suspended",
    observedAt: 0,
    windows: [{ start: 1000, end: 5000 }],
    ...o,
  };
}

test("resumeChurn flags a pushed window and reports its magnitude", () => {
  const r = resumeChurn([
    ver({ observedAt: 100, windows: [{ start: 1000, end: 5000 }] }),
    ver({ observedAt: 200, windows: [{ start: 1000, end: 6800 }] }), // +1800s = +30m
  ]);
  assert.equal(r.windows, 1);
  assert.equal(r.pushed, 1);
  assert.equal(r.pulled, 0);
  assert.equal(r.pushedPct, 1);
  assert.deepEqual(r.pushMagnitudesMin, [30]);
  assert.equal(r.byAlertType[0].alertType, "Planned - Part Suspended");
});

test("resumeChurn calls an unchanged window stable", () => {
  const r = resumeChurn([
    ver({ observedAt: 100, windows: [{ start: 1000, end: 5000 }] }),
    ver({ observedAt: 200, windows: [{ start: 1000, end: 5030 }] }), // +30s < eps
  ]);
  assert.equal(r.windows, 1);
  assert.equal(r.stable, 1);
  assert.equal(r.pushed, 0);
});

test("resumeChurn ignores windows seen in only one version", () => {
  const r = resumeChurn([
    ver({
      observedAt: 100,
      windows: [
        { start: 1000, end: 5000 },
        { start: 100000, end: 105000 }, // only here
      ],
    }),
    ver({ observedAt: 200, windows: [{ start: 1000, end: 6800 }] }),
  ]);
  assert.equal(r.windows, 1); // only the recurring start=1000 slot is assessable
  assert.equal(r.pushed, 1);
});

test("parseAlertVersion accepts planned_work and rejects real-time", () => {
  const body = (id: string, alertType: string) => ({
    observed_at: 123,
    alert: {
      id,
      alert: {
        active_period: [{ start: 1000, end: 5000 }],
        informed_entity: [{ route_id: "Z" }],
        "transit_realtime.mercury_alert": { alert_type: alertType },
      },
    },
  });
  const planned = parseAlertVersion(body("lmm:planned_work:9", "No Scheduled Service"));
  assert.ok(planned);
  assert.equal(planned!.route, "Z");
  assert.equal(planned!.observedAt, 123);
  assert.deepEqual(planned!.windows, [{ start: 1000, end: 5000 }]);
  // Real-time alerts have no trustworthy resume time — not a churn subject.
  assert.equal(parseAlertVersion(body("lmm:alert:9", "Delays")), null);
  assert.equal(parseAlertVersion(null), null);
});

// L: normal[1000,2000), suspended[2000,5000), normal[5000,observedUntil)
const transitions: TransitionRecord[] = [
  {
    ts: 2000,
    route: "L",
    prev_state: "normal",
    new_state: "suspended",
    regime_entered_at: 1000,
    exited_at: 2000,
    dwell_sec: 1000,
    alert_type_at_entry: null,
  },
  {
    ts: 5000,
    route: "L",
    prev_state: "suspended",
    new_state: "normal",
    regime_entered_at: 2000,
    exited_at: 5000,
    dwell_sec: 3000,
    alert_type_at_entry: "Planned - Part Suspended",
  },
];

function spred(o: Partial<PredictionRecord>): PredictionRecord {
  return {
    ts: 2300,
    route: "L",
    condition: "suspended",
    p_normal: 0,
    p_disrupted: 0,
    p_suspended: 1,
    regime_entered_at: 2000,
    recovery_minutes: 45,
    recovery_minutes_low: 45,
    recovery_minutes_high: 45,
    recovery_indeterminate: false,
    p_normal_in_30min: 0,
    p_normal_in_60min: 1,
    p_normal_in_120min: 1,
    primary_alert_type: "Planned - Part Suspended",
    params_version: 1,
    recovery_source: "schedule",
    resumes_at: 4800,
    ...o,
  };
}

test("adherence joins announced resume to actual return, deduped per resume", () => {
  const tls = buildTimelines(transitions, 9000);
  const res = adherence(
    [
      spred({ ts: 2300 }),
      spred({ ts: 2600 }), // same (L,4800) → latest kept, one point
      spred({ ts: 4000, recovery_source: "hmm", resumes_at: null }), // not schedule
    ],
    tls,
  );
  assert.equal(res.n, 1);
  // actual normal at 5000, announced 4800 → +200s ≈ +3.3m overran, within tol.
  assert.ok(Math.abs(res.points[0].errorMin - 200 / 60) < 1e-9);
  assert.equal(res.onTimePct, 1);
  assert.equal(res.overrunPct, 0);
});

test("adherence censors a resume with no observed return-to-normal", () => {
  const tls = buildTimelines(transitions, 9000);
  // A route the transition stream never saw flip back to normal.
  const res = adherence([spred({ route: "Q", resumes_at: 4800 })], tls);
  assert.equal(res.n, 0);
  assert.equal(res.censored, 1);
});
