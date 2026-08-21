import { NextRequest, NextResponse } from "next/server";
import {
  r2Configured,
  listKeys,
  getText,
  getJson,
  parseJsonl,
  utcDateWindow,
  STREAMS,
} from "@/lib/r2";
import {
  buildTimelines,
  reliability,
  recoveryError,
  recoveryOutcomes,
  detectionLatency,
  routeUniverse,
} from "@/lib/calibration";
import { recoveryDistReport, type RecoveryDistSample } from "@/lib/recovery_dist";
import { predictedRecoveryCurve } from "@/lib/dwell";
import {
  adherence,
  parseAlertVersion,
  resumeChurn,
  type AlertVersion,
} from "@/lib/schedule";
import type {
  PredictionRecord,
  TransitionRecord,
  HeatmapEntry,
  GradingResponse,
} from "@/lib/types";
import {
  fetchCalibration,
  calibrationReliability,
  calibrationHeatmap,
  calibrationRoutes,
  calibrationForLine,
} from "@/lib/calibrationFeed";
import { cachedRecords } from "@/lib/r2cache";

export const runtime = "nodejs";
export const dynamic = "force-dynamic";

interface TrainedParamsDoc {
  trained_at?: number;
  routes?: Record<
    string,
    {
      transition?: number[][];
      dwell_quantiles?: Record<
        string,
        {
          curve_sec?: number[];
          tail_ll?: [number, number];
          atom_p?: number;
          atom_sec?: number;
        }
      >;
    }
  >;
}

const STATES = ["normal", "disrupted", "suspended"];
const MAX_DAYS = 21;
const POINT_CAP = 3000; // scatter points returned to the client
const FETCH_CONCURRENCY = 16;
// Bound the alert-archive read (one object per alert version): a runaway window
// shouldn't fan out into unbounded GETs. Surfaced as alertsCapped when hit.
const ALERT_CAP = 30000;

/** Resolve thunks with a bounded concurrency pool. */
async function pool<T>(
  items: (() => Promise<T>)[],
  limit: number,
): Promise<T[]> {
  const out: T[] = new Array(items.length);
  let i = 0;
  const workers = Array.from({ length: Math.min(limit, items.length) }, async () => {
    while (i < items.length) {
      const idx = i++;
      out[idx] = await items[idx]();
    }
  });
  await Promise.all(workers);
  return out;
}

// The dominant cost of this route is the ~1,600 R2 GETs, not the grading math.
// Each object key is immutable, so cachedRecords memoizes it for the process
// lifetime (shared with the movement route via lib/r2cache) — a line switch or
// repeat load re-lists (cheap) and reuses every already-seen object.
const parseAlert = (text: string): AlertVersion[] => {
  try {
    const v = parseAlertVersion(JSON.parse(text));
    return v ? [v] : [];
  } catch {
    return []; // skip a malformed archive object rather than failing the window
  }
};

async function readStream<T>(prefix: string, dates: string[]): Promise<{
  records: T[];
  files: number;
}> {
  const keyLists = await pool(
    dates.map((d) => () => listKeys(`${prefix}/${d}/`)),
    FETCH_CONCURRENCY,
  );
  const keys = keyLists.flat();
  const per = await pool(
    keys.map((k) => () => cachedRecords<T>(k, (t) => parseJsonl<T>(t))),
    FETCH_CONCURRENCY,
  );
  return { records: per.flat(), files: keys.length };
}

/** Planned-work alert archive. Keys are capped BEFORE fetching so a runaway
 * window can't fan out into unbounded GETs; each version is cached by key. */
async function readAlertVersions(dates: string[]): Promise<{
  versions: AlertVersion[];
  files: number;
  capped: boolean;
}> {
  const keyLists = await pool(
    dates.map((d) => () => listKeys(`${STREAMS.alerts}/${d}/`)),
    FETCH_CONCURRENCY,
  );
  let keys = keyLists.flat();
  const capped = keys.length > ALERT_CAP;
  if (capped) keys = keys.slice(0, ALERT_CAP);
  const per = await pool(
    keys.map((k) => () => cachedRecords<AlertVersion>(k, parseAlert)),
    FETCH_CONCURRENCY,
  );
  return { versions: per.flat(), files: keys.length, capped };
}

export async function GET(req: NextRequest) {
  const sp = req.nextUrl.searchParams;
  const days = Math.min(MAX_DAYS, Math.max(1, Number(sp.get("days") ?? 3)));
  const routeFilter = sp.get("route");
  const nowSec = Math.floor(Date.now() / 1000);
  const dates = utcDateWindow(days, Date.now());

  // Default path: the prebuilt static feed (v1/calibration.json). Fast, needs no
  // R2 LIST, and now carries per-line reliability + recovery so line-filtering
  // works here too — including on the credential-less public deploy. The
  // per-point drilldowns (scatter, swimlane, detection, movement) are NOT in the
  // feed; they're the opt-in `?source=streams` credentialed recompute below.
  const wantStreams = sp.get("source") === "streams";
  const configured = await r2Configured();
  if (!wantStreams || !configured) {
    try {
      const doc = await fetchCalibration();
      const line = routeFilter ? calibrationForLine(doc, routeFilter) : null;
      const payload: GradingResponse = {
        configured: true,
        source: "calibration",
        window: { days, from: dates[dates.length - 1], to: dates[0] },
        counts: {
          predictionFiles: 0,
          predictionRecords:
            (routeFilter && doc.by_line?.[routeFilter]?.n_predictions) ||
            doc.predictions_seen,
          transitionFiles: 0,
          transitionRecords: doc.transitions_seen,
          alertFiles: 0,
          alertVersions: 0,
          alertsCapped: false,
          pointsCapped: false,
        },
        routes: calibrationRoutes(doc),
        states: doc.transition_matrices.states ?? STATES,
        reliability: line ? line.reliability : calibrationReliability(doc),
        recovery: line ? line.recovery : doc.recovery,
        resumeChurn: null,
        adherence: null,
        detectionLatency: null,
        timelines: [],
        heatmap: calibrationHeatmap(doc),
        paramsTrainedAt: doc.transition_matrices.trained_at ?? null,
        generatedAt: doc.generated_at ?? null,
        drift: doc.drift,
      };
      return NextResponse.json(payload);
    } catch {
      if (!configured) {
        return NextResponse.json({
          configured: false,
          error:
            "The public calibration feed is unreachable and no R2 credentials are set. " +
            "Add R2_ACCOUNT_ID, R2_ACCESS_KEY_ID, R2_SECRET_ACCESS_KEY to viz/.env.local, " +
            "or set NEXT_PUBLIC_FEED_BASE to a reachable feed.",
          window: { days, from: dates[dates.length - 1], to: dates[0] },
        });
      }
      // Static feed unreadable but credentialed — fall through to the streams
      // recompute rather than fail.
    }
  }

  try {
    const [predRes, transRes, alertRes] = await Promise.all([
      readStream<PredictionRecord>(STREAMS.predictions, dates),
      readStream<TransitionRecord>(STREAMS.transitions, dates),
      readAlertVersions(dates),
    ]);

    let predictions = predRes.records;
    let transitions = transRes.records;
    let alertVersions = alertRes.versions;
    if (routeFilter) {
      predictions = predictions.filter((p) => p.route === routeFilter);
      transitions = transitions.filter((t) => t.route === routeFilter);
      alertVersions = alertVersions.filter((v) => v.route === routeFilter);
    }

    const timelines = buildTimelines(transitions, nowSec);
    const rel = [30, 60, 120].map((h) => reliability(predictions, timelines, h));
    const rec = recoveryError(predictions, timelines);
    const churn = resumeChurn(alertVersions);
    const adher = adherence(predictions, timelines);
    const detection = detectionLatency(predictions);

    // Cap scatter points to keep the payload small.
    const pointsCapped = rec.points.length > POINT_CAP;
    if (pointsCapped) {
      const stride = Math.ceil(rec.points.length / POINT_CAP);
      rec.points = rec.points.filter((_, i) => i % stride === 0);
    }

    // Trained params drive both the transition heatmaps and the reconstructed
    // recovery curves (full dwell curve, not the three published checkpoints).
    let heatmap: HeatmapEntry[] = [];
    let paramsTrainedAt: number | null = null;
    let params: TrainedParamsDoc | null = null;
    try {
      params = await getJson<TrainedParamsDoc>(STREAMS.params);
      paramsTrainedAt = params.trained_at ?? null;
      heatmap = Object.entries(params.routes ?? {})
        .filter(([r]) => !routeFilter || r === routeFilter)
        .map(([route, p]) => ({ route, transition: p.transition ?? [] }))
        .filter((h) => h.transition.length === 3);
    } catch {
      // params may be absent before the first weekly train — non-fatal.
    }

    // Rebuild each gradeable prediction's full recovery curve from its dwell
    // cell, then score the distribution (CRPS / PIT). Outcomes whose cell lacks
    // a curve are skipped (the curve view needs the real distribution).
    const recoverySamples: RecoveryDistSample[] = [];
    for (const o of recoveryOutcomes(predictions, timelines)) {
      const cell = params?.routes?.[o.route]?.dwell_quantiles?.[o.condition];
      if (!cell?.curve_sec || cell.curve_sec.length < 2) continue;
      const elapsedSec = Math.max(0, o.ts - o.regimeEnteredAt);
      // Mixture cells still publish curve_sec (see viz/lib/dwell.ts), but the
      // closed-form atom path supersedes it — only fall back to the curve
      // when the atom fields are missing or malformed.
      const atom =
        cell.atom_p !== undefined &&
        cell.atom_sec !== undefined &&
        cell.atom_p > 0 &&
        cell.atom_p < 1
          ? { p: cell.atom_p, sec: cell.atom_sec }
          : undefined;
      // Where the jump sits differs from training/scorecard.py's producer, and
      // the difference is easy to get wrong. That grader scores each episode
      // once from onset, so the atom lands at the realized duration itself.
      // Here every forecast tick is graded, `actualMin` is the time REMAINING
      // from that tick (recoveryOutcomes: (nn - pr.ts) / 60), and the predicted
      // curve is conditioned on having already survived elapsedSec. So the jump
      // is at atom.sec - elapsedSec in remaining-time terms, i.e. it is the
      // TOTAL dwell that has to land on the atom.
      const totalSec = elapsedSec + o.actualMin * 60;
      // Once the regime has outlived the atom there is no jump left to spread
      // across: the atom cancels out of the conditional and the curve is
      // continuous from here on, so these outcomes must be left alone.
      const predLeft =
        atom !== undefined &&
        elapsedSec < atom.sec &&
        Math.abs(totalSec - atom.sec) < 1.0
          ? 0.0
          : undefined;
      recoverySamples.push({
        predCurve: predictedRecoveryCurve(elapsedSec, cell.curve_sec, cell.tail_ll, atom),
        actualMin: o.actualMin,
        regimeKey: `${o.route}:${o.regimeEnteredAt}`,
        predLeft,
      });
    }
    const recoveryDist = recoveryDistReport(recoverySamples);

    return NextResponse.json({
      configured: true,
      source: "streams",
      window: { days, from: dates[dates.length - 1], to: dates[0] },
      counts: {
        predictionFiles: predRes.files,
        predictionRecords: predRes.records.length,
        transitionFiles: transRes.files,
        transitionRecords: transRes.records.length,
        alertFiles: alertRes.files,
        alertVersions: alertRes.versions.length,
        alertsCapped: alertRes.capped,
        pointsCapped,
      },
      routes: routeUniverse(predRes.records, transRes.records),
      states: STATES,
      reliability: rel,
      recovery: rec,
      recoveryDist,
      resumeChurn: churn,
      adherence: adher,
      detectionLatency: detection,
      timelines: [...timelines.values()].map((t) => ({
        route: t.route,
        segments: t.segments,
        observedUntil: t.observedUntil,
      })),
      heatmap,
      paramsTrainedAt,
      generatedAt: null, // credentialed read runs live up to "now"
    });
  } catch (e) {
    return NextResponse.json(
      { configured: true, error: (e as Error).message },
      { status: 502 },
    );
  }
}
