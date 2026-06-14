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
  detectionLatency,
  routeUniverse,
} from "@/lib/calibration";
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
} from "@/lib/calibrationFeed";

export const runtime = "nodejs";
export const dynamic = "force-dynamic";

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

async function readStream<T>(prefix: string, dates: string[]): Promise<{
  records: T[];
  files: number;
}> {
  const keyLists = await pool(
    dates.map((d) => () => listKeys(`${prefix}/${d}/`)),
    FETCH_CONCURRENCY,
  );
  const keys = keyLists.flat();
  const blobs = await pool(
    keys.map((k) => () => getText(k)),
    FETCH_CONCURRENCY,
  );
  const records = blobs.flatMap((b) => parseJsonl<T>(b));
  return { records, files: keys.length };
}

/** Read the planned-work alert archive (one JSON object per version). */
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
  const blobs = await pool(
    keys.map((k) => () => getText(k)),
    FETCH_CONCURRENCY,
  );
  const versions: AlertVersion[] = [];
  for (const b of blobs) {
    try {
      const v = parseAlertVersion(JSON.parse(b));
      if (v) versions.push(v);
    } catch {
      // skip a malformed archive object rather than failing the whole window
    }
  }
  return { versions, files: keys.length, capped };
}

export async function GET(req: NextRequest) {
  const sp = req.nextUrl.searchParams;
  const days = Math.min(MAX_DAYS, Math.max(1, Number(sp.get("days") ?? 3)));
  const routeFilter = sp.get("route");
  const nowSec = Math.floor(Date.now() / 1000);
  const dates = utcDateWindow(days, Date.now());

  if (!(await r2Configured())) {
    // No credentials → fall back to the public aggregate feed. Powers the
    // reliability, recovery-summary, and transition charts without R2 access;
    // the per-point drilldowns simply aren't in calibration.json.
    try {
      const doc = await fetchCalibration();
      const payload: GradingResponse = {
        configured: true,
        source: "calibration",
        window: { days, from: dates[dates.length - 1], to: dates[0] },
        counts: {
          predictionFiles: 0,
          predictionRecords: doc.predictions_seen,
          transitionFiles: 0,
          transitionRecords: doc.transitions_seen,
          alertFiles: 0,
          alertVersions: 0,
          alertsCapped: false,
          pointsCapped: false,
        },
        routes: [],
        states: doc.transition_matrices.states ?? STATES,
        reliability: calibrationReliability(doc),
        recovery: doc.recovery,
        resumeChurn: null,
        adherence: null,
        detectionLatency: null,
        timelines: [],
        heatmap: calibrationHeatmap(doc),
        paramsTrainedAt: doc.transition_matrices.trained_at ?? null,
      };
      return NextResponse.json(payload);
    } catch {
      return NextResponse.json({
        configured: false,
        error:
          "R2 credentials not set and the public calibration feed is unreachable. " +
          "Add R2_ACCOUNT_ID, R2_ACCESS_KEY_ID, R2_SECRET_ACCESS_KEY to viz/.env.local " +
          "for the full history, or set NEXT_PUBLIC_FEED_BASE to a reachable feed.",
        window: { days, from: dates[dates.length - 1], to: dates[0] },
      });
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

    // Transition matrices from the trained params.
    let heatmap: HeatmapEntry[] = [];
    let paramsTrainedAt: number | null = null;
    try {
      const params = await getJson<{
        trained_at?: number;
        routes?: Record<string, { transition?: number[][] }>;
      }>(STREAMS.params);
      paramsTrainedAt = params.trained_at ?? null;
      heatmap = Object.entries(params.routes ?? {})
        .filter(([r]) => !routeFilter || r === routeFilter)
        .map(([route, p]) => ({ route, transition: p.transition ?? [] }))
        .filter((h) => h.transition.length === 3);
    } catch {
      // params may be absent before the first weekly train — non-fatal.
    }

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
    });
  } catch (e) {
    return NextResponse.json(
      { configured: true, error: (e as Error).message },
      { status: 502 },
    );
  }
}
