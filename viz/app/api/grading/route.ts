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
  routeUniverse,
} from "@/lib/calibration";
import type {
  PredictionRecord,
  TransitionRecord,
  HeatmapEntry,
} from "@/lib/types";

export const runtime = "nodejs";
export const dynamic = "force-dynamic";

const STATES = ["normal", "disrupted", "suspended"];
const MAX_DAYS = 21;
const POINT_CAP = 3000; // scatter points returned to the client
const FETCH_CONCURRENCY = 16;

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

export async function GET(req: NextRequest) {
  const sp = req.nextUrl.searchParams;
  const days = Math.min(MAX_DAYS, Math.max(1, Number(sp.get("days") ?? 3)));
  const routeFilter = sp.get("route");
  const nowSec = Math.floor(Date.now() / 1000);
  const dates = utcDateWindow(days, Date.now());

  if (!r2Configured()) {
    return NextResponse.json({
      configured: false,
      error:
        "R2 credentials not set. Add R2_ACCOUNT_ID, R2_ACCESS_KEY_ID, R2_SECRET_ACCESS_KEY to viz/.env.local.",
      window: { days, from: dates[dates.length - 1], to: dates[0] },
    });
  }

  try {
    const [predRes, transRes] = await Promise.all([
      readStream<PredictionRecord>(STREAMS.predictions, dates),
      readStream<TransitionRecord>(STREAMS.transitions, dates),
    ]);

    let predictions = predRes.records;
    let transitions = transRes.records;
    if (routeFilter) {
      predictions = predictions.filter((p) => p.route === routeFilter);
      transitions = transitions.filter((t) => t.route === routeFilter);
    }

    const timelines = buildTimelines(transitions, nowSec);
    const rel = [30, 60, 120].map((h) => reliability(predictions, timelines, h));
    const rec = recoveryError(predictions, timelines);

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
      window: { days, from: dates[dates.length - 1], to: dates[0] },
      counts: {
        predictionFiles: predRes.files,
        predictionRecords: predRes.records.length,
        transitionFiles: transRes.files,
        transitionRecords: transRes.records.length,
        pointsCapped,
      },
      routes: routeUniverse(predRes.records, transRes.records),
      states: STATES,
      reliability: rel,
      recovery: rec,
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
