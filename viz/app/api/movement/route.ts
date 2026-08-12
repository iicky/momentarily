import { NextRequest, NextResponse } from "next/server";
import { r2Configured, getJson, parseJsonl, utcDateWindow, STREAMS } from "@/lib/r2";
import { fetchDate } from "@/lib/r2cache";
import {
  advanceBaselines,
  buildMovementTruth,
  computeAdvanceBaseline,
  movementConfusion,
  parseThroughStops,
  type ThroughStops,
  type VehicleBody,
} from "@/lib/movement";
import type { PredictionRecord } from "@/lib/types";

export const runtime = "nodejs";
export const dynamic = "force-dynamic";

const MAX_DAYS = 21;
const VEHICLE_PREFIX = "archive/vehicles";
// Advance baseline is trained on this many days ENDING before the scored window
// so a sustained outage can't lower its own normal reference.
const BASELINE_DAYS = 14;

// One vehicle archive object per file; tolerate a malformed object rather than
// failing the whole window.
const parseVehicleBody = (t: string): VehicleBody[] => {
  try {
    return [JSON.parse(t) as VehicleBody];
  } catch {
    return [];
  }
};

// Only the field the movement signal needs from params.json; the grading
// route has its own richer TrainedParamsDoc for the HMM heatmap/dwell fields.
interface ParamsDoc {
  movement_through_stops?: unknown;
}

// Co-versioned with movement_baseline (see training/gtfs_static.through_stops
// + training/load_r2.StopFilter): absent/malformed params.json degrades to
// unfiltered counting, same as any tick before movement_through_stops existed.
async function fetchThroughStops(): Promise<ThroughStops> {
  try {
    const params = await getJson<ParamsDoc>(STREAMS.params);
    return parseThroughStops(params.movement_through_stops);
  } catch {
    return null;
  }
}

export async function GET(req: NextRequest) {
  const sp = req.nextUrl.searchParams;
  const days = Math.min(MAX_DAYS, Math.max(1, Number(sp.get("days") ?? 3)));
  const routeFilter = sp.get("route");
  const dates = utcDateWindow(days, Date.now());

  if (!(await r2Configured())) {
    // Movement archive is a credentialed LIST — no public aggregate exists for
    // it. Surface a clear not-configured state rather than erroring.
    return NextResponse.json({
      configured: false,
      window: { days, from: dates[dates.length - 1], to: dates[0] },
    });
  }

  try {
    // Vehicle archive (archive/vehicles/<date>/<ts>.json) + predictions stream,
    // both per-date cached so past partitions aren't refetched on each window
    // change. Dates run in parallel; today is always live. Through-stop set
    // comes from params.json alongside them; fetchThroughStops fails soft.
    const [vehicleArrays, predArrays, throughStops] = await Promise.all([
      Promise.all(dates.map((d) => fetchDate(VEHICLE_PREFIX, d, parseVehicleBody))),
      Promise.all(
        dates.map((d) => fetchDate(STREAMS.predictions, d, parseJsonl<PredictionRecord>)),
      ),
      fetchThroughStops(),
    ]);
    const bodies = vehicleArrays.flat();
    let predictions = predArrays.flat();
    if (routeFilter) predictions = predictions.filter((p) => p.route === routeFilter);

    const filterByRoute = (bs: VehicleBody[]): VehicleBody[] =>
      routeFilter
        ? bs.map((body) => ({
            ...body,
            rows: body.rows
              ? Object.fromEntries(
                  Object.entries(body.rows).filter(([r]) => r === routeFilter),
                )
              : {},
          }))
        : bs;
    const filteredBodies = filterByRoute(bodies);

    // Causal baseline: train on a window ending before the oldest scored date.
    const oldestScored = dates[dates.length - 1];
    const baselineAnchorMs = Date.parse(`${oldestScored}T00:00:00Z`) - 86_400_000;
    const baselineDates = utcDateWindow(BASELINE_DAYS, baselineAnchorMs);
    const baselineArrays = await Promise.all(
      baselineDates.map((d) => fetchDate(VEHICLE_PREFIX, d, parseVehicleBody)),
    );
    const baseline = computeAdvanceBaseline(filterByRoute(baselineArrays.flat()), throughStops);

    const truth = buildMovementTruth(filteredBodies, baseline, throughStops);
    const confusion = movementConfusion(
      predictions.map((p) => ({ route: p.route, ts: p.ts, condition: p.condition })),
      truth,
    );
    const baselines = advanceBaselines(filteredBodies, throughStops);

    return NextResponse.json({
      configured: true,
      window: { days, from: dates[dates.length - 1], to: dates[0] },
      counts: {
        vehicleTicks: bodies.length,
        predictionRecords: predictions.length,
        judgeableTicks: truth.size,
        baselineCells: baseline.size,
      },
      confusion,
      baselines,
    });
  } catch (e) {
    return NextResponse.json(
      { configured: true, error: (e as Error).message },
      { status: 502 },
    );
  }
}
