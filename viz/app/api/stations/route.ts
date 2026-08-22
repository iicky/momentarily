import { NextResponse } from "next/server";
import type { StationCoord } from "@/lib/stations";

export const runtime = "nodejs";
export const dynamic = "force-dynamic";

// Station coordinates + platform labels come from NYS Open Data 39hk-dx4f (the
// same dataset the Worker reads for station metadata), which the public snapshot
// deliberately doesn't carry — 496 lat/lon pairs would bloat the hot state file.
// Public, no key needed; cached in-module so we hit it at most once a day.
const SOURCE = "https://data.ny.gov/resource/39hk-dx4f.json?$limit=1000";
const TTL_MS = 24 * 60 * 60 * 1000;

interface Row {
  gtfs_stop_id?: string;
  stop_name?: string;
  borough?: string;
  structure?: string;
  line?: string;
  daytime_routes?: string;
  gtfs_latitude?: string;
  gtfs_longitude?: string;
  north_direction_label?: string;
  south_direction_label?: string;
}

// 39hk-dx4f borough codes → full names, matching worker/src/stations_static.ts.
const BOROUGH_NAMES: Record<string, string> = {
  M: "Manhattan",
  Bx: "Bronx",
  Bk: "Brooklyn",
  Q: "Queens",
  SI: "Staten Island",
};

let cache: { at: number; data: Record<string, StationCoord> } | null = null;

async function load(): Promise<Record<string, StationCoord>> {
  const res = await fetch(SOURCE, { cache: "no-store" });
  if (!res.ok) throw new Error(`stations feed ${res.status}`);
  const rows = (await res.json()) as Row[];
  const out: Record<string, StationCoord> = {};
  for (const r of rows) {
    const id = r.gtfs_stop_id;
    const lat = Number(r.gtfs_latitude);
    const lon = Number(r.gtfs_longitude);
    if (!id || !Number.isFinite(lat) || !Number.isFinite(lon)) continue;
    out[id] = {
      stop_id: id,
      name: r.stop_name ?? id,
      lat,
      lon,
      borough: r.borough ? BOROUGH_NAMES[r.borough] ?? r.borough : null,
      structure: r.structure ?? null,
      line: r.line ?? null,
      daytime_routes: (r.daytime_routes ?? "").split(/\s+/).filter(Boolean),
      north_label: r.north_direction_label ?? null,
      south_label: r.south_direction_label ?? null,
    };
  }
  return out;
}

export async function GET() {
  try {
    if (!cache || Date.now() - cache.at > TTL_MS) {
      cache = { at: Date.now(), data: await load() };
    }
    return NextResponse.json({ stations: cache.data });
  } catch (e) {
    return NextResponse.json({ error: (e as Error).message }, { status: 502 });
  }
}
