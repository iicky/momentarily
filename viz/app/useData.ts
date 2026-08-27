"use client";

import { useEffect, useState } from "react";
import { fetchSnapshot } from "@/lib/feed";
import { fetchDiagram } from "@/lib/diagram";
import { fetchStationFacts } from "@/lib/facts";
import type { Snapshot } from "@/lib/types";
import type { StationFacts } from "@/lib/facts";
import type { StationCoord, Topology } from "@/lib/stations";

const SNAP_POLL_MS = 60_000;

export interface Async<T> {
  data: T | null;
  error: string | null;
}

/** The public snapshot, refreshed on the same 60s cadence as the Status page. */
export function useSnapshot(): Async<Snapshot> {
  const [state, setState] = useState<Async<Snapshot>>({ data: null, error: null });
  useEffect(() => {
    let alive = true;
    const load = async () => {
      try {
        const s = await fetchSnapshot();
        if (alive) setState({ data: s, error: null });
      } catch (e) {
        if (alive) setState((p) => ({ data: p.data, error: (e as Error).message }));
      }
    };
    load();
    const id = setInterval(load, SNAP_POLL_MS);
    return () => {
      alive = false;
      clearInterval(id);
    };
  }, []);
  return state;
}

/** Station coordinates keyed by undirected GTFS id (from /api/stations). Fetched
 * once — the source is refreshed daily upstream. */
export function useCoords(): Async<Record<string, StationCoord>> {
  const [state, setState] = useState<Async<Record<string, StationCoord>>>({
    data: null,
    error: null,
  });
  useEffect(() => {
    let alive = true;
    (async () => {
      try {
        const res = await fetch("/api/stations");
        const json = (await res.json()) as {
          stations?: Record<string, StationCoord>;
          error?: string;
        };
        if (!alive) return;
        if (json.error) setState({ data: null, error: json.error });
        else setState({ data: json.stations ?? {}, error: null });
      } catch (e) {
        if (alive) setState({ data: null, error: (e as Error).message });
      }
    })();
    return () => {
      alive = false;
    };
  }, []);
  return state;
}

/** Segment topology + canonical stop order, read off the committed diagram
 * asset (viz/lib/diagram.ts) — one fetch shared with the map overview via
 * fetchDiagram's own module-level cache, not refetched here. */
export function useTopology(): Async<Topology> {
  const [state, setState] = useState<Async<Topology>>({ data: null, error: null });
  useEffect(() => {
    let alive = true;
    fetchDiagram().then(
      (d) => {
        if (!alive) return;
        setState({
          data: {
            topology_source: d.topology_source,
            feed_version: d.feed_version,
            edges: d.adjacency,
            routeStops: d.route_stops,
          },
          error: null,
        });
      },
      (e: Error) => {
        if (alive) setState({ data: null, error: e.message });
      },
    );
    return () => {
      alive = false;
    };
  }, []);
  return state;
}

/** Station reference facts (opening year, photo, landmark, art, ridership rank),
 * read off the committed station_facts.json asset (viz/lib/facts.ts) — one fetch
 * shared across the session via fetchStationFacts's module cache, never polled. */
export function useStationFacts(): Async<StationFacts> {
  const [state, setState] = useState<Async<StationFacts>>({ data: null, error: null });
  useEffect(() => {
    let alive = true;
    fetchStationFacts().then(
      (d) => {
        if (alive) setState({ data: d, error: null });
      },
      (e: Error) => {
        if (alive) setState({ data: null, error: e.message });
      },
    );
    return () => {
      alive = false;
    };
  }, []);
  return state;
}
