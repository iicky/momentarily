"use client";

import { useEffect, useState } from "react";
import { fetchSnapshot } from "@/lib/feed";
import type { Snapshot } from "@/lib/types";
import type { StationCoord, Topology, AdjEdge } from "@/lib/stations";

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

/** Segment adjacency topology (from /api/topology). `configured` is false when
 * the R2 vault isn't loaded, in which case `edges` is empty. */
export function useTopology(): Async<Topology> {
  const [state, setState] = useState<Async<Topology>>({ data: null, error: null });
  useEffect(() => {
    let alive = true;
    (async () => {
      try {
        const res = await fetch("/api/topology");
        const json = (await res.json()) as {
          configured?: boolean;
          trained_at?: number;
          topology_source?: string;
          edges?: AdjEdge[];
          error?: string;
        };
        if (!alive) return;
        setState({
          data: {
            configured: json.configured ?? false,
            trained_at: json.trained_at,
            topology_source: json.topology_source,
            edges: json.edges ?? [],
          },
          error: json.error ?? null,
        });
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
