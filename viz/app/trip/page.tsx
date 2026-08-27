"use client";

// The isolated trip view: pick an origin and a destination station complex,
// choose one of the enumerated candidate journeys, and see ONLY that journey's
// segments as a stripped strip-map, each carrying its current live status. This
// is "my trip", not the network — no other lines, no map of the system, just
// the rides you would actually take, keyed to the same segment_flow cells the
// spatial views read (route|direction|from_stop, direction-aware).
//
// Enumeration is topology-pure (lib/journeys.ts) bridged to the snapshot's
// station_complex_id groupings (lib/complexes.ts); this page owns none of that
// reasoning — it only picks endpoints, lists candidates, and paints the chosen
// one. Scoring/comparing candidates, saved commutes, and history are separate
// surfaces and deliberately absent here.

import { Suspense, useMemo, useState } from "react";
import Link from "next/link";
import { useRouter, useSearchParams } from "next/navigation";
import { useSnapshot, useTopology } from "../useData";
import { PageHeader, RouteBullet } from "../ui";
import { undirected } from "@/lib/stations";
import { fmtMinutes } from "@/lib/feed";
import { indexComplexes, journeysBetween } from "@/lib/complexes";
import { alightStop, boardStop } from "@/lib/journeys";
import type { Journey, JourneyLeg } from "@/lib/journeys";
import type { Snapshot, SegmentStatus } from "@/lib/types";
import type { Topology } from "@/lib/stations";

// One selectable endpoint: a station complex as the snapshot groups it, with a
// display name and the routes it serves to disambiguate same-named complexes.
interface ComplexOption {
  id: string;
  name: string;
  routes: string[];
  borough: string | null;
  search: string;
}

const DIR_LABEL: Record<string, string> = { north: "Northbound", south: "Southbound" };

// One journey's stable identity for the URL: the route sequence alone, matching
// the enumerator's own dedup key (opposite directions of one sequence collapse
// to a single candidate). Route ids never contain a hyphen, so this round-trips.
const journeyId = (j: Journey): string => j.legs.map((l) => l.route).join("-");

export default function TripPage() {
  return (
    <Suspense fallback={<div className="wrap"><div className="sub">loading…</div></div>}>
      <TripView />
    </Suspense>
  );
}

function TripView() {
  const router = useRouter();
  const qp = useSearchParams();
  const { data: snap } = useSnapshot();
  const { data: topo } = useTopology();

  const options = useMemo(() => buildOptions(snap), [snap]);
  const optionById = useMemo(() => new Map(options.map((o) => [o.id, o])), [options]);
  const index = useMemo(() => (snap ? indexComplexes(snap.stations) : null), [snap]);

  const from = qp.get("from") ?? "";
  const to = qp.get("to") ?? "";
  const via = qp.get("via") ?? "";

  const setParams = (next: { from?: string; to?: string; via?: string }) => {
    const p = new URLSearchParams(qp.toString());
    for (const [k, v] of Object.entries(next)) {
      if (v) p.set(k, v);
      else p.delete(k);
    }
    router.replace(`/trip?${p.toString()}`);
  };

  const journeys = useMemo(() => {
    if (!topo || !index || !from || !to || from === to) return [];
    return journeysBetween(topo.routeStops, topo.edges, index, from, to);
  }, [topo, index, from, to]);

  const selected = useMemo(
    () => journeys.find((j) => journeyId(j) === via) ?? null,
    [journeys, via],
  );

  const ready = !!snap && !!topo;
  const bothPicked = !!from && !!to && from !== to;

  return (
    <div className="wrap">
      <PageHeader subtitle="Pick two stations and follow one journey — just its segments, each with how it is moving right now." />

      <div className="trip-pickers">
        <ComplexPicker
          label="From"
          options={options}
          value={from}
          disabled={!ready}
          onChange={(id) => setParams({ from: id, to, via: "" })}
        />
        <ComplexPicker
          label="To"
          options={options}
          value={to}
          disabled={!ready}
          onChange={(id) => setParams({ from, to: id, via: "" })}
        />
      </div>

      {!ready ? (
        <div className="sub">loading…</div>
      ) : !bothPicked ? (
        <div className="note">
          Choose an origin and a destination to see the journeys that connect them.
        </div>
      ) : journeys.length === 0 ? (
        <div className="note">
          No single- or double-transfer journey connects{" "}
          <b>{optionById.get(from)?.name ?? from}</b> and{" "}
          <b>{optionById.get(to)?.name ?? to}</b> over the committed topology.
        </div>
      ) : (
        <div className="trip-layout">
          <CandidateList
            snap={snap}
            journeys={journeys}
            selectedId={selected ? journeyId(selected) : ""}
            onPick={(id) => setParams({ from, to, via: id })}
          />
          {selected ? (
            <StripMap snap={snap} journey={selected} />
          ) : (
            <div className="note muted trip-strip-empty">
              Pick a journey on the left to see its segments.
            </div>
          )}
        </div>
      )}

      {topo?.feed_version && (
        <div className="prov-note">
          timetable · <code>{topo.feed_version.version}</code>
          {topo.topology_source ? ` · ${topo.topology_source}` : ""}
        </div>
      )}
    </div>
  );
}

// --- Endpoint picker -------------------------------------------------------

// Build one selectable option per station_complex_id present in the snapshot.
// The display name is the member station name that occurs most often (ties
// broken toward the longest, so "14 St-Union Sq" wins over a bare "14 St"), and
// the served routes + borough ride along to tell same-named complexes apart.
function buildOptions(snap: Snapshot | null): ComplexOption[] {
  if (!snap) return [];
  const byComplex = new Map<
    string,
    { names: Map<string, number>; routes: Set<string>; borough: string | null }
  >();
  for (const st of Object.values(snap.stations)) {
    const cid = st.station_complex_id;
    if (!cid) continue;
    let agg = byComplex.get(cid);
    if (!agg) {
      agg = { names: new Map(), routes: new Set(), borough: null };
      byComplex.set(cid, agg);
    }
    agg.names.set(st.name, (agg.names.get(st.name) ?? 0) + 1);
    for (const r of st.routes_served) agg.routes.add(r);
    if (!agg.borough && st.borough) agg.borough = st.borough;
  }
  const options: ComplexOption[] = [];
  for (const [id, agg] of byComplex) {
    let name = id;
    let best = -1;
    for (const [n, count] of agg.names) {
      if (count > best || (count === best && n.length > name.length)) {
        best = count;
        name = n;
      }
    }
    const routes = [...agg.routes].sort((a, b) =>
      a.localeCompare(b, undefined, { numeric: true }),
    );
    options.push({
      id,
      name,
      routes,
      borough: agg.borough,
      search: `${name} ${routes.join(" ")} ${agg.borough ?? ""}`.toLowerCase(),
    });
  }
  return options.sort((a, b) => a.name.localeCompare(b.name, undefined, { numeric: true }));
}

function ComplexPicker({
  label,
  options,
  value,
  disabled,
  onChange,
}: {
  label: string;
  options: ComplexOption[];
  value: string;
  disabled: boolean;
  onChange: (id: string) => void;
}) {
  const [open, setOpen] = useState(false);
  const [query, setQuery] = useState("");
  const selected = options.find((o) => o.id === value) ?? null;

  const matches = useMemo(() => {
    const needle = query.trim().toLowerCase();
    const list = needle ? options.filter((o) => o.search.includes(needle)) : options;
    return list.slice(0, 60);
  }, [options, query]);

  return (
    <label className="complex-picker">
      <span className="complex-picker-label">{label}</span>
      <div className="complex-picker-box">
        <input
          type="text"
          className="complex-picker-input"
          disabled={disabled}
          placeholder="search a station…"
          value={open ? query : selected?.name ?? ""}
          onFocus={() => {
            setOpen(true);
            setQuery("");
          }}
          onChange={(e) => {
            setQuery(e.target.value);
            setOpen(true);
          }}
          onBlur={() => setTimeout(() => setOpen(false), 150)}
        />
        {open && matches.length > 0 && (
          <ul className="picker-list">
            {matches.map((o) => (
              <li
                key={o.id}
                className={o.id === value ? "active" : ""}
                onMouseDown={(e) => {
                  e.preventDefault();
                  onChange(o.id);
                  setOpen(false);
                  setQuery("");
                }}
              >
                <span className="picker-name">{o.name}</span>
                <span className="picker-meta">
                  {o.routes.join(" · ")}
                  {o.borough ? ` — ${o.borough}` : ""}
                </span>
              </li>
            ))}
          </ul>
        )}
      </div>
    </label>
  );
}

// --- Candidate list --------------------------------------------------------

function CandidateList({
  snap,
  journeys,
  selectedId,
  onPick,
}: {
  snap: Snapshot;
  journeys: Journey[];
  selectedId: string;
  onPick: (id: string) => void;
}) {
  return (
    <div className="cand-panel">
      <div className="section-title">Journeys ({journeys.length})</div>
      <ul className="cand-list">
        {journeys.map((j) => {
          const id = journeyId(j);
          const stops = j.segments.length + 1;
          return (
            <li key={id}>
              <button
                type="button"
                className={`cand ${id === selectedId ? "active" : ""}`}
                onClick={() => onPick(id)}
              >
                <span className="cand-routes">
                  {j.legs.map((l, i) => (
                    <span className="cand-leg" key={i}>
                      {i > 0 && <span className="cand-arrow">→</span>}
                      <RouteBullet snap={snap} route={l.route} size={18} />
                    </span>
                  ))}
                </span>
                <span className="cand-meta">
                  {j.transfers === 0
                    ? "direct"
                    : `${j.transfers} transfer${j.transfers > 1 ? "s" : ""}`}
                  {" · "}
                  {stops} stops
                </span>
              </button>
            </li>
          );
        })}
      </ul>
    </div>
  );
}

// --- Strip-map of the chosen journey --------------------------------------

// The status verdict published for one segment cell right now, keyed by the
// segment's own route|direction|from_stop — the same cell key segment_flow uses,
// so this is direction-aware by construction. null means the cell was not judged
// this tick (never a healthy read by omission).
const segStatus = (
  snap: Snapshot,
  key: string,
): SegmentStatus | null => snap.segment_flow?.segments[key] ?? null;

function StripMap({ snap, journey }: { snap: Snapshot; journey: Journey }) {
  const nameOf = (stop: string): string => {
    const id = undirected(stop);
    return snap.stations[id]?.name ?? id;
  };
  return (
    <div className="trip-strip">
      {journey.legs.map((leg, li) => (
        <div key={li}>
          <Leg snap={snap} leg={leg} nameOf={nameOf} />
          {li < journey.legs.length - 1 && (
            <div className="trip-transfer">
              <span className="trip-transfer-mark">⇅</span>
              Transfer at {nameOf(alightStop(leg))}
            </div>
          )}
        </div>
      ))}
    </div>
  );
}

function Leg({
  snap,
  leg,
  nameOf,
}: {
  snap: Snapshot;
  leg: JourneyLeg;
  nameOf: (stop: string) => string;
}) {
  return (
    <div className="trip-leg">
      <div className="trip-leg-head">
        <RouteBullet snap={snap} route={leg.route} size={22} />
        <span className="trip-dir">{DIR_LABEL[leg.direction] ?? leg.direction}</span>
        <span className="trip-toward">to {nameOf(alightStop(leg))}</span>
      </div>
      <div className="trip-board">
        <span className="trip-rail trip-rail-board">
          <span className="trip-node board" />
        </span>
        <Link className="trip-stop-name" href={`/stations/${undirected(boardStop(leg))}`}>
          {nameOf(boardStop(leg))}
        </Link>
      </div>
      <ul className="trip-segs">
        {leg.segments.map((seg) => {
          const cell = segStatus(snap, seg.key);
          const status = cell?.status ?? null;
          return (
            <li className="trip-seg" key={seg.key}>
              <span className="trip-rail">
                <span className={`trip-conn ${status ?? "unknown"}`} />
                <span className="trip-node" />
              </span>
              <Link className="trip-stop-name" href={`/stations/${undirected(seg.to)}`}>
                {nameOf(seg.to)}
              </Link>
              {status ? (
                <span className={`cond ${status}`}>
                  {status}
                  {status === "disrupted" && cell?.recovery
                    ? ` · ~${fmtMinutes(cell.recovery.recovery_minutes)}`
                    : ""}
                </span>
              ) : (
                <span className="cond unknown" title="not judged this tick">
                  —
                </span>
              )}
            </li>
          );
        })}
      </ul>
    </div>
  );
}
