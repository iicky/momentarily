"use client";

import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import Nav from "../Nav";
import { fetchSnapshot, fetchTrains, fmtAgo } from "@/lib/feed";
import type { TrainsFeed } from "@/lib/feed";
import { edgePath, fetchDiagram } from "@/lib/diagram";
import type { Diagram, DiagramEdge, ServiceClass } from "@/lib/diagram";
import {
  OVERLAYS,
  SERVICE_CLASSES,
  edgeId,
  markerRadius,
  paintEdges,
  serviceClassNow,
  timeScale,
  trainLayer,
} from "@/lib/overlays";
import type {
  DetailRow,
  LegendItem,
  NoteSpan,
  Overlay,
  OverlayContext,
  OverlayId,
} from "@/lib/overlays";
import type { DirectionFilter } from "@/lib/segments";
import type { Snapshot } from "@/lib/types";

const POLL_MS = 60_000;

const ZOOM_MAX = 24;

// Extra stroke width on the selected unit. Enough to read as "this is what the
// panel is describing" at overview scale, where the thinnest stroke is 1.7 and
// the thickest 3.6, without swamping its neighbours.
const SELECT_WIDTH = 1.8;

// Station dots are an orientation aid, not data. At overview scale 496 of them
// stipple the thin no-reading strokes into dashes, so they appear only once
// zoomed in far enough for individual stations to be the thing you're reading.
const DOT_RADIUS = 1.5;
const DOT_ZOOM = 3;

interface View {
  k: number;
  x: number;
  y: number;
}

const HOME: View = { k: 1, x: 0, y: 0 };

// Overlays whose unit isn't an edge draw their own layer over the resting line
// map. One entry per layer overlay, so a fifth overlay is a registry entry plus
// (only if it needs one) a component here — never a branch in the render path.
const LAYERS: Partial<Record<OverlayId, (p: LayerProps) => React.ReactNode>> = {
  trains: TrainsLayer,
};

export default function MapPage() {
  const [snap, setSnap] = useState<Snapshot | null>(null);
  const [diagram, setDiagram] = useState<Diagram | null>(null);
  const [err, setErr] = useState<string | null>(null);
  const [fetchedAt, setFetchedAt] = useState(0);
  const [filter, setFilter] = useState<DirectionFilter>("both");
  // Opens on today's timetable in New York, because a reader looking at a live
  // map means today. Named in the control either way — see serviceClassNow on
  // why this is a default and not a claim about holidays.
  const [serviceClass, setServiceClass] = useState<ServiceClass>(() =>
    serviceClassNow(Date.now()),
  );
  const [overlayId, setOverlayId] = useState<OverlayId>("movement");
  const [trains, setTrains] = useState<TrainsFeed>({ state: "loading" });
  const [hover, setHover] = useState<string | null>(null);
  const [pinned, setPinned] = useState<string | null>(null);
  const [view, setView] = useState<View>(HOME);
  const svgRef = useRef<SVGSVGElement | null>(null);
  const drag = useRef<{ x: number; y: number; view: View } | null>(null);

  useEffect(() => {
    fetchDiagram().then(setDiagram, (e: Error) => setErr(e.message));
  }, []);

  useEffect(() => {
    let alive = true;
    const load = async () => {
      try {
        const s = await fetchSnapshot();
        if (!alive) return;
        setSnap(s);
        setErr(null);
        setFetchedAt(Math.floor(Date.now() / 1000));
      } catch (e) {
        if (alive) setErr((e as Error).message);
      }
    };
    load();
    const id = setInterval(load, POLL_MS);
    return () => {
      alive = false;
      clearInterval(id);
    };
  }, []);

  // The train object is the largest surface this page reads and three of the
  // four overlays have no use for it, so it is fetched only while its own
  // overlay is selected, then polled like the snapshot — it is republished on
  // the same tick cadence. It never sets `err`: a 404 is a normal state for an
  // object that may not be published yet, and it must not take down the map.
  useEffect(() => {
    if (overlayId !== "trains") return;
    let alive = true;
    const load = async () => {
      const feed = await fetchTrains();
      if (alive) setTrains(feed);
    };
    load();
    const id = setInterval(load, POLL_MS);
    return () => {
      alive = false;
      clearInterval(id);
    };
  }, [overlayId]);

  const active: Overlay = useMemo(
    () => OVERLAYS.find((o) => o.id === overlayId) ?? OVERLAYS[0],
    [overlayId],
  );

  // The scheduled-time bins are a sort over every timed hop in one service
  // class, and they depend on the asset and the class alone — never on the
  // snapshot.
  const scale = useMemo(
    () => (diagram ? timeScale(diagram, serviceClass) : null),
    [diagram, serviceClass],
  );

  // One stable context per set of inputs, so hovering re-renders the JSX
  // without re-deriving 988 readings underneath it.
  const ctx = useMemo<OverlayContext | null>(
    () =>
      diagram === null
        ? null
        : {
            diagram,
            snap,
            filter,
            serviceClass,
            now: fetchedAt,
            time: scale,
            trains,
          },
    [diagram, snap, filter, serviceClass, fetchedAt, scale, trains],
  );

  const painted = useMemo(
    () => (ctx === null ? [] : paintEdges(active, ctx)),
    [active, ctx],
  );
  const note = useMemo(() => (ctx === null ? null : active.note(ctx)), [active, ctx]);
  const legend = useMemo(
    () => (ctx === null ? null : active.legend(ctx)),
    [active, ctx],
  );
  const stamp = useMemo(() => (ctx === null ? null : active.stamp(ctx)), [active, ctx]);
  const caveat = useMemo(
    () => (ctx === null ? null : (active.caveat?.(ctx) ?? null)),
    [active, ctx],
  );

  // Hit targets cover every drawn edge regardless of what the overlay paints:
  // an edge the active overlay has nothing to say about is exactly the one a
  // reader wants to interrogate, and the panel answers for both directions.
  const hits = useMemo(
    () =>
      (diagram?.edges ?? []).map((edge) => ({
        id: edgeId(edge),
        edge,
        d: edgePath(edge),
      })),
    [diagram],
  );
  const byEdge = useMemo(() => {
    const out: Record<string, DiagramEdge> = {};
    for (const hit of hits) out[hit.id] = hit.edge;
    return out;
  }, [hits]);

  const shown = pinned ?? hover;
  const detailEdge = shown === null ? null : (byEdge[shown] ?? null);
  const rows =
    detailEdge === null || ctx === null ? null : active.detail(detailEdge, ctx);

  const onWheel = useCallback(
    (e: React.WheelEvent<SVGSVGElement>) => {
      const svg = svgRef.current;
      if (!diagram || !svg) return;
      e.preventDefault();
      const p = toLocal(e.clientX, e.clientY, svg, diagram.view_box);
      setView((v) => {
        const k = Math.min(ZOOM_MAX, Math.max(1, v.k * Math.exp(-e.deltaY / 400)));
        // Hold the point under the cursor still: t' = t + p·(k − k').
        return { k, x: v.x + p.x * (v.k - k), y: v.y + p.y * (v.k - k) };
      });
    },
    [diagram],
  );

  const onPointerMove = (e: React.PointerEvent<SVGSVGElement>) => {
    const start = drag.current;
    const svg = svgRef.current;
    if (!start || !svg || !diagram) return;
    const rect = svg.getBoundingClientRect();
    const scaled = Math.min(
      rect.width / diagram.view_box[2],
      rect.height / diagram.view_box[3],
    );
    setView({
      k: start.view.k,
      x: start.view.x + (e.clientX - start.x) / scaled,
      y: start.view.y + (e.clientY - start.y) / scaled,
    });
  };

  const Layer = LAYERS[active.id];

  return (
    <div className="wrap">
      <div className="topbar">
        <h1>Segment map</h1>
        <Nav />
      </div>
      <div className="sub">
        {active.label} on the line diagram ·{" "}
        {snap === null ? (
          "loading…"
        ) : (
          <>
            {stamp === null
              ? `static timetable asset (${serviceClass})`
              : stamp.at === null
                ? `no ${stamp.label} available`
                : `${stamp.label} ${fmtAgo(stamp.at, fetchedAt)}`}{" "}
            · snapshot {fmtAgo(snap.generated_at, fetchedAt)}
          </>
        )}
      </div>

      {err && <div className="error">Failed to load: {err}</div>}

      <div className="controls">
        <div className="seg" role="group" aria-label="Overlay">
          {OVERLAYS.map((o) => (
            <button
              key={o.id}
              className={active.id === o.id ? "active" : ""}
              aria-pressed={active.id === o.id}
              onClick={() => setOverlayId(o.id)}
            >
              {o.label}
            </button>
          ))}
        </div>
        {active.classes && (
          <div className="seg" role="group" aria-label="Timetable">
            {SERVICE_CLASSES.map((c) => (
              <button
                key={c.id}
                className={serviceClass === c.id ? "active" : ""}
                aria-pressed={serviceClass === c.id}
                onClick={() => setServiceClass(c.id)}
              >
                {c.label}
              </button>
            ))}
          </div>
        )}
        {/* Hidden, not disabled, for an overlay it doesn't apply to: a control
            that can't do anything is worse than no control. */}
        {active.filters && (
          <div className="seg" role="group" aria-label="Direction">
            {active.filters.map((f) => (
              <button
                key={f.id}
                className={filter === f.id ? "active" : ""}
                aria-pressed={filter === f.id}
                onClick={() => setFilter(f.id)}
              >
                {f.label}
              </button>
            ))}
          </div>
        )}
        <button onClick={() => setView(HOME)} disabled={view === HOME}>
          Reset view
        </button>
      </div>

      {/* A caveat that changes what the picture means goes where the picture
          is, at full width, not in a tooltip. */}
      {caveat && <Prose className="warnbox" spans={caveat} />}

      {note && <Prose className="grp-note" spans={note} />}

      <div className="maprow">
        <div className="mapframe">
          {diagram && ctx ? (
            <svg
              ref={svgRef}
              className="diagram"
              viewBox={diagram.view_box.join(" ")}
              // The asset's own aspect, so the frame shrink-wraps the diagram
              // instead of letterboxing a near-square map in a wide column.
              style={{
                aspectRatio: `${diagram.view_box[2]} / ${diagram.view_box[3]}`,
              }}
              onWheel={onWheel}
              onPointerDown={(e) => {
                drag.current = { x: e.clientX, y: e.clientY, view };
                e.currentTarget.setPointerCapture(e.pointerId);
              }}
              onPointerMove={onPointerMove}
              onPointerUp={() => {
                drag.current = null;
              }}
              onPointerLeave={() => {
                drag.current = null;
                setHover(null);
              }}
            >
              <g transform={`translate(${view.x} ${view.y}) scale(${view.k})`}>
                {painted.map((p) => {
                  const selected = shown === p.id;
                  return (
                    <path
                      key={p.id}
                      d={p.d}
                      stroke={
                        p.paint.color ??
                        diagram.routes[p.edge.route]?.color ??
                        "var(--unknown)"
                      }
                      // Width, not just opacity: `disrupted` already paints at
                      // full opacity, so an opacity-only highlight is invisible
                      // on exactly the strokes a reader most wants to select.
                      strokeWidth={p.paint.width + (selected ? SELECT_WIDTH : 0)}
                      strokeOpacity={selected ? 1 : p.paint.opacity}
                      strokeDasharray={p.paint.dash ?? undefined}
                      strokeLinecap="round"
                      fill="none"
                      vectorEffect="non-scaling-stroke"
                    />
                  );
                })}
                {/* Radius divided by the zoom: strokes hold their screen width
                    via non-scaling-stroke, and a dot has no stroke to do that
                    with, so it would balloon to a blob at full zoom. */}
                {view.k >= DOT_ZOOM &&
                  Object.entries(diagram.stations).map(([id, station]) => (
                    <circle
                      key={id}
                      className="stationdot"
                      cx={station.x}
                      cy={station.y}
                      r={DOT_RADIUS / view.k}
                    />
                  ))}
                {Layer && <Layer ctx={ctx} k={view.k} />}
                {/* Transparent wide strokes on top: thin lines are unhittable. */}
                {hits.map((hit) => (
                  <path
                    key={`hit-${hit.id}`}
                    className="edgehit"
                    d={hit.d}
                    onPointerEnter={() => setHover(hit.id)}
                    onClick={() => setPinned(pinned === hit.id ? null : hit.id)}
                  />
                ))}
                {/* An inset is a component drawn away from where it is. Say so
                    on the map, not just in the asset. */}
                {diagram.insets.map((inset) => (
                  <g key={inset.routes.join()} className="inset">
                    <rect
                      x={inset.box[0] - 10}
                      y={inset.box[1] - 10}
                      width={inset.box[2] + 20}
                      height={inset.box[3] + 26}
                      rx={4}
                      vectorEffect="non-scaling-stroke"
                    />
                    <text
                      x={inset.box[0] - 10}
                      y={inset.box[1] + inset.box[3] + 12}
                      fontSize={11 / view.k}
                    >
                      {inset.routes
                        .map((r) => diagram.routes[r]?.name ?? r)
                        .join(", ")}{" "}
                      — inset, {Math.round(inset.scale * 100)}% scale
                    </text>
                  </g>
                ))}
              </g>
            </svg>
          ) : (
            <div className="chart-empty muted">loading diagram…</div>
          )}
          {legend && (
            <div className="legend legend-wrap">
              {legend.map((item) => (
                <span key={item.label} title={item.title}>
                  <Swatch item={item} />
                  {item.label}
                </span>
              ))}
            </div>
          )}
          <Prose className="mapcaption" spans={active.caption} />
        </div>

        <aside className="card mapdetail">
          {detailEdge && rows && diagram ? (
            <EdgeDetail
              diagram={diagram}
              edge={detailEdge}
              rows={rows}
              pinned={pinned !== null}
            />
          ) : (
            <p className="muted">
              Hover a segment for its reading, click to pin it. Scroll to zoom,
              drag to pan.
            </p>
          )}
        </aside>
      </div>

      {diagram && (
        <div className="chart-meta">
          <span className="chart-chip">
            timetable {diagram.feed_version.version}
          </span>
          <span className="chart-chip">
            {Object.keys(diagram.stations).length} stations ·{" "}
            {diagram.edges.length} drawn edges
          </span>
          <span className="chart-chip chart-chip-muted">
            layout derived from MTA GTFS stop coordinates + timetable
          </span>
        </div>
      )}
    </div>
  );
}

interface LayerProps {
  ctx: OverlayContext;
  k: number;
}

// Two concentric marks per stop: a filled disc for the trains standing at the
// platform, and a ring for those plus the ones heading there. Nothing is drawn
// between stations — the feed names a stop, not a segment, and which segment a
// moving train is on is ambiguous at a branch.
function TrainsLayer({ ctx, k }: LayerProps) {
  const feed = ctx.trains;
  const layer = trainLayer(
    ctx.diagram,
    feed.state === "ready" ? feed.trains : null,
  );
  return (
    <g className="trainlayer">
      {layer.markers.map((m) => (
        <g key={m.station}>
          <circle
            className="trainring"
            cx={m.x}
            cy={m.y}
            // Radius over the zoom, for the same reason the station dots do it:
            // a circle has no stroke for non-scaling-stroke to hold.
            r={markerRadius(m.stopped + m.inbound) / k}
          />
          {m.stopped > 0 && (
            <circle
              className="traindot"
              cx={m.x}
              cy={m.y}
              r={markerRadius(m.stopped) / k}
            />
          )}
        </g>
      ))}
    </g>
  );
}

function Prose({ className, spans }: { className: string; spans: NoteSpan[] }) {
  return (
    <p className={className}>
      {spans.map((span, i) =>
        typeof span === "string" ? (
          <span key={i}>{span}</span>
        ) : (
          <em key={i}>{span.em}</em>
        ),
      )}
    </p>
  );
}

// Shape is a second encoding beside hue: a dashed swatch marks a coarser unit,
// a ring marks a count that includes trains not yet arrived, and the dimmed
// swatch is the no-reading ghost every overlay shares.
function Swatch({ item }: { item: LegendItem }) {
  if (item.shape === "ring") {
    return (
      <i className="lg-ring" style={{ borderColor: item.color ?? "var(--text)" }} />
    );
  }
  if (item.color === null) return <i className={`lg-${item.shape} ghost`} />;
  if (item.shape === "dash") {
    const dashes =
      `repeating-linear-gradient(90deg, ${item.color} 0 5px, ` +
      "transparent 5px 8px)";
    return <i className="lg-dash" style={{ backgroundImage: dashes }} />;
  }
  return <i className={`lg-${item.shape}`} style={{ background: item.color }} />;
}

function EdgeDetail({
  diagram,
  edge,
  rows,
  pinned,
}: {
  diagram: Diagram;
  edge: DiagramEdge;
  rows: DetailRow[];
  pinned: boolean;
}) {
  return (
    <>
      <div className="card-head">
        <span
          className="bullet"
          style={{ background: diagram.routes[edge.route]?.color ?? "#6e6e73" }}
        >
          {edge.route}
        </span>
        <span className="segpair">
          {diagram.stations[edge.a]?.name ?? edge.a} ↔{" "}
          {diagram.stations[edge.b]?.name ?? edge.b}
        </span>
      </div>
      {rows.map((row, i) => (
        <div className="segrow" key={`${row.key}-${i}`}>
          <span className="k">{row.key}</span>
          <span className="cond" style={{ color: row.color ?? "var(--muted)" }}>
            {row.value}
          </span>
          {row.note && <span className="muted">{row.note}</span>}
        </div>
      ))}
      {pinned && (
        <p className="muted section-note">pinned · click again to release</p>
      )}
    </>
  );
}

// Client point -> diagram coordinates, accounting for the letterboxing the
// default preserveAspectRatio applies when the frame isn't the box's aspect.
function toLocal(
  clientX: number,
  clientY: number,
  svg: SVGSVGElement,
  box: readonly [number, number, number, number],
): { x: number; y: number } {
  const rect = svg.getBoundingClientRect();
  const scale = Math.min(rect.width / box[2], rect.height / box[3]);
  const ox = (rect.width - box[2] * scale) / 2;
  const oy = (rect.height - box[3] * scale) / 2;
  return {
    x: (clientX - rect.left - ox) / scale + box[0],
    y: (clientY - rect.top - oy) / scale + box[1],
  };
}
