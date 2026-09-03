// Which lines the network diagram draws, and the one honest rule for hiding the
// rest. The map draws ~28 routes at once; a reader who cares about two of them
// needs the other 26 to stop competing for attention.
//
// The subtlety this module exists to hold: a drawn EDGE carries exactly one
// route (the diagram draws a shared trunk as one parallel stroke per route, each
// keyed on its own segment cell — see viz/lib/diagram.ts DiagramEdge.route), so
// hiding an edge is a per-route decision with no attribution to get wrong. The
// UNION lives one level up, at the STATION: a transfer stop is served by several
// routes and must survive as long as ANY selected route still calls there, or a
// filtered map leaves orphan dots floating with no line beneath them.
//
// Filtering is deliberately a subset of what the overlays already painted, never
// a re-derivation: a segment verdict keyed on its from_stop alone can belong to
// several drawn edges, and the overlay has already attributed it to the one
// successor it was measured toward (viz/lib/segments.ts placesOn). Selecting
// routes only chooses which of those already-correct strokes to show — it must
// never move a reading onto a sibling branch it was not measured for.

import type { Diagram, DiagramEdge, DiagramStation } from "./diagram";

/** The selected lines, or `null` for the default "every line shown". `null` is
 * canonical for all-on: it keeps the resting map byte-identical to the
 * unfiltered one and drops the URL param entirely. */
export type RouteSelection = ReadonlySet<string> | null;

/** Compare route ids the way a rider reads them: 1,2,…,10 numerically, letters
 * lexically. Matches the trip page's route ordering so the two never disagree. */
export function compareRoutes(a: string, b: string): number {
  return a.localeCompare(b, undefined, { numeric: true });
}

/** The distinct routes the diagram actually draws an edge for, sorted. The
 * route registry can carry more than the geometry does; only a drawn line earns
 * a filter bullet. */
export function drawnRoutes(diagram: Diagram): string[] {
  const set = new Set<string>();
  for (const edge of diagram.edges) set.add(edge.route);
  return [...set].sort(compareRoutes);
}

/** Whether a route is currently shown. `null` selection = all shown. */
export function isRouteOn(sel: RouteSelection, route: string): boolean {
  return sel === null || sel.has(route);
}

/** Whether this drawn edge shows under the selection. One route per edge, so
 * this is the whole rule — no union, no re-attribution. */
export function edgeShown(edge: DiagramEdge, sel: RouteSelection): boolean {
  return sel === null || sel.has(edge.route);
}

/** Whether a station's dot shows: the UNION over the routes it serves. A stop
 * survives while any selected line calls there, and drops only once every line
 * through it is hidden — so a transfer never disappears while a line still runs
 * through it, and a stop served only by hidden lines leaves no orphan dot. */
export function stationShown(
  station: DiagramStation,
  sel: RouteSelection,
): boolean {
  return sel === null || station.routes.some((route) => sel.has(route));
}

/** Toggle one route in or out. Re-canonicalises to `null` the moment every
 * drawn route is on again, so "all" always has one representation. */
export function toggleRoute(
  sel: RouteSelection,
  route: string,
  all: readonly string[],
): RouteSelection {
  const next = new Set(sel ?? all);
  if (next.has(route)) next.delete(route);
  else next.add(route);
  return next.size === all.length ? null : next;
}

/** Read a selection from a URL query string. Absent param = all (`null`); an
 * explicit empty list or the `none` sentinel = the empty selection, which is a
 * real, shareable state distinct from "all". */
export function parseSelection(search: string): RouteSelection {
  const params = new URLSearchParams(search);
  if (!params.has("routes")) return null;
  const raw = params.get("routes") ?? "";
  if (raw === "" || raw === "none") return new Set();
  return new Set(raw.split(",").filter((route) => route.length > 0));
}

/** The `routes` param value for a selection, or `null` to omit the param.
 * Canonicalises against the drawn set: all-on omits the param, none-on writes
 * the `none` sentinel, and unknown ids are dropped so the link stays clean.
 * Route ids never contain a comma, so a comma-joined list round-trips. */
export function serializeSelection(
  sel: RouteSelection,
  all: readonly string[],
): string | null {
  if (sel === null) return null;
  const on = all.filter((route) => sel.has(route));
  if (on.length === all.length) return null;
  if (on.length === 0) return "none";
  return on.join(",");
}
