import type { Snapshot } from "./types";

// The public snapshot. Override with NEXT_PUBLIC_FEED_BASE to point at a local
// Worker or a staging feed.
export const FEED_BASE =
  process.env.NEXT_PUBLIC_FEED_BASE ?? "https://feed.momentarily.nyc";

export async function fetchSnapshot(): Promise<Snapshot> {
  const res = await fetch(`${FEED_BASE}/v1/snapshot.json`, { cache: "no-store" });
  if (!res.ok) throw new Error(`snapshot fetch failed: ${res.status}`);
  return res.json();
}

const CONDITION_RANK: Record<string, number> = {
  suspended: 3,
  disrupted: 2,
  normal: 1,
  unknown: 0,
};

export function conditionRank(c: string | null | undefined): number {
  return CONDITION_RANK[c ?? "unknown"] ?? 0;
}

// The severity the MTA alert *cause* implies, independent of the HMM. Lets us
// flag when the model's condition (severity axis) diverges from the alert's
// label (cause axis) — e.g. planned "No Scheduled Service" reads as a
// suspension for display but the HMM treats it as disrupted.
export function impliedCondition(category: string | null | undefined): string {
  if (category === "service_suspension") return "suspended";
  if (!category || category === "none") return "normal";
  return "disrupted";
}

// Fallback colors for routes the compat layer doesn't carry. Standard MTA hues.
const FALLBACK_COLOR = "#6e6e73";

export function routeColor(snap: Snapshot, routeId: string): string {
  return snap.compat?.subwaynow_routes?.[routeId]?.color ?? FALLBACK_COLOR;
}

export function routeLabel(snap: Snapshot, routeId: string): string {
  return snap.compat?.subwaynow_routes?.[routeId]?.name ?? routeId;
}

export function fmtAgo(epochSec: number | null | undefined, nowSec: number): string {
  if (epochSec == null) return "—";
  const d = Math.max(0, nowSec - epochSec);
  if (d < 90) return `${d}s ago`;
  if (d < 5400) return `${Math.round(d / 60)}m ago`;
  if (d < 172800) return `${Math.round(d / 3600)}h ago`;
  return `${Math.round(d / 86400)}d ago`;
}

export function fmtMinutes(min: number): string {
  if (min <= 0) return "—";
  if (min < 60) return `${Math.round(min)}m`;
  const h = Math.floor(min / 60);
  const m = Math.round(min % 60);
  return m ? `${h}h ${m}m` : `${h}h`;
}
