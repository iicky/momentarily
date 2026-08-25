"use client";

import Link from "next/link";
import Nav from "./Nav";
import type { Snapshot, StationServiceFlow } from "@/lib/types";

// Station-flow status -> the CSS state class its badge and stop node share.
// 'flowing' reuses the .normal palette; 'quiet' gets its own muted one, so a
// page full of stations with nothing scheduled doesn't read as a green
// all-clear. Shared so the line page and the station page can't drift.
export const FLOW_CLASS: Record<StationServiceFlow["status"], string> = {
  flowing: "normal",
  quiet: "quiet",
  degraded: "disrupted",
};

/** Shared topbar for the non-Status pages: wordmark home link + nav + subtitle. */
export function PageHeader({ subtitle }: { subtitle?: React.ReactNode }) {
  return (
    <>
      <div className="topbar">
        <h1>
          <Link href="/" className="brand">
            Momentarily
          </Link>
        </h1>
        <Nav />
      </div>
      {subtitle != null && <div className="sub">{subtitle}</div>}
    </>
  );
}

// Standard MTA route hues for lines the compat layer doesn't carry (shuttles,
// SIR), so a bullet is never the fallback grey when a real colour exists.
const ROUTE_COLORS: Record<string, string> = {
  FS: "#6cbe45",
  GS: "#808183",
  H: "#808183",
  SI: "#053159",
  SS: "#808183",
};

export function routeHue(snap: Snapshot | null, route: string): string {
  if (snap) {
    const c = snap.compat?.subwaynow_routes?.[route]?.color;
    if (c && c !== "#6e6e73") return c;
  }
  return ROUTE_COLORS[route] ?? "#6e6e73";
}

/** A colored MTA route bullet. Links to the line page when `href` is set. */
export function RouteBullet({
  snap,
  route,
  size = 26,
  href,
}: {
  snap: Snapshot | null;
  route: string;
  size?: number;
  href?: string;
}) {
  const color = routeHue(snap, route);
  const dot = (
    <span
      className="bullet"
      style={{ background: color, width: size, height: size, fontSize: size * 0.5 }}
      title={`${route} line`}
    >
      {route}
    </span>
  );
  return href ? (
    <Link href={href} aria-label={`${route} line`}>
      {dot}
    </Link>
  ) : (
    dot
  );
}
