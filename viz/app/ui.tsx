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

// Text colour for a solid route bullet: black on light hues (MTA yellow/orange/
// lime), white on the dark ones — the same split MTA signage uses (N/Q/R/W ride
// black on yellow). Chosen by WCAG relative luminance so the pick is identical
// at every render site and always the higher-contrast option against the exact
// MTA background hex. Non-hex input (a CSS var fallback) keeps white.
export function bulletTextColor(bg: string): "#000" | "#fff" {
  const raw = bg.replace("#", "");
  const hex =
    raw.length === 3
      ? raw
          .split("")
          .map((c) => c + c)
          .join("")
      : raw;
  if (!/^[0-9a-fA-F]{6}$/.test(hex)) return "#fff";
  const n = Number.parseInt(hex, 16);
  const toLin = (c: number) => {
    const s = c / 255;
    return s <= 0.03928 ? s / 12.92 : ((s + 0.055) / 1.055) ** 2.4;
  };
  const lum =
    0.2126 * toLin((n >> 16) & 0xff) +
    0.7152 * toLin((n >> 8) & 0xff) +
    0.0722 * toLin(n & 0xff);
  const onWhite = 1.05 / (lum + 0.05);
  const onBlack = (lum + 0.05) / 0.05;
  return onBlack > onWhite ? "#000" : "#fff";
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
      style={{ background: color, color: bulletTextColor(color), width: size, height: size, fontSize: size * 0.5 }}
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
