"use client";

// The saved-commutes view: every NAMED segment collection the site remembers,
// each showing live status scoped to EXACTLY its own (route, direction,
// from_stop) cells. The scoping is direction-aware by construction — a commute
// stores directional cell keys, so a southbound problem never colours a
// northbound commute — and where the movement read and the alert feed disagree
// for a saved cell, that contradiction is surfaced on the card rather than
// silently resolved to one signal. Persistence is localStorage only (see
// lib/commutes.ts); there are no accounts and no server round-trip.

import { useEffect, useMemo, useState } from "react";
import Link from "next/link";
import { useSnapshot } from "../useData";
import { PageHeader, RouteBullet } from "../ui";
import { undirected } from "@/lib/stations";
import { fmtMinutes } from "@/lib/feed";
import {
  commuteStatus,
  loadCommutes,
  persistCommutes,
  type Commute,
  type CommuteStatus,
  type CommuteRollup,
  type SegmentReading,
} from "@/lib/commutes";
import type { Snapshot } from "@/lib/types";
import { HeadwayRead } from "../HeadwayRead";
import { headwayFor } from "@/lib/headway";

const DIR_LABEL: Record<string, string> = { north: "Northbound", south: "Southbound" };

// The collection headline: state class (shared with the segment palette) plus a
// plain-language verdict for the whole commute.
const ROLLUP: Record<CommuteRollup, { cls: string; label: string }> = {
  disrupted: { cls: "disrupted", label: "Disrupted" },
  normal: { cls: "normal", label: "Moving" },
  quiet: { cls: "quiet", label: "Quiet" },
  unknown: { cls: "unknown", label: "No reading" },
};

// localStorage-backed list, reloaded on cross-tab writes. Same-tab saves happen
// on /trip, which navigates here and remounts, so a fresh load already reflects
// them; the storage listener only keeps a second open tab in sync.
function useCommutes() {
  const [commutes, setCommutes] = useState<Commute[]>([]);
  useEffect(() => {
    setCommutes(loadCommutes());
    const reload = () => setCommutes(loadCommutes());
    window.addEventListener("storage", reload);
    return () => window.removeEventListener("storage", reload);
  }, []);
  const remove = (id: string) => {
    const next = loadCommutes().filter((c) => c.id !== id);
    persistCommutes(next);
    setCommutes(next);
  };
  const rename = (id: string, name: string) => {
    const next = loadCommutes().map((c) => (c.id === id ? { ...c, name } : c));
    persistCommutes(next);
    setCommutes(next);
  };
  return { commutes, remove, rename };
}

export default function CommutesPage() {
  const { data: snap } = useSnapshot();
  const { commutes, remove, rename } = useCommutes();

  return (
    <div className="wrap">
      <PageHeader subtitle="Journeys you saved, each watched on just its own segments — the direction you actually ride." />

      {!snap ? (
        <div className="sub">loading…</div>
      ) : commutes.length === 0 ? (
        <div className="note">
          No saved commutes yet. Build a journey on the{" "}
          <Link href="/trip">Trip</Link> view and save it — it will show up here
          with live status for exactly its segments.
        </div>
      ) : (
        <div className="commute-list">
          {commutes.map((c) => (
            <CommuteCard
              key={c.id}
              snap={snap}
              commute={c}
              onRemove={() => remove(c.id)}
              onRename={(name) => rename(c.id, name)}
            />
          ))}
        </div>
      )}

      {snap?.segment_flow && (
        <div className="prov-note">
          movement · segment_flow observed{" "}
          <code>{new Date(snap.segment_flow.observed_at * 1000).toLocaleTimeString()}</code>
        </div>
      )}
    </div>
  );
}

function CommuteCard({
  snap,
  commute,
  onRemove,
  onRename,
}: {
  snap: Snapshot;
  commute: Commute;
  onRemove: () => void;
  onRename: (name: string) => void;
}) {
  const status = useMemo(() => commuteStatus(snap, commute), [snap, commute]);
  const [editing, setEditing] = useState(false);
  const [draft, setDraft] = useState(commute.name);
  const [confirmDelete, setConfirmDelete] = useState(false);

  const badge = ROLLUP[status.rollup];

  const commitName = () => {
    const next = draft.trim();
    if (next && next !== commute.name) onRename(next);
    setEditing(false);
  };

  return (
    <section className="commute-card">
      <header className="commute-head">
        <div className="commute-title">
          {editing ? (
            <input
              className="commute-name-input"
              value={draft}
              autoFocus
              onChange={(e) => setDraft(e.target.value)}
              onKeyDown={(e) => {
                if (e.key === "Enter") commitName();
                if (e.key === "Escape") setEditing(false);
              }}
              onBlur={commitName}
            />
          ) : (
            <h2 className="commute-name">{commute.name}</h2>
          )}
          <span className={`cond ${badge.cls}`}>{badge.label}</span>
        </div>
        <div className="commute-actions">
          {!editing && (
            <button
              type="button"
              className="commute-btn"
              onClick={() => {
                setDraft(commute.name);
                setEditing(true);
              }}
            >
              Rename
            </button>
          )}
          {confirmDelete ? (
            <>
              <button type="button" className="commute-btn danger" onClick={onRemove}>
                Remove?
              </button>
              <button type="button" className="commute-btn" onClick={() => setConfirmDelete(false)}>
                Keep
              </button>
            </>
          ) : (
            <button type="button" className="commute-btn" onClick={() => setConfirmDelete(true)}>
              Delete
            </button>
          )}
        </div>
      </header>

      <Coverage status={status} />

      {status.disagreements.length > 0 && (
        <ul className="commute-flags">
          {status.disagreements.map((r) => (
            <Disagreement key={r.segment.key} snap={snap} reading={r} />
          ))}
        </ul>
      )}

      <CommuteStrip snap={snap} commute={commute} status={status} />
    </section>
  );
}

// One line under the headline that never lets a verdict overclaim health for the
// cells it could not judge: disrupted counts and recovery when something is
// broken; otherwise coverage — how many of the commute's cells were actually
// read this tick.
function Coverage({ status }: { status: CommuteStatus }) {
  const { total, judged, disruptedCount, unknownCount, worstRecoveryMinutes } = status;
  if (disruptedCount > 0) {
    return (
      <div className="commute-coverage">
        {disruptedCount} of {total} segment{total === 1 ? "" : "s"} disrupted
        {worstRecoveryMinutes != null ? ` · ~${fmtMinutes(worstRecoveryMinutes)} to clear` : ""}
      </div>
    );
  }
  if (judged === 0) {
    return (
      <div className="commute-coverage muted">
        No live reading for {total === 1 ? "this segment" : `any of ${total} segments`} this tick.
      </div>
    );
  }
  return (
    <div className="commute-coverage">
      Moving on {judged} of {total} segment{total === 1 ? "" : "s"}
      {unknownCount > 0 ? ` · ${unknownCount} not judged this tick` : ""}
    </div>
  );
}

function Disagreement({ snap, reading }: { snap: Snapshot; reading: SegmentReading }) {
  const nameOf = (stop: string): string => snap.stations[undirected(stop)]?.name ?? undirected(stop);
  const { segment, disagreement, alert } = reading;
  const where = `${nameOf(segment.from)} → ${nameOf(segment.to)}`;
  const dir = DIR_LABEL[segment.direction] ?? segment.direction;
  const detail =
    disagreement === "alert-only"
      ? `advisory up${alert ? ` (“${alert}”)` : ""}, but trains are moving`
      : "trains slowed here, no advisory posted";
  return (
    <li className={`commute-flag ${disagreement}`}>
      <span className="commute-flag-mark">⚠</span>
      <span>
        <b>
          {segment.route} {dir}
        </b>{" "}
        — {detail} · {where}
      </span>
    </li>
  );
}

// The commute's segments as a strip, one row per hop coloured by its movement
// read, grouped by leg with a transfer break — the same visual language the
// trip view uses. Status is read from the scoped `status.readings`, keyed by
// segment cell key so direction is never lost.
function CommuteStrip({
  snap,
  commute,
  status,
}: {
  snap: Snapshot;
  commute: Commute;
  status: CommuteStatus;
}) {
  const nameOf = (stop: string): string => snap.stations[undirected(stop)]?.name ?? undirected(stop);
  const byKey = useMemo(
    () => new Map(status.readings.map((r) => [r.segment.key, r])),
    [status],
  );
  return (
    <div className="trip-strip commute-strip">
      {commute.legs.map((leg, li) => {
        const board = leg.segments[0].from;
        const alight = leg.segments[leg.segments.length - 1].to;
        return (
          <div key={li}>
            <div className="trip-leg">
              <div className="trip-leg-head">
                <RouteBullet snap={snap} route={leg.route} size={22} />
                <span className="trip-dir">{DIR_LABEL[leg.direction] ?? leg.direction}</span>
                <span className="trip-toward">to {nameOf(alight)}</span>
              </div>
              {(leg.direction === "north" || leg.direction === "south") && (
                <HeadwayRead obs={headwayFor(snap, leg.route, leg.direction)} />
              )}
              <div className="trip-board">
                <span className="trip-rail trip-rail-board">
                  <span className="trip-node board" />
                </span>
                <Link className="trip-stop-name" href={`/stations/${undirected(board)}`}>
                  {nameOf(board)}
                </Link>
              </div>
              <ul className="trip-segs">
                {leg.segments.map((s) => {
                  const reading = byKey.get(s.key);
                  const cell = reading?.status ?? null;
                  const recovery = reading?.recoveryMinutes ?? null;
                  return (
                    <li className="trip-seg" key={s.key}>
                      <span className="trip-rail">
                        <span className={`trip-conn ${cell ?? "unknown"}`} />
                        <span className="trip-node" />
                      </span>
                      <Link className="trip-stop-name" href={`/stations/${undirected(s.to)}`}>
                        {nameOf(s.to)}
                      </Link>
                      {cell ? (
                        <span className={`cond ${cell}`}>
                          {cell}
                          {cell === "disrupted" && recovery != null ? ` · ~${fmtMinutes(recovery)}` : ""}
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
            {li < commute.legs.length - 1 && (
              <div className="trip-transfer">
                <span className="trip-transfer-mark">⇅</span>
                Transfer at {nameOf(alight)}
              </div>
            )}
          </div>
        );
      })}
    </div>
  );
}
