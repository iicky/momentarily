"use client";

// The one /trip touch this feature needs: a small affordance under a chosen
// journey that saves it — whole, or a hand-picked subset of its segments — as a
// NAMED commute in localStorage. Owned here (not in /trip) so the trip view only
// imports and renders it; all of the saving, naming, and segment-picking lives
// on this side of the boundary.

import { useMemo, useState } from "react";
import Link from "next/link";
import { undirected } from "@/lib/stations";
import { boardStop, alightStop } from "@/lib/journeys";
import type { Journey } from "@/lib/journeys";
import type { Snapshot } from "@/lib/types";
import { addCommute, newCommuteId, type CommuteLeg } from "@/lib/commutes";

const DIR_LABEL: Record<string, string> = { north: "Northbound", south: "Southbound" };

export function SaveJourney({ snap, journey }: { snap: Snapshot; journey: Journey }) {
  const [open, setOpen] = useState(false);
  const [name, setName] = useState("");
  // Which segment cells to keep, by key. Membership is toggled at runtime, so a
  // Set fits; seeded to the whole journey (the common "save all of it" path).
  const [selected, setSelected] = useState<Set<string>>(
    () => new Set(journey.segments.map((s) => s.key)),
  );
  const [savedAs, setSavedAs] = useState<string | null>(null);

  const nameOf = useMemo(() => {
    return (stop: string): string => {
      const id = undirected(stop);
      return snap.stations[id]?.name ?? id;
    };
  }, [snap]);

  const defaultName = useMemo(() => {
    const origin = nameOf(boardStop(journey.legs[0]));
    const dest = nameOf(alightStop(journey.legs[journey.legs.length - 1]));
    return `${origin} → ${dest}`;
  }, [journey, nameOf]);

  const selectedLegs = useMemo<CommuteLeg[]>(
    () =>
      journey.legs
        .map((l) => ({
          route: l.route,
          direction: l.direction,
          segments: l.segments
            .filter((s) => selected.has(s.key))
            .map((s) => ({ route: s.route, direction: s.direction, from: s.from, to: s.to, key: s.key })),
        }))
        .filter((l) => l.segments.length > 0),
    [journey, selected],
  );

  const trimmed = name.trim() || defaultName;
  const canSave = selectedLegs.length > 0 && trimmed.length > 0;

  const toggle = (key: string) => {
    setSelected((prev) => {
      const next = new Set(prev);
      if (next.has(key)) next.delete(key);
      else next.add(key);
      return next;
    });
  };

  const save = () => {
    if (!canSave) return;
    addCommute({ id: newCommuteId(), name: trimmed, createdAt: Date.now(), legs: selectedLegs });
    setSavedAs(trimmed);
    setOpen(false);
  };

  if (savedAs) {
    return (
      <div className="save-journey save-done">
        <span className="save-done-mark">✓</span>
        Saved <b>{savedAs}</b> ·{" "}
        <Link href="/commutes">View commutes →</Link>
      </div>
    );
  }

  if (!open) {
    return (
      <div className="save-journey">
        <button type="button" className="save-toggle" onClick={() => setOpen(true)}>
          ☆ Save as commute
        </button>
      </div>
    );
  }

  return (
    <div className="save-journey save-panel">
      <label className="save-name">
        <span className="save-name-label">Name</span>
        <input
          type="text"
          value={name}
          placeholder={defaultName}
          onChange={(e) => setName(e.target.value)}
          autoFocus
        />
      </label>

      <div className="save-seg-head">Segments to remember</div>
      <div className="save-seg-list">
        {journey.legs.map((l, li) => (
          <div className="save-seg-leg" key={li}>
            <div className="save-seg-leg-head">
              <span className="save-seg-route">{l.route}</span>
              <span className="save-seg-dir">{DIR_LABEL[l.direction] ?? l.direction}</span>
            </div>
            {l.segments.map((s) => (
              <label className="save-seg" key={s.key}>
                <input
                  type="checkbox"
                  checked={selected.has(s.key)}
                  onChange={() => toggle(s.key)}
                />
                <span>
                  {nameOf(s.from)} → {nameOf(s.to)}
                </span>
              </label>
            ))}
          </div>
        ))}
      </div>

      <div className="save-actions">
        <button type="button" className="save-do" disabled={!canSave} onClick={save}>
          Save
        </button>
        <button type="button" className="save-cancel" onClick={() => setOpen(false)}>
          Cancel
        </button>
      </div>
    </div>
  );
}
