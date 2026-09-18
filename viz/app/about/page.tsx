"use client";

import Link from "next/link";
import { PageHeader } from "../ui";
import { fmtAgo } from "@/lib/feed";
import { useSnapshot } from "../useData";

// A plain-English orientation page: what Momentarily is, where its data comes
// from, how it decides a line's status, what freshness means, and what it can't
// see. The honest framing is lifted from the Models intro (models/page.tsx) so
// the two pages can't drift; the full methodology stays on /models.
export default function AboutPage() {
  const { data: snap } = useSnapshot();
  return (
    <div className="wrap">
      <PageHeader subtitle="What this is, where the data comes from, and what it can't tell you." />

      <div style={{ maxWidth: 680 }}>
        <h2 className="grp">What this is</h2>
        <p>
          Momentarily is a live status board for the New York City subway. It
          watches where the trains actually are, minute to minute, and says
          whether each line is running normally, disrupted, or suspended. It is a
          public, independent project. It is not affiliated with the MTA.
        </p>

        <h2 className="grp">Where the data comes from</h2>
        <p>
          Every reading is built from the MTA&apos;s own public feeds: the
          GTFS-RT vehicle positions (where each train is right now), the GTFS-RT
          service alerts (what the MTA is announcing), and the
          elevator-and-escalator status feed. These feeds are public and need no
          API key. We add no private data.
        </p>
        {snap?.attribution && <p className="grp-note">{snap.attribution}</p>}

        <h2 className="grp">How status is decided</h2>
        <p>
          A line&apos;s condition is set by the severity of what the MTA has
          announced. An active non-planned alert at a severe tier — Severe Delays
          or a suspension — reads as disrupted or suspended. A planned No
          Scheduled Service alert reads as not scheduled. Everything else reads
          as normal. The alert feed drives the condition call; train movement
          does not.
        </p>
        <p>
          Train movement drives separate surfaces: the per-segment and
          per-station movement reads, the observed headway at each
          line&apos;s reference stop, and the platform crowding estimate. These
          describe what is happening on the tracks without changing the
          condition label.
        </p>
        <p>
          Recovery is how long disruptions like this one have actually lasted,
          conditioned on how long this one has already been going. It is an
          empirical count from past incidents of the same type — not a fitted
          model.
        </p>

        <h2 className="grp">What freshness means</h2>
        <p>
          Every reading has an age. The board refreshes on a 60-second cadence,
          but the MTA feeds behind it can lag or briefly stall. On a tick where a
          feed fails, we keep the last good reading rather than publish an empty
          one, so a value can be older than 60 seconds. That is why every reading
          carries its own age: check it before you trust a quiet line.
        </p>
        {snap && (
          <p className="grp-note">
            The snapshot on this page was generated{" "}
            {fmtAgo(snap.generated_at, Math.floor(Date.now() / 1000))}.
          </p>
        )}

        <h2 className="grp">What we do not know</h2>
        <ul>
          <li>
            Condition follows the MTA alert feed. If the MTA has not posted an
            alert, a real disruption will read as normal even if the movement
            surfaces show trains are stuck.
          </li>
          <li>
            There is no archived truth for unannounced disruptions, so we cannot
            measure how often condition misses something real. Only incidents the
            alert feed captured can be graded.
          </li>
          <li>
            Grades on the Models page compare the model against its own published
            stream. That is self-consistency, not independent ground truth.
          </li>
          <li>
            A line the snapshot carries no status for (some shuttle badges) is
            never judged. It is shown as unknown, not healthy.
          </li>
          <li>
            When a feed lags or fails, a reading can be stale. The timestamp is
            the only thing that says how stale.
          </li>
        </ul>

        <h2 className="grp">The full methodology</h2>
        <p>
          The <Link href="/models">Models page</Link> shows how well the calls
          hold up, line by line, with the numbers behind every claim on this
          page.
        </p>
      </div>
    </div>
  );
}
