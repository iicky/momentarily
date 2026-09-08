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
          The status we publish follows train movement first. A line reads
          disrupted when its trains stop moving the way they should, not because
          an alert was posted. The alerts feed is the cross-reference: it names
          the likely cause behind a call, but it does not trigger the call. The
          two are different signals, so when they disagree, the disagreement is
          itself the signal.
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
            If the trains keep moving but riders are stuck on a platform, the
            movement signal can miss it.
          </li>
          <li>
            We can measure how often the movement signal calls a moving line
            stuck, but not how often it misses a stuck line, because no archived
            movement truth exists to grade the misses against.
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
