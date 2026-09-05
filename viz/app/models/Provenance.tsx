"use client";

// "What produced this" as a derivation chain, read from the trainer run's W3C
// PROV-JSON document (training/prov.py is the producer; viz/lib/prov.ts is the
// reader). The snapshot the page is looking at points at the document through
// provenance.prov_ref — a PUBLIC v1/prov/ URL the Worker attaches ONLY when the
// served params carry one.
//
// Degradation, in order of how little we have:
//   prov_ref absent   render nothing. This is today's page, unchanged: params
//                     trained before the emitter existed carry no document, and
//                     we never synthesize a key to go looking for one.
//   fetch fails       the section renders, labeled unavailable; the rest of the
//                     page is untouched.
//   body malformed    same, labeled unreadable.
//   a full document   the derivation chain.
//
// FACTS-ONLY: we draw exactly the nodes the document carries. A run whose GTFS
// fetch failed has no feed entity, so no feed node appears — an "unknown feed"
// node would fabricate lineage the run does not have.

import { useEffect, useState } from "react";
import { useSnapshot } from "../useData";
import { fetchProvChain, type ProvChain, type ProvChainState } from "@/lib/prov";
import type { Snapshot } from "@/lib/types";

function shortHash(hash: string): string {
  return hash.length > 12 ? `${hash.slice(0, 12)}…` : hash;
}

function fmtDate(epochSeconds: number): string {
  return new Date(epochSeconds * 1000).toLocaleDateString(undefined, {
    year: "numeric",
    month: "short",
    day: "numeric",
  });
}

function fmtDuration(seconds: number): string {
  if (seconds < 60) return `${seconds}s`;
  const m = Math.floor(seconds / 60);
  const s = seconds % 60;
  return s === 0 ? `${m}m` : `${m}m ${s}s`;
}

// A labeled downward edge between two stages, carrying the PROV relation that
// justifies it. The label is the vocabulary term; the caption above defines it.
function Edge({ relation }: { relation: string }) {
  return (
    <div className="prov-edge" aria-hidden>
      <span className="prov-edge-rel">{relation}</span>
    </div>
  );
}

function fieldRow(label: string, value: string) {
  return (
    <div className="prov-field" key={label}>
      <span className="prov-field-k">{label}</span>
      <span className="prov-field-v">{value}</span>
    </div>
  );
}

function FeedNode({ chain }: { chain: ProvChain }) {
  const feed = chain.feed;
  if (feed === null) return null;
  const window =
    feed.start && feed.end
      ? `${feed.start} → ${feed.end}`
      : feed.start ?? feed.end ?? null;
  return (
    <div className="prov-node">
      <div className="prov-node-head">
        <span className="prov-node-title">GTFS static feed</span>
        <span className="prov-kind">Entity</span>
      </div>
      {fieldRow("version", feed.version)}
      {fieldRow("digest", `sha256:${shortHash(feed.sha256)}`)}
      {window !== null && fieldRow("timetable", window)}
    </div>
  );
}

function ManifestNode({ chain }: { chain: ProvChain }) {
  const m = chain.manifest;
  if (m === null) return null;
  return (
    <div className="prov-node">
      <div className="prov-node-head">
        <span className="prov-node-title">Input manifest</span>
        <span className="prov-kind">Entity</span>
      </div>
      {fieldRow("blake3", shortHash(m.blake3))}
      {fieldRow(
        "archive keys",
        `${m.nAlertKeys.toLocaleString()} alert · ${m.nVehicleKeys.toLocaleString()} vehicle`,
      )}
      {fieldRow("manifest", `v${m.manifestVersion}`)}
    </div>
  );
}

function RunNode({ chain }: { chain: ProvChain }) {
  const { run, agent } = chain;
  return (
    <div className="prov-node prov-node-run">
      <div className="prov-node-head">
        <span className="prov-node-title">Trainer run</span>
        <span className="prov-kind">Activity</span>
      </div>
      {fieldRow("trained", fmtDate(run.trainedAt))}
      {fieldRow("duration", fmtDuration(Math.max(0, run.endedAt - run.startedAt)))}
      {agent && (
        <div className="prov-agent">
          <span className="prov-field-k">software agent</span>
          <span className="prov-field-v">
            code {shortHash(agent.codeSha)}
            {agent.dirty === true && <span className="prov-dirty"> · dirty tree</span>}
            {agent.producer && <span className="prov-agent-producer"> · {agent.producer}</span>}
          </span>
        </div>
      )}
    </div>
  );
}

function ArtifactsNode({ chain, activeKey }: { chain: ProvChain; activeKey: string | null }) {
  if (chain.artifacts.length === 0) return null;
  return (
    <div className="prov-node">
      <div className="prov-node-head">
        <span className="prov-node-title">Published artifacts</span>
        <span className="prov-kind">Entity</span>
      </div>
      <ul className="prov-artifacts">
        {chain.artifacts.map((a) => {
          const active = activeKey !== null && a.bucketKey === activeKey;
          const from: string[] = [];
          if (a.derivedFromFeed) from.push("feed");
          if (a.derivedFromManifest) from.push("manifest");
          return (
            <li key={a.bucketKey} className={active ? "prov-artifact active" : "prov-artifact"}>
              <span className="prov-artifact-name">{a.name}</span>
              {from.length > 0 && (
                <span className="prov-derived">wasDerivedFrom {from.join(" + ")}</span>
              )}
              {active && <span className="prov-artifact-tag">this model</span>}
            </li>
          );
        })}
      </ul>
    </div>
  );
}

function SnapshotNode({ snap }: { snap: Snapshot }) {
  const sha = snap.provenance.code_sha.slice(0, 7);
  const trainedAt = snap.provenance.params?.trained_at ?? null;
  return (
    <div className="prov-node prov-node-snapshot">
      <div className="prov-node-head">
        <span className="prov-node-title">This snapshot</span>
        <span className="prov-kind">Entity · live</span>
      </div>
      {fieldRow("generated", fmtDate(snap.generated_at))}
      {trainedAt !== null && fieldRow("model", `v${trainedAt}`)}
      {fieldRow("worker code", sha)}
    </div>
  );
}

function ChainBody({ chain, snap }: { chain: ProvChain; snap: Snapshot }) {
  const activeKey = snap.provenance.params?.key ?? null;
  const hasFeed = chain.feed !== null;
  const hasManifest = chain.manifest !== null;
  return (
    <div className="prov-chain">
      {(hasFeed || hasManifest) && (
        <>
          <div className="prov-stage prov-inputs">
            <FeedNode chain={chain} />
            <ManifestNode chain={chain} />
          </div>
          <Edge relation="used" />
        </>
      )}
      <RunNode chain={chain} />
      {chain.artifacts.length > 0 && <Edge relation="wasGeneratedBy" />}
      <ArtifactsNode chain={chain} activeKey={activeKey} />
      <Edge relation="this snapshot runs on the params artifact" />
      <SnapshotNode snap={snap} />
    </div>
  );
}

export default function Provenance() {
  const { data: snap } = useSnapshot();
  const provRef = snap?.provenance.prov_ref;
  const [state, setState] = useState<ProvChainState | { state: "loading" }>({ state: "loading" });

  useEffect(() => {
    if (!provRef) return;
    let live = true;
    setState({ state: "loading" });
    fetchProvChain(provRef).then((s) => {
      if (live) setState(s);
    });
    return () => {
      live = false;
    };
  }, [provRef]);

  // No document to point at — today's page, unchanged. Also covers a snapshot
  // that itself failed to load: with no snapshot there is no prov_ref, and we
  // render nothing rather than an empty scaffold.
  if (!snap || !provRef) return null;

  return (
    <section className="prov-section">
      <h3 className="grp">What produced this</h3>
      <p className="grp-note">
        The lineage behind the snapshot on this page, read from the trainer run&apos;s{" "}
        <a href={provRef} target="_blank" rel="noreferrer">
          W3C PROV document
        </a>
        . It reads bottom-up as cause to effect. The edge labels are PROV terms:{" "}
        <em>used</em> is an input a step read, <em>wasGeneratedBy</em> a thing a step
        produced, and <em>wasDerivedFrom</em> which input an artifact was built from.
        Only facts the document actually records appear — a missing GTFS feed means
        that run&apos;s feed fetch failed, not that the feed is unknown.
      </p>
      {state.state === "loading" && (
        <div className="muted">reading the provenance document…</div>
      )}
      {state.state === "unavailable" && (
        <div className="prov-degraded" role="note">
          Provenance document unavailable — {state.reason}. The document is referenced by
          this snapshot but could not be fetched; the rest of the page is unaffected.
        </div>
      )}
      {state.state === "malformed" && (
        <div className="prov-degraded" role="note">
          Provenance document unreadable — the referenced document did not parse as a
          trainer-run PROV document. The rest of the page is unaffected.
        </div>
      )}
      {state.state === "ready" && <ChainBody chain={state.chain} snap={snap} />}
    </section>
  );
}
