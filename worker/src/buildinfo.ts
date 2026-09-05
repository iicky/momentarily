/**
 * Code provenance for the live Worker. __GIT_SHA__ / __GIT_DIRTY__ are replaced
 * at deploy time by wrangler `--define` (see package.json deploy script) with
 * string literals from git. The `typeof` guard keeps it safe when they're NOT
 * defined (dev, or a deploy that forgot the flags): the identifier is absent, so
 * provenance degrades to "unknown" rather than throwing a ReferenceError.
 */

declare const __GIT_SHA__: string;
declare const __GIT_DIRTY__: string;

export const CODE_SHA: string =
  typeof __GIT_SHA__ === 'string' ? __GIT_SHA__ : 'unknown';

export const CODE_DIRTY: boolean | null =
  typeof __GIT_DIRTY__ === 'string' ? __GIT_DIRTY__ === 'true' : null;

// Identity of the trained params.json that produced a snapshot's inference —
// the one thing code_sha can't say, since the model version moves independently
// of the deployed Worker. `trained_at` is the trainer's own version stamp (the
// `trained_at` the params doc carries); `key` is the immutable versioned R2
// object that stamp maps to (state/params/v<trained_at>.json), so a consumer can
// pin the exact params without a LIST or a read of the live pointer. Both null
// means the Worker is running on BOOTSTRAP params — no params.json published yet
// — not that identity was unavailable. Snapshot carries this; trains.json does
// NOT (it has no inference, only positions), so it stays off the shared block
// below and is attached by snapshot.ts alone.
export interface ParamsProvenance {
  trained_at: number | null;
  key: string | null;
}

export interface Provenance {
  code_sha: string;
  dirty: boolean | null;
  producer: string;
  // Present on the snapshot (always, even on bootstrap — see ParamsProvenance);
  // absent on trains.json. Optional so codeProvenance() stays the shared base
  // both artifacts build from.
  params?: ParamsProvenance | null;
  // Public URL of the W3C PROV-JSON document for the params behind this
  // snapshot's inference (v1/prov/v<trained_at>.json). Optional and attached by
  // snapshot.ts alone: present only when the served params carry a prov_ref
  // (proof the trainer published a PROV doc for them); absent for params
  // trained before the emitter existed, and absent on trains.json.
  prov_ref?: string;
}

export function codeProvenance(): Provenance {
  return { code_sha: CODE_SHA, dirty: CODE_DIRTY, producer: 'worker' };
}
