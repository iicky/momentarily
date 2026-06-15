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

export interface Provenance {
  code_sha: string;
  dirty: boolean | null;
  producer: string;
}

export function codeProvenance(): Provenance {
  return { code_sha: CODE_SHA, dirty: CODE_DIRTY, producer: 'worker' };
}
