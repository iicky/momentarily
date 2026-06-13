// Server-only R2 access via the S3-compatible API.
//
// Why S3 and not the public Worker: the grading streams are timestamped JSONL
// files (v1/predictions/<date>/<ts>.jsonl) and reading a window means LISTing a
// prefix, which the public Worker deliberately doesn't expose. We own the bucket,
// this tool runs locally, so we read R2 directly.
//
// Credentials come from the project's murk vault — the same secrets the Worker
// deploy uses (R2_ACCOUNT_ID, R2_ACCESS_KEY_ID, R2_SECRET_ACCESS_KEY, R2_BUCKET).
// Two ways to surface them, tried in order:
//   1. process.env — set when launched via `murk exec -- next dev` (default), or
//      from .env.local / CI.
//   2. The @iicky/murk-secrets bindings reading ../.murk directly, for a plain
//      `next dev` after `source .env` (needs MURK_KEY/MURK_KEY_FILE in the env).

import path from "node:path";
import {
  S3Client,
  ListObjectsV2Command,
  GetObjectCommand,
} from "@aws-sdk/client-s3";

export interface R2Creds {
  accountId: string;
  accessKeyId: string;
  secretAccessKey: string;
  bucket: string;
}

function fromEnv(): R2Creds | null {
  const a = process.env.R2_ACCOUNT_ID;
  const k = process.env.R2_ACCESS_KEY_ID;
  const s = process.env.R2_SECRET_ACCESS_KEY;
  if (a && k && s)
    return {
      accountId: a,
      accessKeyId: k,
      secretAccessKey: s,
      bucket: process.env.R2_BUCKET ?? "momentarily",
    };
  return null;
}

async function fromVault(): Promise<R2Creds | null> {
  try {
    // Dynamic so a missing native binding degrades gracefully instead of
    // crashing the route — and so Next never bundles the .node.
    const murk = await import("@iicky/murk-secrets");
    const vaultPath =
      process.env.MURK_VAULT ?? path.resolve(process.cwd(), "..", ".murk");
    const vault = murk.load(vaultPath); // reads MURK_KEY / MURK_KEY_FILE
    const a = vault.get("R2_ACCOUNT_ID");
    const k = vault.get("R2_ACCESS_KEY_ID");
    const s = vault.get("R2_SECRET_ACCESS_KEY");
    if (a && k && s)
      return {
        accountId: a,
        accessKeyId: k,
        secretAccessKey: s,
        bucket: vault.get("R2_BUCKET") ?? "momentarily",
      };
  } catch {
    // bindings unavailable (no key in env, or native binary not published) —
    // fall through to "not configured".
  }
  return null;
}

let _creds: R2Creds | null | undefined;
async function creds(): Promise<R2Creds | null> {
  if (_creds === undefined) _creds = fromEnv() ?? (await fromVault());
  return _creds;
}

export async function r2Configured(): Promise<boolean> {
  return (await creds()) !== null;
}

let _client: S3Client | null = null;
async function client(): Promise<S3Client> {
  if (_client) return _client;
  const c = await creds();
  if (!c) throw new Error("R2 credentials unavailable");
  _client = new S3Client({
    region: "auto",
    endpoint: `https://${c.accountId}.r2.cloudflarestorage.com`,
    credentials: { accessKeyId: c.accessKeyId, secretAccessKey: c.secretAccessKey },
  });
  return _client;
}

async function bucket(): Promise<string> {
  return (await creds())!.bucket;
}

/** List every key under a prefix (follows continuation tokens). */
export async function listKeys(prefix: string): Promise<string[]> {
  const c = await client();
  const Bucket = await bucket();
  const out: string[] = [];
  let token: string | undefined;
  do {
    const res = await c.send(
      new ListObjectsV2Command({ Bucket, Prefix: prefix, ContinuationToken: token }),
    );
    for (const o of res.Contents ?? []) if (o.Key) out.push(o.Key);
    token = res.IsTruncated ? res.NextContinuationToken : undefined;
  } while (token);
  return out;
}

export async function getText(key: string): Promise<string> {
  const c = await client();
  const res = await c.send(
    new GetObjectCommand({ Bucket: await bucket(), Key: key }),
  );
  return (await res.Body!.transformToString()) as string;
}

export async function getJson<T>(key: string): Promise<T> {
  return JSON.parse(await getText(key)) as T;
}

/** Parse a JSONL blob into records, skipping blank lines. */
export function parseJsonl<T>(text: string): T[] {
  const out: T[] = [];
  for (const line of text.split("\n")) {
    const t = line.trim();
    if (t) out.push(JSON.parse(t) as T);
  }
  return out;
}

/** UTC YYYY-MM-DD strings for the last `days` days, inclusive of today. */
export function utcDateWindow(days: number, nowMs: number): string[] {
  const out: string[] = [];
  for (let i = 0; i < days; i++) {
    const d = new Date(nowMs - i * 86_400_000);
    out.push(d.toISOString().slice(0, 10));
  }
  return out;
}

export const STREAMS = {
  predictions: "v1/predictions",
  transitions: "v1/regime_transitions",
  params: "state/params.json",
} as const;
