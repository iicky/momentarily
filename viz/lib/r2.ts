// Server-only R2 access via the S3-compatible API.
//
// Why S3 and not the public Worker: the grading streams are timestamped JSONL
// files (v1/predictions/<date>/<ts>.jsonl) and reading a window means LISTing a
// prefix, which the public Worker deliberately doesn't expose. We own the bucket,
// this tool runs locally, so we read R2 directly. Needs an R2 API token (S3
// access key/secret) — the Workers deploy token can't read objects.
//
// Required env (.env.local):
//   R2_ACCOUNT_ID, R2_ACCESS_KEY_ID, R2_SECRET_ACCESS_KEY
//   R2_BUCKET (optional, defaults to "momentarily")

import {
  S3Client,
  ListObjectsV2Command,
  GetObjectCommand,
} from "@aws-sdk/client-s3";

const BUCKET = process.env.R2_BUCKET ?? "momentarily";

export function r2Configured(): boolean {
  return Boolean(
    process.env.R2_ACCOUNT_ID &&
      process.env.R2_ACCESS_KEY_ID &&
      process.env.R2_SECRET_ACCESS_KEY,
  );
}

let _client: S3Client | null = null;
function client(): S3Client {
  if (_client) return _client;
  const accountId = process.env.R2_ACCOUNT_ID!;
  _client = new S3Client({
    region: "auto",
    endpoint: `https://${accountId}.r2.cloudflarestorage.com`,
    credentials: {
      accessKeyId: process.env.R2_ACCESS_KEY_ID!,
      secretAccessKey: process.env.R2_SECRET_ACCESS_KEY!,
    },
  });
  return _client;
}

/** List every key under a prefix (follows continuation tokens). */
export async function listKeys(prefix: string): Promise<string[]> {
  const out: string[] = [];
  let token: string | undefined;
  do {
    const res = await client().send(
      new ListObjectsV2Command({
        Bucket: BUCKET,
        Prefix: prefix,
        ContinuationToken: token,
      }),
    );
    for (const o of res.Contents ?? []) if (o.Key) out.push(o.Key);
    token = res.IsTruncated ? res.NextContinuationToken : undefined;
  } while (token);
  return out;
}

export async function getText(key: string): Promise<string> {
  const res = await client().send(
    new GetObjectCommand({ Bucket: BUCKET, Key: key }),
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
