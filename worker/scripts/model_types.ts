/**
 * Generate shared/model/s1/*.ts from schema/model/s1/*.schema.json.
 *
 * The JSON Schemas are exported from Pydantic (scripts/export_model_schema.py),
 * the source of truth; these types are derived from them, never hand-edited.
 * worker/test/model_network.test.ts re-renders and diffs against the committed
 * files, so a stale TS mirror fails `bun run test`.
 *
 * Run:  bun run gen:model   (scripts/gen_model_types.ts)
 */

/// <reference types="node" />

import { readFileSync, readdirSync, rmSync, writeFileSync } from 'node:fs';
import { dirname, join, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';

import { compile } from 'json-schema-to-typescript';

const ROOT = resolve(dirname(fileURLToPath(import.meta.url)), '..', '..');
export const SCHEMA_DIR = join(ROOT, 'schema', 'model', 's1');
export const TYPES_DIR = join(ROOT, 'shared', 'model', 's1');

const BANNER =
  '/* Generated from schema/model/s1 by worker/scripts/model_types.ts. Do not edit.\n' +
  ' * Regenerate: uv run python -m scripts.export_model_schema && (cd worker && bun run gen:model) */';

type Json = null | boolean | number | string | Json[] | { [key: string]: Json };

/**
 * Pydantic titles every property ("Complex Id"); json-schema-to-typescript
 * would turn each into a named alias (`type ComplexId = string`). Keep titles
 * only where they name a model (objects under $defs and the root), and turn
 * 2020-12 `prefixItems` tuples into the draft-07 `items` array form the
 * generator understands.
 */
function normalise(node: Json, keepTitle: boolean): Json {
  if (Array.isArray(node)) return node.map((n) => normalise(n, false));
  if (node === null || typeof node !== 'object') return node;
  const out: { [key: string]: Json } = {};
  for (const [key, value] of Object.entries(node)) {
    if (key === 'title' && !keepTitle) continue;
    if (key === '$defs' && value !== null && typeof value === 'object' && !Array.isArray(value)) {
      out[key] = Object.fromEntries(
        Object.entries(value).map(([name, def]) => [name, normalise(def, true)]),
      );
    } else if (key === 'prefixItems') {
      out.items = normalise(value, false);
      out.minItems = Array.isArray(value) ? value.length : 0;
      out.maxItems = Array.isArray(value) ? value.length : 0;
    } else if (key === 'properties' && value !== null && typeof value === 'object' && !Array.isArray(value)) {
      out[key] = Object.fromEntries(
        Object.entries(value).map(([name, prop]) => [name, normalise(prop, false)]),
      );
    } else {
      out[key] = normalise(value, false);
    }
  }
  return out;
}

export async function renderModelTypes(): Promise<Map<string, string>> {
  const rendered = new Map<string, string>();
  const files = readdirSync(SCHEMA_DIR)
    .filter((f) => f.endsWith('.schema.json'))
    .sort();
  for (const file of files) {
    const schema = JSON.parse(readFileSync(join(SCHEMA_DIR, file), 'utf8')) as Json;
    const ts = await compile(normalise(schema, true) as object, '', {
      bannerComment: BANNER,
      additionalProperties: false,
      strictIndexSignatures: true,
      unreachableDefinitions: false,
      format: true,
      style: { singleQuote: true, printWidth: 100 },
    });
    rendered.set(file.replace(/\.schema\.json$/, '.ts'), ts);
  }
  return rendered;
}

export async function writeModelTypes(): Promise<void> {
  const rendered = await renderModelTypes();
  for (const stale of readdirSync(TYPES_DIR, { withFileTypes: true })) {
    if (stale.isFile() && stale.name.endsWith('.ts') && !rendered.has(stale.name)) {
      rmSync(join(TYPES_DIR, stale.name));
    }
  }
  for (const [name, text] of rendered) {
    writeFileSync(join(TYPES_DIR, name), text);
    console.log(`wrote ${join(TYPES_DIR, name)}`);
  }
}
