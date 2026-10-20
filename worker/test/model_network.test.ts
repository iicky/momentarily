/**
 * Network layer (schema network/s1) across languages.
 *
 *   * shared/model/s1/*.ts must be exactly what json-schema-to-typescript
 *     renders from schema/model/s1/*.schema.json today (no hand edits, no
 *     stale mirror). Regenerate with `bun run gen:model`.
 *   * The network built by the Python builder from the fixture feed must
 *     validate against the Pydantic-exported JSON Schema, and read cleanly
 *     through the generated types.
 */

/// <reference types="node" />

import { readFileSync } from 'node:fs';
import { join } from 'node:path';

import Ajv2020, { type ValidateFunction } from 'ajv/dist/2020';
import { describe, expect, test } from 'vitest';

import dayFeedMapSchema from '../../schema/model/s1/day_feed_map.schema.json';
import schema from '../../schema/model/s1/network.schema.json';
import indexSchema from '../../schema/model/s1/network_index.schema.json';
import type { DayFeedMap } from '../../shared/model/s1/day_feed_map';
import type { NetworkSnapshot, Pattern, Segment } from '../../shared/model/s1/network';
import type { NetworkIndex } from '../../shared/model/s1/network_index';
import dayFeedMapFixture from '../../tests/fixtures/model/day_feed_map_s1.json';
import fixture from '../../tests/fixtures/model/network_s1.json';
import indexFixture from '../../tests/fixtures/model/network_index_s1.json';
import { TYPES_DIR, listFiles, renderModelTypes } from '../scripts/model_types';

describe('generated model types', () => {
  test('shared/model/s1 (every layer) matches the committed JSON Schemas', async () => {
    const rendered = await renderModelTypes();
    expect(rendered.size).toBeGreaterThan(0);
    const onDisk = listFiles(TYPES_DIR, '.ts');
    expect(onDisk, 'shared/model/s1 has files no schema generates: run bun run gen:model').toEqual(
      [...rendered.keys()].sort(),
    );
    for (const [name, text] of rendered) {
      const committed = readFileSync(join(TYPES_DIR, name), 'utf8');
      expect(committed, `shared/model/s1/${name} is stale: run bun run gen:model`).toBe(text);
    }
  });
});

describe('index and day map conform to their generated contracts', () => {
  const ajv = new Ajv2020({ allErrors: true, strict: false });

  test('the fixture index validates and reads through its type', () => {
    const validate: ValidateFunction<NetworkIndex> = ajv.compile<NetworkIndex>(indexSchema);
    expect(validate(indexFixture), JSON.stringify(validate.errors)).toBe(true);
    if (!validate(indexFixture)) return;
    const entries = Object.values(indexFixture.feeds);
    expect(entries.map((e) => e?.build_id)).toEqual([fixture.build_id]);
  });

  test('the fixture day map validates and reads through its type', () => {
    const validate: ValidateFunction<DayFeedMap> = ajv.compile<DayFeedMap>(dayFeedMapSchema);
    expect(validate(dayFeedMapFixture), JSON.stringify(validate.errors)).toBe(true);
    if (!validate(dayFeedMapFixture)) return;
    const verified = Object.values(dayFeedMapFixture.days).filter((d) => d?.verified);
    expect(verified.map((d) => d?.feed_id)).toEqual([fixture.feed.feed_id]);
  });
});

describe('network snapshot conforms to the generated contract', () => {
  const ajv = new Ajv2020({ allErrors: true, strict: false });
  const validate: ValidateFunction<NetworkSnapshot> = ajv.compile<NetworkSnapshot>(schema);

  function parse(doc: unknown): NetworkSnapshot {
    if (!validate(doc)) {
      throw new Error(`network snapshot failed schema:\n${JSON.stringify(validate.errors, null, 2)}`);
    }
    return doc;
  }

  test('the fixture build validates', () => {
    expect(() => parse(fixture)).not.toThrow();
  });

  test('a typed read walks every pattern along its segments', () => {
    const snap = parse(fixture);
    const segments = new Map<string, Segment>(snap.segments.map((s) => [s.segment_id, s]));
    const hops = new Map<string, string[]>();
    for (const ps of [...snap.pattern_segments].sort((a, b) => a.seq - b.seq)) {
      const list = hops.get(ps.pattern_id) ?? [];
      list.push(ps.segment_id);
      hops.set(ps.pattern_id, list);
    }
    const walk = (p: Pattern): string[] => {
      const stops = [p.stops[0]];
      for (const id of hops.get(p.pattern_id) ?? []) {
        const seg = segments.get(id);
        expect(seg?.from_stop).toBe(stops[stops.length - 1]);
        if (seg) stops.push(seg.to_stop);
      }
      return stops;
    };
    for (const p of snap.patterns) expect(walk(p)).toEqual(p.stops);

    const pastMidnight = snap.trips.filter((t) => t.stop_times.some(([, arr]) => arr >= 86_400));
    expect(pastMidnight.map((t) => t.service_id)).toEqual(['Weekday']);
  });

  test('a document missing a field, or carrying an extra one, is rejected', () => {
    const { trips: _trips, ...missing } = fixture;
    expect(validate(missing)).toBe(false);
    expect(validate({ ...fixture, extra: true })).toBe(false);
  });
});
