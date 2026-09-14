/**
 * arrivals.json: the published sibling artifact to snapshot.json — the per-stop
 * upcoming-arrivals countdown, at its own URL rather than a Snapshot field and
 * on the 1-minute cron rather than the 5-minute pipeline (see arrivals.ts's
 * PublishedArrivals doc comment). This file covers buildArrivals/publishArrivals
 * in isolation, plus that the built per-stop map validates against the committed
 * contract schema. The fail-soft publish gate (all-feeds-failed vs. partial) and
 * the "publishes every minute, both objects on a boundary" cadence are covered
 * end-to-end through the real scheduled() handler in index.test.ts, where the
 * decision to call publishArrivals at all is made.
 */

import Ajv2020 from 'ajv/dist/2020';
import { describe, expect, test } from 'vitest';

import schema from '../../schema/snapshot.schema.json';
import { buildArrivals, publishArrivals } from '../src/snapshot';
import type { TripLite } from '../src/gtfsrt';

const NOW = 1_700_000_000;

function trip(over: Partial<TripLite>): TripLite {
  return {
    routeId: 'Q',
    tripId: 't',
    isAssigned: true,
    direction: 3,
    stopCount: 0,
    stopTimes: [],
    ...over,
  };
}

function fakeBucket() {
  const store = new Map<string, { body: string; httpMetadata?: R2HTTPMetadata }>();
  return {
    bucket: {
      async put(key: string, body: string, opts?: { httpMetadata?: R2HTTPMetadata }) {
        store.set(key, { body, ...(opts?.httpMetadata ? { httpMetadata: opts.httpMetadata } : {}) });
        return {} as unknown;
      },
    } as unknown as R2Bucket,
    store,
  };
}

const SAMPLE_TRIPS: TripLite[] = [
  trip({
    tripId: 'q1',
    stopTimes: [{ stopId: 'Q05S', arrival: NOW + 120, departure: null, scheduleRelationship: 0 }],
  }),
];

describe('buildArrivals', () => {
  test('carries its own observed_at and a provenance block, independent of any Snapshot', () => {
    const arr = buildArrivals(NOW, SAMPLE_TRIPS, ['nqrw'], ['ace', 'nqrw']);
    expect(arr.observed_at).toBe(NOW);
    expect(arr.provenance).toEqual({ code_sha: 'unknown', dirty: null, producer: 'worker' });
    expect(arr.arrivals).toEqual({
      Q05S: [{ route: 'Q', eta_epoch: NOW + 120, seconds_away: 120, trip_id: 'q1' }],
    });
  });

  test('a complete feed set (fresh_feeds === expected_feeds) with no upcoming trains is a genuine "none due" reading', () => {
    const arr = buildArrivals(NOW, [], ['ace', 'nqrw'], ['ace', 'nqrw']);
    expect(arr.fresh_feeds).toEqual(arr.expected_feeds);
    expect(arr.arrivals).toEqual({});
  });

  test('fresh_feeds shorter than expected_feeds flags the arrivals as a partial, not a complete, read', () => {
    const arr = buildArrivals(NOW, SAMPLE_TRIPS, ['nqrw'], ['ace', 'nqrw', 'si']);
    expect(arr.fresh_feeds).toEqual(['nqrw']);
    expect(arr.expected_feeds).toEqual(['ace', 'nqrw', 'si']);
    expect(arr.fresh_feeds.length).toBeLessThan(arr.expected_feeds.length);
  });
});

describe('publishArrivals', () => {
  test('writes to v1/arrivals.json with a short per-minute cache-control, not the 5-minute artifacts\' policy', async () => {
    const { bucket, store } = fakeBucket();
    const arr = buildArrivals(NOW, SAMPLE_TRIPS, ['nqrw'], ['nqrw']);
    await publishArrivals(bucket, arr);

    const rec = store.get('v1/arrivals.json');
    expect(rec).toBeDefined();
    expect(JSON.parse(rec!.body)).toEqual(arr);
    expect(rec!.httpMetadata?.contentType).toBe('application/json');
    // Deliberately shorter than snapshot.json/trains.json (max-age=60,s-maxage=300).
    expect(rec!.httpMetadata?.cacheControl).toBe('public, max-age=30, s-maxage=30');
  });
});

describe('the published per-stop map conforms to the committed contract schema', () => {
  const ajv = new Ajv2020({ allErrors: true, strict: false });
  const validate = ajv.compile(schema);

  test('buildArrivals().arrivals validates as Snapshot.arrivals', () => {
    const arr = buildArrivals(NOW, SAMPLE_TRIPS, ['nqrw'], ['nqrw']);
    // The arrivals surface is the same row type Snapshot.arrivals carries, so
    // the committed snapshot.schema.json is its contract of record: attach the
    // built map to an otherwise-empty snapshot and validate the whole thing.
    const snapshot = {
      schema_version: '1',
      generated_at: NOW,
      arrivals: arr.arrivals,
    };
    const ok = validate(snapshot);
    expect(ok, `arrivals failed schema:\n${JSON.stringify(validate.errors, null, 2)}`).toBe(true);
  });
});
