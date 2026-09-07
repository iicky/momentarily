/**
 * fetchProtobuf's decode-sanity floor. A 200 is necessary but not sufficient:
 * the MTA gateway occasionally serves a truncated or empty body that the
 * tolerant GTFS-RT reader decodes to ZERO entities without throwing. Recorded
 * as-is that would land in vehicleFreshFeeds as a fresh feed of partial/garbage
 * rows; the floor reclassifies it as a failed feed (a gap) instead. These tests
 * pin that a zero-entity 200 rejects while a body carrying a real FeedEntity
 * resolves.
 */

import { afterEach, describe, expect, test, vi } from 'vitest';

import { fetchProtobuf } from '../src/fetch';

const URL = 'https://example.test/gtfs';

/** Encode `body` bytes as a length-delimited protobuf field. Field 2 is
 * FeedMessage.entity; field 1 is the header — used to build a header-only feed
 * that carries no entity. */
function lenField(field: number, body: number[]): number[] {
  return [field * 8 + 2, body.length, ...body];
}

function respond(body: Uint8Array, ok = true, status = 200): void {
  vi.stubGlobal(
    'fetch',
    vi.fn(async () =>
      ({
        ok,
        status,
        arrayBuffer: async () => body.buffer.slice(body.byteOffset, body.byteOffset + body.byteLength),
      }) as unknown as Response,
    ),
  );
}

afterEach(() => {
  vi.unstubAllGlobals();
});

describe('fetchProtobuf decode-sanity floor', () => {
  test('a 200 with a body carrying a FeedEntity resolves to its bytes', async () => {
    // One entity (field 2), itself carrying a length-1 sub-field — the reader
    // needs to reach field 2 past the header to accept.
    const body = new Uint8Array([
      ...lenField(1, [0x08, 0x01]), // header (field 1), skipped
      ...lenField(2, [...lenField(1, [...new TextEncoder().encode('e0')])]), // an entity
    ]);
    respond(body);
    await expect(fetchProtobuf(URL)).resolves.toEqual(body);
  });

  test('a 200 with an empty body is a failed feed, not a fresh empty one', async () => {
    respond(new Uint8Array([]));
    await expect(fetchProtobuf(URL)).rejects.toThrow(/zero entities/);
  });

  test('a 200 whose body decodes to zero entities (header only) is rejected', async () => {
    // A well-formed FeedMessage header and nothing else: the exact truncated
    // read that decodes cleanly to no entities and would otherwise be recorded
    // as a fresh feed.
    respond(new Uint8Array(lenField(1, [0x08, 0x01])));
    await expect(fetchProtobuf(URL)).rejects.toThrow(/zero entities/);
  });

  test('a non-2xx response still rejects on status before the floor runs', async () => {
    respond(new Uint8Array([]), false, 502);
    await expect(fetchProtobuf(URL)).rejects.toThrow(/HTTP 502/);
  });
});
