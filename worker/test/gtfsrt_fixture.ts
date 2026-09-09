/**
 * Hand-rolled GTFS-realtime encoder for the decoder tests. Keeping the wire
 * bytes hand-built (rather than committing a captured feed) is what pins the
 * exact field numbers: a wrong tag would silently mis-decode real feeds.
 */

export function varint(n: number): number[] {
  const out: number[] = [];
  while (n > 0x7f) {
    out.push((n & 0x7f) | 0x80);
    n = Math.floor(n / 128);
  }
  out.push(n);
  return out;
}
export const tag = (field: number, wire: number): number[] => varint(field * 8 + wire);
export const lenField = (field: number, body: number[]): number[] => [
  ...tag(field, 2),
  ...varint(body.length),
  ...body,
];
export const strField = (field: number, s: string): number[] =>
  lenField(field, [...new TextEncoder().encode(s)]);
export const varField = (field: number, n: number): number[] => [
  ...tag(field, 0),
  ...varint(n),
];

export function nyct(isAssigned: boolean, direction?: number): number[] {
  return [
    ...strField(1, 'TRAIN_ID'),
    ...varField(2, isAssigned ? 1 : 0),
    ...(direction !== undefined ? varField(3, direction) : []),
  ];
}

export function tripDescriptor(opts: {
  tripId?: string;
  routeId?: string;
  isAssigned?: boolean;
  direction?: number;
  withNyct?: boolean;
}): number[] {
  return [
    ...strField(1, opts.tripId ?? '000000_A..N'),
    ...(opts.routeId !== undefined ? strField(5, opts.routeId) : []),
    ...(opts.withNyct === false
      ? []
      : lenField(1001, nyct(opts.isAssigned ?? false, opts.direction))),
  ];
}

/** StopTimeUpdate: arrival(2), departure(3), stop_id(4), schedule_relationship(5).
 * Each StopTimeEvent carries time in field 2; `arrivalDelayOnly` emits an event
 * with only delay(1) set, which is how a feed reports an event with no time. */
export interface StopSpec {
  stopId?: string;
  arrival?: number;
  departure?: number;
  rel?: number;
  arrivalDelayOnly?: boolean;
  extraField?: number[]; // unknown/skipped bytes, to prove we stay in sync
}

export function stopTimeUpdate(s: StopSpec): number[] {
  return [
    ...(s.extraField ?? []),
    ...(s.arrival !== undefined ? lenField(2, varField(2, s.arrival)) : []),
    ...(s.arrivalDelayOnly ? lenField(2, varField(1, 42)) : []),
    ...(s.departure !== undefined ? lenField(3, varField(2, s.departure)) : []),
    ...(s.stopId !== undefined ? strField(4, s.stopId) : []),
    ...(s.rel !== undefined ? varField(5, s.rel) : []),
  ];
}

/** `stops` as a number builds that many plain rows (stop id only), which is all
 * the pre-arrivals tests needed. */
export function tripUpdate(desc: number[], stops: number | StopSpec[]): number[] {
  const specs: StopSpec[] =
    typeof stops === 'number'
      ? Array.from({ length: stops }, (_, i) => ({ stopId: `STOP${i}` }))
      : stops;
  const rows: number[] = [];
  for (const s of specs) rows.push(...lenField(2, stopTimeUpdate(s)));
  return [...lenField(1, desc), ...rows];
}

export function entity(body: number[], field = 3): number[] {
  return lenField(2, [...strField(1, 'entity-id'), ...lenField(field, body)]);
}

export function feed(...entities: number[][]): Uint8Array {
  // FeedMessage.header (field 1) + entities (field 2)
  return new Uint8Array([...lenField(1, varField(1, 2)), ...entities.flat()]);
}

/**
 * A small multi-route trip-updates feed: route A with an assigned northbound
 * trip that is moving, an assigned southbound trip parked at its terminal, and
 * an unassigned (scheduled-only) trip; plus a 6 whose direction has to come
 * from the trip_id `..S` fallback. Shared by the decoder and the service-metric
 * tests so both read the same bytes.
 */
export const SERVICE_FIXTURE: Uint8Array = feed(
  entity(
    tripUpdate(tripDescriptor({ routeId: 'A', isAssigned: true, direction: 1 }), [
      { stopId: 'A15N', arrival: 1780000010, departure: 1780000040 },
      { stopId: 'A12N', arrival: 1780000160, departure: 1780000190 },
    ]),
  ),
  entity(tripUpdate(tripDescriptor({ routeId: 'A', isAssigned: true, direction: 3 }), [])),
  entity(
    tripUpdate(tripDescriptor({ routeId: 'A', isAssigned: false, direction: 3 }), [
      { stopId: 'A31S', arrival: 1780000900 },
    ]),
  ),
  entity(
    tripUpdate(
      tripDescriptor({ routeId: '6', isAssigned: true, tripId: '012345_6..S01R' }),
      [
        { stopId: '626S', arrival: 1780000020, departure: 1780000050, rel: 0 },
        { stopId: '627S', rel: 1 }, // SKIPPED: no times at all
        { stopId: '628S', departure: 1780000300, rel: 0 }, // departure only
      ],
    ),
  ),
);
