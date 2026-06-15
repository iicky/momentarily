/**
 * Minimal GTFS-realtime protobuf reader — just the slice we need from the NYCT
 * subway trip-updates feeds. No dependency: we read ~6 fields, so a full
 * protobuf lib (reflection + NYCT-extension registration) would be all cost and
 * no benefit. We never materialize the StopTimeUpdate rows — only count them —
 * which is what keeps decoding 8 feeds/tick cheap.
 *
 * Field numbers verified against a live ACE feed (2026-06-14):
 *   FeedMessage.entity            = 2  (repeated message)
 *   FeedEntity.trip_update        = 3  (message)
 *   TripUpdate.trip               = 1  (TripDescriptor)
 *   TripUpdate.stop_time_update   = 2  (repeated — we count these)
 *   TripDescriptor.trip_id        = 1  (string)
 *   TripDescriptor.route_id       = 5  (string)
 *   TripDescriptor.<nyct ext>     = 1001 (NyctTripDescriptor)
 *   NyctTripDescriptor.is_assigned= 2  (bool)   <-- a dispatched, running train
 *   NyctTripDescriptor.direction  = 3  (enum: 1=N, 2=E, 3=S, 4=W)
 * Everything else is skipped by wire type.
 */

const WIRE_VARINT = 0;
const WIRE_I64 = 1;
const WIRE_LEN = 2;
const WIRE_I32 = 5;

export interface TripLite {
  routeId: string;
  tripId: string;
  isAssigned: boolean;
  direction: number | null; // NYCT enum: 1=N, 3=S; null when absent
  stopCount: number; // remaining stop_time_update entries
}

/** Cursor over a byte view; reads protobuf wire primitives, skips the rest. */
class Reader {
  private p = 0;
  private readonly len: number;
  constructor(private readonly buf: Uint8Array) {
    this.len = buf.length;
  }

  get done(): boolean {
    return this.p >= this.len;
  }

  /** Read a base-128 varint. Multiplication (not <<) so >32-bit values we skip
   * over don't corrupt the cursor. */
  varint(): number {
    let result = 0;
    let shift = 0;
    let b: number;
    do {
      b = this.buf[this.p++]!;
      result += (b & 0x7f) * 2 ** shift;
      shift += 7;
    } while (b & 0x80);
    return result;
  }

  tag(): { field: number; wire: number } {
    const t = this.varint();
    return { field: Math.floor(t / 8), wire: t & 7 };
  }

  /** A length-delimited field as a sub-view (no copy). */
  lenView(): Uint8Array {
    const n = this.varint();
    const start = this.p;
    this.p = Math.min(this.p + n, this.len);
    return this.buf.subarray(start, this.p);
  }

  string(): string {
    return new TextDecoder().decode(this.lenView());
  }

  skip(wire: number): void {
    switch (wire) {
      case WIRE_VARINT:
        this.varint();
        break;
      case WIRE_I64:
        this.p += 8;
        break;
      case WIRE_LEN: {
        // Read the length first: `this.p += this.varint()` would use the stale
        // pre-read this.p and land a varint-byte short.
        const n = this.varint();
        this.p += n;
        break;
      }
      case WIRE_I32:
        this.p += 4;
        break;
      default:
        // Unknown wire type — bail out of this message to stay in sync.
        this.p = this.len;
    }
  }
}

/** Decode one feed's bytes into the lite per-trip rows we care about. */
export function decodeTripUpdates(buf: Uint8Array): TripLite[] {
  const out: TripLite[] = [];
  const r = new Reader(buf);
  while (!r.done) {
    const { field, wire } = r.tag();
    if (field === 2 && wire === WIRE_LEN) {
      parseEntity(r.lenView(), out);
    } else {
      r.skip(wire);
    }
  }
  return out;
}

function parseEntity(view: Uint8Array, out: TripLite[]): void {
  const r = new Reader(view);
  while (!r.done) {
    const { field, wire } = r.tag();
    if (field === 3 && wire === WIRE_LEN) {
      const trip = parseTripUpdate(r.lenView());
      if (trip) out.push(trip);
    } else {
      r.skip(wire);
    }
  }
}

function parseTripUpdate(view: Uint8Array): TripLite | null {
  const r = new Reader(view);
  let descriptor: ReturnType<typeof parseTripDescriptor> | null = null;
  let stopCount = 0;
  while (!r.done) {
    const { field, wire } = r.tag();
    if (field === 1 && wire === WIRE_LEN) {
      descriptor = parseTripDescriptor(r.lenView());
    } else if (field === 2 && wire === WIRE_LEN) {
      stopCount += 1;
      r.skip(wire); // count only; never materialize the stop row
    } else {
      r.skip(wire);
    }
  }
  if (!descriptor || !descriptor.routeId) return null;
  return {
    routeId: descriptor.routeId,
    tripId: descriptor.tripId,
    isAssigned: descriptor.isAssigned,
    direction: descriptor.direction,
    stopCount,
  };
}

function parseTripDescriptor(view: Uint8Array): {
  routeId: string;
  tripId: string;
  isAssigned: boolean;
  direction: number | null;
} {
  const r = new Reader(view);
  let routeId = '';
  let tripId = '';
  let isAssigned = false;
  let direction: number | null = null;
  while (!r.done) {
    const { field, wire } = r.tag();
    if (field === 5 && wire === WIRE_LEN) {
      routeId = r.string();
    } else if (field === 1 && wire === WIRE_LEN) {
      tripId = r.string();
    } else if (field === 1001 && wire === WIRE_LEN) {
      const nyct = parseNyct(r.lenView());
      isAssigned = nyct.isAssigned;
      direction = nyct.direction;
    } else {
      r.skip(wire);
    }
  }
  return { routeId, tripId, isAssigned, direction };
}

function parseNyct(view: Uint8Array): {
  isAssigned: boolean;
  direction: number | null;
} {
  const r = new Reader(view);
  let isAssigned = false;
  let direction: number | null = null;
  while (!r.done) {
    const { field, wire } = r.tag();
    if (field === 2 && wire === WIRE_VARINT) {
      isAssigned = r.varint() !== 0;
    } else if (field === 3 && wire === WIRE_VARINT) {
      direction = r.varint();
    } else {
      r.skip(wire);
    }
  }
  return { isAssigned, direction };
}
