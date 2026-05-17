/**
 * Parse MTA elevator/escalator (E&E) feed records — mirrors
 * src/momentarily/ene.py. Three feeds:
 *
 *   nyct_ene.json (outages)         — one record per *active* outage
 *   nyct_ene_equipments.json (catalog) — every elevator/escalator on the system
 *   nyct_ene_upcoming.json          — scheduled future outages; ignored by
 *                                      the Worker for now
 *
 * Wall-time strings come in ET. We parse to UTC epoch seconds without
 * pulling a tz library by using Date with a synthetic UTC string and
 * adjusting for ET offset — DST-aware approximation via Intl.DateTimeFormat.
 */

export type EquipmentType = 'elevator' | 'escalator';

export interface EquipmentOutage {
  /** Free-text reason from MTA (e.g. "Capital Replacement"). */
  reason: string | null;
  /** Epoch seconds. May be far in the future for long-running outages. */
  est_return: number | null;
  /** Epoch seconds of when the outage began per MTA. */
  since: number | null;
}

export interface EquipmentCatalogEntry {
  equipment_id: string;
  station: string;
  /** Numeric station-complex ID per MTA (e.g. "119"); falls back to station
   * name when the catalog row is missing it. */
  station_complex_id: string;
  /** GTFS stop_id when present in the catalog — useful for alert mapping
   * downstream. May be null for non-NYCT equipment. */
  gtfs_stop_id: string | null;
  type: EquipmentType;
  /** True when this elevator/escalator is part of an ADA-accessible pathway. */
  ada_pathway: boolean;
  /** False when MTA flags the unit as decommissioned. We keep them in the
   * catalog so an outage record on a stale ID still resolves, but don't
   * count them toward station totals. */
  is_active: boolean;
}

export interface ActiveOutage {
  equipment_id: string;
  /** Station name from the outage record (display string like "1 Av"). The
   * catalog's stationcomplexid wins when an entry is found. */
  station: string;
  type: EquipmentType;
  ada_pathway: boolean;
  outage: EquipmentOutage;
}

const ET_OFFSET_FORMATTER = new Intl.DateTimeFormat('en-US', {
  timeZone: 'America/New_York',
  timeZoneName: 'shortOffset',
});

function etOffsetMinutes(epochSec: number): number {
  // "GMT-4" or "GMT-5"; falls back to -5 (standard time) if parse fails.
  const parts = ET_OFFSET_FORMATTER.formatToParts(new Date(epochSec * 1000));
  const tz = parts.find((p) => p.type === 'timeZoneName')?.value ?? 'GMT-5';
  const match = /GMT([+-]\d+)/.exec(tz);
  return match ? Number.parseInt(match[1]!, 10) * 60 : -300;
}

const DATE_FMT_RE =
  /^(\d{1,2})\/(\d{1,2})\/(\d{4}) (\d{1,2}):(\d{2}):(\d{2}) (AM|PM)$/;

/**
 * Parse "MM/DD/YYYY HH:MM:SS AM/PM" in ET to UTC epoch seconds.
 * Returns null for empty, whitespace, or unparseable inputs.
 */
export function parseEtEpoch(raw: string | null | undefined): number | null {
  if (!raw) return null;
  const trimmed = raw.trim();
  if (!trimmed) return null;
  const m = DATE_FMT_RE.exec(trimmed);
  if (!m) return null;
  let month = Number.parseInt(m[1]!, 10);
  const day = Number.parseInt(m[2]!, 10);
  const year = Number.parseInt(m[3]!, 10);
  let hour = Number.parseInt(m[4]!, 10);
  const minute = Number.parseInt(m[5]!, 10);
  const second = Number.parseInt(m[6]!, 10);
  const meridiem = m[7]!;
  if (meridiem === 'PM' && hour < 12) hour += 12;
  if (meridiem === 'AM' && hour === 12) hour = 0;
  // Provisional UTC epoch with no offset, then correct for ET offset at that
  // wall time (handles DST: offset depends on the date).
  const naiveUtcSec = Math.floor(
    Date.UTC(year, month - 1, day, hour, minute, second) / 1000,
  );
  const offsetMin = etOffsetMinutes(naiveUtcSec);
  return naiveUtcSec - offsetMin * 60;
}

function equipmentType(raw: unknown): EquipmentType | null {
  if (raw === 'EL') return 'elevator';
  if (raw === 'ES') return 'escalator';
  return null;
}

function asString(raw: unknown): string | null {
  return typeof raw === 'string' && raw.trim() !== '' ? raw.trim() : null;
}

/** Parse one record from nyct_ene.json (active outages). */
export function parseOutageRecord(record: unknown): ActiveOutage | null {
  if (!record || typeof record !== 'object') return null;
  const r = record as Record<string, unknown>;
  const type = equipmentType(r.equipmenttype);
  if (type === null) return null;
  const equipment_id = asString(r.equipment);
  if (!equipment_id) return null;
  const station = asString(r.station) ?? '';
  const since = parseEtEpoch(asString(r.outagedate));
  const est_return = parseEtEpoch(asString(r.estimatedreturntoservice));
  const reason = asString(r.reason);
  const ada_pathway = r.ADA === 'Y';
  return {
    equipment_id,
    station,
    type,
    ada_pathway,
    outage: { reason, est_return, since },
  };
}

/** Parse one record from nyct_ene_equipments.json (full catalog). */
export function parseEquipmentRecord(record: unknown): EquipmentCatalogEntry | null {
  if (!record || typeof record !== 'object') return null;
  const r = record as Record<string, unknown>;
  const type = equipmentType(r.equipmenttype);
  if (type === null) return null;
  const equipment_id = asString(r.equipmentno);
  if (!equipment_id) return null;
  const station = asString(r.station) ?? '';
  const station_complex_id = asString(r.stationcomplexid) ?? station;
  const gtfs_stop_id = asString(r.elevatorsgtfsstopid);
  const ada_pathway = r.ADA === 'Y';
  const is_active = r.isactive !== 'N';
  return {
    equipment_id,
    station,
    station_complex_id,
    gtfs_stop_id,
    type,
    ada_pathway,
    is_active,
  };
}

export function parseOutageFeed(payload: unknown): ActiveOutage[] {
  if (!Array.isArray(payload)) return [];
  const out: ActiveOutage[] = [];
  for (const record of payload) {
    const parsed = parseOutageRecord(record);
    if (parsed) out.push(parsed);
  }
  return out;
}

export function parseEquipmentFeed(payload: unknown): EquipmentCatalogEntry[] {
  if (!Array.isArray(payload)) return [];
  const out: EquipmentCatalogEntry[] = [];
  for (const record of payload) {
    const parsed = parseEquipmentRecord(record);
    if (parsed) out.push(parsed);
  }
  return out;
}

/**
 * Whether this outage should currently count against the station. Mirrors
 * is_active_outage in ene.py: drop the small forgive window where the est
 * return has already passed and the feed hasn't caught up.
 */
export function isActiveOutage(outage: EquipmentOutage, now: number): boolean {
  if (outage.since === null) return false;
  if (outage.est_return !== null && outage.est_return < now) return false;
  return true;
}
