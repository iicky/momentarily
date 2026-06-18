/**
 * Per-station status derivation from the parsed E&E feeds — mirror of
 * src/momentarily/derive.py:derive_station_status.
 *
 * State here is observable: we count what's out, surface the earliest
 * reported est_return, and flag the longest-running outage so UIs can
 * distinguish "out for an hour" from "out for six months." No HMM.
 *
 * Alerts-to-station matching is deferred — needs a gtfs_stop_id →
 * station_complex_id registry that the Worker doesn't yet load (the
 * catalog carries elevatorsgtfsstopid, but a complete picture also wants
 * station-platform stop_ids the alerts feed actually references). For
 * now `alerts: []` ships and downstream consumers fall back to
 * route-level alerts. See momentarily-dik.
 */

import type { ActiveOutage, EquipmentCatalogEntry, EquipmentOutage, EquipmentType } from './ene';
import { isActiveOutage } from './ene';

export type AdaStatus = 'operational' | 'ada_degraded' | 'non_ada';

/** One elevator/escalator currently out of service, for the snapshot's
 * `equipment` array. We publish only units with an active outage — the full
 * catalog's working units are summarized by the station_status totals, so
 * shipping ~2k healthy entries every tick would bloat the feed for no signal. */
export interface EquipmentOut {
  equipment_id: string;
  type: EquipmentType;
  station_complex_id: string | null;
  location_text: string | null;
  ada_pathway: boolean;
  outage: EquipmentOutage;
}

/**
 * Equipment with an active outage at `now`, enriched from the catalog (canonical
 * complex id, type, ADA flag) when the outage's equipment_id resolves there.
 */
export function buildEquipmentList(
  catalog: EquipmentCatalogEntry[],
  outages: ActiveOutage[],
  now: number,
): EquipmentOut[] {
  const byEquipmentId = new Map<string, EquipmentCatalogEntry>();
  for (const e of catalog) byEquipmentId.set(e.equipment_id, e);

  const out: EquipmentOut[] = [];
  for (const o of outages) {
    if (!isActiveOutage(o.outage, now)) continue;
    const cat = byEquipmentId.get(o.equipment_id);
    out.push({
      equipment_id: o.equipment_id,
      type: cat?.type ?? o.type,
      station_complex_id: cat?.station_complex_id ?? o.station ?? null,
      location_text: o.station ?? null,
      ada_pathway: cat?.ada_pathway ?? o.ada_pathway,
      outage: o.outage,
    });
  }
  return out;
}

export interface StationStatus {
  station_complex_id: string;
  alerts: string[];
  ada_status: AdaStatus;
  elevators_total: number;
  elevators_out: number;
  escalators_total: number;
  escalators_out: number;
  earliest_elevator_return: number | null;
  oldest_outage_since: number | null;
}

/**
 * Group equipment + outages by station_complex_id and emit one
 * StationStatus per distinct station the catalog mentions.
 *
 * Outage records carry only the station display name. We resolve the
 * canonical complex_id via the catalog by equipment_id when available;
 * outages whose equipment_id isn't in the catalog fall back to grouping
 * by the display-name string (still useful even though the resulting
 * key isn't the numeric MRN). The catalog is the spine — stations that
 * appear only in an outage record but never in the catalog get a
 * minimal entry with zero totals.
 */
export function deriveStationStatuses(
  catalog: EquipmentCatalogEntry[],
  outages: ActiveOutage[],
  now: number,
): Map<string, StationStatus> {
  // byEquipmentId keeps inactive entries — it only resolves an outage to its
  // canonical station, and outages on decommissioned units still belong to a
  // real station. byStationComplex is active-only: totals and ADA status
  // shouldn't count decommissioned units.
  const byEquipmentId = new Map<string, EquipmentCatalogEntry>();
  const byStationComplex = new Map<string, EquipmentCatalogEntry[]>();
  for (const e of catalog) {
    byEquipmentId.set(e.equipment_id, e);
    if (!e.is_active) continue;
    const existing = byStationComplex.get(e.station_complex_id);
    if (existing) existing.push(e);
    else byStationComplex.set(e.station_complex_id, [e]);
  }

  // Bucket outages by the resolved station_complex_id. Active outages only.
  const outagesByStation = new Map<string, ActiveOutage[]>();
  for (const outage of outages) {
    if (!isActiveOutage(outage.outage, now)) continue;
    const catEntry = byEquipmentId.get(outage.equipment_id);
    const complexId = catEntry?.station_complex_id ?? outage.station;
    if (!complexId) continue;
    const existing = outagesByStation.get(complexId);
    if (existing) existing.push(outage);
    else outagesByStation.set(complexId, [outage]);
  }

  // Union of stations appearing in the catalog and the outage-resolved keys.
  const allStationIds = new Set<string>(byStationComplex.keys());
  for (const k of outagesByStation.keys()) allStationIds.add(k);

  const out = new Map<string, StationStatus>();
  for (const stationId of allStationIds) {
    const entries = byStationComplex.get(stationId) ?? [];
    const outagesHere = outagesByStation.get(stationId) ?? [];

    const elevatorsTotal = entries.filter((e) => e.type === 'elevator').length;
    const escalatorsTotal = entries.filter((e) => e.type === 'escalator').length;
    const elevatorsOut = outagesHere.filter((o) => o.type === 'elevator').length;
    const escalatorsOut = outagesHere.filter((o) => o.type === 'escalator').length;

    const hasAnyAdaElevator = entries.some((e) => e.type === 'elevator' && e.ada_pathway);
    const adaElevatorOut = outagesHere.some((o) => o.type === 'elevator' && o.ada_pathway);
    let ada_status: AdaStatus;
    if (!hasAnyAdaElevator) ada_status = 'non_ada';
    else if (adaElevatorOut) ada_status = 'ada_degraded';
    else ada_status = 'operational';

    const estReturns = outagesHere
      .map((o) => o.outage.est_return)
      .filter((r): r is number => r !== null);
    const earliest_elevator_return = estReturns.length > 0 ? Math.min(...estReturns) : null;

    const sinces = outagesHere
      .map((o) => o.outage.since)
      .filter((s): s is number => s !== null);
    const oldest_outage_since = sinces.length > 0 ? Math.min(...sinces) : null;

    out.set(stationId, {
      station_complex_id: stationId,
      alerts: [],
      ada_status,
      elevators_total: elevatorsTotal,
      elevators_out: elevatorsOut,
      escalators_total: escalatorsTotal,
      escalators_out: escalatorsOut,
      earliest_elevator_return,
      oldest_outage_since,
    });
  }
  return out;
}
