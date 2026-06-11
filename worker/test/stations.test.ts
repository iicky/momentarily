/**
 * Tests for derive_station_status TS port — checks per-station totals,
 * outage attribution via the equipment catalog, ada_status calculation,
 * and the earliest_return / oldest_since rollups.
 */

import { describe, expect, test } from 'vitest';

import type { ActiveOutage, EquipmentCatalogEntry } from '../src/ene';
import { deriveStationStatuses } from '../src/stations';

function eq(
  equipment_id: string,
  station_complex_id: string,
  type: 'elevator' | 'escalator',
  opts: { ada?: boolean; active?: boolean } = {},
): EquipmentCatalogEntry {
  return {
    equipment_id,
    station: station_complex_id, // display name = id in tests for simplicity
    station_complex_id,
    gtfs_stop_id: null,
    type,
    ada_pathway: opts.ada ?? false,
    is_active: opts.active ?? true,
  };
}

function out(
  equipment_id: string,
  station: string,
  type: 'elevator' | 'escalator',
  opts: { ada?: boolean; since?: number | null; est?: number | null; reason?: string | null } = {},
): ActiveOutage {
  return {
    equipment_id,
    station,
    type,
    ada_pathway: opts.ada ?? false,
    outage: {
      reason: opts.reason ?? null,
      since: opts.since ?? 100,
      est_return: opts.est ?? 5000,
    },
  };
}

const NOW = 1000;

describe('deriveStationStatuses', () => {
  test('counts totals from catalog and outages from outage records', () => {
    const catalog = [
      eq('EL1', 'S1', 'elevator', { ada: true }),
      eq('EL2', 'S1', 'elevator', { ada: false }),
      eq('ES1', 'S1', 'escalator'),
      eq('EL3', 'S2', 'elevator', { ada: true }),
    ];
    const outages = [out('EL1', 'S1', 'elevator', { ada: true })];
    const m = deriveStationStatuses(catalog, outages, NOW);

    const s1 = m.get('S1')!;
    expect(s1.elevators_total).toBe(2);
    expect(s1.escalators_total).toBe(1);
    expect(s1.elevators_out).toBe(1);
    expect(s1.escalators_out).toBe(0);
    expect(m.get('S2')!.elevators_total).toBe(1);
    expect(m.get('S2')!.elevators_out).toBe(0);
  });

  test('ada_status: operational when no ADA elevator is out', () => {
    const catalog = [eq('EL1', 'S1', 'elevator', { ada: true })];
    const m = deriveStationStatuses(catalog, [], NOW);
    expect(m.get('S1')!.ada_status).toBe('operational');
  });

  test('ada_status: ada_degraded when any ADA elevator is out', () => {
    const catalog = [
      eq('EL1', 'S1', 'elevator', { ada: true }),
      eq('EL2', 'S1', 'elevator', { ada: false }),
    ];
    const outages = [out('EL1', 'S1', 'elevator', { ada: true })];
    expect(deriveStationStatuses(catalog, outages, NOW).get('S1')!.ada_status).toBe('ada_degraded');
  });

  test('ada_status: non_ada when station has no ADA elevators at all', () => {
    const catalog = [eq('EL1', 'S1', 'elevator', { ada: false })];
    const outages = [out('EL1', 'S1', 'elevator')];
    expect(deriveStationStatuses(catalog, outages, NOW).get('S1')!.ada_status).toBe('non_ada');
  });

  test('skips outages whose est_return has already passed', () => {
    const catalog = [eq('EL1', 'S1', 'elevator', { ada: true })];
    const outages = [out('EL1', 'S1', 'elevator', { since: 100, est: 500 })];
    expect(deriveStationStatuses(catalog, outages, NOW).get('S1')!.elevators_out).toBe(0);
  });

  test('earliest_elevator_return picks the min est_return across outages', () => {
    const catalog = [
      eq('EL1', 'S1', 'elevator'),
      eq('EL2', 'S1', 'elevator'),
    ];
    const outages = [
      out('EL1', 'S1', 'elevator', { est: 8000 }),
      out('EL2', 'S1', 'elevator', { est: 3000 }),
    ];
    expect(
      deriveStationStatuses(catalog, outages, NOW).get('S1')!.earliest_elevator_return,
    ).toBe(3000);
  });

  test('oldest_outage_since picks the min since across outages', () => {
    const catalog = [
      eq('EL1', 'S1', 'elevator'),
      eq('ES1', 'S1', 'escalator'),
    ];
    const outages = [
      out('EL1', 'S1', 'elevator', { since: 300 }),
      out('ES1', 'S1', 'escalator', { since: 150 }),
    ];
    expect(
      deriveStationStatuses(catalog, outages, NOW).get('S1')!.oldest_outage_since,
    ).toBe(150);
  });

  test('outage on an equipment_id missing from catalog falls back to display-name key', () => {
    const catalog = [eq('EL1', 'S1', 'elevator')];
    const outages = [out('STRAGGLER', 'S99-display', 'elevator')];
    const m = deriveStationStatuses(catalog, outages, NOW);
    // S1 from catalog, plus a fallback entry keyed on the display name
    expect(m.has('S1')).toBe(true);
    expect(m.get('S99-display')!.elevators_out).toBe(1);
    expect(m.get('S99-display')!.elevators_total).toBe(0);
  });

  test('outage on a catalog equipment_id resolves to the canonical complex id, not the outage station string', () => {
    const catalog = [
      { ...eq('EL1', '119', 'elevator', { ada: true }), station: '1 Av' },
    ];
    // Outage record carries display name "1 Av" but we want it to land under "119"
    const outages = [out('EL1', '1 Av', 'elevator', { ada: true })];
    const m = deriveStationStatuses(catalog, outages, NOW);
    expect(m.get('119')!.elevators_out).toBe(1);
    expect(m.has('1 Av')).toBe(false);
  });

  test('inactive equipment is excluded from totals', () => {
    const catalog = [
      eq('EL1', 'S1', 'elevator', { active: true }),
      eq('EL2', 'S1', 'elevator', { active: false }),
    ];
    expect(deriveStationStatuses(catalog, [], NOW).get('S1')!.elevators_total).toBe(1);
  });

  test('outage on inactive equipment still resolves to the canonical station id', () => {
    const catalog = [
      eq('EL1', '119', 'elevator', { active: true }),
      { ...eq('EL2', '119', 'elevator', { active: false }), station: '1 Av' },
    ];
    const outages = [out('EL2', '1 Av', 'elevator')];
    const m = deriveStationStatuses(catalog, outages, NOW);
    // Lands under the canonical id, no bogus display-name row, and the
    // decommissioned unit still doesn't count toward totals.
    expect(m.get('119')!.elevators_out).toBe(1);
    expect(m.get('119')!.elevators_total).toBe(1);
    expect(m.has('1 Av')).toBe(false);
  });
});
