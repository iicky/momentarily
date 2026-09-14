/**
 * Entrances static: parsing i9wp-a4ja Socrata rows, resolution against the
 * stations catalog (direct gtfs_stop_id + complex_id fallback), and the
 * published v1/entrances.json shape including coverage metadata.
 *
 * Fixture: 20 verbatim rows from the live Socrata feed, including 4 rows
 * with compound gtfs_stop_ids ("A12; D13", "A32; D20", "718; R09") — the
 * real cause of the ~14-row miss: shared entrances serving multiple stations
 * within a complex encode both ids semicolon-separated, which doesn't match
 * any single station entry.
 */

import Ajv2020 from 'ajv/dist/2020';
import { describe, expect, test } from 'vitest';

import entrancesSchema from '../../schema/entrances.schema.json';
import type { StationOut } from '../src/stations_static';
import {
  parseEntrancesFeed,
  resolveEntrances,
  buildEntrances,
} from '../src/entrances_static';
import FIXTURE_ROWS from './fixtures/entrances_sample.json';

const ajv = new Ajv2020({ allErrors: true, strict: false });
const validate = ajv.compile(entrancesSchema);

// ---------------------------------------------------------------------------
// Station catalog for resolution tests.
//
// Includes the real gtfs_stop_ids present in the fixture's direct-match rows,
// plus stations with complex_ids matching the compound-miss rows. The compound
// ids ("A12; D13" etc.) are intentionally absent — they don't exist in the
// real 39hk-dx4f feed either.
// ---------------------------------------------------------------------------
function stationsCatalog(): Record<string, StationOut> {
  const catalog: Record<string, StationOut> = {};
  function add(id: string, complexId: string | null, name: string): void {
    catalog[id] = {
      gtfs_stop_id: id,
      station_complex_id: complexId,
      name,
      borough: null,
      routes_served: [],
      ada: 0,
      ada_northbound: false,
      ada_southbound: false,
    };
  }
  // Direct-match stations (all simple gtfs_stop_ids from the fixture)
  add('101', '293', 'Van Cortlandt Park-242 St');
  add('103', '294', '238 St');
  add('104', '295', '231 St');
  add('R04', '3', '30 Av');
  add('R05', '4', 'Broadway');
  add('M22', '628', 'Fulton St');
  add('626', '397', '86 St');
  add('R16', '611', 'Times Sq-42 St');
  add('A27', '611', '42 St-Port Authority Bus Terminal');

  // Stations whose complex_id rescues the compound-miss rows.
  // 145 St complex 151: A12 and D13 are separate stations
  add('A12', '151', '145 St');
  add('D13', '151', '145 St');
  // W 4 St complex 167: A32 and D20 are separate stations
  add('A32', '167', 'W 4 St-Wash Sq');
  add('D20', '167', 'W 4 St-Wash Sq');
  // Queensboro Plaza complex 461: 718 and R09 are separate stations
  add('718', '461', 'Queensboro Plaza');
  add('R09', '461', 'Queensboro Plaza');

  return catalog;
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

describe('parseEntrancesFeed', () => {
  test('maps fields from a real Socrata row with uppercase YES/NO', () => {
    // Row 0: Van Cortlandt Park, entry_allowed=YES
    const [e] = parseEntrancesFeed([FIXTURE_ROWS[0]]);
    expect(e).toEqual({
      gtfs_stop_id: '101',
      complex_id: '293',
      station_id: '293',
      entrance_type: 'Stair',
      entry_allowed: true,
      exit_allowed: true,
      lat: 40.889783,
      lon: -73.898432,
    });
  });

  test('entry_allowed=NO parses as false', () => {
    // Row 4: 238 St, entry_allowed=NO
    const parsed = parseEntrancesFeed([FIXTURE_ROWS[4]]);
    expect(parsed[0]?.entry_allowed).toBe(false);
    expect(parsed[0]?.exit_allowed).toBe(true);
  });

  test('compound gtfs_stop_id passes through verbatim', () => {
    // Row 16: 145 St, gtfs_stop_id="A12; D13"
    const parsed = parseEntrancesFeed([FIXTURE_ROWS[16]]);
    expect(parsed[0]?.gtfs_stop_id).toBe('A12; D13');
  });

  test('skips rows missing gtfs_stop_id', () => {
    expect(parseEntrancesFeed([{ ...FIXTURE_ROWS[0], gtfs_stop_id: '' }])).toHaveLength(0);
  });

  test('skips rows missing latitude or longitude', () => {
    expect(parseEntrancesFeed([{ ...FIXTURE_ROWS[0], entrance_latitude: null }])).toHaveLength(0);
    expect(parseEntrancesFeed([{ ...FIXTURE_ROWS[0], entrance_longitude: 'bad' }])).toHaveLength(0);
  });

  test('tolerates a non-array payload', () => {
    expect(parseEntrancesFeed({ not: 'an array' })).toEqual([]);
  });

  test('parses the full 20-row fixture', () => {
    expect(parseEntrancesFeed(FIXTURE_ROWS)).toHaveLength(20);
  });

  test('drops entrance_georeference blob from output', () => {
    // Row 0 carries the real georeference blob; parsed output must not include it.
    const [e] = parseEntrancesFeed([FIXTURE_ROWS[0]]);
    expect(e).not.toHaveProperty('entrance_georeference');
    expect(Object.keys(e!)).toEqual(
      expect.arrayContaining(['gtfs_stop_id', 'lat', 'lon']),
    );
  });
});

describe('resolveEntrances', () => {
  test('direct match keys entrance under its own gtfs_stop_id', () => {
    const raws = parseEntrancesFeed([FIXTURE_ROWS[0]]);
    const { entrances, coverage } = resolveEntrances(raws, stationsCatalog());
    expect(entrances['101']).toHaveLength(1);
    expect(coverage.direct_match).toBe(1);
    expect(coverage.complex_fallback).toBe(0);
  });

  test('compound-ID rows land in complex_entrances keyed by complex_id', () => {
    const raws = parseEntrancesFeed(FIXTURE_ROWS);
    const stations = stationsCatalog();
    const { entrances, complexEntrances, coverage } = resolveEntrances(raws, stations);

    // "A12; D13" (2 rows) → complex_entrances["151"]
    expect(complexEntrances['151']).toHaveLength(2);
    expect(complexEntrances['151']![0]!.gtfs_stop_id).toBe('A12; D13');

    // "A32; D20" (1 row) → complex_entrances["167"]
    expect(complexEntrances['167']).toHaveLength(1);
    expect(complexEntrances['167']![0]!.gtfs_stop_id).toBe('A32; D20');

    // "718; R09" (1 row) → complex_entrances["461"]
    expect(complexEntrances['461']).toHaveLength(1);
    expect(complexEntrances['461']![0]!.gtfs_stop_id).toBe('718; R09');

    // None of the compound ids appear as keys in the direct entrances map
    expect(entrances['A12; D13']).toBeUndefined();
    expect(entrances['A32; D20']).toBeUndefined();
    expect(entrances['718; R09']).toBeUndefined();

    // Coverage: 16 direct + 4 complex fallback = 20
    expect(coverage.direct_match).toBe(16);
    expect(coverage.complex_fallback).toBe(4);
    expect(coverage.unresolved).toBe(0);
    expect(coverage.complex_keys).toBe(3);
  });

  test('unresolved entrances keep their source id and are counted', () => {
    const raws = parseEntrancesFeed([{
      ...FIXTURE_ROWS[16],
      complex_id: '99999', // unknown complex
    }]);
    const { entrances, coverage } = resolveEntrances(raws, stationsCatalog());
    expect(entrances['A12; D13']).toHaveLength(1);
    expect(coverage.unresolved).toBe(1);
    expect(coverage.unresolved_ids).toEqual(['A12; D13']);
  });

  test('coverage reports both direct and effective station coverage', () => {
    const raws = parseEntrancesFeed(FIXTURE_ROWS);
    const catalog = stationsCatalog();
    const { coverage } = resolveEntrances(raws, catalog);
    // 9 distinct direct-match gtfs_stop_ids: 101, 103, 104, R04, R05, M22, 626, R16, A27
    expect(coverage.station_keys).toBe(9);
    expect(coverage.row_count).toBe(20);
    expect(coverage.catalog_stations).toBe(Object.keys(catalog).length); // 15
    expect(coverage.stations_with_entrances).toBe(9);
    expect(coverage.station_coverage).toBeCloseTo(9 / 15);
    // Effective: 9 direct + stations in complexes 151 (A12,D13), 167 (A32,D20), 461 (718,R09) = 9+6=15
    expect(coverage.stations_resolved).toBe(15);
    expect(coverage.station_coverage_effective).toBeCloseTo(1.0);
  });
});

describe('buildEntrances', () => {
  const NOW = 1_700_000_000;

  test('produces a valid PublishedEntrances object', () => {
    const raws = parseEntrancesFeed(FIXTURE_ROWS);
    const published = buildEntrances(NOW, raws, stationsCatalog());

    expect(published.fetched_at).toBe(NOW);
    expect(published.provenance.producer).toBe('worker');
    expect(published.coverage.row_count).toBe(20);
    expect(published.coverage.direct_match).toBe(16);
    expect(published.coverage.complex_fallback).toBe(4);
    expect(published.coverage.unresolved).toBe(0);
  });

  test('validates against the JSON schema', () => {
    const raws = parseEntrancesFeed(FIXTURE_ROWS);
    const published = buildEntrances(NOW, raws, stationsCatalog());
    expect(validate(published), JSON.stringify(validate.errors, null, 2)).toBe(true);
  });

  test('complex_entrances carry source provenance ids', () => {
    const raws = parseEntrancesFeed(FIXTURE_ROWS);
    const published = buildEntrances(NOW, raws, stationsCatalog());

    const c151 = published.complex_entrances['151'];
    expect(c151).toBeDefined();
    expect(c151![0]!.gtfs_stop_id).toBe('A12; D13');
    expect(c151![0]!.complex_id).toBe('151');
    expect(c151![0]!.station_id).toBe('151');
  });

  test('direct-match entrances retain their source ids', () => {
    const raws = parseEntrancesFeed(FIXTURE_ROWS);
    const published = buildEntrances(NOW, raws, stationsCatalog());

    // 86 St has 2 entrances keyed under '626'
    expect(published.entrances['626']).toHaveLength(2);
    expect(published.entrances['626']![0]!.gtfs_stop_id).toBe('626');
  });
});
