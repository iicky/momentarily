/**
 * Tests for the E&E feed parser — parity with src/momentarily/ene.py.
 * Covers the ET wall-time decoder (handles DST), record shape variations
 * the worker has actually seen in production, and the active-outage filter.
 */

import { describe, expect, test } from 'vitest';

import {
  isActiveOutage,
  parseEquipmentFeed,
  parseEquipmentRecord,
  parseEtEpoch,
  parseOutageFeed,
  parseOutageRecord,
} from '../src/ene';

describe('parseEtEpoch', () => {
  test('handles standard time (January, GMT-5)', () => {
    // 2026-01-15 12:00:00 ET = 2026-01-15 17:00:00 UTC = 1768546800
    const epoch = parseEtEpoch('01/15/2026 12:00:00 PM');
    expect(epoch).toBe(Math.floor(Date.UTC(2026, 0, 15, 17, 0, 0) / 1000));
  });

  test('handles daylight time (May, GMT-4)', () => {
    // 2026-05-15 12:00:00 ET = 2026-05-15 16:00:00 UTC
    const epoch = parseEtEpoch('05/15/2026 12:00:00 PM');
    expect(epoch).toBe(Math.floor(Date.UTC(2026, 4, 15, 16, 0, 0) / 1000));
  });

  test('converts 12 AM → hour 0 not 12', () => {
    // 2026-05-15 12:00:00 AM ET = 2026-05-15 00:00:00 ET = 2026-05-15 04:00:00 UTC
    const epoch = parseEtEpoch('05/15/2026 12:00:00 AM');
    expect(epoch).toBe(Math.floor(Date.UTC(2026, 4, 15, 4, 0, 0) / 1000));
  });

  test('returns null for empty/whitespace/garbage', () => {
    expect(parseEtEpoch('')).toBeNull();
    expect(parseEtEpoch('   ')).toBeNull();
    expect(parseEtEpoch(null)).toBeNull();
    expect(parseEtEpoch(undefined)).toBeNull();
    expect(parseEtEpoch('not a date')).toBeNull();
    expect(parseEtEpoch('2026-05-15')).toBeNull();
  });
});

describe('parseOutageRecord', () => {
  const sample = {
    station: '61 St-Woodside',
    borough: '',
    trainno: '7/LIRR',
    equipment: 'ES448',
    equipmenttype: 'ES',
    serving: '61 St & Roosevelt Ave (SE corner) to mezzanine',
    ADA: 'N',
    outagedate: '09/30/2024 12:05:00 PM',
    estimatedreturntoservice: '05/31/2026 11:59:00 PM',
    reason: 'Capital Replacement',
    isupcomingoutage: 'N',
    ismaintenanceoutage: 'N',
  };

  test('parses a well-formed outage record', () => {
    const out = parseOutageRecord(sample);
    expect(out).not.toBeNull();
    expect(out!.equipment_id).toBe('ES448');
    expect(out!.type).toBe('escalator');
    expect(out!.station).toBe('61 St-Woodside');
    expect(out!.ada_pathway).toBe(false);
    expect(out!.outage.reason).toBe('Capital Replacement');
    expect(out!.outage.since).not.toBeNull();
    expect(out!.outage.est_return).not.toBeNull();
  });

  test('maps EL to elevator, ES to escalator, drops other types', () => {
    expect(parseOutageRecord({ ...sample, equipmenttype: 'EL' })?.type).toBe('elevator');
    expect(parseOutageRecord({ ...sample, equipmenttype: 'ES' })?.type).toBe('escalator');
    expect(parseOutageRecord({ ...sample, equipmenttype: 'WHEELCHAIR' })).toBeNull();
  });

  test('drops records missing equipment id or non-object', () => {
    expect(parseOutageRecord({ ...sample, equipment: '' })).toBeNull();
    expect(parseOutageRecord({ ...sample, equipment: undefined })).toBeNull();
    expect(parseOutageRecord(null)).toBeNull();
    expect(parseOutageRecord('not an object')).toBeNull();
  });

  test('null reason / est_return when missing', () => {
    const out = parseOutageRecord({
      ...sample, reason: '', estimatedreturntoservice: '',
    });
    expect(out!.outage.reason).toBeNull();
    expect(out!.outage.est_return).toBeNull();
  });

  test('parseOutageFeed skips malformed entries', () => {
    const list = parseOutageFeed([sample, null, 'garbage', { ...sample, equipment: '' }]);
    expect(list).toHaveLength(1);
  });

  test('parseOutageFeed returns [] for non-array payloads', () => {
    expect(parseOutageFeed({ entity: [] })).toEqual([]);
    expect(parseOutageFeed(null)).toEqual([]);
  });
});

describe('parseEquipmentRecord', () => {
  const sample = {
    station: '1 Av',
    borough: '',
    trainno: 'L',
    equipmentno: 'EL293',
    equipmenttype: 'EL',
    serving: 'E 14 St and Avenue A (SW corner) to Canarsie-bound platform',
    ADA: 'Y',
    isactive: 'Y',
    nonNYCT: 'N',
    elevatorsgtfsstopid: 'L06',
    stationcomplexid: '119',
  };

  test('parses a well-formed catalog record', () => {
    const out = parseEquipmentRecord(sample);
    expect(out).not.toBeNull();
    expect(out!.equipment_id).toBe('EL293');
    expect(out!.type).toBe('elevator');
    expect(out!.station_complex_id).toBe('119');
    expect(out!.gtfs_stop_id).toBe('L06');
    expect(out!.ada_pathway).toBe(true);
    expect(out!.is_active).toBe(true);
  });

  test('falls back to station name when stationcomplexid is missing', () => {
    const out = parseEquipmentRecord({ ...sample, stationcomplexid: '' });
    expect(out!.station_complex_id).toBe('1 Av');
  });

  test('marks isactive=N as inactive', () => {
    expect(parseEquipmentRecord({ ...sample, isactive: 'N' })!.is_active).toBe(false);
  });

  test('parseEquipmentFeed skips malformed entries', () => {
    const list = parseEquipmentFeed([sample, {}, null, { ...sample, equipmentno: '' }]);
    expect(list).toHaveLength(1);
  });
});

describe('isActiveOutage', () => {
  test('false when since is null', () => {
    expect(isActiveOutage({ reason: null, est_return: null, since: null }, 1000)).toBe(false);
  });

  test('false when est_return has already passed', () => {
    expect(
      isActiveOutage({ reason: null, est_return: 500, since: 100 }, 1000),
    ).toBe(false);
  });

  test('true when since is set and est_return is in the future', () => {
    expect(
      isActiveOutage({ reason: null, est_return: 5000, since: 100 }, 1000),
    ).toBe(true);
  });

  test('true when since is set and est_return is null (open-ended outage)', () => {
    expect(
      isActiveOutage({ reason: null, est_return: null, since: 100 }, 1000),
    ).toBe(true);
  });
});
