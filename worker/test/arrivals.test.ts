/**
 * Behaviour of deriveArrivals: the observable rules of the per-stop arrivals
 * surface, each pinned with a hand-computed expectation. Cross-language parity
 * with the Python mirror lives in arrivals_parity.test.ts.
 */

import { describe, expect, test } from 'vitest';

import { deriveArrivals } from '../src/arrivals';
import type { StopTimeLite, TripLite } from '../src/gtfsrt';

const NOW = 1_000_000;

function st(over: Partial<StopTimeLite>): StopTimeLite {
  return { stopId: 'Q05S', arrival: null, departure: null, scheduleRelationship: 0, ...over };
}

function trip(over: Partial<TripLite>): TripLite {
  return {
    routeId: 'Q',
    tripId: 't',
    isAssigned: true,
    direction: 3, // S, matches the default Q05S suffix
    stopCount: 0,
    stopTimes: [],
    ...over,
  };
}

describe('deriveArrivals', () => {
  test('sorts a stop ascending by time and caps at 6', () => {
    const offsets = [600, 120, 900, 300, 1500, 60, 1200, 1800];
    const trips = offsets.map((off, i) =>
      trip({ tripId: `q${i}`, stopTimes: [st({ arrival: NOW + off })] }),
    );
    const out = deriveArrivals(trips, NOW);
    expect(out['Q05S']!.map((a) => a.seconds_away)).toEqual([60, 120, 300, 600, 900, 1200]);
    expect(out['Q05S']!.every((a) => a.eta_epoch === NOW + a.seconds_away)).toBe(true);
  });

  test('falls back to departure when arrival is absent, and arrival wins when both set', () => {
    const trips = [
      trip({ tripId: 'dep', direction: 1, stopTimes: [st({ stopId: 'F10N', departure: NOW + 500 })] }),
      trip({
        tripId: 'arr',
        direction: 1,
        stopTimes: [st({ stopId: 'F10N', arrival: NOW + 300, departure: NOW + 999 })],
      }),
    ];
    const out = deriveArrivals(trips, NOW);
    expect(out['F10N']!.map((a) => [a.trip_id, a.eta_epoch])).toEqual([
      ['arr', NOW + 300],
      ['dep', NOW + 500],
    ]);
  });

  test('drops SKIPPED rows but keeps scheduled ones at the same stop', () => {
    const trips = [
      trip({ tripId: 'skip', stopTimes: [st({ stopId: 'A15S', arrival: NOW + 250, scheduleRelationship: 1 })] }),
      trip({ tripId: 'ok', stopTimes: [st({ stopId: 'A15S', arrival: NOW + 700 })] }),
    ];
    const out = deriveArrivals(trips, NOW);
    expect(out['A15S']!.map((a) => a.trip_id)).toEqual(['ok']);
  });

  test('drops past and beyond-horizon rows entirely', () => {
    const trips = [
      trip({ tripId: 'past', stopTimes: [st({ stopId: 'B20S', arrival: NOW - 1 })] }),
      trip({ tripId: 'far', stopTimes: [st({ stopId: 'C30S', arrival: NOW + 3601 })] }),
      trip({ tripId: 'edge', stopTimes: [st({ stopId: 'C30S', arrival: NOW + 3600 })] }),
    ];
    const out = deriveArrivals(trips, NOW);
    expect(out['B20S']).toBeUndefined();
    // now and now+HORIZON are both inclusive; only the +1 over is dropped.
    expect(out['C30S']!.map((a) => a.seconds_away)).toEqual([3600]);
  });

  test('keeps one entry per trip per stop — the first occurrence wins', () => {
    const trips = [
      trip({
        tripId: 'dup',
        direction: 1,
        stopTimes: [st({ stopId: 'R10N', arrival: NOW + 400 }), st({ stopId: 'R10N', arrival: NOW + 200 })],
      }),
    ];
    const out = deriveArrivals(trips, NOW);
    expect(out['R10N']!.map((a) => a.eta_epoch)).toEqual([NOW + 400]);
  });

  test('folds the express variant to its base route', () => {
    const out = deriveArrivals(
      [trip({ routeId: '6X', tripId: '6x', direction: 1, stopTimes: [st({ stopId: 'D40N', arrival: NOW + 350 })] })],
      NOW,
    );
    expect(out['D40N']![0]!.route).toBe('6');
  });

  test('emits a null trip_id when the feed omitted it, and skips unkeyable empty stop ids', () => {
    const out = deriveArrivals(
      [
        trip({
          tripId: '',
          direction: 1,
          stopTimes: [st({ stopId: 'E50N', arrival: NOW + 150 }), st({ stopId: '', arrival: NOW + 150 })],
        }),
      ],
      NOW,
    );
    expect(out['E50N']).toEqual([{ route: 'Q', eta_epoch: NOW + 150, seconds_away: 150, trip_id: null }]);
    expect(out['']).toBeUndefined();
  });

  test('two anonymous trips at one stop are two arrivals: dedupe needs an id', () => {
    const out = deriveArrivals(
      [
        trip({ tripId: '', direction: 1, stopTimes: [st({ stopId: 'E50N', arrival: NOW + 450 })] }),
        trip({ tripId: '', direction: 1, stopTimes: [st({ stopId: 'E50N', arrival: NOW + 150 })] }),
      ],
      NOW,
    );
    expect(out['E50N']!.map((a) => a.eta_epoch)).toEqual([NOW + 150, NOW + 450]);
    expect(out['E50N']!.every((a) => a.trip_id === null)).toBe(true);
  });
});
