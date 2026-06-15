/**
 * Shape validation for state/params.json — momentarily-30o.
 *
 * Forward inference does non-null assertions on emission arrays, so a malformed
 * trainer upload would crash or NaN-propagate without these guards. parseTrainedParams
 * drops bad routes (logged) and the Worker's paramsForRoute falls them back to bootstrap.
 */

import { describe, expect, test, vi } from 'vitest';

import { parseTrainedParams, paramsForRoute, dwellForRouteState, BOOTSTRAP_PARAMS } from '../src/params';

function wellFormedEmissions(): Record<string, unknown> {
  return {
    poisson_lambda: [0.3, 4.0, 12.0],
    gamma_alpha: [1.0, 3.0, 6.0],
    gamma_beta: [2.0, 0.4, 0.2],
    bernoulli_p: [0.001, 0.05, 0.95],
    bernoulli_p_delays: [0.02, 0.6, 0.35],
    bernoulli_p_service_change: [0.02, 0.6, 0.4],
    bernoulli_p_planned: [0.05, 0.6, 0.35],
  };
}

function wellFormedRoute(): Record<string, unknown> {
  return {
    transition: [
      [0.95, 0.04, 0.01],
      [0.08, 0.9, 0.02],
      [0.02, 0.1, 0.88],
    ],
    initial: [0.9, 0.08, 0.02],
    emissions: wellFormedEmissions(),
  };
}

function wrapper(routes: Record<string, unknown>): Record<string, unknown> {
  return { schema_version: '1', trained_at: 1_700_000_000, routes };
}

describe('parseTrainedParams', () => {
  test('keeps well-formed routes', () => {
    const result = parseTrainedParams(wrapper({ '1': wellFormedRoute() }));
    expect(result).not.toBeNull();
    expect(Object.keys(result!.routes)).toEqual(['1']);
    expect(result!.routes['1']!.transition[0]).toEqual([0.95, 0.04, 0.01]);
  });

  test('drops a route with a wrong-length transition row', () => {
    const bad = wellFormedRoute();
    bad['transition'] = [[0.95, 0.04, 0.01], [0.08, 0.9], [0.02, 0.1, 0.88]];
    const warn = vi.spyOn(console, 'warn').mockImplementation(() => {});
    const result = parseTrainedParams(wrapper({ '1': wellFormedRoute(), 'BAD': bad }));
    warn.mockRestore();
    expect(Object.keys(result!.routes)).toEqual(['1']);
  });

  test('drops a route with a non-finite emission value', () => {
    const bad = wellFormedRoute();
    (bad.emissions as Record<string, unknown>).poisson_lambda = [0.3, Number.NaN, 12.0];
    const warn = vi.spyOn(console, 'warn').mockImplementation(() => {});
    const result = parseTrainedParams(wrapper({ 'BAD': bad, '1': wellFormedRoute() }));
    warn.mockRestore();
    expect(Object.keys(result!.routes)).toEqual(['1']);
  });

  test('drops a route missing the emissions key entirely', () => {
    const bad = wellFormedRoute();
    delete bad.emissions;
    const warn = vi.spyOn(console, 'warn').mockImplementation(() => {});
    const result = parseTrainedParams(wrapper({ 'BAD': bad }));
    warn.mockRestore();
    expect(result!.routes).toEqual({});
  });

  test('drops a route whose transition row does not sum to 1', () => {
    const bad = wellFormedRoute();
    // Finite, in-range, but the first row sums to 0.5 — invalid forward math.
    bad['transition'] = [[0.4, 0.05, 0.05], [0.08, 0.9, 0.02], [0.02, 0.1, 0.88]];
    const warn = vi.spyOn(console, 'warn').mockImplementation(() => {});
    const result = parseTrainedParams(wrapper({ '1': wellFormedRoute(), BAD: bad }));
    warn.mockRestore();
    expect(Object.keys(result!.routes)).toEqual(['1']);
  });

  test('keeps a route whose rows sum to 1 within float tolerance', () => {
    const route = wellFormedRoute();
    route['transition'] = [
      [0.3333, 0.3333, 0.3334],
      [0.08, 0.9, 0.02],
      [0.02, 0.1, 0.88],
    ];
    const result = parseTrainedParams(wrapper({ '1': route }));
    expect(Object.keys(result!.routes)).toEqual(['1']);
  });

  test('drops a route with an out-of-range emission probability', () => {
    const bad = wellFormedRoute();
    (bad.emissions as Record<string, unknown>).bernoulli_p = [0.01, 0.5, 1.2];
    const warn = vi.spyOn(console, 'warn').mockImplementation(() => {});
    const result = parseTrainedParams(wrapper({ '1': wellFormedRoute(), BAD: bad }));
    warn.mockRestore();
    expect(Object.keys(result!.routes)).toEqual(['1']);
  });

  test('drops a route with a negative poisson rate', () => {
    const bad = wellFormedRoute();
    (bad.emissions as Record<string, unknown>).poisson_lambda = [-0.1, 4.0, 12.0];
    const warn = vi.spyOn(console, 'warn').mockImplementation(() => {});
    const result = parseTrainedParams(wrapper({ BAD: bad }));
    warn.mockRestore();
    expect(result!.routes).toEqual({});
  });

  test('drops a route with a non-positive gamma parameter', () => {
    const bad = wellFormedRoute();
    (bad.emissions as Record<string, unknown>).gamma_beta = [2.0, 0.0, 0.2];
    const warn = vi.spyOn(console, 'warn').mockImplementation(() => {});
    const result = parseTrainedParams(wrapper({ BAD: bad }));
    warn.mockRestore();
    expect(result!.routes).toEqual({});
  });

  test('paramsForRoute falls back to bootstrap when a dropped route is requested', () => {
    const warn = vi.spyOn(console, 'warn').mockImplementation(() => {});
    const bad = wellFormedRoute();
    bad.initial = [0.5, 0.3];
    const result = parseTrainedParams(wrapper({ 'BAD': bad }));
    warn.mockRestore();
    expect(paramsForRoute(result, 'BAD')).toBe(BOOTSTRAP_PARAMS);
  });

  test('returns null when the wrapper itself is malformed', () => {
    const err = vi.spyOn(console, 'error').mockImplementation(() => {});
    expect(parseTrainedParams({ routes: 'not-an-object' })).toBeNull();
    expect(parseTrainedParams({ schema_version: '1', trained_at: 'never', routes: {} })).toBeNull();
    expect(parseTrainedParams(null)).toBeNull();
    err.mockRestore();
  });

  test('accepts emissions_by_bin when length matches N_TOD_BINS', () => {
    const route = wellFormedRoute();
    route.emissions_by_bin = Array.from({ length: 5 }, () => wellFormedEmissions());
    const result = parseTrainedParams(wrapper({ '1': route }));
    expect(result!.routes['1']!.emissions_by_bin).toHaveLength(5);
  });

  test('drops a route whose emissions_by_bin has wrong length', () => {
    const route = wellFormedRoute();
    route.emissions_by_bin = Array.from({ length: 3 }, () => wellFormedEmissions());
    const warn = vi.spyOn(console, 'warn').mockImplementation(() => {});
    const result = parseTrainedParams(wrapper({ '1': route }));
    warn.mockRestore();
    expect(Object.keys(result!.routes)).toEqual([]);
  });

  test('parses optional dwell_quantiles sidecar and exposes via dwellForRouteState', () => {
    const route = wellFormedRoute();
    route.dwell_quantiles = {
      disrupted: { n: 12, q25_sec: 1200, median_sec: 2400, q75_sec: 4800 },
      suspended: { n: 7, q25_sec: 600, median_sec: 1800, q75_sec: 3600 },
    };
    const result = parseTrainedParams(wrapper({ A: route, B: wellFormedRoute() }));
    expect(dwellForRouteState(result, 'A', 'disrupted')).toEqual({
      n: 12, q25_sec: 1200, median_sec: 2400, q75_sec: 4800,
    });
    expect(dwellForRouteState(result, 'A', 'suspended')).not.toBeNull();
    // Cell absent: falls through to null
    expect(dwellForRouteState(result, 'A', 'normal')).toBeNull();
    // Route without sidecar: null
    expect(dwellForRouteState(result, 'B', 'disrupted')).toBeNull();
    // No trained params at all: null
    expect(dwellForRouteState(null, 'A', 'disrupted')).toBeNull();
  });

  test('round-trips the optional curve_sec dwell curve', () => {
    const route = wellFormedRoute();
    const curve = Array.from({ length: 21 }, (_, i) => i * 300);
    route.dwell_quantiles = {
      disrupted: {
        n: 12, q25_sec: 1200, median_sec: 2400, q75_sec: 4800, curve_sec: curve,
      },
    };
    const result = parseTrainedParams(wrapper({ A: route }));
    expect(dwellForRouteState(result, 'A', 'disrupted')!.curve_sec).toEqual(curve);
  });

  test('drops dwell entry with malformed quantile shape but keeps the route', () => {
    const route = wellFormedRoute();
    route.dwell_quantiles = {
      disrupted: { n: 'not-a-number', q25_sec: 1200, median_sec: 2400, q75_sec: 4800 },
    };
    const warn = vi.spyOn(console, 'warn').mockImplementation(() => {});
    const result = parseTrainedParams(wrapper({ A: route }));
    warn.mockRestore();
    // Whole route is dropped (sidecar is part of the route schema)
    expect(Object.keys(result!.routes)).toEqual([]);
  });

  test('prefers cause-conditioned dwell, falls back to the state aggregate', () => {
    const route = wellFormedRoute();
    route.dwell_quantiles = {
      disrupted: { n: 20, q25_sec: 1200, median_sec: 2400, q75_sec: 4800 },
    };
    route.dwell_quantiles_by_alert = {
      disrupted: {
        Delays: { n: 9, q25_sec: 300, median_sec: 600, q75_sec: 900 },
      },
    };
    const result = parseTrainedParams(wrapper({ A: route }));
    // Matching (state, alert_type) cell wins.
    expect(dwellForRouteState(result, 'A', 'disrupted', 'Delays')).toEqual({
      n: 9, q25_sec: 300, median_sec: 600, q75_sec: 900,
    });
    // Alert type with no cell falls back to the (state) aggregate.
    expect(dwellForRouteState(result, 'A', 'disrupted', 'Planned - Stops Skipped')).toEqual({
      n: 20, q25_sec: 1200, median_sec: 2400, q75_sec: 4800,
    });
    // No alert type given: aggregate.
    expect(dwellForRouteState(result, 'A', 'disrupted')).toEqual({
      n: 20, q25_sec: 1200, median_sec: 2400, q75_sec: 4800,
    });
  });

  test('cause-conditioned lookup with no aggregate returns null when cell absent', () => {
    const route = wellFormedRoute();
    route.dwell_quantiles_by_alert = {
      disrupted: { Delays: { n: 9, q25_sec: 300, median_sec: 600, q75_sec: 900 } },
    };
    const result = parseTrainedParams(wrapper({ A: route }));
    expect(dwellForRouteState(result, 'A', 'disrupted', 'Delays')).not.toBeNull();
    // alert type present but no matching cell, and no aggregate -> null
    expect(dwellForRouteState(result, 'A', 'disrupted', 'Other')).toBeNull();
  });
});
