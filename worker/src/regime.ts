/**
 * Generic regime clock over a keyed stream of per-tick calls.
 *
 * A classifier answers "what is this cell doing right now", one tick at a time.
 * A regime is the debounced run of that answer plus the timestamp it began. The
 * same code clocks a route (key = route id) and a segment (key =
 * `route|direction|from_stop`); the abstraction is the clock, not the thing
 * being clocked.
 *
 * Debounce mirrors derive_actual_recovery in training/load_r2.py: a change
 * commits only after `debounceTicks` consecutive calls agree, and the new
 * regime is back-dated to the first tick of that run rather than the tick the
 * run completed. An abstention — the key absent from this tick's calls — resets
 * the candidate run but never ends an open regime: no reading is not a reading
 * of change.
 *
 * A cell whose first-ever call arrives mid-window opens its regime at that
 * tick. There is no prior regime to protect, so there is nothing to debounce
 * against.
 *
 * This is a hand-port of training/regime.py. The two must segment identically
 * or curves fitted offline describe regimes the Worker never enters; the pin is
 * tests/fixtures/parity_regime.json (worker/test/regime_parity.test.ts).
 */

/**
 * Consecutive agreeing calls required to commit a regime change.
 *
 * One, not two. The movement classifier is already conservative: over
 * 2026-08-04..08-11 it opened 17 route episodes in 65,685 route-ticks, ~193x
 * rarer than its nominal alpha=0.05 tail would produce, because the
 * DISRUPTED_RATIO posterior gate binds long before the significance test does.
 * A second confirming tick erased 13 of those 17 episodes and every episode
 * under 10 minutes, against a population whose own median is 5.0 min — one
 * tick. Debouncing that is deleting signal, not filtering noise. The mechanism
 * stays (callers may raise it, and the parity fixture pins it at 2); the
 * default does not use it.
 */
export const DEBOUNCE_TICKS = 1;

/**
 * A cell unobserved for longer than this drops out. A brief abstention holds
 * the regime open; an hour of blindness means the regime that resumes is not
 * knowably the one that stopped.
 */
export const MAX_IDLE_SEC = 3600;

/** One cell's debounced regime and the clock for it. */
export interface RegimeEntry<C extends string = string> {
  state: C;
  entered_at: number;
  last_seen_at: number;
  // Candidate state seen but not yet debounced, and the tick its run started.
  // A committed change back-dates entered_at to pending_since.
  pending: C | null;
  pending_since: number;
  pending_run: number;
}

/**
 * A committed regime change. Field names match TransitionRecord so the movement
 * stream and the filter stream grade through one code path.
 */
export interface RegimeChange<C extends string = string> {
  key: string;
  prev_state: C;
  new_state: C;
  entered_at: number;
  exited_at: number;
  dwell_sec: number;
}

export interface AdvanceRegimesOptions {
  debounceTicks?: number;
  maxIdleSec?: number;
}

/**
 * Drop cells unobserved for longer than `maxIdleSec` — the idle-expiry half of
 * advanceRegimes, exported so a stateful classifier (the service hysteresis
 * band) can expire a stale regime BEFORE it reads the prior state, instead of
 * letting advanceRegimes expire it only afterward and recreate it from a call
 * the stale state already biased.
 */
export function pruneIdleRegimes<C extends string>(
  prev: Record<string, RegimeEntry<C>> | null | undefined,
  observedAt: number,
  maxIdleSec: number = MAX_IDLE_SEC,
): Record<string, RegimeEntry<C>> {
  const out: Record<string, RegimeEntry<C>> = {};
  for (const [key, entry] of Object.entries(prev ?? {})) {
    if (observedAt - entry.last_seen_at <= maxIdleSec) out[key] = entry;
  }
  return out;
}

/**
 * Advance every cell's regime by one tick.
 *
 * `observed` holds this tick's raw classifier calls; a key absent from it is an
 * abstention, not a state. Returns the new entry map and the changes that
 * committed this tick, ordered by key so both languages emit the same list.
 */
export function advanceRegimes<C extends string>(
  prev: Record<string, RegimeEntry<C>> | null | undefined,
  observed: Record<string, C>,
  observedAt: number,
  options: AdvanceRegimesOptions = {},
): { entries: Record<string, RegimeEntry<C>>; changes: RegimeChange<C>[] } {
  const debounceTicks = options.debounceTicks ?? DEBOUNCE_TICKS;
  const maxIdleSec = options.maxIdleSec ?? MAX_IDLE_SEC;

  const entries: Record<string, RegimeEntry<C>> = {};
  for (const [key, entry] of Object.entries(pruneIdleRegimes(prev, observedAt, maxIdleSec))) {
    entries[key] = Object.prototype.hasOwnProperty.call(observed, key)
      ? entry
      : { ...entry, pending: null, pending_since: 0, pending_run: 0 };
  }

  const changes: RegimeChange<C>[] = [];
  for (const key of Object.keys(observed).sort()) {
    const call = observed[key]!;
    const entry = entries[key];
    if (entry === undefined) {
      entries[key] = {
        state: call,
        entered_at: observedAt,
        last_seen_at: observedAt,
        pending: null,
        pending_since: 0,
        pending_run: 0,
      };
      continue;
    }
    if (call === entry.state) {
      entries[key] = {
        ...entry,
        last_seen_at: observedAt,
        pending: null,
        pending_since: 0,
        pending_run: 0,
      };
      continue;
    }
    const sameCandidate = entry.pending === call;
    const run = sameCandidate ? entry.pending_run + 1 : 1;
    const since = sameCandidate ? entry.pending_since : observedAt;
    if (run >= debounceTicks) {
      changes.push({
        key,
        prev_state: entry.state,
        new_state: call,
        entered_at: entry.entered_at,
        exited_at: since,
        dwell_sec: since - entry.entered_at,
      });
      entries[key] = {
        state: call,
        entered_at: since,
        last_seen_at: observedAt,
        pending: null,
        pending_since: 0,
        pending_run: 0,
      };
    } else {
      entries[key] = {
        ...entry,
        last_seen_at: observedAt,
        pending: call,
        pending_since: since,
        pending_run: run,
      };
    }
  }
  return { entries, changes };
}
