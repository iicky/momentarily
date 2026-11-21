/**
 * Backup trigger for the minute pipeline: a Durable Object alarm watchdog
 * (docs/model-observations.md §4.1, "As built (backup trigger)").
 *
 * The Cron Trigger stays the primary trigger. Cloudflare's cron path has
 * skipped minutes and started runs hours late, and a skipped minute is lost
 * for good, so one SQLite-backed Durable Object, `MinuteClock`, keeps an alarm
 * armed for 45 s past every minute and runs any minute the cron did not run
 * and is no longer running.
 *
 * Both triggers claim a minute ("slot", `floor(ms / 60 s)`) in this object
 * before running it, and the first claim wins. A claim names the run that
 * holds it (a random run id, the fencing token):
 *
 *   - the cron claims its slot first (`claimForCron`). A lost claim returns
 *     at once; a claim that errors or times out (3 s) runs the minute anyway,
 *     since the claim must never cost a minute. The claim also re-arms an
 *     alarm that is missing or more than a minute overdue (local workerd keeps
 *     a stored alarm across a restart without firing it);
 *   - a cron claim is a lease of `LEASE_MS`. The cron renews it before every
 *     write stage of the minute and every `HEARTBEAT_MS` from a timer
 *     (`CronLease`), and marks the slot done when the minute has run to its
 *     end (`reportCronDone`). A renewal that finds the slot held by another
 *     run answers "fenced", and the cron stops writing for that minute;
 *   - at :45 the alarm re-arms itself for the next minute before anything
 *     else, then looks at the last `BACKLOG_SLOTS` + 1 slots, oldest first. It
 *     takes a slot that nobody claimed (the slot whose deadline just passed,
 *     or one whose deadline passed while the previous alarm was still running
 *     a minute), or a cron claim that was never marked done and whose lease
 *     has expired (a cron that died or stopped renewing). A live cron, however
 *     slow, keeps renewing and is never taken over. A takeover replaces the
 *     run id, which fences the old run. The alarm runs the whole minute
 *     (`runMinute`, the cron's own body) with `scheduledTime` = slot + 19 s,
 *     the cron's dispatch offset, so the tick's `scheduled_at`, the 5-minute
 *     gate and coverage's slot grid read as for a cron run. It never takes
 *     over an alarm claim and catches every error, so Cloudflare's alarm
 *     retries never run a minute twice.
 *
 * The fence is a check, not a lock: a write already past its check can still
 * land after a takeover. Every object two runs of one minute can both write is
 * therefore also protected at the write itself (a conditional put, or a key
 * per run); §4.1 lists them.
 *
 * Each claim, renewal and done mark is a `get` then a `put` with no other I/O
 * between them; the input gate makes the pair atomic against every other
 * request to the object. Claims older than a day are pruned by the alarm.
 */

import { z } from 'zod';

import type { Env } from './index';
import { runMinute } from './index';
import type { Trigger } from './recorder';

const ClaimAnswer = z.object({ won: z.boolean() });
const RenewAnswer = z.object({ held: z.boolean() });

/** The one watchdog object's name. */
export const CLOCK_NAME = 'minute';

export const MINUTE_MS = 60_000;
/** Offset into a minute at which the alarm checks the minute's claim. */
export const DEADLINE_MS = 45_000;
/**
 * The cron's dispatch offset: its `event.scheduledTime` was 19 s past the
 * minute in 18,317 of 18,317 recorded ticks. Alarm runs use the same offset.
 */
export const CRON_OFFSET_MS = 19_000;
/** How long the cron waits on the watchdog before it runs the minute anyway. */
export const CALL_TIMEOUT_MS = 3_000;
/** How often a running cron renews its lease from a timer. */
export const HEARTBEAT_MS = 10_000;
/**
 * How long a cron claim stays held after its last renewal. A live cron renews
 * at least every 43 s: the 10 s heartbeat, plus a renewal call's 3 s budget,
 * plus the longest stretch of synchronous work that can hold the timer back,
 * which the Worker's 30 s CPU limit bounds. A cron that stops renewing is
 * taken over at the first deadline after its lease ends: 60-120 s after it
 * stopped.
 */
export const LEASE_MS = 60_000;
/**
 * How many slots before the current one the alarm still looks at. Five
 * minutes is longer than the longest recorded run (237 s), so a cron that dies
 * late in a long run, or a minute missed while an alarm run was still going,
 * is still inside the window when the alarm next looks.
 */
export const BACKLOG_SLOTS = 5;
/** Claims kept, in minutes: one day, also a per-minute log of who ran it. */
const KEEP_MINUTES = 24 * 60;
/** Storage key of the slots whose deadline passed during the last alarm run. */
const MISSED_KEY = 'missed';
/** A run id: what `crypto.randomUUID` returns, or a short test name. */
const RUN_ID = /^[A-Za-z0-9-]{1,64}$/;

/** The slot (minutes since the epoch) of an instant in epoch milliseconds. */
export function minuteSlot(ms: number): number {
  return Math.floor(ms / MINUTE_MS);
}

/** The slot whose deadline is the latest at or before `now`. */
export function alarmSlot(now: number): number {
  return minuteSlot(now - DEADLINE_MS);
}

/** The first deadline strictly after `now`. */
export function nextDeadline(now: number): number {
  return (alarmSlot(now) + 1) * MINUTE_MS + DEADLINE_MS;
}

/** One slot's claim: who holds it, whether its run has ended, and its lease. */
export interface Claim {
  trigger: Trigger;
  done: boolean;
  /** The run holding the slot. Absent on a claim stored before leases. */
  run?: string;
  /**
   * Epoch ms until which a cron claim is held. Absent on an alarm claim
   * (never taken over) and on a claim stored before leases (expired).
   */
  leaseUntil?: number;
}

/** The slots whose deadline passed while an alarm run was still going. */
interface Missed {
  from: number;
  to: number;
}

export function slotKey(slot: number): string {
  return `slot:${String(slot).padStart(12, '0')}`;
}

export class MinuteClock implements DurableObject {
  constructor(
    protected readonly state: DurableObjectState,
    protected readonly env: Env,
  ) {}

  /** The object's clock; a seam for tests. */
  protected now(): number {
    return Date.now();
  }

  async fetch(request: Request): Promise<Response> {
    const url = new URL(request.url);
    const slot = Number(url.searchParams.get('slot'));
    const run = url.searchParams.get('run') ?? '';
    if (request.method !== 'POST' || !Number.isSafeInteger(slot) || !RUN_ID.test(run)) {
      return new Response('Bad Request', { status: 400 });
    }
    if (url.pathname === '/claim') {
      const won = await this.claimForCron(slot, run);
      await this.rearmIfLost();
      return Response.json({ won });
    }
    if (url.pathname === '/renew') {
      return Response.json({ held: await this.renew(slot, run) });
    }
    if (url.pathname === '/done') {
      await this.markCronDone(slot, run);
      return Response.json({ done: true });
    }
    return new Response('Not Found', { status: 404 });
  }

  async alarm(): Promise<void> {
    const now = this.now();
    try {
      // First, so nothing below can end the loop.
      await this.state.storage.setAlarm(nextDeadline(now));
    } catch (err) {
      console.error('watchdog: re-arm failed; the next cron claim re-arms:', err);
    }
    const current = alarmSlot(now);
    let missed: Missed | undefined;
    try {
      missed = await this.state.storage.get<Missed>(MISSED_KEY);
    } catch (err) {
      console.error('watchdog: read of missed slots failed:', err);
    }
    // Oldest first, so a minute never runs after a later one it skipped.
    for (let slot = current - BACKLOG_SLOTS; slot <= current; slot += 1) {
      // An unclaimed slot is run only when its deadline is this wake's, or
      // passed while the previous wake was busy running a minute. Not one
      // that passed while the object was unreachable: a cron whose claim
      // failed open may have run it, unrecorded.
      const due = slot === current || (missed !== undefined && slot >= missed.from && slot <= missed.to);
      let run: string | null = null;
      try {
        run = await this.claimForAlarm(slot, due);
      } catch (err) {
        console.error(`watchdog: claim of minute ${slot} failed:`, err);
      }
      if (run === null) continue;
      try {
        await runMinute(slot * MINUTE_MS + CRON_OFFSET_MS, this.env, this.state, 'alarm', ALWAYS_HELD);
      } catch (err) {
        console.error(`watchdog: alarm run of minute ${slot} failed (not retried):`, err);
      }
      try {
        await this.markAlarmDone(slot, run);
      } catch (err) {
        console.error(`watchdog: done mark of minute ${slot} failed:`, err);
      }
    }
    try {
      const next: Missed = { from: current + 1, to: alarmSlot(this.now()) };
      await this.state.storage.put(MISSED_KEY, next);
    } catch (err) {
      console.error('watchdog: write of missed slots failed:', err);
    }
    try {
      await this.prune(current);
    } catch (err) {
      console.error('watchdog: prune failed:', err);
    }
  }

  /** Take `slot` for the cron run `run`, if nobody holds it. */
  private async claimForCron(slot: number, run: string): Promise<boolean> {
    const key = slotKey(slot);
    if ((await this.state.storage.get<Claim>(key)) !== undefined) return false;
    const claim: Claim = { trigger: 'cron', done: false, run, leaseUntil: this.now() + LEASE_MS };
    await this.state.storage.put(key, claim);
    return true;
  }

  /**
   * Extend `run`'s lease on `slot`. False when another run holds the slot:
   * the alarm took it over, and `run` must stop writing. A slot nobody holds
   * is claimed for `run`, the case of a claim whose answer was lost.
   */
  private async renew(slot: number, run: string): Promise<boolean> {
    const key = slotKey(slot);
    const held = await this.state.storage.get<Claim>(key);
    if (held !== undefined && held.run !== run) return false;
    if (held?.done === true) return true;
    const claim: Claim = { trigger: 'cron', done: false, run, leaseUntil: this.now() + LEASE_MS };
    await this.state.storage.put(key, claim);
    return true;
  }

  /**
   * Mark the cron run `run` of `slot` ended. A cron whose claim failed open
   * marks a slot nobody holds, so the alarm does not run that minute a second
   * time; a fenced run marks nothing.
   */
  private async markCronDone(slot: number, run: string): Promise<void> {
    const key = slotKey(slot);
    const held = await this.state.storage.get<Claim>(key);
    if (held !== undefined && held.run !== run) return;
    const claim: Claim = { trigger: 'cron', done: true, run };
    await this.state.storage.put(key, claim);
  }

  /**
   * Take `slot` for an alarm run, returning its run id, or null. Wins a slot
   * nobody holds when `due`, and a not-done cron claim whose lease expired.
   */
  private async claimForAlarm(slot: number, due: boolean): Promise<string | null> {
    const key = slotKey(slot);
    const held = await this.state.storage.get<Claim>(key);
    if (held === undefined) {
      if (!due) return null;
      console.log(`watchdog: minute ${slot} not claimed by the cron; the alarm runs it`);
    } else {
      if (held.done || held.trigger === 'alarm') return null;
      if ((held.leaseUntil ?? 0) > this.now()) return null; // a live cron
      console.log(`watchdog: the cron's lease on minute ${slot} expired; the alarm takes it over`);
    }
    const run = crypto.randomUUID();
    const claim: Claim = { trigger: 'alarm', done: false, run };
    await this.state.storage.put(key, claim);
    return run;
  }

  private async markAlarmDone(slot: number, run: string): Promise<void> {
    const key = slotKey(slot);
    const held = await this.state.storage.get<Claim>(key);
    if (held?.run !== run) return;
    await this.state.storage.put(key, { ...held, done: true });
  }

  private async rearmIfLost(): Promise<void> {
    const now = this.now();
    const at = await this.state.storage.getAlarm();
    if (at === null || at < now - MINUTE_MS) {
      await this.state.storage.setAlarm(nextDeadline(now));
    }
  }

  private async prune(slot: number): Promise<void> {
    const old = await this.state.storage.list({
      prefix: 'slot:',
      end: slotKey(slot - KEEP_MINUTES),
      limit: 128,
    });
    if (old.size > 0) await this.state.storage.delete([...old.keys()]);
  }
}

/**
 * Whether a run still owns its minute. `runMinute` asks before each write
 * stage and stops writing at the first no.
 */
export interface Lease {
  hold(): Promise<boolean>;
}

/** An alarm run's lease: an alarm claim is never taken over. */
export const ALWAYS_HELD: Lease = { hold: async () => true };

/**
 * A cron run's lease on its slot. `hold` renews it and reports whether the
 * run still owns the slot; a timer renews it every `heartbeatMs` while the
 * run is going, so a long stage does not let it lapse. A renewal that fails
 * counts as held (the cron fails open, as its claim does). After one that got
 * no answer within the call budget, `hold` skips the call for one heartbeat
 * period, so an unresponsive object costs one call budget per period rather
 * than one per stage; an error answer costs nothing, so `hold` keeps asking.
 */
export class CronLease implements Lease {
  private fenced = false;
  private silentAt: number | null = null;
  private timer: ReturnType<typeof setInterval> | undefined;

  constructor(
    private readonly env: Env,
    private readonly slot: number,
    private readonly run: string,
    private readonly heartbeatMs: number = HEARTBEAT_MS,
  ) {}

  start(): void {
    this.timer = setInterval(() => void this.renew(), this.heartbeatMs);
  }

  stop(): void {
    clearInterval(this.timer);
  }

  async hold(): Promise<boolean> {
    const backingOff = this.silentAt !== null && Date.now() - this.silentAt < this.heartbeatMs;
    if (!this.fenced && !backingOff) await this.renew();
    return !this.fenced;
  }

  private async renew(): Promise<void> {
    if (this.fenced) return;
    try {
      const res = await callClock(this.env, `/renew?slot=${this.slot}&run=${this.run}`);
      const { held } = RenewAnswer.parse(await res.json());
      this.silentAt = null;
      if (!held) {
        this.fenced = true;
        console.warn(`watchdog: minute ${this.slot} was taken over by the alarm; this cron run stops writing`);
      }
    } catch (err) {
      if (err instanceof NoAnswer) this.silentAt = Date.now();
      console.warn(`watchdog: lease renewal of minute ${this.slot} failed; the cron runs on:`, err);
    }
  }
}

/** The watchdog did not answer within `CALL_TIMEOUT_MS`. */
class NoAnswer extends Error {}

/** POST `path` to the watchdog, failing after `CALL_TIMEOUT_MS`. */
async function callClock(env: Env, path: string): Promise<Response> {
  const ns = env.MINUTE_CLOCK;
  if (ns === undefined) throw new Error('no MINUTE_CLOCK binding');
  const stub = ns.get(ns.idFromName(CLOCK_NAME), { locationHint: 'enam' });
  // The platform's handle type: workerd types say number, Node's say Timeout.
  let timer: ReturnType<typeof setTimeout> | undefined;
  const timeout = new Promise<never>((_, reject) => {
    timer = setTimeout(() => reject(new NoAnswer(`no answer in ${CALL_TIMEOUT_MS} ms`)), CALL_TIMEOUT_MS);
  });
  try {
    const res = await Promise.race([stub.fetch(`https://minute-clock${path}`, { method: 'POST' }), timeout]);
    if (!res.ok) throw new Error(`watchdog answered ${res.status}`);
    return res;
  } finally {
    clearTimeout(timer);
  }
}

/**
 * Claim `slot` for the cron run `run`. False only when another run holds it;
 * any failure to reach the watchdog is true (run the minute anyway).
 */
export async function claimForCron(env: Env, slot: number, run: string): Promise<boolean> {
  try {
    const { won } = ClaimAnswer.parse(await (await callClock(env, `/claim?slot=${slot}&run=${run}`)).json());
    if (!won) console.log(`watchdog: minute ${slot} already claimed; the cron skips it`);
    return won;
  } catch (err) {
    console.warn(`watchdog: claim of minute ${slot} failed; the cron runs it anyway:`, err);
    return true;
  }
}

/** Mark the cron run `run` of `slot` ended. A failure leaves the lease to expire. */
export async function reportCronDone(env: Env, slot: number, run: string): Promise<void> {
  try {
    await callClock(env, `/done?slot=${slot}&run=${run}`);
  } catch (err) {
    console.warn(`watchdog: done mark of minute ${slot} failed; the alarm may run it again:`, err);
  }
}
