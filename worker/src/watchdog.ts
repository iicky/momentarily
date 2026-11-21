/**
 * Backup trigger for the minute pipeline: a Durable Object alarm watchdog
 * (docs/model-observations.md §4.1, "As built (backup trigger)").
 *
 * The Cron Trigger stays the primary trigger. Cloudflare's cron path has
 * skipped minutes and started runs hours late, and a skipped minute is lost
 * for good, so one SQLite-backed Durable Object, `MinuteClock`, keeps an alarm
 * armed for 45 s past every minute and runs any minute the cron did not run
 * to its end.
 *
 * Both triggers claim a minute ("slot", `floor(ms / 60 s)`) in this object
 * before running it, and the first claim wins:
 *
 *   - the cron claims its slot first (`claimForCron`). A lost claim returns
 *     at once; a claim that errors or times out (3 s) runs the minute anyway,
 *     since a duplicate is harmless and the claim must never cost a minute.
 *     The claim also re-arms an alarm that is missing or more than a minute
 *     overdue (local workerd keeps a stored alarm across a restart without
 *     firing it). When the minute has run to its end, the cron marks the slot
 *     done (`reportCronDone`);
 *   - at :45 the alarm re-arms itself for the next minute before anything
 *     else, then claims the slot whose deadline just passed: unclaimed, or
 *     claimed by a cron that never marked it done (killed or stuck after its
 *     claim). It runs the whole minute (`runMinute`, the cron's own body) with
 *     `scheduledTime` = slot + 19 s, the cron's dispatch offset, so the tick's
 *     `scheduled_at`, the 5-minute gate and coverage's slot grid read as for a
 *     cron run. It never takes over an alarm claim and catches every error,
 *     so Cloudflare's alarm retries never run a minute twice.
 *
 * A claim is a `get` then a `put` with no other I/O between them; the input
 * gate makes the pair atomic against every other request to the object.
 * Claims older than a day are pruned by the alarm.
 */

import { z } from 'zod';

import type { Env } from './index';
import { runMinute } from './index';
import type { Trigger } from './recorder';

const ClaimAnswer = z.object({ won: z.boolean() });

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
/** Claims kept, in minutes: one day, also a per-minute log of who ran it. */
const KEEP_MINUTES = 24 * 60;

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

/** One slot's claim: who holds it, and whether its run has ended. */
export interface Claim {
  trigger: Trigger;
  done: boolean;
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
    if (request.method !== 'POST' || !Number.isSafeInteger(slot)) {
      return new Response('Bad Request', { status: 400 });
    }
    if (url.pathname === '/claim') {
      const won = await this.claim(slot, 'cron');
      await this.rearmIfLost();
      return Response.json({ won });
    }
    if (url.pathname === '/done') {
      await this.markDone(slot, 'cron');
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
    // The current slot only, never a backlog: the feeds show the present.
    const slot = alarmSlot(now);
    let won = false;
    try {
      won = await this.claim(slot, 'alarm');
    } catch (err) {
      console.error(`watchdog: claim of minute ${slot} failed:`, err);
    }
    if (won) {
      console.log(`watchdog: minute ${slot} not run by the cron; the alarm runs it`);
      try {
        await runMinute(slot * MINUTE_MS + CRON_OFFSET_MS, this.env, this.state, 'alarm');
      } catch (err) {
        console.error(`watchdog: alarm run of minute ${slot} failed (not retried):`, err);
      }
      try {
        await this.markDone(slot, 'alarm');
      } catch (err) {
        console.error(`watchdog: done mark of minute ${slot} failed:`, err);
      }
    }
    try {
      await this.prune(slot);
    } catch (err) {
      console.error('watchdog: prune failed:', err);
    }
  }

  /**
   * Take `slot` for `trigger`. The first claim wins; the alarm also takes a
   * cron claim that was never marked done.
   */
  private async claim(slot: number, trigger: Trigger): Promise<boolean> {
    const key = slotKey(slot);
    const held = await this.state.storage.get<Claim>(key);
    if (held !== undefined && (held.done || trigger === 'cron' || held.trigger === 'alarm')) {
      return false;
    }
    const claim: Claim = { trigger, done: false };
    await this.state.storage.put(key, claim);
    return true;
  }

  /**
   * Mark `slot`'s run ended. A cron whose claim failed open still marks it,
   * so the alarm does not run that minute a second time.
   */
  private async markDone(slot: number, trigger: Trigger): Promise<void> {
    const key = slotKey(slot);
    const held = await this.state.storage.get<Claim>(key);
    const claim: Claim = { trigger: held?.trigger ?? trigger, done: true };
    await this.state.storage.put(key, claim);
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

/** POST `path` to the watchdog, failing after `CALL_TIMEOUT_MS`. */
async function callClock(env: Env, path: string): Promise<Response> {
  const ns = env.MINUTE_CLOCK;
  if (ns === undefined) throw new Error('no MINUTE_CLOCK binding');
  const stub = ns.get(ns.idFromName(CLOCK_NAME), { locationHint: 'enam' });
  // The platform's handle type: workerd types say number, Node's say Timeout.
  let timer: ReturnType<typeof setTimeout> | undefined;
  const timeout = new Promise<never>((_, reject) => {
    timer = setTimeout(() => reject(new Error(`no answer in ${CALL_TIMEOUT_MS} ms`)), CALL_TIMEOUT_MS);
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
 * Claim `slot` for the cron. False only when another run holds it; any
 * failure to reach the watchdog is true (run the minute anyway).
 */
export async function claimForCron(env: Env, slot: number): Promise<boolean> {
  try {
    const { won } = ClaimAnswer.parse(await (await callClock(env, `/claim?slot=${slot}`)).json());
    if (!won) console.log(`watchdog: minute ${slot} already claimed; the cron skips it`);
    return won;
  } catch (err) {
    console.warn(`watchdog: claim of minute ${slot} failed; the cron runs it anyway:`, err);
    return true;
  }
}

/** Mark the cron's run of `slot` ended. A failure means a possible duplicate. */
export async function reportCronDone(env: Env, slot: number): Promise<void> {
  try {
    await callClock(env, `/done?slot=${slot}`);
  } catch (err) {
    console.warn(`watchdog: done mark of minute ${slot} failed; the alarm may run it again:`, err);
  }
}
