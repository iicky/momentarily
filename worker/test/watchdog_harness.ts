/**
 * Test entry for watchdog.test.ts: the production Worker (its `scheduled` and
 * `fetch`) plus `HarnessClock`, the production `MinuteClock` with test hooks,
 * bundled and run in real workerd through Miniflare.
 *
 * Hooks, all under /harness/ (forwarded to the one watchdog object):
 *   offset?ms=N     the object's clock reads Date.now() + N
 *   alarm           run the alarm handler now, as the runtime would
 *   state           its stored alarm and claims
 *   delete-alarm    drop its alarm
 *   claim?slot=S    the cron's claim request, alone (a cron that died at once)
 *   fail?mode=M     'claim-500' | 'claim-hang' | 'state-read' | 'none'
 *   noop            nothing (the dispatch cost the wake measurement subtracts)
 */

import worker, { MinuteClock } from '../src/index';
import type { Env } from '../src/index';
import { CLOCK_NAME } from '../src/watchdog';

type FailMode = 'none' | 'claim-500' | 'claim-hang' | 'state-read';
const faults: { mode: FailMode } = { mode: 'none' };

function isFailMode(value: string | null): value is FailMode {
  return value === 'none' || value === 'claim-500' || value === 'claim-hang' || value === 'state-read';
}

/** `env` whose bucket fails every state/ read while mode is 'state-read'. */
function faultyEnv(env: Env): Env {
  const bucket = new Proxy(env.MOMENTARILY, {
    get(target, prop, receiver) {
      const value: unknown = Reflect.get(target, prop, receiver);
      if (prop === 'get') {
        return async (key: string, opts?: R2GetOptions) => {
          if (faults.mode === 'state-read' && key.startsWith('state/')) throw new Error(`injected read failure: ${key}`);
          return target.get(key, opts);
        };
      }
      return typeof value === 'function' ? value.bind(target) : value;
    },
  });
  return { ...env, MOMENTARILY: bucket };
}

export class HarnessClock extends MinuteClock {
  private offsetMs = 0;

  constructor(state: DurableObjectState, env: Env) {
    super(state, faultyEnv(env));
  }

  protected override now(): number {
    return Date.now() + this.offsetMs;
  }

  override async fetch(request: Request): Promise<Response> {
    const url = new URL(request.url);
    switch (url.pathname) {
      case '/harness/offset':
        this.offsetMs = Number(url.searchParams.get('ms'));
        return Response.json({ now: this.now() });
      case '/harness/alarm':
        await this.alarm();
        return Response.json({ ran: true });
      case '/harness/state':
        return Response.json({
          now: this.now(),
          alarm: await this.state.storage.getAlarm(),
          claims: Object.fromEntries(await this.state.storage.list({ prefix: 'slot:' })),
        });
      case '/harness/delete-alarm':
        await this.state.storage.deleteAlarm();
        return Response.json({ alarm: null });
      case '/harness/claim':
        return super.fetch(new Request(`https://minute-clock/claim${url.search}`, { method: 'POST' }));
      case '/harness/fail': {
        const mode = url.searchParams.get('mode');
        if (!isFailMode(mode)) return new Response('bad mode', { status: 400 });
        faults.mode = mode;
        return Response.json({ mode });
      }
      case '/harness/noop':
        return Response.json({});
      default:
        break;
    }
    if (url.pathname === '/claim' && faults.mode === 'claim-500') return new Response('injected', { status: 500 });
    if (url.pathname === '/claim' && faults.mode === 'claim-hang') {
      await new Promise((resolve) => setTimeout(resolve, 5_000));
    }
    return super.fetch(request);
  }
}

export default {
  async fetch(request: Request, env: Env): Promise<Response> {
    const ns = env.MINUTE_CLOCK;
    if (ns !== undefined && new URL(request.url).pathname.startsWith('/harness/')) {
      return ns.get(ns.idFromName(CLOCK_NAME)).fetch(request);
    }
    return worker.fetch(request, env);
  },
  scheduled: worker.scheduled,
} satisfies ExportedHandler<Env>;
