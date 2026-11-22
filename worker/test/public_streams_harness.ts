/**
 * Test entry for public_streams.test.ts: the production Worker's `fetch`
 * (the public read route) plus one hook that runs a production grading-stream
 * writer against the Worker's own R2 binding, inside workerd.
 *
 *   POST /harness/write  {writer, at, run, records}
 *     writer: 'predictions' | 'transitions' | 'movement'
 */

import worker from '../src/index';
import type { Env } from '../src/index';
import { writeMovementTransitions, writePredictions, writeTransitions } from '../src/grading';

const WRITERS = {
  predictions: writePredictions,
  transitions: writeTransitions,
  movement: writeMovementTransitions,
} as const;

interface WriteRequest {
  writer: keyof typeof WRITERS;
  at: number;
  run: string;
  records: never[];
}

export default {
  async fetch(request: Request, env: Env): Promise<Response> {
    if (new URL(request.url).pathname === '/harness/write') {
      const { writer, at, run, records } = await request.json<WriteRequest>();
      await WRITERS[writer](env.MOMENTARILY, at, run, records);
      return new Response('ok');
    }
    return worker.fetch(request, env);
  },
};
