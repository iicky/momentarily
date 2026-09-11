/**
 * Headway cell sizing constant, the single source shared by
 * worker/src/headway.ts (which bounds the published rolling window to this many
 * passings) and viz/lib/headway.ts (which reads it back to tell a full window
 * from one still filling). One definition so the client's "short window" cutoff
 * cannot drift from the size the worker actually publishes.
 */

// How many recent passings a cell's rolling headway window holds. A shorter
// published window is a cell that has not yet filled, not a wider one.
export const HEADWAY_WINDOW_SIZE = 12;
