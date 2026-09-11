/**
 * The supply/service axis hysteresis thresholds — the SINGLE source of truth for
 * the ratio at which a route degrades and the higher ratio at which it recovers.
 * worker/src/movement_state.ts uses these to compute the debounced service
 * regime; viz marks them on the drawer meter so a reader sees where a route sits
 * relative to what actually flips the axis. Ported from the offline degradation
 * label (load_r2.derive_actual_recovery) so the published axis and the grading
 * truth agree.
 */

// A route degrades once its supply ratio drops strictly below this.
export const SERVICE_DEGRADE_RATIO = 0.5;
// Once degraded, a route only recovers back to normal above this higher ratio —
// the hysteresis gap that keeps a route riding the threshold from flapping.
export const SERVICE_RECOVER_RATIO = 0.8;
