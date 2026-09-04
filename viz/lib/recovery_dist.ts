// Recovery time as a distribution — grade the model's full predicted recovery
// curve as one object instead of three horizon Brier scores.
//
// Each sample carries the model's recovery CDF (reconstructed from the
// params.json dwell curve, sampled at every integer minute) and the realized
// time-to-normal. We score with:
//   - CRPS: ∫ (F_pred(t) − 1{t ≥ actual})² dt — one proper score over the whole
//     curve, in minutes. Two climatology baselines turn that into a skill score,
//     and they are NOT interchangeable: the ORACLE baseline is the empirical CDF
//     of the graded population's OWN realized durations (hindsight — a
//     forecaster who already knew this window), kept because every recovery
//     number on the record was quoted in it; the CAUSAL baseline is the same
//     empirical-CDF forecast fitted on a training window the caller passes in,
//     so both sides of the ratio are forecasts. causalSkill is null (never a
//     silent fallback to the oracle) when no training durations are supplied.
//     Measured gap on the movement dwell arm: a causally-fitted climatology
//     scores oracleSkill −0.1157 on its own graded window.
//   - PIT: F_pred(actual). Calibrated ⇒ uniform on [0,1]; the average (meanPit)
//     is a single readable "lean": <0.5 the model is too pessimistic (recoveries
//     beat its forecast), >0.5 too optimistic.
//
// Graded only on cases that did recover, so the predicted object is the timing
// of recovery *given it recovers* — see predictedRecoveryCurve.

export interface RecoveryDistSample {
  predCurve: number[]; // F_pred at integer minutes 0..TMAX
  actualMin: number; // realized minutes until the route next returned to normal
  // Ties every tick from one disruption episode together (route + regime onset)
  // so scoring can weight per incident, not per forecast tick.
  regimeKey: string;
  // F_pred immediately below the realized duration, present only when the
  // predictive distribution jumps there. PIT is only uniform against a
  // continuous predictive CDF; against a point mass every episode landing on
  // the atom returns the identical F value and the histogram collapses into
  // one bin, which reads as gross miscalibration even though the forecast may
  // be fine. The fix (see jumpFraction below) spreads each observation across
  // its own jump. Leave unset for a continuous cell, where the left limit
  // equals F and the correction is a no-op.
  predLeft?: number;
}

// CRPS/PIT under one weighting. Per-tick weights every prediction tick equally
// (operational forecast load — long incidents dominate); per-regime averages
// each episode's ticks, then weights episodes equally (incident-level quality).
export interface RecoveryWeighting {
  n: number; // ticks (per-tick) or distinct regimes (per-regime)
  meanCrps: number; // minutes, lower better
  // Hindsight: the graded population's own empirical duration CDF. Not a
  // forecast — read the header before quoting oracleSkill.
  oracleBaselineCrps: number;
  oracleSkill: number; // 1 − meanCrps/oracleBaselineCrps
  // Causal: the same empirical-CDF forecast fitted on the caller's training
  // window. null when the caller supplied none.
  causalBaselineCrps: number | null;
  causalSkill: number | null; // 1 − meanCrps/causalBaselineCrps
  meanPit: number; // <0.5 pessimistic, >0.5 optimistic, 0.5 calibrated
}

export interface RecoveryDistReport {
  // Per-tick headline kept at the top level for the curve view's back-compat.
  n: number;
  meanCrps: number;
  oracleBaselineCrps: number;
  oracleSkill: number;
  causalBaselineCrps: number | null;
  causalSkill: number | null;
  meanPit: number;
  perTick: RecoveryWeighting; // mirrors the top-level fields, named explicitly
  perRegime: RecoveryWeighting; // each disruption episode weighted equally
  pit: number[]; // 10-bin per-tick PIT histogram counts
  grid: number[]; // minutes (display sampling)
  predictedCurve: number[]; // mean F_pred at each grid minute
  empiricalCurve: number[]; // realized recovery CDF at each grid minute
  horizons: { h: number; predicted: number; observed: number }[];
}

const GRID_STEP = 5; // curve display sampling (min)

/** Empirical CDF (fraction ≤ t) over a sorted array, via binary search. */
function ecdf(sortedAsc: number[], t: number): number {
  let lo = 0;
  let hi = sortedAsc.length;
  while (lo < hi) {
    const mid = (lo + hi) >> 1;
    if (sortedAsc[mid] <= t) lo = mid + 1;
    else hi = mid;
  }
  return sortedAsc.length ? lo / sortedAsc.length : 0;
}

/**
 * A stable uniform in [0, 1) keyed on the regime, for placing an observation
 * inside its own probability jump.
 *
 * Deterministic by construction: the same episode must land in the same PIT
 * bin on every run, or the grade stops being reproducible and the dashboard
 * drifts for no reason. Keyed off a digest of the regime rather than an RNG
 * so it doesn't depend on iteration order either.
 *
 * FNV-1a 32-bit specifically, because this has to be reproduced exactly in
 * training/recovery_dist.py's _jump_fraction — it's a few lines of integer
 * arithmetic in either language, where a real digest would drag a crypto
 * dependency into the browser bundle for no benefit. Measured indistinguishable
 * from uniform at these sample sizes. Mirrored from there; keep in sync.
 */
export function jumpFraction(key: string): number {
  let h = 0x811c9dc5;
  // UTF-8 bytes, not UTF-16 code units — matches Python's key.encode().
  for (const byte of new TextEncoder().encode(key)) {
    // Plain `*` promotes through float64 and silently loses bits above 2**53;
    // Math.imul forces the exact 32-bit wraparound Python gets for free from
    // `& 0xFFFFFFFF`.
    h = Math.imul(h ^ byte, 0x01000193) >>> 0;
  }
  return h / 2 ** 32;
}

const EMPTY_WEIGHTING: RecoveryWeighting = {
  n: 0,
  meanCrps: NaN,
  oracleBaselineCrps: NaN,
  oracleSkill: NaN,
  causalBaselineCrps: null,
  causalSkill: null,
  meanPit: NaN,
};

function emptyReport(tMax: number): RecoveryDistReport {
  const grid: number[] = [];
  for (let t = 0; t <= tMax; t += GRID_STEP) grid.push(t);
  return {
    n: 0,
    meanCrps: NaN,
    oracleBaselineCrps: NaN,
    oracleSkill: NaN,
    causalBaselineCrps: null,
    causalSkill: null,
    meanPit: NaN,
    perTick: { ...EMPTY_WEIGHTING },
    perRegime: { ...EMPTY_WEIGHTING },
    pit: new Array(10).fill(0),
    grid,
    predictedCurve: grid.map(() => 0),
    empiricalCurve: grid.map(() => 0),
    horizons: [30, 60, 120].map((h) => ({ h, predicted: NaN, observed: NaN })),
  };
}

export function recoveryDistReport(
  samples: RecoveryDistSample[],
  // Realized recovery durations, in minutes, from a window that closes before
  // the graded one. Used only to build a rival FORECAST (their empirical CDF),
  // never to score, so nothing about the graded population reaches them.
  //
  // Required, with no default, on purpose (mirrors the Python port): null is
  // allowed and leaves causalSkill null, but it has to be typed by someone who
  // decided this caller has no pre-window population. A default would let the
  // next caller write recoveryDistReport(samples), render a number that looks
  // like forecast skill, and never meet the distinction.
  baselineDurationsMin: number[] | null,
): RecoveryDistReport {
  // An empty training window is a caller bug, not an absent baseline: null
  // means "no pre-window population, decided"; [] means someone built a causal
  // window and it came out empty, which must fail loudly rather than silently
  // degrade to oracle-only reporting (mirrors the Python port's ValueError).
  if (baselineDurationsMin !== null && baselineDurationsMin.length === 0) {
    throw new Error(
      "baselineDurationsMin is empty: pass null only when no pre-window population exists",
    );
  }
  const n = samples.length;
  if (!n) return emptyReport(240);

  const tMax = samples[0].predCurve.length - 1;
  const grid: number[] = [];
  for (let t = 0; t <= tMax; t += GRID_STEP) grid.push(t);

  const actualsAsc = samples.map((s) => s.actualMin).sort((a, b) => a - b);
  const empAt = (t: number) => ecdf(actualsAsc, t);
  // Both baselines are step functions of integer t only, so evaluate each once
  // per minute instead of re-bisecting inside every sample's loop.
  const oracleAt: number[] = [];
  for (let t = 0; t < tMax; t++) oracleAt.push(empAt(t));
  const causalAsc = baselineDurationsMin
    ? [...baselineDurationsMin].sort((a, b) => a - b)
    : null;
  const causalAt: number[] | null = causalAsc ? [] : null;
  if (causalAsc && causalAt) {
    for (let t = 0; t < tMax; t++) causalAt.push(ecdf(causalAsc, t));
  }

  const pit = new Array(10).fill(0);
  let crpsSum = 0;
  let baseSum = 0;
  let causalSum = 0;
  let pitSum = 0;
  const predAccum = grid.map(() => 0);

  // Per-regime accumulators: each episode's per-tick scores are averaged first,
  // then episodes are weighted equally so one long incident can't dominate.
  const byRegime = new Map<
    string,
    { crps: number; base: number; causal: number; pit: number; count: number }
  >();

  for (const s of samples) {
    const f = s.predCurve;
    const y = s.actualMin;
    // CRPS at 1-min integration steps.
    let crps = 0;
    let base = 0;
    let causal = 0;
    for (let t = 0; t < tMax; t++) {
      const ind = t >= y ? 1 : 0;
      const dp = f[t] - ind;
      crps += dp * dp;
      const db = oracleAt[t] - ind;
      base += db * db;
      if (causalAt) {
        const dc = causalAt[t] - ind;
        causal += dc * dc;
      }
    }
    crpsSum += crps;
    baseSum += base;
    causalSum += causal;
    const idx = Math.min(tMax, Math.max(0, Math.round(y)));
    let u = f[idx];
    // Spread the observation across the predictive jump it landed on, if any.
    // left === u for a continuous curve, which leaves this exactly as it was.
    const left = s.predLeft;
    if (left !== undefined && left < u) {
      u = left + jumpFraction(s.regimeKey) * (u - left);
    }
    pitSum += u;
    pit[Math.min(9, Math.max(0, Math.floor(u * 10)))] += 1;
    grid.forEach((t, i) => (predAccum[i] += f[t]));

    const r = byRegime.get(s.regimeKey) ?? {
      crps: 0,
      base: 0,
      causal: 0,
      pit: 0,
      count: 0,
    };
    r.crps += crps;
    r.base += base;
    r.causal += causal;
    r.pit += u;
    r.count += 1;
    byRegime.set(s.regimeKey, r);
  }

  const meanCrps = crpsSum / n;
  const oracleBaselineCrps = baseSum / n;
  const causalBaselineCrps = causalAt ? causalSum / n : null;
  const perTick: RecoveryWeighting = {
    n,
    meanCrps,
    oracleBaselineCrps,
    oracleSkill: oracleBaselineCrps > 0 ? 1 - meanCrps / oracleBaselineCrps : NaN,
    causalBaselineCrps,
    causalSkill:
      causalBaselineCrps === null
        ? null
        : causalBaselineCrps > 0
          ? 1 - meanCrps / causalBaselineCrps
          : NaN,
    meanPit: pitSum / n,
  };

  // Average within each regime, then across regimes (equal weight per episode).
  const regimes = byRegime.size;
  let rCrps = 0;
  let rBase = 0;
  let rCausal = 0;
  let rPit = 0;
  for (const r of byRegime.values()) {
    rCrps += r.crps / r.count;
    rBase += r.base / r.count;
    rCausal += r.causal / r.count;
    rPit += r.pit / r.count;
  }
  const regimeBaseline = rBase / regimes;
  const regimeCausal = causalAt ? rCausal / regimes : null;
  const regimeCrps = rCrps / regimes;
  const perRegime: RecoveryWeighting = {
    n: regimes,
    meanCrps: regimeCrps,
    oracleBaselineCrps: regimeBaseline,
    oracleSkill: regimeBaseline > 0 ? 1 - rCrps / rBase : NaN,
    causalBaselineCrps: regimeCausal,
    causalSkill:
      regimeCausal === null
        ? null
        : regimeCausal > 0
          ? 1 - regimeCrps / regimeCausal
          : NaN,
    meanPit: rPit / regimes,
  };

  return {
    n,
    meanCrps,
    oracleBaselineCrps,
    oracleSkill: perTick.oracleSkill,
    causalBaselineCrps: perTick.causalBaselineCrps,
    causalSkill: perTick.causalSkill,
    meanPit: perTick.meanPit,
    perTick,
    perRegime,
    pit,
    grid,
    predictedCurve: predAccum.map((v) => v / n),
    empiricalCurve: grid.map((t) => empAt(t)),
    horizons: [30, 60, 120].map((h) => ({
      h,
      predicted: predAccum[grid.indexOf(h)] / n,
      observed: empAt(h),
    })),
  };
}

// --- Verdict: read the calibration story off the PIT shape ---

// Minimum distinct incidents before the PIT shape is worth reading. Below this
// the histogram is noise, so the card says so rather than inventing a verdict.
export const VERDICT_MIN_INCIDENTS = 8;

export interface RecoveryVerdict {
  verdict: string;
  explain: string;
  tone: "good" | "warn" | "muted";
  // Surfaced when calibration shape and baseline skill tell different stories.
  warning?: string;
}

// Derive the verdict from the actual PIT histogram shape (not a fixed sentence):
// left/right lean, U-shape (overconfident) vs hump (underconfident), with a
// small-n guard and a skill-vs-shape conflict check.
export function recoveryVerdict(result: RecoveryDistReport): RecoveryVerdict {
  const pit = result.pit;
  const total = pit.reduce((a, b) => a + b, 0);
  if (!total || Number.isNaN(result.meanPit))
    return {
      verdict: "Not enough data yet",
      explain: "No recovery forecasts scored in this window yet.",
      tone: "muted",
    };

  const incidents = result.perRegime.n;
  if (incidents < VERDICT_MIN_INCIDENTS)
    return {
      verdict: "Inconclusive",
      explain: `Only ${incidents} distinct incident${incidents === 1 ? "" : "s"} recovered in this window — too few to read the calibration shape. Widen the window.`,
      tone: "muted",
    };

  const expected = total / pit.length;
  const ends = pit[0] + pit[pit.length - 1];
  const mid = pit[3] + pit[4] + pit[5] + pit[6];
  const lean = result.meanPit;
  const off = Math.abs(lean - 0.5);
  const uShape = ends > expected * 2 * 1.6; // extremes overweight → too narrow
  const humped = mid > expected * 4 * 1.3; // middle overweight → too wide
  // Prefer the causal comparison when the caller supplied a training window;
  // the oracle number is a comparison against hindsight and the sentences below
  // have to say so rather than call it "the baseline".
  const causal = result.perRegime.causalSkill;
  const skill = causal ?? result.perRegime.oracleSkill;
  const against =
    causal === null
      ? "a baseline that already knew this window's durations"
      : "a climatology forecast fitted before this window";

  let verdict: string;
  let explain: string;
  let tone: "good" | "warn";
  if (uShape && !humped) {
    verdict = "Overconfident";
    explain =
      "Outcomes pile up at the edges of the model's predicted range — its recovery intervals are too narrow, so reality lands outside them more often than it should.";
    tone = "warn";
  } else if (humped && !uShape) {
    verdict = "Underconfident";
    explain =
      "Outcomes cluster in the middle of the predicted range — the intervals are wider than they need to be.";
    tone = "warn";
  } else if (off < 0.05) {
    verdict = "Well calibrated";
    explain =
      "Recovery outcomes fall about evenly across the model's predicted range — the timing odds are honest.";
    tone = "good";
  } else if (lean < 0.5) {
    verdict = "Leans cautious";
    explain = "Lines tend to recover a little sooner than the model expects.";
    tone = "warn";
  } else {
    verdict = "Leans optimistic";
    explain =
      "Lines tend to take a little longer to recover than the model expects.";
    tone = "warn";
  }

  let warning: string | undefined;
  if (tone === "good" && skill < 0)
    warning = `But it scores ${Math.abs(skill * 100).toFixed(0)}% worse than ${against} — calibrated, yet no sharper. Calibration isn't skill.`;
  else if (tone === "warn" && skill >= 0.1)
    warning = `Even so, it beats ${against} by ${(skill * 100).toFixed(0)}% — miscalibrated but still more informative.`;

  return { verdict, explain, tone, warning };
}
