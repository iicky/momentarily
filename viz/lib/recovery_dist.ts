// Recovery time as a distribution — grade the model's full predicted recovery
// curve as one object instead of three horizon Brier scores.
//
// Each sample carries the model's recovery CDF (reconstructed from the
// params.json dwell curve, sampled at every integer minute) and the realized
// time-to-normal. We score with:
//   - CRPS: ∫ (F_pred(t) − 1{t ≥ actual})² dt — one proper score over the whole
//     curve, in minutes. A climatology baseline (the empirical realized CDF used
//     as everyone's forecast) gives a skill score.
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
}

// CRPS/PIT under one weighting. Per-tick weights every prediction tick equally
// (operational forecast load — long incidents dominate); per-regime averages
// each episode's ticks, then weights episodes equally (incident-level quality).
export interface RecoveryWeighting {
  n: number; // ticks (per-tick) or distinct regimes (per-regime)
  meanCrps: number; // minutes, lower better
  baselineCrps: number; // climatology (empirical CDF) CRPS, minutes
  skill: number; // 1 − meanCrps/baselineCrps; >0 beats climatology
  meanPit: number; // <0.5 pessimistic, >0.5 optimistic, 0.5 calibrated
}

export interface RecoveryDistReport {
  // Per-tick headline kept at the top level for the curve view's back-compat.
  n: number;
  meanCrps: number;
  baselineCrps: number;
  skill: number;
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

const EMPTY_WEIGHTING: RecoveryWeighting = {
  n: 0,
  meanCrps: NaN,
  baselineCrps: NaN,
  skill: NaN,
  meanPit: NaN,
};

function emptyReport(tMax: number): RecoveryDistReport {
  const grid: number[] = [];
  for (let t = 0; t <= tMax; t += GRID_STEP) grid.push(t);
  return {
    n: 0,
    meanCrps: NaN,
    baselineCrps: NaN,
    skill: NaN,
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

export function recoveryDistReport(samples: RecoveryDistSample[]): RecoveryDistReport {
  const n = samples.length;
  if (!n) return emptyReport(240);

  const tMax = samples[0].predCurve.length - 1;
  const grid: number[] = [];
  for (let t = 0; t <= tMax; t += GRID_STEP) grid.push(t);

  const actualsAsc = samples.map((s) => s.actualMin).sort((a, b) => a - b);
  const empAt = (t: number) => ecdf(actualsAsc, t);

  const pit = new Array(10).fill(0);
  let crpsSum = 0;
  let baseSum = 0;
  let pitSum = 0;
  const predAccum = grid.map(() => 0);

  // Per-regime accumulators: each episode's per-tick scores are averaged first,
  // then episodes are weighted equally so one long incident can't dominate.
  const byRegime = new Map<
    string,
    { crps: number; base: number; pit: number; count: number }
  >();

  for (const s of samples) {
    const f = s.predCurve;
    const y = s.actualMin;
    // CRPS at 1-min integration steps.
    let crps = 0;
    let base = 0;
    for (let t = 0; t < tMax; t++) {
      const ind = t >= y ? 1 : 0;
      const dp = f[t] - ind;
      crps += dp * dp;
      const db = empAt(t) - ind;
      base += db * db;
    }
    crpsSum += crps;
    baseSum += base;
    const u = f[Math.min(tMax, Math.max(0, Math.round(y)))];
    pitSum += u;
    pit[Math.min(9, Math.max(0, Math.floor(u * 10)))] += 1;
    grid.forEach((t, i) => (predAccum[i] += f[t]));

    const r = byRegime.get(s.regimeKey) ?? { crps: 0, base: 0, pit: 0, count: 0 };
    r.crps += crps;
    r.base += base;
    r.pit += u;
    r.count += 1;
    byRegime.set(s.regimeKey, r);
  }

  const meanCrps = crpsSum / n;
  const baselineCrps = baseSum / n;
  const perTick: RecoveryWeighting = {
    n,
    meanCrps,
    baselineCrps,
    skill: baselineCrps > 0 ? 1 - meanCrps / baselineCrps : NaN,
    meanPit: pitSum / n,
  };

  // Average within each regime, then across regimes (equal weight per episode).
  const regimes = byRegime.size;
  let rCrps = 0;
  let rBase = 0;
  let rPit = 0;
  for (const r of byRegime.values()) {
    rCrps += r.crps / r.count;
    rBase += r.base / r.count;
    rPit += r.pit / r.count;
  }
  const regimeBaseline = rBase / regimes;
  const perRegime: RecoveryWeighting = {
    n: regimes,
    meanCrps: rCrps / regimes,
    baselineCrps: regimeBaseline,
    skill: regimeBaseline > 0 ? 1 - rCrps / rBase : NaN,
    meanPit: rPit / regimes,
  };

  return {
    n,
    meanCrps,
    baselineCrps,
    skill: perTick.skill,
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
  const skill = result.perRegime.skill;

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
    warning = `But it scores ${Math.abs(skill * 100).toFixed(0)}% worse than the dead-simple baseline — calibrated, yet no sharper than guessing the average. Calibration isn't skill.`;
  else if (tone === "warn" && skill >= 0.1)
    warning = `Even so, it beats the simple baseline by ${(skill * 100).toFixed(0)}% — miscalibrated but still more informative than guessing the average.`;

  return { verdict, explain, tone, warning };
}
