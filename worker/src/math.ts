/**
 * Numerical primitives the HMM needs. JavaScript doesn't ship lgamma or gamma,
 * so we implement what we need.
 */

/**
 * Lanczos approximation to log Γ(x), accurate to ~15 digits for x > 0.
 * Standard textbook coefficients (Numerical Recipes / mathlib).
 */
const LG_COEF = [
  0.99999999999980993, 676.5203681218851, -1259.1392167224028, 771.32342877765313,
  -176.61502916214059, 12.507343278686905, -0.13857109526572012,
  9.9843695780195716e-6, 1.5056327351493116e-7,
];

export function lgamma(x: number): number {
  if (x < 0.5) {
    // Reflection formula: Γ(x)·Γ(1−x) = π / sin(π·x)
    return Math.log(Math.PI / Math.sin(Math.PI * x)) - lgamma(1 - x);
  }
  const z = x - 1;
  let sum = LG_COEF[0]!;
  for (let i = 1; i < LG_COEF.length; i += 1) {
    sum += LG_COEF[i]! / (z + i);
  }
  const t = z + LG_COEF.length - 1.5;
  return 0.5 * Math.log(2 * Math.PI) + (z + 0.5) * Math.log(t) - t + Math.log(sum);
}

/**
 * log P(k | Poisson(λ)) = −λ + k·log(λ) − log(k!)
 */
export function logPoisson(k: number, lambda: number): number {
  if (lambda <= 0) {
    return k === 0 ? 0 : -Infinity;
  }
  return -lambda + k * Math.log(lambda) - lgamma(k + 1);
}

/**
 * log Gamma(x; α=shape, β=rate) with the same +0.5 shift hmm.py uses,
 * so x=0 doesn't blow up log(x).
 */
export function logGamma(x: number, alpha: number, beta: number): number {
  const shifted = Math.max(x + 0.5, 1e-9);
  return (
    alpha * Math.log(beta)
    + (alpha - 1) * Math.log(shifted)
    - beta * shifted
    - lgamma(alpha)
  );
}

/**
 * log P(value | Bernoulli(p)) with clipping so log(0) is bounded.
 */
export function logBernoulli(value: boolean, p: number): number {
  const clipped = Math.min(Math.max(p, 1e-12), 1 - 1e-12);
  return value ? Math.log(clipped) : Math.log1p(-clipped);
}

/**
 * log P(k of n | Binomial(n, p)) with the n-choose-k coefficient included, so
 * the value is a true log-likelihood. n <= 0 returns 0 (the channel drops out).
 * Mirrors _log_binomial in src/momentarily/hmm.py.
 */
export function logBinomial(k: number, n: number, p: number): number {
  if (n <= 0) return 0;
  const clipped = Math.min(Math.max(p, 1e-12), 1 - 1e-12);
  const logCoef = lgamma(n + 1) - lgamma(k + 1) - lgamma(n - k + 1);
  return logCoef + k * Math.log(clipped) + (n - k) * Math.log1p(-clipped);
}
