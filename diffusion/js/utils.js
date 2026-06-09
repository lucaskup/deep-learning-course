/* Utilidades compartilhadas: RNG com semente, schedules de ruído e datasets. */
(function () {
  'use strict';
  window.DM = window.DM || {};

  /* PRNG determinístico (mulberry32) */
  function mulberry32(seed) {
    return function () {
      seed |= 0; seed = (seed + 0x6D2B79F5) | 0;
      let t = Math.imul(seed ^ (seed >>> 15), 1 | seed);
      t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
      return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
  }

  /* Box-Muller: amostra de N(0,1) */
  function randn(rng) {
    let u1 = 0;
    while (u1 === 0) u1 = rng();
    return Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * rng());
  }

  /* Schedules: retornam {beta, alpha} indexados 1..T, com alpha[0] = 1.
     Notação do curso (Prince cap. 18): alpha_t = prod_{s<=t} (1 - beta_s). */
  function linearSchedule(T, betaMin, betaMax) {
    const beta = new Float64Array(T + 1);
    const alpha = new Float64Array(T + 1);
    alpha[0] = 1;
    for (let t = 1; t <= T; t++) {
      beta[t] = T === 1 ? betaMax : betaMin + (betaMax - betaMin) * (t - 1) / (T - 1);
      alpha[t] = alpha[t - 1] * (1 - beta[t]);
    }
    return { beta, alpha, T };
  }

  function cosineSchedule(T) {
    const s = 0.008;
    const f = (t) => Math.pow(Math.cos(((t / T + s) / (1 + s)) * Math.PI / 2), 2);
    const f0 = f(0);
    const beta = new Float64Array(T + 1);
    const alpha = new Float64Array(T + 1);
    alpha[0] = 1;
    for (let t = 1; t <= T; t++) {
      let b = 1 - (f(t) / f0) / (f(t - 1) / f0);
      b = Math.min(Math.max(b, 1e-5), 0.999);
      beta[t] = b;
      alpha[t] = alpha[t - 1] * (1 - b);
    }
    return { beta, alpha, T };
  }

  /* Densidade gaussiana univariada */
  function gaussPdf(x, mu, sigma2) {
    return Math.exp(-(x - mu) * (x - mu) / (2 * sigma2)) / Math.sqrt(2 * Math.PI * sigma2);
  }

  /* Mistura 1D bimodal usada nos slides: 0.5 N(-1.5, 0.3^2) + 0.5 N(1.5, 0.3^2) */
  const BIMODAL = { w: [0.5, 0.5], mu: [-1.5, 1.5], sigma: [0.3, 0.3] };

  function sampleBimodal(rng) {
    const k = rng() < BIMODAL.w[0] ? 0 : 1;
    return BIMODAL.mu[k] + BIMODAL.sigma[k] * randn(rng);
  }

  /* Marginal exata da difusão sobre a mistura:
     q(z_t) = sum_k w_k N(sqrt(alpha_t) mu_k, alpha_t sigma_k^2 + 1 - alpha_t) */
  function bimodalMarginalPdf(z, alphaT) {
    const sa = Math.sqrt(alphaT);
    let p = 0;
    for (let k = 0; k < 2; k++) {
      p += BIMODAL.w[k] * gaussPdf(z, sa * BIMODAL.mu[k],
        alphaT * BIMODAL.sigma[k] * BIMODAL.sigma[k] + (1 - alphaT));
    }
    return p;
  }

  /* Two moons padronizado (media zero, escala ~unitária) com rótulos.
     A padronização é essencial: a difusão reversa parte de N(0,I). */
  function twoMoons(n, noise, rng) {
    const pts = [], lab = [];
    for (let i = 0; i < n; i++) {
      const a = Math.PI * rng();
      let x, y, c;
      if (i % 2 === 0) { x = Math.cos(a); y = Math.sin(a); c = 0; }
      else { x = 1 - Math.cos(a); y = 0.5 - Math.sin(a); c = 1; }
      pts.push([x + noise * randn(rng), y + noise * randn(rng)]);
      lab.push(c);
    }
    let mx = 0, my = 0;
    for (const p of pts) { mx += p[0]; my += p[1]; }
    mx /= n; my /= n;
    let sx = 0, sy = 0;
    for (const p of pts) { sx += (p[0] - mx) ** 2; sy += (p[1] - my) ** 2; }
    sx = Math.sqrt(sx / n); sy = Math.sqrt(sy / n);
    for (const p of pts) { p[0] = (p[0] - mx) / sx; p[1] = (p[1] - my) / sy; }
    return { pts, lab };
  }

  DM.utils = { mulberry32, randn, linearSchedule, cosineSchedule, gaussPdf, BIMODAL, sampleBimodal, bimodalMarginalPdf, twoMoons };
})();
