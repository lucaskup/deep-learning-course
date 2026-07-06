/* Utilidades compartilhadas das demos: RNG com semente e datasets 2D/1D. */
(function () {
  'use strict';
  window.DL = window.DL || {};

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

  /* Densidade gaussiana univariada */
  function gaussPdf(x, mu, sigma2) {
    return Math.exp(-(x - mu) * (x - mu) / (2 * sigma2)) / Math.sqrt(2 * Math.PI * sigma2);
  }

  /* Padroniza pontos 2D para média zero e desvio unitário por eixo (in place). */
  function standardize(pts) {
    const n = pts.length;
    let mx = 0, my = 0;
    for (const p of pts) { mx += p[0]; my += p[1]; }
    mx /= n; my /= n;
    let sx = 0, sy = 0;
    for (const p of pts) { sx += (p[0] - mx) ** 2; sy += (p[1] - my) ** 2; }
    sx = Math.sqrt(sx / n) || 1; sy = Math.sqrt(sy / n) || 1;
    for (const p of pts) { p[0] = (p[0] - mx) / sx; p[1] = (p[1] - my) / sy; }
    return pts;
  }

  /* Two moons padronizado com rótulos 0/1. */
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
    standardize(pts);
    return { pts, lab };
  }

  /* Dois anéis concêntricos padronizados com rótulos 0/1. */
  function circles(n, noise, rng) {
    const pts = [], lab = [];
    for (let i = 0; i < n; i++) {
      const a = 2 * Math.PI * rng();
      const c = i % 2;
      const r = c === 0 ? 0.6 : 1.6;
      pts.push([r * Math.cos(a) + noise * randn(rng), r * Math.sin(a) + noise * randn(rng)]);
      lab.push(c);
    }
    standardize(pts);
    return { pts, lab };
  }

  /* Padrão XOR: quatro blobs gaussianos nos quadrantes, rótulo = sinal de x·y. */
  function xorData(n, noise, rng) {
    const pts = [], lab = [];
    for (let i = 0; i < n; i++) {
      const sx = rng() < 0.5 ? -1 : 1;
      const sy = rng() < 0.5 ? -1 : 1;
      pts.push([sx * 1.2 + noise * randn(rng), sy * 1.2 + noise * randn(rng)]);
      lab.push(sx * sy > 0 ? 0 : 1);
    }
    standardize(pts);
    return { pts, lab };
  }

  /* Dois blobs gaussianos linearmente separáveis (ou não, conforme a distância). */
  function blobs(n, dist, noise, rng) {
    const pts = [], lab = [];
    for (let i = 0; i < n; i++) {
      const c = i % 2;
      const cx = c === 0 ? -dist / 2 : dist / 2;
      const cy = c === 0 ? -dist / 4 : dist / 4;
      pts.push([cx + noise * randn(rng), cy + noise * randn(rng)]);
      lab.push(c);
    }
    return { pts, lab };
  }

  /* Regressão 1D: y = f(x) + ruído, x em [xmin, xmax]. */
  function noisyCurve(n, f, noise, xmin, xmax, rng) {
    const xs = [], ys = [];
    for (let i = 0; i < n; i++) {
      const x = xmin + (xmax - xmin) * rng();
      xs.push(x);
      ys.push(f(x) + noise * randn(rng));
    }
    return { xs, ys };
  }

  DL.utils = { mulberry32, randn, gaussPdf, standardize, twoMoons, circles, xorData, blobs, noisyCurve };
})();
