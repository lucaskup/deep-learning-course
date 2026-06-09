/* Utilidades compartilhadas: RNG determinístico com semente. */
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

  DM.utils = { mulberry32, randn };
})();
