/* Seção 4: variância das ativações em blocos residuais, sem e com batch
   norm. Propaga um batch N(0,1) por blocos ReLU seguida de Linear (He) e
   mede a variância empírica após cada bloco. */
(function () {
  'use strict';
  window.DL = window.DL || {};
  DL.sections = DL.sections || [];

  const WID = 32, BATCH = 128;

  function init() {
    const P = DL.plot, U = DL.utils;
    const $ = (id) => document.getElementById(id);

    const cvVar = $('s4-var'), cvHist = $('s4-hist');
    const slDepth = $('s4-depth');
    const btnSeed = $('s4-seed');

    let seed = 21;
    let depth = +slDepth.value;
    let simNo, simBN;

    function variance(h) {
      let mu = 0;
      for (let i = 0; i < h.length; i++) mu += h[i];
      mu /= h.length;
      let v = 0;
      for (let i = 0; i < h.length; i++) v += (h[i] - mu) * (h[i] - mu);
      return v / h.length;
    }

    /* Padroniza cada feature (coluna) sobre o batch, como o batch norm
       com γ = 1 e β = 0. */
    function bnStandardize(h, out) {
      for (let j = 0; j < WID; j++) {
        let mu = 0;
        for (let n = 0; n < BATCH; n++) mu += h[n * WID + j];
        mu /= BATCH;
        let v = 0;
        for (let n = 0; n < BATCH; n++) {
          const d = h[n * WID + j] - mu;
          v += d * d;
        }
        const s = Math.sqrt(v / BATCH) + 1e-5;
        for (let n = 0; n < BATCH; n++) out[n * WID + j] = (h[n * WID + j] - mu) / s;
      }
    }

    function simulate(useBN) {
      const rng = U.mulberry32(seed);
      const std = Math.sqrt(2 / WID);
      const Ws = [];
      for (let l = 0; l < depth; l++) {
        const Wl = new Float64Array(WID * WID);
        for (let i = 0; i < WID * WID; i++) Wl[i] = U.randn(rng) * std;
        Ws.push(Wl);
      }
      const h = new Float64Array(BATCH * WID);
      for (let i = 0; i < h.length; i++) h[i] = U.randn(rng);
      const hb = new Float64Array(BATCH * WID);
      const z = new Float64Array(WID);
      const vars = [variance(h)];
      for (let l = 0; l < depth; l++) {
        const Wl = Ws[l];
        let src = h;
        if (useBN) {
          bnStandardize(h, hb);
          src = hb;
        }
        for (let n = 0; n < BATCH; n++) {
          const base = n * WID;
          for (let i = 0; i < WID; i++) {
            let s = 0;
            for (let j = 0; j < WID; j++) {
              const a = src[base + j];
              if (a > 0) s += Wl[i * WID + j] * a;
            }
            z[i] = s;
          }
          for (let i = 0; i < WID; i++) h[base + i] += z[i];
        }
        vars.push(variance(h));
      }
      return { vars, final: h };
    }

    function compute() {
      simNo = simulate(false);
      simBN = simulate(true);
    }

    function drawVar() {
      const th = P.theme();
      const { ctx, w, h } = P.setup(cvVar);
      P.clear(ctx, w, h);
      const lg = (v) => Math.log2(Math.max(v, 1e-6));
      let hi = 1;
      for (const v of simNo.vars) hi = Math.max(hi, lg(v));
      for (const v of simBN.vars) hi = Math.max(hi, lg(v));
      const fr = P.frame(ctx, w, h, 0, Math.max(depth, 1), -0.5, hi + 1);
      P.axes(ctx, fr, {
        xlabel: 'bloco k', ylabel: 'log₂ σ²',
        xticks: [0, Math.round(depth / 2), depth],
        yticks: [0, Math.round(hi / 2), Math.round(hi)],
      });
      const ks = [];
      for (let k = 0; k <= depth; k++) ks.push(k);
      /* Referência 2^k: log₂ = k. */
      P.line(ctx, fr, ks, ks, th.green, { width: 1.2, dash: [5, 4], alpha: 0.7 });
      P.line(ctx, fr, ks, simNo.vars.map(lg), th.cyan, { width: 2 });
      P.line(ctx, fr, ks, simBN.vars.map(lg), th.orange, { width: 2 });
      P.scatter(ctx, fr, ks.map((k) => [k, lg(simNo.vars[k])]), th.cyan, { r: 2.6, alpha: 1 });
      P.scatter(ctx, fr, ks.map((k) => [k, lg(simBN.vars[k])]), th.orange, { r: 2.6, alpha: 1 });
      P.label(ctx, fr.X(0) + 8, fr.Y(fr.ymax) + 14, 'sem batch norm', th.cyan);
      P.label(ctx, fr.X(0) + 8, fr.Y(fr.ymax) + 28, 'com batch norm', th.orange);
      P.label(ctx, fr.X(0) + 8, fr.Y(fr.ymax) + 42, 'referência 2^k', th.green);
    }

    function drawHist() {
      const th = P.theme();
      const { ctx, w, h } = P.setup(cvHist);
      P.clear(ctx, w, h);
      const sdNo = Math.sqrt(simNo.vars[depth]);
      const sdBN = Math.sqrt(simBN.vars[depth]);
      const R = Math.max(4 * sdNo, 6);
      const ymax = 0.55 / sdBN;
      const fr = P.frame(ctx, w, h, -R, R, 0, ymax);
      P.axes(ctx, fr, {
        xlabel: 'ativação', ylabel: 'densidade',
        xticks: [-Math.round(R * 0.75), 0, Math.round(R * 0.75)],
        yticks: [],
      });
      P.histogram(ctx, fr, simNo.final, 60, th.cyan, { alpha: 0.5 });
      P.histogram(ctx, fr, simBN.final, 60, th.orange, { alpha: 0.5 });
      P.label(ctx, fr.X(-R) + 8, fr.Y(ymax) + 14, 'σ sem BN = ' + sdNo.toFixed(1), th.cyan);
      P.label(ctx, fr.X(-R) + 8, fr.Y(ymax) + 28, 'σ com BN = ' + sdBN.toFixed(1), th.orange);
    }

    function updateReadout() {
      $('s4-readout').textContent =
        'σ² final sem BN = ' + simNo.vars[depth].toFixed(1) +
        ' (2^' + depth + ' = ' + (1 << depth) + ')' +
        ' · com BN = ' + simBN.vars[depth].toFixed(1) +
        ' (k+1 = ' + (depth + 1) + ')';
    }

    function redraw() {
      drawVar();
      drawHist();
      updateReadout();
    }

    slDepth.addEventListener('input', () => {
      depth = +slDepth.value;
      $('s4-depth-val').textContent = depth;
      compute();
      redraw();
    });
    btnSeed.addEventListener('click', () => {
      seed += 1;
      compute();
      redraw();
    });

    compute();
    redraw();
    P.onRedraw(redraw);
    P.observeResize(cvVar, redraw);
    P.observeResize(cvHist, redraw);
  }

  DL.sections.push({ name: 's4-variance-batchnorm', init });
})();
