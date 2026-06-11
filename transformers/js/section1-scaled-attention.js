/* Seção 1: por que scaled dot product attention. Histograma de q·k vs q·k/√d
   e saturação da softmax quando d cresce (var[q·k] = d, slides da aula). */
(function () {
  'use strict';
  window.DL = window.DL || {};
  DL.sections = DL.sections || [];

  const N_PAIRS = 1500, M_KEYS = 10;

  function init() {
    const P = DL.plot;
    const U = DL.utils;
    const $ = (id) => document.getElementById(id);

    const cvHist = $('s1-hist'), cvSoft = $('s1-softmax');
    const slD = $('s1-d');

    let seed = 7;
    let d = Math.pow(2, +slD.value);
    let scoresRaw, scoresScaled, softRaw, softScaled;

    function dot(rng, dim) {
      let s = 0;
      for (let i = 0; i < dim; i++) s += U.randn(rng) * U.randn(rng);
      return s;
    }

    function softmax(xs) {
      const mx = Math.max(...xs);
      const ex = xs.map((v) => Math.exp(v - mx));
      const Z = ex.reduce((a, b) => a + b, 0);
      return ex.map((v) => v / Z);
    }

    function recompute() {
      /* Histograma: N_PAIRS produtos escalares de pares q, k ~ N(0,1)^d */
      const rng = U.mulberry32(seed);
      scoresRaw = new Array(N_PAIRS);
      for (let p = 0; p < N_PAIRS; p++) scoresRaw[p] = dot(rng, d);
      scoresScaled = scoresRaw.map((s) => s / Math.sqrt(d));

      /* Softmax: uma query fixa contra M_KEYS keys */
      const rng2 = U.mulberry32(seed + 1);
      const q = new Array(d), keys = [];
      for (let i = 0; i < d; i++) q[i] = U.randn(rng2);
      for (let k = 0; k < M_KEYS; k++) {
        const kv = new Array(d);
        for (let i = 0; i < d; i++) kv[i] = U.randn(rng2);
        keys.push(kv);
      }
      const raw = keys.map((kv) => {
        let s = 0;
        for (let i = 0; i < d; i++) s += q[i] * kv[i];
        return s;
      });
      softRaw = softmax(raw);
      softScaled = softmax(raw.map((s) => s / Math.sqrt(d)));
    }

    function variance(xs) {
      let m = 0;
      for (const v of xs) m += v;
      m /= xs.length;
      let s = 0;
      for (const v of xs) s += (v - m) * (v - m);
      return s / xs.length;
    }

    function drawHist() {
      const th = P.theme();
      const { ctx, w, h } = P.setup(cvHist);
      P.clear(ctx, w, h);
      const span = Math.max(8, 3.6 * Math.sqrt(d));
      /* ymax: densidade do pico da curva escalada (~N(0,1)) com folga */
      const fr = P.frame(ctx, w, h, -span, span, 0, 0.48);
      P.axes(ctx, fr, {
        xlabel: 'score', ylabel: 'densidade',
        xticks: [-Math.round(span * 0.8), 0, Math.round(span * 0.8)],
        yticks: [0, 0.2, 0.4],
      });
      P.histogram(ctx, fr, scoresRaw, 60, th.orange, { alpha: 0.45 });
      P.histogram(ctx, fr, scoresScaled, 60, th.cyan, { alpha: 0.45 });
      /* Curva teórica N(0,1) sobre o histograma escalado */
      const xs = [], ys = [];
      for (let i = 0; i <= 120; i++) {
        const x = -4 + (8 * i) / 120;
        xs.push(x);
        ys.push(U.gaussPdf(x, 0, 1));
      }
      P.line(ctx, fr, xs, ys, th.cyan, { width: 1.4, dash: [4, 3] });
      P.label(ctx, fr.X(fr.xmin) + 8, fr.Y(fr.ymax) + 14, 'q·k (var ≈ d)', th.orange);
      P.label(ctx, fr.X(fr.xmin) + 8, fr.Y(fr.ymax) + 28, 'q·k/√d (var ≈ 1)', th.cyan);
    }

    function drawSoftmax() {
      const th = P.theme();
      const { ctx, w, h } = P.setup(cvSoft);
      P.clear(ctx, w, h);
      const fr = P.frame(ctx, w, h, 0, M_KEYS, 0, 1.05);
      P.axes(ctx, fr, { xlabel: 'key i', ylabel: 'a_i', yticks: [0, 0.5, 1] });
      const bw = fr.iw / M_KEYS;
      for (let i = 0; i < M_KEYS; i++) {
        const x0 = fr.X(i);
        ctx.globalAlpha = 0.85;
        ctx.fillStyle = th.orange;
        ctx.fillRect(x0 + bw * 0.12, fr.Y(softRaw[i]), bw * 0.34, fr.Y(0) - fr.Y(softRaw[i]));
        ctx.fillStyle = th.cyan;
        ctx.fillRect(x0 + bw * 0.52, fr.Y(softScaled[i]), bw * 0.34, fr.Y(0) - fr.Y(softScaled[i]));
        ctx.globalAlpha = 1;
        ctx.fillStyle = th.comment;
        ctx.font = '10px sans-serif';
        ctx.textAlign = 'center';
        ctx.fillText(String(i + 1), x0 + bw / 2, fr.Y(0) + 13);
      }
      P.label(ctx, fr.X(0) + 8, fr.Y(1.05) + 14, 'sem escala', th.orange);
      P.label(ctx, fr.X(0) + 8, fr.Y(1.05) + 28, 'com escala (÷√d)', th.cyan);
    }

    function updateReadout() {
      const vRaw = variance(scoresRaw), vSc = variance(scoresScaled);
      const maxRaw = Math.max(...softRaw);
      $('s1-readout').textContent =
        'var(q·k) = ' + vRaw.toFixed(1) + ' (teoria: ' + d + ') · var(q·k/√d) = ' +
        vSc.toFixed(2) + ' · maior peso sem escala = ' + maxRaw.toFixed(2);
    }

    function redraw() {
      drawHist();
      drawSoftmax();
      updateReadout();
    }

    slD.addEventListener('input', () => {
      d = Math.pow(2, +slD.value);
      $('s1-d-val').textContent = d;
      recompute();
      redraw();
    });
    $('s1-resample').addEventListener('click', () => {
      seed = (seed * 1664525 + 1013904223) >>> 0;
      recompute();
      redraw();
    });

    recompute();
    redraw();
    P.onRedraw(redraw);
    P.observeResize(cvHist, redraw);
    P.observeResize(cvSoft, redraw);
  }

  DL.sections.push({ name: 's1-scaled-attention', init });
})();
