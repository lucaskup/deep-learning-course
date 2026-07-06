/* Seção 4: tarefa sintética de memória de longo prazo. Um bit ±1 chega no
   passo t = 1 com um marcador de escrita; depois só distratores ruidosos.
   A LSTM (gates dirigidas pelo marcador, f = σ(b_f) fora da escrita) segura
   o bit na esteira de c_t; a RNN vanilla sobrescreve o estado a cada passo. */
(function () {
  'use strict';
  window.DL = window.DL || {};
  DL.sections = DL.sections || [];

  const GATE_OPEN = 4; /* σ(4) ≈ 0.982: gate "aberta" no passo de escrita */

  function init() {
    const P = DL.plot;
    const $ = (id) => document.getElementById(id);
    const sig = (x) => 1 / (1 + Math.exp(-x));

    const cvTime = $('s4-timeline'), cvGates = $('s4-gates');
    const slT = $('s4-T'), slNoise = $('s4-noise'), slBf = $('s4-bf');
    const btnSample = $('s4-resample');

    let T = +slT.value, noise = +slNoise.value, bf = +slBf.value;
    let seed = 21;
    let bit = 1;
    let xs = [], cs = [], fs = [], is = [], hr = [];

    function genInputs() {
      const rng = DL.utils.mulberry32(seed);
      bit = rng() < 0.5 ? -1 : 1;
      xs = [0, bit];
      for (let k = 2; k <= T; k++) xs.push(noise * (2 * rng() - 1));
    }

    function compute() {
      cs = [0]; fs = [0]; is = [0]; hr = [0];
      for (let k = 1; k <= T; k++) {
        const write = (k === 1);
        /* LSTM: no passo de escrita i abre e f fecha; fora dele i fecha e
           f = σ(b_f) segura o que está na esteira */
        const f = write ? sig(-GATE_OPEN) : sig(bf);
        const i = write ? sig(GATE_OPEN) : sig(-GATE_OPEN);
        const cand = Math.tanh(2 * xs[k]);
        cs.push(f * cs[k - 1] + i * cand);
        fs.push(f);
        is.push(i);
        /* RNN vanilla: h_t = tanh(0.9·h_{t−1} + x_t) */
        hr.push(Math.tanh(0.9 * hr[k - 1] + xs[k]));
      }
    }

    function drawTimeline() {
      const th = P.theme();
      const { ctx, w, h } = P.setup(cvTime);
      P.clear(ctx, w, h);
      const ym = 1.15;
      const fr = P.frame(ctx, w, h, 0, T, -ym, ym);
      P.axes(ctx, fr, {
        xlabel: 't',
        xticks: [0, Math.round(T / 2), T],
        yticks: [-1, 0, 1],
      });
      /* barras: bit roxo em t = 1, distratores em ciano */
      const y0 = fr.Y(0);
      const bw = Math.max(2, fr.iw / T * 0.35);
      for (let k = 1; k <= T; k++) {
        if (xs[k] === 0) continue;
        const py = fr.Y(xs[k]);
        ctx.fillStyle = k === 1 ? th.purple : th.cyan;
        ctx.globalAlpha = k === 1 ? 0.9 : 0.4;
        ctx.fillRect(fr.X(k) - bw / 2, Math.min(py, y0), bw, Math.max(1, Math.abs(py - y0)));
      }
      ctx.globalAlpha = 1;
      const ks = cs.map((_, k) => k);
      P.line(ctx, fr, ks, hr, th.pink, { width: 1.8 });
      P.line(ctx, fr, ks, cs, th.orange, { width: 2.4 });
      P.scatter(ctx, fr, [[T, cs[T]], [T, hr[T]]], th.fg, { r: 3, alpha: 1 });
      P.label(ctx, fr.X(0) + 8, fr.Y(ym) + 14, 'bit x_1', th.purple);
      P.label(ctx, fr.X(0) + 8, fr.Y(ym) + 28, 'c_t (LSTM)', th.orange);
      P.label(ctx, fr.X(0) + 8, fr.Y(ym) + 42, 'h_t (RNN)', th.pink);
    }

    function drawGates() {
      const th = P.theme();
      const { ctx, w, h } = P.setup(cvGates);
      P.clear(ctx, w, h);
      const fr = P.frame(ctx, w, h, 0, T, 0, 1.08);
      P.axes(ctx, fr, {
        xlabel: 't',
        xticks: [0, Math.round(T / 2), T],
        yticks: [0, 0.5, 1],
      });
      P.line(ctx, fr, [0, T], [1, 1], th.comment, { width: 1, dash: [4, 4], alpha: 0.7 });
      const ks = [];
      for (let k = 1; k <= T; k++) ks.push(k);
      P.line(ctx, fr, ks, fs.slice(1), th.cyan, { width: 2 });
      P.line(ctx, fr, ks, is.slice(1), th.purple, { width: 2 });
      P.label(ctx, fr.X(T) - 60, fr.Y(1.08) + 14, 'f_t', th.cyan);
      P.label(ctx, fr.X(T) - 32, fr.Y(1.08) + 14, 'i_t', th.purple);
    }

    function verdict(v) {
      const ok = Math.abs(v) > 0.05 && Math.sign(v) === Math.sign(bit);
      return v.toFixed(3) + (ok ? ' (lembra ✓)' : ' (esqueceu ✗)');
    }

    function updateReadout() {
      $('s4-readout').textContent =
        'bit = ' + (bit > 0 ? '+1' : '−1') +
        ' · no passo final: LSTM c_T = ' + verdict(cs[T]) +
        ' · RNN h_T = ' + verdict(hr[T]);
    }

    function redraw() {
      drawTimeline();
      drawGates();
      updateReadout();
    }

    function refresh() {
      genInputs();
      compute();
      redraw();
    }

    slT.addEventListener('input', () => {
      T = +slT.value;
      $('s4-T-val').textContent = T;
      refresh();
    });
    slNoise.addEventListener('input', () => {
      noise = +slNoise.value;
      $('s4-noise-val').textContent = noise.toFixed(2);
      refresh();
    });
    slBf.addEventListener('input', () => {
      bf = +slBf.value;
      $('s4-bf-val').textContent = bf.toFixed(1);
      compute(); /* b_f não muda a sequência, só a célula */
      redraw();
    });
    btnSample.addEventListener('click', () => {
      seed = (seed * 1103515245 + 12345) >>> 0;
      refresh();
    });

    refresh();
    P.onRedraw(redraw);
    P.observeResize(cvTime, redraw);
    P.observeResize(cvGates, redraw);
  }

  DL.sections.push({ name: 's4-memoria', init });
})();
