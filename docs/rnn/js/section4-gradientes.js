/* Seção 4: vanishing e exploding gradient em BPTT.
   Uma RNN escalar processa T entradas aleatórias com loss no último passo.
   Esquerda: |∂J/∂h_t| ao retropropagar de T até t. Ciano: a rede tanh real,
   produto dos fatores θ_hh·(1 − tanh²(z_k)). Cinza tracejado: a linearização
   dos slides, em que cada fator vale θ_hh (como se 1 − tanh² = 1), que
   explode para θ_hh > 1. Direita: |∂J/∂θ_ih| em função do comprimento,
   com o gradient clipping by norm (limiar c) aplicado à curva que explode. */
(function () {
  'use strict';
  window.DL = window.DL || {};
  DL.sections = DL.sections || [];

  const LMAX = 60;          /* comprimento máximo (gráfico da direita) */
  const THIH = 0.5, THO = 1, TARGET = 1.5;
  const LOGMIN = -24, LOGMAX = 26;

  function init() {
    const P = DL.plot;
    const $ = (id) => document.getElementById(id);

    const cvFlow = $('s4-flow'), cvNorm = $('s4-norm');
    const slHh = $('s4-thhh'), slT = $('s4-T'), slClip = $('s4-clip');
    const btnRand = $('s4-rand');

    let thhh = +slHh.value, T = +slT.value, clip = +slClip.value;
    let seed = 11;
    let xs = [];

    function resample() {
      const rng = DL.utils.mulberry32(seed);
      xs = [];
      for (let i = 0; i < LMAX; i++) xs.push(2 * rng() - 1);
    }

    /* Forward de L passos: h_t = tanh(x_t·θ_ih + h_{t−1}·θ_hh), h_0 = 0.
       Backward da loss J = ½(ŷ_L − y)² do último passo:
       gh[t] = ∂J/∂h_t (rede tanh) e ghLin[t] = linearização com fator θ_hh. */
    function backprop(w, L) {
      const hs = [0];
      for (let t = 1; t <= L; t++) {
        hs.push(Math.tanh(xs[t - 1] * THIH + hs[t - 1] * w));
      }
      const e = hs[L] * THO - TARGET;
      const gh = new Array(L + 1).fill(0);
      const ghLin = new Array(L + 1).fill(0);
      gh[L] = e * THO;
      ghLin[L] = e * THO;
      for (let t = L - 1; t >= 1; t--) {
        gh[t] = gh[t + 1] * (1 - hs[t + 1] * hs[t + 1]) * w;
        ghLin[t] = ghLin[t + 1] * w;
      }
      return { hs, gh, ghLin };
    }

    /* |∂J/∂θ_ih| somando as contribuições de todos os passos de tempo.
       Devolve a versão tanh (real) e a linearizada (fatores θ_hh). */
    function gradThih(w, L) {
      const { hs, gh, ghLin } = backprop(w, L);
      let g = 0, gLin = 0;
      for (let t = 1; t <= L; t++) {
        g += gh[t] * (1 - hs[t] * hs[t]) * xs[t - 1];
        gLin += ghLin[t] * xs[t - 1];
      }
      return { g: Math.abs(g), gLin: Math.abs(gLin) };
    }

    const log10 = (v) => {
      const a = Math.abs(v);
      if (!isFinite(a)) return LOGMAX;
      if (a <= 0) return LOGMIN;
      return Math.max(LOGMIN, Math.min(LOGMAX, Math.log10(a)));
    };

    function niceTicks(lo, hi) {
      const span = hi - lo;
      const step = span > 30 ? 12 : span > 15 ? 6 : span > 8 ? 3 : 1;
      const ticks = [];
      for (let v = Math.ceil(lo / step) * step; v <= hi; v += step) ticks.push(v);
      return ticks;
    }

    function clipPlot(ctx, fr) {
      ctx.save();
      ctx.beginPath();
      ctx.rect(fr.X(fr.xmin), fr.Y(fr.ymax), fr.iw, fr.ih);
      ctx.clip();
    }

    function drawFlow() {
      const th = P.theme();
      const { ctx, w, h } = P.setup(cvFlow);
      P.clear(ctx, w, h);

      const { gh, ghLin } = backprop(thhh, T);
      const ts = [], yr = [], yi = [];
      for (let t = 1; t <= T; t++) {
        ts.push(t);
        yr.push(log10(gh[t]));
        yi.push(log10(ghLin[t]));
      }
      const all = yr.concat(yi);
      const lo = Math.max(LOGMIN - 1, Math.floor(Math.min(-2, ...all)) - 1);
      const hi = Math.min(LOGMAX + 1, Math.ceil(Math.max(2, ...all)) + 1);
      const fr = P.frame(ctx, w, h, 1, Math.max(2, T), lo, hi, { l: 46, r: 10, t: 10, b: 26 });
      P.axes(ctx, fr, {
        xlabel: 't', ylabel: 'log₁₀ |∂J/∂h_t|',
        xticks: [1, Math.round(T / 2), T],
        yticks: niceTicks(lo, hi),
      });

      clipPlot(ctx, fr);
      /* Linha de referência |g| = 1 */
      P.line(ctx, fr, [1, T], [0, 0], th.line, { width: 1, dash: [3, 3] });
      /* Linearização: fatores θ_hh, sem a saturação do tanh */
      P.line(ctx, fr, ts, yi, th.comment, { width: 1.4, dash: [5, 4], alpha: 0.9 });
      /* Rede tanh real */
      P.line(ctx, fr, ts, yr, th.cyan, { width: 2 });
      P.scatter(ctx, fr, ts.map((t, i) => [t, yr[i]]), th.cyan, { r: 2.2, alpha: 0.9 });
      ctx.restore();

      P.label(ctx, fr.X(1) + 6, fr.Y(hi) + 12, 'RNN tanh', th.cyan);
      P.label(ctx, fr.X(1) + 6, fr.Y(hi) + 26, 'fatores θ_{hh} (sem saturação)', th.comment);
    }

    function drawNorm() {
      const th = P.theme();
      const { ctx, w, h } = P.setup(cvNorm);
      P.clear(ctx, w, h);

      const Ls = [], yr = [], yi = [], yc = [];
      for (let L = 1; L <= LMAX; L++) {
        const { g, gLin } = gradThih(thhh, L);
        Ls.push(L);
        yr.push(log10(g));
        yi.push(log10(gLin));
        yc.push(log10(Math.min(gLin, clip)));
      }
      const all = yr.concat(yi);
      const lo = Math.max(LOGMIN - 1, Math.floor(Math.min(-2, ...all)) - 1);
      const hi = Math.min(LOGMAX + 1, Math.ceil(Math.max(2, ...all)) + 1);
      const fr = P.frame(ctx, w, h, 1, LMAX, lo, hi, { l: 46, r: 10, t: 10, b: 26 });
      P.axes(ctx, fr, {
        xlabel: 'T', ylabel: 'log₁₀ |∂J/∂θ_ih|',
        xticks: [1, 20, 40, 60],
        yticks: niceTicks(lo, hi),
      });

      clipPlot(ctx, fr);
      /* Limiar do clipping */
      const lc = Math.log10(clip);
      P.line(ctx, fr, [1, LMAX], [lc, lc], th.green, { width: 1, dash: [4, 4], alpha: 0.8 });
      /* Comprimento atual T do gráfico da esquerda */
      P.line(ctx, fr, [T, T], [lo, hi], th.orange, { width: 1, dash: [3, 4], alpha: 0.7 });
      /* Sem saturação (explode para θ_hh > 1) e a versão após clipping */
      P.line(ctx, fr, Ls, yi, th.comment, { width: 1.4, dash: [5, 4], alpha: 0.9 });
      P.line(ctx, fr, Ls, yc, th.green, { width: 1.8 });
      /* Rede tanh real */
      P.line(ctx, fr, Ls, yr, th.cyan, { width: 2 });
      ctx.restore();

      P.label(ctx, fr.X(LMAX) - 4, fr.Y(lc) - 6, 'limiar c', th.green, 'right');
      P.label(ctx, fr.X(T) + 4, fr.Y(lo) - 6, 'T', th.orange);
      P.label(ctx, fr.X(1) + 6, fr.Y(hi) + 12, 'RNN tanh', th.cyan);
      P.label(ctx, fr.X(1) + 6, fr.Y(hi) + 26, 'sem saturação', th.comment);
      P.label(ctx, fr.X(1) + 6, fr.Y(hi) + 40, 'após clipping', th.green);
    }

    function updateReadout() {
      const { hs, gh } = backprop(thhh, T);
      let fac = 0;
      for (let t = 1; t <= T; t++) fac += Math.abs(thhh * (1 - hs[t] * hs[t]));
      fac /= T;
      const { g, gLin } = gradThih(thhh, T);
      const num = (v) => (Math.abs(v) >= 1e3 || (Math.abs(v) < 1e-3 && v !== 0)
        ? v.toExponential(1) : v.toFixed(3));
      $('s4-readout').textContent =
        'fator médio θ_hh·(1−tanh²) = ' + fac.toFixed(3) +
        ' · |∂J/∂h_1| = ' + num(Math.abs(gh[1])) +
        ' · |∂J/∂θ_ih| tanh = ' + num(g) +
        ' · sem saturação = ' + num(gLin) +
        (gLin > clip ? ' → clipping: ' + num(clip) : '');
    }

    function redraw() {
      drawFlow();
      drawNorm();
      updateReadout();
    }

    slHh.addEventListener('input', () => {
      thhh = +slHh.value;
      $('s4-thhh-val').textContent = thhh.toFixed(2);
      redraw();
    });
    slT.addEventListener('input', () => {
      T = +slT.value;
      $('s4-T-val').textContent = T;
      redraw();
    });
    slClip.addEventListener('input', () => {
      clip = +slClip.value;
      $('s4-clip-val').textContent = clip.toFixed(1);
      redraw();
    });
    btnRand.addEventListener('click', () => {
      seed = (seed * 1103515245 + 12345) >>> 0;
      resample();
      redraw();
    });

    resample();
    redraw();
    P.onRedraw(redraw);
    P.observeResize(cvFlow, redraw);
    P.observeResize(cvNorm, redraw);
  }

  DL.sections.push({ name: 's4-gradientes', init });
})();
