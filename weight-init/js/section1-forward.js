/* Seção 1: sinal no forward. Também define DL.winit, o estado compartilhado
   (configuração + rede + helpers de desenho) usado pela seção 2 (backward). */
(function () {
  'use strict';
  window.DL = window.DL || {};
  DL.sections = DL.sections || [];

  const N = 64;          // largura (= fan-in n^(l-1) de todas as camadas)
  const B = 256;         // tamanho do batch de entradas x ~ N(0, I)
  const FLOOR = 1e-13;   // piso para a escala log (zeros dão desvio 0)
  const NBINS = 36;

  /* Estado compartilhado entre as seções 1 e 2. */
  const S = DL.winit = {
    cfg: { init: 'normal', sigma: 1.0, act: 'tanh', L: 10, seed: 42 },
    data: null,
    listeners: [],
  };
  S.notify = function () { for (const fn of S.listeners) fn(); };

  /* ── helpers numéricos ── */

  function stdOf(arr) {
    const n = arr.length;
    let m = 0;
    for (let i = 0; i < n; i++) m += arr[i];
    m /= n;
    let v = 0;
    for (let i = 0; i < n; i++) { const d = arr[i] - m; v += d * d; }
    return Math.sqrt(v / n);
  }

  const SUPS = { '-': '⁻', '0': '⁰', '1': '¹', '2': '²', '3': '³', '4': '⁴', '5': '⁵', '6': '⁶', '7': '⁷', '8': '⁸', '9': '⁹' };
  function sup(e) {
    let out = '';
    for (const ch of String(e)) out += SUPS[ch] || ch;
    return out;
  }

  /* Número compacto para canvas: decimal no meio da escala, científico fora. */
  function sciUni(x) {
    if (x === 0) return '0';
    const a = Math.abs(x);
    if (a >= 0.01 && a < 1000) {
      const s = a >= 100 ? x.toFixed(0) : a >= 10 ? x.toFixed(1) : x.toFixed(2);
      return s.replace('-', '−');
    }
    const e = Math.floor(Math.log10(a));
    const m = x / Math.pow(10, e);
    return (m.toFixed(1) + '·10' + sup(e)).replace('-', '−');
  }
  S.sup = sup;
  S.sciUni = sciUni;

  /* ── rede: pesos, forward e backward, tudo com semente ── */

  S.recompute = function () {
    const U = DL.utils;
    const cfg = S.cfg;
    const L = cfg.L;
    const rng = U.mulberry32(cfg.seed);

    let sw;                                      // desvio dos pesos σ_θ
    if (cfg.init === 'zeros') sw = 0;
    else if (cfg.init === 'normal') sw = cfg.sigma;
    else if (cfg.init === 'xavier') sw = Math.sqrt(1 / N);   // σ_θ² = 1/n^(l-1)
    else sw = Math.sqrt(2 / N);                              // He: σ_θ² = 2/n^(l-1)

    /* batch de entradas */
    const X = new Float32Array(B * N);
    for (let i = 0; i < X.length; i++) X[i] = U.randn(rng);

    /* pesos θ^(l), bias = 0 */
    const Ws = [];
    for (let l = 0; l < L; l++) {
      const W = new Float32Array(N * N);
      if (sw > 0) for (let i = 0; i < W.length; i++) W[i] = sw * U.randn(rng);
      Ws.push(W);
    }

    /* forward: acts[l] = a^(l) (acts[0] = x), derivs[l-1] = φ'(z^(l)) */
    const acts = [X], derivs = [];
    const act = cfg.act;
    let aPrev = X;
    for (let l = 0; l < L; l++) {
      const W = Ws[l];
      const a = new Float32Array(B * N), d = new Float32Array(B * N);
      for (let b = 0; b < B; b++) {
        const off = b * N;
        for (let j = 0; j < N; j++) {
          let s = 0;
          const wr = j * N;
          for (let k = 0; k < N; k++) s += W[wr + k] * aPrev[off + k];
          let av, dv;
          if (act === 'relu') { av = s > 0 ? s : 0; dv = s > 0 ? 1 : 0; }
          else if (act === 'sigmoid') { av = 1 / (1 + Math.exp(-s)); dv = av * (1 - av); }
          else { av = Math.tanh(s); dv = 1 - av * av; }
          a[off + j] = av; d[off + j] = dv;
        }
      }
      acts.push(a); derivs.push(d);
      aPrev = a;
    }

    /* backward: gradiente com desvio 1 injetado em a^(L),
       g^(l) = (θ^(l+1))^T (g^(l+1) ⊙ φ'(z^(l+1))) */
    const grads = new Array(L + 1);
    const gTop = new Float32Array(B * N);
    for (let i = 0; i < gTop.length; i++) gTop[i] = U.randn(rng);
    grads[L] = gTop;
    for (let l = L - 1; l >= 1; l--) {
      const W = Ws[l];                  // θ^(l+1)
      const gNext = grads[l + 1], d = derivs[l];   // φ'(z^(l+1))
      const g = new Float32Array(B * N);
      for (let b = 0; b < B; b++) {
        const off = b * N;
        for (let j = 0; j < N; j++) {
          const t = gNext[off + j] * d[off + j];
          if (t === 0) continue;
          const wr = j * N;
          for (let k = 0; k < N; k++) g[off + k] += W[wr + k] * t;
        }
      }
      grads[l] = g;
    }

    /* estatísticas por camada */
    const actStd = new Array(L + 1), gradStd = new Array(L + 1);
    for (let l = 0; l <= L; l++) actStd[l] = stdOf(acts[l]);
    gradStd[0] = NaN;
    for (let l = 1; l <= L; l++) gradStd[l] = stdOf(grads[l]);

    /* camadas exibidas nos histogramas: 1, ⌈L/4⌉, ⌈L/2⌉, ⌈3L/4⌉, L (sem repetição) */
    const sel = [];
    for (const l of [1, Math.ceil(L / 4), Math.ceil(L / 2), Math.ceil(3 * L / 4), L]) {
      if (!sel.includes(l)) sel.push(l);
    }

    S.data = { L, acts, grads, actStd, gradStd, sel };
  };

  /* ── helpers de desenho compartilhados ── */

  /* Pequenos múltiplos: um histograma por camada selecionada, lado a lado,
     todos com o mesmo intervalo horizontal [lo, hi]. */
  S.drawHistRow = function (cv, sel, getVals, getStd, lo, hi, color) {
    const P = DL.plot;
    const th = P.theme();
    const { ctx, w, h } = P.setup(cv);
    P.clear(ctx, w, h);
    const k = sel.length, cw = w / k, bw = (hi - lo) / NBINS;
    sel.forEach(function (l, i) {
      const vals = getVals(l);
      /* binagem prévia só para achar a densidade máxima da célula */
      const counts = new Float64Array(NBINS);
      for (let j = 0; j < vals.length; j++) {
        const b = Math.floor((vals[j] - lo) / bw);
        if (b >= 0 && b < NBINS) counts[b]++;
      }
      let maxD = 0;
      for (let b = 0; b < NBINS; b++) if (counts[b] > maxD) maxD = counts[b];
      maxD = maxD / (vals.length * bw);
      if (maxD <= 0) maxD = 1;
      const pad = { l: i * cw + 8, r: w - (i + 1) * cw + 8, t: 20, b: 32 };
      const fr = P.frame(ctx, w, h, lo, hi, 0, maxD * 1.06, pad);
      P.line(ctx, fr, [lo, hi], [0, 0], th.line, { width: 1 });
      if (lo < 0 && hi > 0) {
        P.line(ctx, fr, [0, 0], [0, maxD * 1.06], th.line, { dash: [3, 4], width: 1 });
      }
      P.histogram(ctx, fr, vals, NBINS, color, { alpha: 0.6, range: [lo, hi] });
      P.mathText(ctx, 'camada ' + l, i * cw + cw / 2, 13, th.fg, 'center');
      P.mathText(ctx, sciUni(lo), i * cw + 10, h - 19, th.comment, 'left', 9);
      P.mathText(ctx, sciUni(hi), (i + 1) * cw - 10, h - 19, th.comment, 'right', 9);
      P.mathText(ctx, 'σ = ' + sciUni(getStd(l)), i * cw + cw / 2, h - 6, th.comment, 'center', 10);
    });
  };

  /* Curva do desvio padrão por camada em escala log (eixo y = log10 σ),
     com linha de referência tracejada em σ = 1 e pontos nas camadas exibidas. */
  S.drawStdCurve = function (cv, stds, l0, sel, color, refLabel) {
    const P = DL.plot;
    const th = P.theme();
    const { ctx, w, h } = P.setup(cv);
    P.clear(ctx, w, h);
    const L = stds.length - 1;
    const xs = [], ys = [];
    for (let l = l0; l <= L; l++) {
      xs.push(l);
      ys.push(Math.log10(Math.max(stds[l], FLOOR)));
    }
    let lo = 0, hi = 0;
    for (const y of ys) { if (y < lo) lo = y; if (y > hi) hi = y; }
    lo -= 0.6; hi += 0.6;
    const fr = P.frame(ctx, w, h, l0, L, lo, hi, { l: 56, r: 14, t: 12, b: 26 });
    const step = Math.max(1, Math.ceil((hi - lo) / 6));
    for (let e = Math.ceil(lo); e <= Math.floor(hi); e += step) {
      P.line(ctx, fr, [l0, L], [e, e], th.line, { dash: [3, 4], width: 1 });
      P.label(ctx, fr.X(l0) - 6, fr.Y(e) + 4, '10' + sup(e), th.comment, 'right');
    }
    const xticks = [];
    for (const v of [l0, Math.ceil(L / 2), L]) if (!xticks.includes(v)) xticks.push(v);
    P.axes(ctx, fr, { xlabel: 'camada l', xticks });
    P.line(ctx, fr, [l0, L], [0, 0], th.comment, { dash: [6, 4], width: 1.3 });
    P.label(ctx, fr.X(l0) + 8, fr.Y(0) - 6, refLabel, th.comment);
    P.line(ctx, fr, xs, ys, color, { width: 2.2 });
    P.scatter(ctx, fr, sel.map((l) => [l, ys[l - l0]]), color, { r: 3.5, alpha: 1 });
  };

  /* ── seção 1: controles e desenho ── */

  function init() {
    const P = DL.plot;
    const $ = (id) => document.getElementById(id);

    const cvHist = $('s1-hist'), cvStd = $('s1-std');
    const selInit = $('s1-init'), selAct = $('s1-act');
    const sliderSigma = $('s1-sigma'), sigmaWrap = $('s1-sigma-wrap');
    const sliderDepth = $('s1-depth');
    const btnResample = $('s1-resample');

    function updateReadout() {
      const cfg = S.cfg;
      let txt;
      if (cfg.init === 'zeros') txt = 'θ = 0';
      else if (cfg.init === 'normal') txt = 'σ<sub>θ</sub> = ' + cfg.sigma.toFixed(2);
      else if (cfg.init === 'xavier') txt = 'σ<sub>θ</sub>² = 1/64, σ<sub>θ</sub> ≈ 0.125';
      else txt = 'σ<sub>θ</sub>² = 2/64, σ<sub>θ</sub> ≈ 0.177';
      $('s1-readout').innerHTML = txt;
    }

    function redraw() {
      const d = S.data;
      if (!d) return;
      const th = P.theme();
      const cfg = S.cfg;
      let lo, hi;
      if (cfg.act === 'tanh') { lo = -1.15; hi = 1.15; }
      else if (cfg.act === 'sigmoid') { lo = -0.05; hi = 1.05; }
      else {
        let m = 0;
        for (const l of d.sel) if (d.actStd[l] > m) m = d.actStd[l];
        lo = 0; hi = Math.max(3.5 * m, 1e-6);
      }
      S.drawHistRow(cvHist, d.sel, (l) => d.acts[l], (l) => d.actStd[l], lo, hi, th.cyan);
      S.drawStdCurve(cvStd, d.actStd, 0, d.sel, th.cyan, 'σ do input = 1');
      updateReadout();
    }

    /* Coalesce eventos rápidos dos sliders: recompute (caro em L = 20)
       roda no máximo uma vez por frame. */
    let pending = false;
    function update() {
      if (pending) return;
      pending = true;
      requestAnimationFrame(() => {
        pending = false;
        S.recompute();
        S.notify();
      });
    }

    selInit.addEventListener('change', () => {
      S.cfg.init = selInit.value;
      sigmaWrap.classList.toggle('hidden', S.cfg.init !== 'normal');
      update();
    });
    sliderSigma.addEventListener('input', () => {
      S.cfg.sigma = +sliderSigma.value;
      $('s1-sigma-val').textContent = S.cfg.sigma.toFixed(2);
      update();
    });
    selAct.addEventListener('change', () => {
      S.cfg.act = selAct.value;
      update();
    });
    sliderDepth.addEventListener('input', () => {
      S.cfg.L = +sliderDepth.value;
      $('s1-depth-val').textContent = S.cfg.L;
      update();
    });
    btnResample.addEventListener('click', () => {
      S.cfg.seed = (S.cfg.seed * 1664525 + 1013904223) >>> 0;
      update();
    });

    sigmaWrap.classList.toggle('hidden', S.cfg.init !== 'normal');
    S.listeners.push(redraw);
    S.recompute();
    redraw();
    P.onRedraw(redraw);
    P.observeResize(cvHist, redraw);
    P.observeResize(cvStd, redraw);
  }

  DL.sections.push({ name: 's1-forward', init });
})();
