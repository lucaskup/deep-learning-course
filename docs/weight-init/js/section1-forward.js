/* Seção 1: sinal no forward. Também define DL.winit, o estado compartilhado
   (configuração + rede + helpers de desenho) usado pela seção 2 (backward). */
(function () {
  'use strict';
  window.DL = window.DL || {};
  DL.sections = DL.sections || [];

  const FLOOR = 1e-13;   // piso para a escala log (zeros dão desvio 0)
  const NBINS = 36;

  /* Estado compartilhado entre as seções 1 e 2. */
  const S = DL.winit = {
    cfg: { init: 'normal', sigma: 1.0, act: 'tanh', L: 10, seed: 42, input: 'mnist' },
    data: null,
    listeners: [],
  };
  S.notify = function () { for (const fn of S.listeners) fn(); };

  /* ── helpers de formatação ── */

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

  /* ── rede: pesos, forward e backward, tudo com semente (winit-core.js) ── */

  S.recompute = function () {
    S.data = DL.winitCore.simulate(S.cfg);
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
    const selInit = $('s1-init'), selAct = $('s1-act'), selInput = $('s1-input');
    const sliderSigma = $('s1-sigma'), sigmaWrap = $('s1-sigma-wrap');
    const sliderDepth = $('s1-depth');
    const btnResample = $('s1-resample');

    /* Presets: as tentativas do estudo de caso dos slides, com a mesma semente
       usada por scripts/export_init_histograms.js (histogramas idênticos). */
    const PRESET_BASE = { act: 'tanh', L: 4, seed: 42, input: 'mnist' };
    const PRESETS = {
      zeros: { init: 'zeros' },
      normal: { init: 'normal', sigma: 1 },
      uniform: { init: 'uniform' },
      xavier: { init: 'xavier' },
    };

    /* Copia S.cfg para os controles (usado depois de aplicar um preset). */
    function syncControls() {
      const cfg = S.cfg;
      selInit.value = cfg.init;
      selAct.value = cfg.act;
      selInput.value = cfg.input;
      sliderSigma.value = cfg.sigma;
      $('s1-sigma-val').textContent = cfg.sigma.toFixed(2);
      sliderDepth.value = cfg.L;
      $('s1-depth-val').textContent = cfg.L;
      sigmaWrap.classList.toggle('hidden', cfg.init !== 'normal');
    }

    /* Destaca o preset que coincide com a configuração atual, se houver. */
    function updatePresetTabs() {
      const cfg = S.cfg;
      for (const key of Object.keys(PRESETS)) {
        const p = Object.assign({}, PRESET_BASE, PRESETS[key]);
        const match = Object.keys(p).every((k) => cfg[k] === p[k]);
        $('s1-preset-' + key).classList.toggle('active', match);
      }
    }

    function updateReadout() {
      const cfg = S.cfg;
      let txt;
      if (cfg.init === 'zeros') txt = 'θ = 0';
      else if (cfg.init === 'normal') txt = 'σ<sub>θ</sub> = ' + cfg.sigma.toFixed(2);
      else if (cfg.init === 'uniform') txt = 'σ<sub>θ</sub>² = 1/(3n)';
      else if (cfg.init === 'xavier') txt = 'σ<sub>θ</sub>² = 1/n';
      else txt = 'σ<sub>θ</sub>² = 2/n';
      if (cfg.init !== 'zeros' && cfg.init !== 'normal') {
        txt += cfg.input === 'mnist' ? ' (n = 784 na camada 1, 300 nas demais)' : ' (n = 300)';
      }
      $('s1-readout').innerHTML = txt;
      updatePresetTabs();
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
      S.drawStdCurve(cvStd, d.actStd, 0, d.sel, th.cyan, 'σ do input ≈ 1');
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

    for (const key of Object.keys(PRESETS)) {
      $('s1-preset-' + key).addEventListener('click', () => {
        Object.assign(S.cfg, PRESET_BASE, PRESETS[key]);
        syncControls();
        update();
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
    selInput.addEventListener('change', () => {
      S.cfg.input = selInput.value;
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
