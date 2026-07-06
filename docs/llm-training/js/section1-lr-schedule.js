/* Seção 1: schedule de LR do pré-treinamento (warmup linear + cosine decay)
   aplicado a uma otimização estocástica de brinquedo, comparado com η constante. */
(function () {
  'use strict';
  window.DL = window.DL || {};
  DL.sections = DL.sections || [];

  const C = 12;        /* número de condição do vale ℒ = ½(θ₁² + c·θ₂²) */
  const SIGMA = 0.3;   /* desvio do ruído aditivo no gradiente */
  const START = [-1.8, 1.4];

  function init() {
    const P = DL.plot;
    const $ = (id) => document.getElementById(id);

    const cvSched = $('s1-sched'), cvLoss = $('s1-loss');
    const slEtaMax = $('s1-etamax'), slWarmup = $('s1-warmup');
    const slEtaMin = $('s1-etamin'), slT = $('s1-T');
    const btnStep = $('s1-step'), btnRun = $('s1-run');
    const btnReset = $('s1-reset'), btnSeed = $('s1-seed');

    let etaMax = +slEtaMax.value;
    let wuFrac = +slWarmup.value / 100;
    let etaMinRatio = +slEtaMin.value;
    let T = +slT.value;
    let seed = 7;

    let t, thC, thS, lossC, lossS, noise, rng;
    let running = false, acc = 0;

    const L = (p) => 0.5 * (p[0] * p[0] + C * p[1] * p[1]);
    const clamp = (v) => (isFinite(v) ? Math.max(-60, Math.min(60, v)) : 60);

    function eta(step) {
      const Twu = Math.round(wuFrac * T);
      const etaMin = etaMinRatio * etaMax;
      if (step < Twu) return etaMax * (step + 1) / Twu;
      const u = (step - Twu) / Math.max(1, T - Twu);
      return etaMin + 0.5 * (etaMax - etaMin) * (1 + Math.cos(Math.PI * u));
    }

    /* O mesmo ruído é usado nas duas trajetórias para a comparação ser justa. */
    function noiseAt(step) {
      while (noise.length <= step) {
        noise.push([DL.utils.randn(rng), DL.utils.randn(rng)]);
      }
      return noise[step];
    }

    function resetTraj() {
      t = 0;
      thC = START.slice();
      thS = START.slice();
      lossC = [L(START)];
      lossS = [L(START)];
      noise = [];
      rng = DL.utils.mulberry32(seed);
    }

    function sgdStep(p, lr, eps) {
      return [
        clamp(p[0] - lr * (p[0] + SIGMA * eps[0])),
        clamp(p[1] - lr * (C * p[1] + SIGMA * eps[1])),
      ];
    }

    function stepOnce() {
      if (t >= T) return;
      const eps = noiseAt(t);
      thC = sgdStep(thC, etaMax, eps);
      thS = sgdStep(thS, eta(t), eps);
      lossC.push(L(thC));
      lossS.push(L(thS));
      t++;
    }

    function drawSched() {
      const th = P.theme();
      const { ctx, w, h } = P.setup(cvSched);
      P.clear(ctx, w, h);
      const fr = P.frame(ctx, w, h, 0, T, 0, etaMax * 1.15);
      P.axes(ctx, fr, {
        xlabel: 'passo t', ylabel: 'η_t',
        xticks: [0, Math.round(T / 2), T],
        yticks: [0, +(etaMax / 2).toFixed(2), +etaMax.toFixed(2)],
      });
      const Twu = Math.round(wuFrac * T);
      const xsW = [], ysW = [], xsD = [], ysD = [];
      for (let k = 0; k <= T; k++) {
        const v = eta(Math.min(k, T - 1));
        if (k <= Twu) { xsW.push(k); ysW.push(v); }
        if (k >= Twu) { xsD.push(k); ysD.push(v); }
      }
      if (xsW.length > 1) P.line(ctx, fr, xsW, ysW, th.cyan, { width: 2 });
      P.line(ctx, fr, xsD, ysD, th.orange, { width: 2 });
      /* linha do passo atual */
      P.line(ctx, fr, [t, t], [0, etaMax * 1.15], th.fg, { width: 1, alpha: 0.6, dash: [4, 4] });
      /* limite de estabilidade 2/c do GD determinístico */
      if (2 / C < etaMax * 1.15) {
        P.line(ctx, fr, [0, T], [2 / C, 2 / C], th.red, { width: 1, alpha: 0.7, dash: [2, 4] });
        P.label(ctx, fr.X(T) - 4, fr.Y(2 / C) - 5, '2/c', th.red, 'right');
      }
      if (Twu > 0) P.label(ctx, fr.X(Twu / 2), fr.Y(etaMax * 0.45), 'warmup', th.cyan, 'center');
      P.label(ctx, fr.X(Twu + (T - Twu) / 2), fr.Y(etaMax * 1.02), 'cosine decay', th.orange, 'center');
    }

    function drawLoss() {
      const th = P.theme();
      const { ctx, w, h } = P.setup(cvLoss);
      P.clear(ctx, w, h);
      const logv = (v) => (isFinite(v) && v > 0 ? Math.min(8, Math.log10(Math.max(v, 1e-10))) : (v === 0 ? -10 : 8));
      const lc = lossC.map(logv), ls = lossS.map(logv);
      const ymax = Math.min(8, Math.ceil(Math.max(1, ...lc, ...ls)) + 1);
      const fr = P.frame(ctx, w, h, 0, T, -10, ymax);
      P.axes(ctx, fr, {
        xlabel: 't', ylabel: 'log₁₀ ℒ',
        xticks: [0, Math.round(T / 2), T],
        yticks: [-10, -5, 0, 5],
      });
      ctx.save();
      ctx.beginPath();
      ctx.rect(fr.X(fr.xmin), fr.Y(fr.ymax), fr.iw, fr.ih);
      ctx.clip();
      const ks = lc.map((_, i) => i);
      P.line(ctx, fr, ks, lc, th.cyan, { width: 1.8 });
      P.line(ctx, fr, ks, ls, th.orange, { width: 1.8 });
      ctx.restore();
      P.label(ctx, fr.X(0) + 8, fr.Y(fr.ymax) + 14, 'η constante', th.cyan);
      P.label(ctx, fr.X(0) + 8, fr.Y(fr.ymax) + 28, 'warmup + cosine', th.orange);
    }

    function updateReadout() {
      const num = (v) => (isFinite(v) ? v.toExponential(1) : '∞');
      const cur = t < T ? eta(t) : eta(T - 1);
      $('s1-readout').textContent =
        't = ' + t + ' · η_t = ' + cur.toFixed(3) +
        ' · ℒ const = ' + num(lossC[lossC.length - 1]) +
        ' · ℒ schedule = ' + num(lossS[lossS.length - 1]);
    }

    function redraw() {
      drawSched();
      drawLoss();
      updateReadout();
    }

    function stopRun() {
      running = false;
      btnRun.textContent = '▶ Rodar';
      DL.stopTicker(tick);
    }

    function tick(dt) {
      acc += dt * Math.max(30, T / 5);
      while (acc >= 1 && t < T) { acc -= 1; stepOnce(); }
      redraw();
      if (t >= T) stopRun();
    }

    btnStep.addEventListener('click', () => { stepOnce(); redraw(); });
    btnRun.addEventListener('click', () => {
      if (running) { stopRun(); return; }
      if (t >= T) resetTraj();
      running = true;
      acc = 0;
      btnRun.textContent = '⏸ Pausar';
      DL.startTicker(tick);
    });
    btnReset.addEventListener('click', () => { stopRun(); resetTraj(); redraw(); });
    btnSeed.addEventListener('click', () => {
      stopRun();
      seed = (seed * 1664525 + 1013904223) >>> 0;
      resetTraj();
      redraw();
    });

    function bindSlider(sl, span, fmt, apply) {
      sl.addEventListener('input', () => {
        apply(+sl.value);
        $(span).textContent = fmt(+sl.value);
        stopRun();
        resetTraj();
        redraw();
      });
    }
    bindSlider(slEtaMax, 's1-etamax-val', (v) => v.toFixed(3), (v) => { etaMax = v; });
    bindSlider(slWarmup, 's1-warmup-val', (v) => String(v), (v) => { wuFrac = v / 100; });
    bindSlider(slEtaMin, 's1-etamin-val', (v) => v.toFixed(2), (v) => { etaMinRatio = v; });
    bindSlider(slT, 's1-T-val', (v) => String(v), (v) => { T = v; });

    resetTraj();
    redraw();
    P.onRedraw(redraw);
    P.observeResize(cvSched, redraw);
    P.observeResize(cvLoss, redraw);
  }

  DL.sections.push({ name: 's1-lr-schedule', init });
})();
