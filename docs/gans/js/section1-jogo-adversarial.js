/* Seção 1: o jogo adversarial no exemplo toy 1D dos slides.
   Dados reais ~ N(7,1); gerador x* = z + θ (um parâmetro); discriminador
   logístico σ(a·x + b) treinado em alternância (k passos de D, 1 de G).
   Tabs comparam a loss original do gerador com a non-saturating. */
(function () {
  'use strict';
  window.DL = window.DL || {};
  DL.sections = DL.sections || [];

  const MU_REAL = 7, XMIN = -7, XMAX = 15;
  const BATCH = 64, ETA_D = 0.04, ETA_G = 0.06, MAX_STEPS = 600;

  function init() {
    const P = DL.plot;
    const U = DL.utils;
    const $ = (id) => document.getElementById(id);

    const cvDist = $('s1-dist'), cvTheta = $('s1-theta');
    const slTheta0 = $('s1-theta0'), slK = $('s1-k');
    const btnStep = $('s1-step'), btnRun = $('s1-run');
    const btnReset = $('s1-reset'), btnDice = $('s1-dice');
    const tabOrig = $('s1-tab-orig'), tabNS = $('s1-tab-ns');

    let theta0 = +slTheta0.value, k = +slK.value;
    let nsLoss = true;
    let seed = 7;
    let rng, theta, a, b, hist, gradMag, accD, realBatch, fakeBatch;
    let running = false, acc = 0;
    let frDist = null, dragging = false;

    const sigmoid = (d) => 1 / (1 + Math.exp(-d));

    function sampleBatches() {
      realBatch = [];
      fakeBatch = [];
      for (let i = 0; i < BATCH; i++) {
        realBatch.push(MU_REAL + U.randn(rng));
        fakeBatch.push(theta + U.randn(rng));
      }
    }

    function reset() {
      rng = U.mulberry32(seed);
      theta = theta0;
      a = 0;
      b = 0;
      hist = [theta];
      gradMag = 0;
      accD = 0.5;
      sampleBatches();
    }

    function discStep() {
      sampleBatches();
      let ga = 0, gb = 0, hits = 0;
      for (const x of realBatch) {
        const s = sigmoid(a * x + b);
        ga += (s - 1) * x;
        gb += (s - 1);
        if (s > 0.5) hits++;
      }
      for (const x of fakeBatch) {
        const s = sigmoid(a * x + b);
        ga += s * x;
        gb += s;
        if (s < 0.5) hits++;
      }
      a -= ETA_D * ga / (2 * BATCH);
      b -= ETA_D * gb / (2 * BATCH);
      /* clamp suave: mantém o discriminador longe do regime degenerado */
      a = Math.max(-6, Math.min(6, a));
      b = Math.max(-60, Math.min(60, b));
      accD = hits / (2 * BATCH);
    }

    function genStep() {
      /* loss original: dL/dθ = média de −σ(d)·a (gradiente some quando σ≈0);
         non-saturating: dL/dθ = média de −(1−σ(d))·a. */
      let g = 0;
      for (let i = 0; i < BATCH; i++) {
        const x = theta + U.randn(rng);
        const s = sigmoid(a * x + b);
        g += nsLoss ? -(1 - s) * a : -s * a;
      }
      g /= BATCH;
      gradMag = Math.abs(g);
      theta -= ETA_G * g;
      theta = Math.max(XMIN + 1, Math.min(XMAX - 1, theta));
      hist.push(theta);
    }

    function stepOnce() {
      if (done()) return;
      for (let t = 0; t < k; t++) discStep();
      genStep();
    }

    function done() { return hist.length > MAX_STEPS; }

    function drawDist() {
      const th = P.theme();
      const { ctx, w, h } = P.setup(cvDist);
      P.clear(ctx, w, h);
      const YMAX = 0.55;
      const fr = P.frame(ctx, w, h, XMIN, XMAX, 0, YMAX);
      frDist = fr;
      P.axes(ctx, fr, { xlabel: 'x', ylabel: 'densidade', xticks: [-4, 0, 4, 7, 12], yticks: [0, 0.2, 0.4] });

      /* minibatches como histogramas */
      P.histogram(ctx, fr, realBatch, 36, th.cyan, { alpha: 0.18 });
      P.histogram(ctx, fr, fakeBatch, 36, th.orange, { alpha: 0.22 });

      /* densidades analíticas */
      const n = 240, xs = [], pr = [], pg = [], dd = [];
      for (let i = 0; i <= n; i++) {
        const x = XMIN + (XMAX - XMIN) * i / n;
        xs.push(x);
        pr.push(U.gaussPdf(x, MU_REAL, 1));
        pg.push(U.gaussPdf(x, theta, 1));
        dd.push(sigmoid(a * x + b) * YMAX);
      }
      P.line(ctx, fr, xs, pr, th.cyan, { width: 2 });
      P.line(ctx, fr, xs, pg, th.orange, { width: 2 });
      P.line(ctx, fr, xs, dd, th.purple, { width: 2, dash: [6, 3] });

      /* eixo da direita para a sigmóide (0 a 1) */
      P.label(ctx, fr.X(XMAX) - 4, fr.Y(YMAX) + 12, '1', th.purple, 'right');
      P.label(ctx, fr.X(XMAX) - 4, fr.Y(YMAX / 2) + 4, '0.5', th.purple, 'right');
      P.label(ctx, fr.X(XMAX) - 4, fr.Y(0) - 4, '0', th.purple, 'right');

      /* marcador arrastável de θ */
      ctx.beginPath();
      ctx.arc(fr.X(theta), fr.Y(0), 6, 0, 2 * Math.PI);
      ctx.fillStyle = th.orange;
      ctx.fill();
      ctx.strokeStyle = th.fg;
      ctx.lineWidth = 1.5;
      ctx.stroke();
      P.label(ctx, fr.X(theta) + 8, fr.Y(0) - 8, 'θ', th.orange);

      P.label(ctx, fr.X(XMIN) + 8, fr.Y(YMAX) + 14, 'P(x) real', th.cyan);
      P.label(ctx, fr.X(XMIN) + 8, fr.Y(YMAX) + 28, 'P*(x) gerada', th.orange);
      P.label(ctx, fr.X(XMIN) + 8, fr.Y(YMAX) + 42, 'σ(f[x,ϕ])', th.purple);
    }

    function drawTheta() {
      const th = P.theme();
      const { ctx, w, h } = P.setup(cvTheta);
      P.clear(ctx, w, h);
      const n = Math.max(40, hist.length - 1);
      const fr = P.frame(ctx, w, h, 0, n, -5, 13);
      P.axes(ctx, fr, { xlabel: 'passo', ylabel: 'θ', xticks: [0, Math.round(n / 2), n], yticks: [-4, 0, 7, 12] });
      P.line(ctx, fr, [0, n], [MU_REAL, MU_REAL], th.green, { width: 1.2, dash: [5, 4] });
      const ks = hist.map((_, i) => i);
      P.line(ctx, fr, ks, hist, th.orange, { width: 1.8 });
      P.scatter(ctx, fr, [[hist.length - 1, hist[hist.length - 1]]], th.orange, { r: 3.5, alpha: 1 });
      P.label(ctx, fr.X(n) - 4, fr.Y(MU_REAL) - 6, 'θ = 7', th.green, 'right');
    }

    function updateReadout() {
      $('s1-readout').textContent =
        'passo = ' + (hist.length - 1) +
        ' · θ = ' + theta.toFixed(2) +
        ' · |∂L/∂θ| = ' + gradMag.toExponential(1) +
        ' · acurácia de D = ' + (100 * accD).toFixed(0) + '%';
    }

    function redraw() {
      drawDist();
      drawTheta();
      updateReadout();
    }

    function stopRun() {
      running = false;
      btnRun.textContent = '▶ Rodar';
      DL.stopTicker(tick);
    }

    function tick(dt) {
      acc += dt * 25;
      while (acc >= 1) { acc -= 1; stepOnce(); }
      redraw();
      if (done()) stopRun();
    }

    cvDist.addEventListener('pointerdown', (e) => {
      if (!frDist) return;
      dragging = true;
      cvDist.setPointerCapture(e.pointerId);
      const r = cvDist.getBoundingClientRect();
      theta = Math.max(XMIN + 1, Math.min(XMAX - 1, frDist.invX(e.clientX - r.left)));
      hist.push(theta);
      sampleBatches();
      redraw();
    });
    cvDist.addEventListener('pointermove', (e) => {
      if (!dragging || !frDist) return;
      const r = cvDist.getBoundingClientRect();
      theta = Math.max(XMIN + 1, Math.min(XMAX - 1, frDist.invX(e.clientX - r.left)));
      hist[hist.length - 1] = theta;
      sampleBatches();
      redraw();
    });
    cvDist.addEventListener('pointerup', () => { dragging = false; });

    btnStep.addEventListener('click', () => { stepOnce(); redraw(); });
    btnRun.addEventListener('click', () => {
      if (running) { stopRun(); return; }
      if (done()) reset();
      running = true;
      acc = 0;
      btnRun.textContent = '⏸ Pausar';
      DL.startTicker(tick);
    });
    btnReset.addEventListener('click', () => { stopRun(); reset(); redraw(); });
    btnDice.addEventListener('click', () => {
      seed = (seed * 16807) % 2147483647;
      stopRun();
      reset();
      redraw();
    });

    slTheta0.addEventListener('input', () => {
      theta0 = +slTheta0.value;
      $('s1-theta0-val').textContent = theta0.toFixed(1);
      stopRun();
      reset();
      redraw();
    });
    slK.addEventListener('input', () => {
      k = +slK.value;
      $('s1-k-val').textContent = k;
    });

    function setLoss(ns) {
      nsLoss = ns;
      tabNS.classList.toggle('active', ns);
      tabOrig.classList.toggle('active', !ns);
    }
    tabOrig.addEventListener('click', () => setLoss(false));
    tabNS.addEventListener('click', () => setLoss(true));

    reset();
    redraw();
    P.onRedraw(redraw);
    P.observeResize(cvDist, redraw);
    P.observeResize(cvTheta, redraw);
  }

  DL.sections.push({ name: 's1-jogo-adversarial', init });
})();
