/* Seção 4: vanishing gradient da Jensen-Shannon vs distância de Wasserstein.
   P = N(0,1) fixa, P* = N(d,1). Para esse par, W(d) = d exatamente, enquanto
   D_JS(d) satura em ln 2. Tabs treinam d por descida de gradiente sobre cada
   distância: a JS congela quando os suportes se separam, a W desliza sempre. */
(function () {
  'use strict';
  window.DL = window.DL || {};
  DL.sections = DL.sections || [];

  const XMIN = -6, XMAX = 18, NGRID = 600;
  const D_MAX = 12, LN2 = Math.LN2, MAX_STEPS = 600;

  function init() {
    const P = DL.plot;
    const U = DL.utils;
    const $ = (id) => document.getElementById(id);

    const cvDist = $('s4-dist'), cvCurves = $('s4-curves'), cvGrad = $('s4-grad');
    const slD = $('s4-d');
    const btnStep = $('s4-step'), btnRun = $('s4-run'), btnReset = $('s4-reset');
    const tabJS = $('s4-tab-js'), tabW = $('s4-tab-w');

    let d = +slD.value;
    let mode = 'js';
    let running = false, accT = 0, steps = 0;
    let frDist = null, dragging = false;

    const DX = (XMAX - XMIN) / NGRID;
    const xs = new Float64Array(NGRID);
    const pReal = new Float64Array(NGRID);
    for (let i = 0; i < NGRID; i++) {
      xs[i] = XMIN + (i + 0.5) * DX;
      pReal[i] = U.gaussPdf(xs[i], 0, 1);
    }

    function jsAt(dd) {
      let acc = 0;
      for (let i = 0; i < NGRID; i++) {
        const p = pReal[i];
        const q = U.gaussPdf(xs[i], dd, 1);
        const m = 0.5 * (p + q);
        if (q > 1e-300 && m > 1e-300) acc += q * Math.log(q / m);
        if (p > 1e-300 && m > 1e-300) acc += p * Math.log(p / m);
      }
      return 0.5 * acc * DX;
    }

    function jsGrad(dd) {
      const h = 0.05;
      return (jsAt(dd + h) - jsAt(Math.max(0, dd - h))) / (dd - h < 0 ? h + dd : 2 * h);
    }

    /* curvas pré-computadas: D_JS(d) e |dD_JS/dd| */
    const ND = 140;
    const dsAxis = [], jsVals = [], jsGradVals = [];
    for (let i = 0; i <= ND; i++) {
      const dd = D_MAX * i / ND;
      dsAxis.push(dd);
      jsVals.push(jsAt(dd));
    }
    for (let i = 0; i <= ND; i++) {
      const a = jsVals[Math.max(0, i - 1)], b = jsVals[Math.min(ND, i + 1)];
      const span = (Math.min(ND, i + 1) - Math.max(0, i - 1)) * D_MAX / ND;
      jsGradVals.push(Math.abs((b - a) / span));
    }

    function trainStep() {
      if (steps >= MAX_STEPS || d <= 1e-3) return false;
      const g = mode === 'w' ? 1 : jsGrad(d);
      d = Math.max(0, d - 0.15 * g);
      steps++;
      syncSlider();
      return true;
    }

    function syncSlider() {
      slD.value = d;
      $('s4-d-val').textContent = d.toFixed(1);
    }

    function drawDist() {
      const th = P.theme();
      const { ctx, w, h } = P.setup(cvDist);
      P.clear(ctx, w, h);
      const YMAX = 0.5;
      const fr = P.frame(ctx, w, h, XMIN, XMAX, 0, YMAX);
      frDist = fr;
      P.axes(ctx, fr, { xlabel: 'x', ylabel: 'densidade', xticks: [-4, 0, 4, 8, 12, 16], yticks: [0, 0.2, 0.4] });

      const n = 280, gx = [], gp = [], gq = [], gd = [], gf = [];
      for (let i = 0; i <= n; i++) {
        const x = XMIN + (XMAX - XMIN) * i / n;
        const p = U.gaussPdf(x, 0, 1);
        const q = U.gaussPdf(x, d, 1);
        gx.push(x);
        gp.push(p);
        gq.push(q);
        gd.push((p + q > 1e-300 ? p / (p + q) : 0.5) * YMAX);
        /* crítico ótimo 1-Lipschitz: reta de inclinação −1, em escala própria */
        gf.push(Math.max(0.02, Math.min(0.98, 0.5 + (d / 2 - x) / 30)) * YMAX);
      }
      P.line(ctx, fr, gx, gp, th.cyan, { width: 2 });
      P.line(ctx, fr, gx, gq, th.orange, { width: 2 });
      P.line(ctx, fr, gx, gd, th.purple, { width: 2, dash: [6, 3] });
      P.line(ctx, fr, gx, gf, th.green, { width: 2 });

      ctx.beginPath();
      ctx.arc(fr.X(d), fr.Y(0), 6, 0, 2 * Math.PI);
      ctx.fillStyle = th.orange;
      ctx.fill();
      ctx.strokeStyle = th.fg;
      ctx.lineWidth = 1.5;
      ctx.stroke();
      P.label(ctx, fr.X(d) + 8, fr.Y(0) - 8, 'd', th.orange);

      P.label(ctx, fr.X(XMIN) + 8, fr.Y(YMAX) + 14, 'P(x) real', th.cyan);
      P.label(ctx, fr.X(XMIN) + 8, fr.Y(YMAX) + 28, 'P*(x) gerada', th.orange);
      P.label(ctx, fr.X(XMIN) + 8, fr.Y(YMAX) + 42, 'D*(x) saturado', th.purple);
      P.label(ctx, fr.X(XMIN) + 8, fr.Y(YMAX) + 56, 'crítico f(x), |∂f/∂x| = 1', th.green);
    }

    function drawCurves() {
      const th = P.theme();
      const { ctx, w, h } = P.setup(cvCurves);
      P.clear(ctx, w, h);
      const YMAX = 0.8;
      const fr = P.frame(ctx, w, h, 0, D_MAX, 0, YMAX);
      P.axes(ctx, fr, { xlabel: 'd', ylabel: 'D_{JS} (nats)', xticks: [0, 4, 8, 12], yticks: [0, 0.35, 0.69] });
      P.line(ctx, fr, [0, D_MAX], [LN2, LN2], th.comment, { width: 1.2, dash: [5, 4] });
      P.label(ctx, fr.X(D_MAX) - 4, fr.Y(LN2) - 6, 'ln 2', th.comment, 'right');
      P.line(ctx, fr, dsAxis, jsVals, th.purple, { width: 2 });
      /* W(d) = d, escala própria à direita (0 a 12) */
      const wScaled = dsAxis.map((v) => v / D_MAX * YMAX);
      P.line(ctx, fr, dsAxis, wScaled, th.green, { width: 2 });
      P.label(ctx, fr.X(D_MAX) - 4, fr.Y(YMAX) + 12, '12', th.green, 'right');
      P.label(ctx, fr.X(D_MAX) - 4, fr.Y(0) - 16, '0', th.green, 'right');
      P.label(ctx, fr.X(0.4), fr.Y(LN2) + 26, 'D_{JS}(d)', th.purple);
      P.label(ctx, fr.X(D_MAX / 2) + 6, fr.Y(YMAX / 2) + 12, 'W(d) = d', th.green);
      P.scatter(ctx, fr, [[d, jsAt(d)]], th.purple, { r: 4.5, alpha: 1 });
      P.scatter(ctx, fr, [[d, d / D_MAX * YMAX]], th.green, { r: 4.5, alpha: 1 });
    }

    function drawGrad() {
      const th = P.theme();
      const { ctx, w, h } = P.setup(cvGrad);
      P.clear(ctx, w, h);
      const fr = P.frame(ctx, w, h, 0, D_MAX, 0, 1.15);
      P.axes(ctx, fr, { xlabel: 'd', ylabel: '|∂/∂d|', xticks: [0, 4, 8, 12], yticks: [0, 0.5, 1] });
      P.line(ctx, fr, [0, D_MAX], [1, 1], th.green, { width: 2 });
      P.line(ctx, fr, dsAxis, jsGradVals, th.purple, { width: 2 });
      P.label(ctx, fr.X(0.4), fr.Y(1) - 8, 'Wasserstein: |∂W/∂d| = 1', th.green);
      P.label(ctx, fr.X(D_MAX / 2), fr.Y(0.32), 'JS: |∂D_{JS}/∂d|', th.purple);
      const gNow = mode === 'w' ? 1 : Math.abs(jsGrad(d));
      P.scatter(ctx, fr, [[d, Math.min(1.1, gNow)]], mode === 'w' ? th.green : th.purple, { r: 4.5, alpha: 1 });
    }

    function updateReadout() {
      const g = mode === 'w' ? 1 : Math.abs(jsGrad(d));
      $('s4-readout').textContent =
        'passo = ' + steps +
        ' · d = ' + d.toFixed(2) +
        ' · D_JS = ' + jsAt(d).toFixed(3) + ' nats' +
        ' · W = ' + d.toFixed(2) +
        ' · |∇| (' + (mode === 'w' ? 'W' : 'JS') + ') = ' + g.toExponential(1);
    }

    function redraw() {
      drawDist();
      drawCurves();
      drawGrad();
      updateReadout();
    }

    function stopRun() {
      running = false;
      btnRun.textContent = '▶ Treinar';
      DL.stopTicker(tick);
    }

    function tick(dt) {
      accT += dt * 20;
      let alive = true;
      while (accT >= 1) { accT -= 1; alive = trainStep(); }
      redraw();
      if (!alive) stopRun();
    }

    function setMode(m) {
      mode = m;
      tabJS.classList.toggle('active', m === 'js');
      tabW.classList.toggle('active', m === 'w');
      redraw();
    }
    tabJS.addEventListener('click', () => setMode('js'));
    tabW.addEventListener('click', () => setMode('w'));

    function setD(v) {
      d = Math.max(0, Math.min(D_MAX, v));
      syncSlider();
      redraw();
    }

    cvDist.addEventListener('pointerdown', (e) => {
      if (!frDist) return;
      dragging = true;
      cvDist.setPointerCapture(e.pointerId);
      stopRun();
      const r = cvDist.getBoundingClientRect();
      setD(frDist.invX(e.clientX - r.left));
    });
    cvDist.addEventListener('pointermove', (e) => {
      if (!dragging || !frDist) return;
      const r = cvDist.getBoundingClientRect();
      setD(frDist.invX(e.clientX - r.left));
    });
    cvDist.addEventListener('pointerup', () => { dragging = false; });

    btnStep.addEventListener('click', () => { trainStep(); redraw(); });
    btnRun.addEventListener('click', () => {
      if (running) { stopRun(); return; }
      running = true;
      accT = 0;
      steps = 0;
      btnRun.textContent = '⏸ Pausar';
      DL.startTicker(tick);
    });
    btnReset.addEventListener('click', () => {
      stopRun();
      steps = 0;
      setD(8);
    });

    slD.addEventListener('input', () => {
      d = +slD.value;
      $('s4-d-val').textContent = d.toFixed(1);
      redraw();
    });

    redraw();
    P.onRedraw(redraw);
    P.observeResize(cvDist, redraw);
    P.observeResize(cvCurves, redraw);
    P.observeResize(cvGrad, redraw);
  }

  DL.sections.push({ name: 's4-wasserstein', init });
})();
