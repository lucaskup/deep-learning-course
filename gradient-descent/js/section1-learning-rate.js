/* Seção 1: taxa de aprendizado η em uma função 1D não convexa.
   f(θ) = (θ⁴ − 3θ² + θ)/4 tem mínimo global em θ ≈ −1.30 e local em θ ≈ 1.13. */
(function () {
  'use strict';
  window.DL = window.DL || {};
  DL.sections = DL.sections || [];

  const XMIN = -2.5, XMAX = 2.5, YMIN = -1.3, YMAX = 6;
  const MAX_STEPS = 300, TRAIL = 40;

  function f(x) { return (x * x * x * x - 3 * x * x + x) / 4; }
  function df(x) { return x * x * x - 1.5 * x + 0.25; }

  function init() {
    const P = DL.plot;
    const $ = (id) => document.getElementById(id);

    const cvFn = $('s1-fn'), cvLoss = $('s1-loss');
    const slEta = $('s1-eta');
    const btnStep = $('s1-step'), btnRun = $('s1-run'), btnReset = $('s1-reset');

    /* Slider em escala log: η ∈ [0.001, 1.2] */
    const sliderToEta = (v) => 0.001 * Math.pow(1200, v / 100);

    let eta = sliderToEta(+slEta.value);
    let x0 = 2.3;
    let hist = [x0];
    let diverged = false;
    let running = false, acc = 0;
    let frFn = null, dragging = false;

    const last = () => hist[hist.length - 1];

    function stepOnce() {
      if (diverged || hist.length > MAX_STEPS) return;
      const x = last();
      const xn = x - eta * df(x);
      hist.push(xn);
      if (!isFinite(xn) || Math.abs(xn) > 1e3) diverged = true;
    }

    function converged() {
      return !diverged && isFinite(last()) && Math.abs(df(last())) < 1e-7;
    }

    function clipPlot(ctx, fr) {
      ctx.save();
      ctx.beginPath();
      ctx.rect(fr.X(fr.xmin), fr.Y(fr.ymax), fr.iw, fr.ih);
      ctx.clip();
    }

    function arrowSeg(ctx, ax, ay, bx, by, color, alpha) {
      ctx.strokeStyle = color;
      ctx.fillStyle = color;
      ctx.globalAlpha = alpha;
      ctx.lineWidth = 1.5;
      ctx.beginPath();
      ctx.moveTo(ax, ay);
      ctx.lineTo(bx, by);
      ctx.stroke();
      if (Math.hypot(bx - ax, by - ay) > 7) {
        const ang = Math.atan2(by - ay, bx - ax), hl = 5;
        ctx.beginPath();
        ctx.moveTo(bx, by);
        ctx.lineTo(bx - hl * Math.cos(ang - 0.45), by - hl * Math.sin(ang - 0.45));
        ctx.lineTo(bx - hl * Math.cos(ang + 0.45), by - hl * Math.sin(ang + 0.45));
        ctx.closePath();
        ctx.fill();
      }
      ctx.globalAlpha = 1;
    }

    function drawFn() {
      const th = P.theme();
      const { ctx, w, h } = P.setup(cvFn);
      P.clear(ctx, w, h);
      const fr = P.frame(ctx, w, h, XMIN, XMAX, YMIN, YMAX);
      frFn = fr;
      P.axes(ctx, fr, { xlabel: 'θ', ylabel: 'f(θ)', xticks: [-2, -1, 0, 1, 2], yticks: [0, 2, 4, 6] });

      const xs = [], ys = [];
      for (let x = XMIN; x <= XMAX + 1e-9; x += 0.02) { xs.push(x); ys.push(f(x)); }
      P.line(ctx, fr, xs, ys, th.purple, { width: 2.2 });

      clipPlot(ctx, fr);
      /* Rastro com setas e alpha crescente (passos antigos esmaecem) */
      const lo = Math.max(0, hist.length - 1 - TRAIL);
      for (let i = lo; i < hist.length - 1; i++) {
        const a = 0.15 + 0.75 * (i - lo + 1) / (hist.length - lo);
        arrowSeg(ctx, fr.X(hist[i]), fr.Y(f(hist[i])), fr.X(hist[i + 1]), fr.Y(f(hist[i + 1])), th.cyan, a);
      }
      const xc = last();
      if (isFinite(xc)) {
        /* Tangente no ponto atual */
        const g = df(xc), t0 = xc - 0.65, t1 = xc + 0.65;
        P.line(ctx, fr, [t0, t1], [f(xc) + g * (t0 - xc), f(xc) + g * (t1 - xc)],
          th.orange, { dash: [5, 4], width: 1.6 });
        /* Bola no iterado atual (arrastável) */
        ctx.beginPath();
        ctx.arc(fr.X(xc), fr.Y(f(xc)), 7, 0, 2 * Math.PI);
        ctx.fillStyle = th.pink;
        ctx.fill();
        ctx.lineWidth = 1.5;
        ctx.strokeStyle = th.bg;
        ctx.stroke();
      }
      ctx.restore();

      P.label(ctx, fr.X(XMIN) + 8, fr.Y(YMAX) + 14, 'f(θ)', th.purple);
      P.label(ctx, fr.X(XMIN) + 8, fr.Y(YMAX) + 28, 'tangente f′(θ_k)', th.orange);
      if (diverged) P.label(ctx, fr.X(0), fr.Y(YMAX) + 18, 'divergiu!', th.red, 'center');
    }

    function drawLoss() {
      const th = P.theme();
      const { ctx, w, h } = P.setup(cvLoss);
      P.clear(ctx, w, h);
      const n = hist.length - 1;
      const kmax = Math.max(20, n);
      const vals = hist.map((x) => f(x));
      let vmax = 0;
      for (const v of vals) if (isFinite(v)) vmax = Math.max(vmax, Math.min(v, 50));
      const ymax = Math.max(1, Math.min(8, vmax * 1.08));
      const fr = P.frame(ctx, w, h, 0, kmax, -1.2, ymax);
      P.axes(ctx, fr, {
        xlabel: 'k', ylabel: 'f(θ_k)',
        xticks: [0, Math.round(kmax / 2), kmax],
        yticks: [0, Math.round(ymax * 10) / 10],
      });
      clipPlot(ctx, fr);
      const ks = vals.map((_, i) => i);
      const safe = vals.map((v) => (isFinite(v) ? v : ymax * 2));
      P.line(ctx, fr, ks, safe, th.cyan, { width: 1.8 });
      P.scatter(ctx, fr, safe.map((v, i) => [i, v]), th.cyan, { r: 2, alpha: 0.9 });
      ctx.restore();
    }

    function updateReadout() {
      const xc = last(), k = hist.length - 1;
      const num = (v) => (isFinite(v) ? v.toFixed(3) : '∞');
      $('s1-readout').textContent =
        'k = ' + k + ' · θ = ' + num(xc) + ' · f(θ) = ' + num(f(xc)) +
        (diverged ? ' · divergiu' : '');
    }

    function redraw() {
      drawFn();
      drawLoss();
      updateReadout();
    }

    function stopRun() {
      running = false;
      btnRun.textContent = '▶ Rodar';
      DL.stopTicker(tick);
    }

    function tick(dt) {
      acc += dt * 10;
      while (acc >= 1) { acc -= 1; stepOnce(); }
      redraw();
      if (diverged || hist.length > MAX_STEPS || converged()) stopRun();
    }

    function setStart(x) {
      x0 = Math.min(XMAX - 0.05, Math.max(XMIN + 0.05, x));
      hist = [x0];
      diverged = false;
      redraw();
    }

    cvFn.addEventListener('pointerdown', (e) => {
      if (!frFn) return;
      dragging = true;
      cvFn.setPointerCapture(e.pointerId);
      stopRun();
      const r = cvFn.getBoundingClientRect();
      setStart(frFn.invX(e.clientX - r.left));
    });
    cvFn.addEventListener('pointermove', (e) => {
      if (!dragging || !frFn) return;
      const r = cvFn.getBoundingClientRect();
      setStart(frFn.invX(e.clientX - r.left));
    });
    cvFn.addEventListener('pointerup', () => { dragging = false; });

    btnStep.addEventListener('click', () => { stepOnce(); redraw(); });
    btnRun.addEventListener('click', () => {
      if (running) { stopRun(); return; }
      if (diverged || hist.length > MAX_STEPS || converged()) { hist = [x0]; diverged = false; }
      running = true;
      acc = 0;
      btnRun.textContent = '⏸ Pausar';
      DL.startTicker(tick);
    });
    btnReset.addEventListener('click', () => {
      stopRun();
      hist = [x0];
      diverged = false;
      redraw();
    });
    slEta.addEventListener('input', () => {
      eta = sliderToEta(+slEta.value);
      $('s1-eta-val').textContent = eta.toFixed(3);
    });

    $('s1-eta-val').textContent = eta.toFixed(3);
    redraw();
    P.onRedraw(redraw);
    P.observeResize(cvFn, redraw);
    P.observeResize(cvLoss, redraw);
  }

  DL.sections.push({ name: 's1-learning-rate', init });
})();
