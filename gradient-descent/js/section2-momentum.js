/* Seção 2: vale alongado ℒ(θ₁, θ₂) = ½(θ₁² + c·θ₂²), zigue-zague do GD
   e o efeito do momentum (convenção dos slides: v ← βv + (1−β)∇ℒ). */
(function () {
  'use strict';
  window.DL = window.DL || {};
  DL.sections = DL.sections || [];

  const DOM = 2.2, MAX_ITERS = 200;

  function init() {
    const P = DL.plot;
    const $ = (id) => document.getElementById(id);

    const cvSurf = $('s2-surface'), cvLoss = $('s2-loss');
    const slC = $('s2-c'), slEta = $('s2-eta'), slBeta = $('s2-beta');
    const btnStep = $('s2-step'), btnRun = $('s2-run'), btnReset = $('s2-reset');

    let c = +slC.value, eta = +slEta.value, beta = +slBeta.value;
    let start = [-1.9, 1.5];
    let gd, mo, vel, lossGD, lossMo;
    let running = false, acc = 0;
    let frSurf = null, dragging = false;

    const L = (p) => 0.5 * (p[0] * p[0] + c * p[1] * p[1]);

    function resetTraj() {
      gd = [start.slice()];
      mo = [start.slice()];
      vel = [0, 0];
      lossGD = [L(start)];
      lossMo = [L(start)];
    }

    function stepOnce() {
      if (done()) return;
      /* GD puro: ∇ℒ = (θ₁, c·θ₂) */
      const p = gd[gd.length - 1];
      const pn = [p[0] - eta * p[0], p[1] - eta * c * p[1]];
      gd.push(pn);
      lossGD.push(L(pn));
      /* Momentum: v ← βv + (1−β)∇ℒ; θ ← θ − ηv */
      const q = mo[mo.length - 1];
      vel = [beta * vel[0] + (1 - beta) * q[0], beta * vel[1] + (1 - beta) * c * q[1]];
      const qn = [q[0] - eta * vel[0], q[1] - eta * vel[1]];
      mo.push(qn);
      lossMo.push(L(qn));
    }

    function done() {
      if (gd.length > MAX_ITERS) return true;
      const a = lossGD[lossGD.length - 1], b = lossMo[lossMo.length - 1];
      const settled = (v) => !isFinite(v) || v < 1e-11 || v > 1e8;
      return settled(a) && settled(b);
    }

    function clipPlot(ctx, fr) {
      ctx.save();
      ctx.beginPath();
      ctx.rect(fr.X(fr.xmin), fr.Y(fr.ymax), fr.iw, fr.ih);
      ctx.clip();
    }

    /* Pontos divergidos são truncados para fora da janela em vez de virar NaN no path */
    const clamp = (v) => (isFinite(v) ? Math.max(-50, Math.min(50, v)) : 50);

    function drawTraj(ctx, fr, traj, color) {
      const xs = traj.map((p) => clamp(p[0]));
      const ys = traj.map((p) => clamp(p[1]));
      P.line(ctx, fr, xs, ys, color, { width: 1.8, alpha: 0.9 });
      P.scatter(ctx, fr, xs.map((x, i) => [x, ys[i]]), color, { r: 2.2, alpha: 0.9 });
    }

    function drawSurface() {
      const th = P.theme();
      const { ctx, w, h } = P.setup(cvSurf);
      P.clear(ctx, w, h);
      const fr = P.frame(ctx, w, h, -DOM, DOM, -DOM, DOM);
      frSurf = fr;
      const fmax = 0.5 * DOM * DOM * (1 + c);
      P.heatmap(ctx, fr, (x, y) => 0.5 * (x * x + c * y * y), 150, 150, (v) => {
        const u = Math.pow(Math.min(1, v / fmax), 0.3);
        const col = P.viridis(u);
        const lift = (u * 9) % 1 < 0.13 ? 42 : 0;
        return [Math.min(255, col[0] + lift), Math.min(255, col[1] + lift), Math.min(255, col[2] + lift), 235];
      });
      P.axes(ctx, fr, { xlabel: 'θ₁', ylabel: 'θ₂', xticks: [-2, 0, 2], yticks: [-2, 0, 2] });

      clipPlot(ctx, fr);
      drawTraj(ctx, fr, gd, th.cyan);
      drawTraj(ctx, fr, mo, th.orange);
      ctx.restore();

      /* Mínimo global em (0, 0) */
      P.scatter(ctx, fr, [[0, 0]], th.green, { r: 4, alpha: 1 });
      P.label(ctx, fr.X(0) + 8, fr.Y(0) + 4, 'mínimo', th.green);
      /* Marcador do ponto inicial (arrastável) */
      ctx.beginPath();
      ctx.arc(fr.X(start[0]), fr.Y(start[1]), 6, 0, 2 * Math.PI);
      ctx.strokeStyle = th.fg;
      ctx.lineWidth = 2;
      ctx.stroke();
      P.label(ctx, fr.X(start[0]) + 9, fr.Y(start[1]) - 8, 'θ_0', th.fg);

      P.label(ctx, fr.X(-DOM) + 8, fr.Y(DOM) + 14, 'GD', th.cyan);
      P.label(ctx, fr.X(-DOM) + 8, fr.Y(DOM) + 28, 'momentum', th.orange);
    }

    function drawLoss() {
      const th = P.theme();
      const { ctx, w, h } = P.setup(cvLoss);
      P.clear(ctx, w, h);
      const n = Math.max(20, lossGD.length - 1);
      const logv = (v) => (isFinite(v) && v > 0 ? Math.min(9, Math.log10(Math.max(v, 1e-12))) : (v === 0 ? -12 : 9));
      const lg = lossGD.map(logv), lm = lossMo.map(logv);
      const ymax = Math.min(10, Math.ceil(Math.max(2, ...lg, ...lm)) + 1);
      const fr = P.frame(ctx, w, h, 0, n, -12, ymax);
      P.axes(ctx, fr, {
        xlabel: 'k', ylabel: 'log₁₀ ℒ',
        xticks: [0, Math.round(n / 2), n],
        yticks: [-12, -6, 0, 6],
      });
      clipPlot(ctx, fr);
      const ks = lg.map((_, i) => i);
      P.line(ctx, fr, ks, lg, th.cyan, { width: 1.8 });
      P.line(ctx, fr, ks, lm, th.orange, { width: 1.8 });
      ctx.restore();
    }

    function updateReadout() {
      const a = lossGD[lossGD.length - 1], b = lossMo[lossMo.length - 1];
      const num = (v) => (isFinite(v) ? v.toExponential(1) : '∞');
      $('s2-readout').textContent =
        'k = ' + (gd.length - 1) + ' · ℒ GD = ' + num(a) + ' · ℒ momentum = ' + num(b);
    }

    function redraw() {
      drawSurface();
      drawLoss();
      updateReadout();
    }

    function stopRun() {
      running = false;
      btnRun.textContent = '▶ Rodar';
      DL.stopTicker(tick);
    }

    function tick(dt) {
      acc += dt * 12;
      while (acc >= 1) { acc -= 1; stepOnce(); }
      redraw();
      if (done()) stopRun();
    }

    function setStart(x, y) {
      const lim = DOM - 0.05;
      start = [Math.max(-lim, Math.min(lim, x)), Math.max(-lim, Math.min(lim, y))];
      resetTraj();
      redraw();
    }

    cvSurf.addEventListener('pointerdown', (e) => {
      if (!frSurf) return;
      dragging = true;
      cvSurf.setPointerCapture(e.pointerId);
      stopRun();
      const r = cvSurf.getBoundingClientRect();
      setStart(frSurf.invX(e.clientX - r.left), frSurf.invY(e.clientY - r.top));
    });
    cvSurf.addEventListener('pointermove', (e) => {
      if (!dragging || !frSurf) return;
      const r = cvSurf.getBoundingClientRect();
      setStart(frSurf.invX(e.clientX - r.left), frSurf.invY(e.clientY - r.top));
    });
    cvSurf.addEventListener('pointerup', () => { dragging = false; });

    btnStep.addEventListener('click', () => { stepOnce(); redraw(); });
    btnRun.addEventListener('click', () => {
      if (running) { stopRun(); return; }
      if (done()) resetTraj();
      running = true;
      acc = 0;
      btnRun.textContent = '⏸ Pausar';
      DL.startTicker(tick);
    });
    btnReset.addEventListener('click', () => { stopRun(); resetTraj(); redraw(); });

    slC.addEventListener('input', () => {
      c = +slC.value;
      $('s2-c-val').textContent = c;
      resetTraj();
      redraw();
    });
    slEta.addEventListener('input', () => {
      eta = +slEta.value;
      $('s2-eta-val').textContent = eta.toFixed(3);
      resetTraj();
      redraw();
    });
    slBeta.addEventListener('input', () => {
      beta = +slBeta.value;
      $('s2-beta-val').textContent = beta.toFixed(2);
      resetTraj();
      redraw();
    });

    resetTraj();
    redraw();
    P.onRedraw(redraw);
    P.observeResize(cvSurf, redraw);
    P.observeResize(cvLoss, redraw);
  }

  DL.sections.push({ name: 's2-momentum', init });
})();
