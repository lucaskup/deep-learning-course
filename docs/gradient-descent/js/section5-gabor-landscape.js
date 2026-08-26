/* Seção 5: paisagem completa L(θ₀, θ₁) do modelo de Gabor (DL.gaborDemo,
   definido na seção 4) e descida de gradiente a partir de um ponto arrastável. */
(function () {
  'use strict';
  window.DL = window.DL || {};
  DL.sections = DL.sections || [];

  const MAX_ITERS = 300, GRAD_TOL = 2e-3, RES = 160;

  function init() {
    const P = DL.plot;
    const $ = (id) => document.getElementById(id);
    const G = DL.gaborDemo;

    const cvSurf = $('s5-surface'), cvData = $('s5-data');
    const slEta = $('s5-eta');
    const btnStep = $('s5-step'), btnRun = $('s5-run'), btnReset = $('s5-reset');

    let eta = +slEta.value;
    let start = [-4, 18];
    let traj;
    let running = false, acc = 0;
    let frSurf = null, dragging = false;

    /* A paisagem é fixa (dataset fixo) e o colormap viridis não depende do
       tema, então o heatmap é rasterizado uma única vez num canvas offscreen
       e apenas redesenhado escalado a cada frame. */
    const surf = document.createElement('canvas');
    surf.width = RES; surf.height = RES;
    (function buildSurface() {
      let lmin = Infinity, lmax = -Infinity;
      const vals = new Float64Array(RES * RES);
      for (let j = 0; j < RES; j++) {
        const t1 = G.T1MAX - (j + 0.5) / RES * (G.T1MAX - G.T1MIN);
        for (let i = 0; i < RES; i++) {
          const t0 = G.T0MIN + (i + 0.5) / RES * (G.T0MAX - G.T0MIN);
          const v = G.loss(t0, t1);
          vals[j * RES + i] = v;
          if (v < lmin) lmin = v;
          if (v > lmax) lmax = v;
        }
      }
      const octx = surf.getContext('2d');
      const img = octx.createImageData(RES, RES);
      for (let k = 0; k < RES * RES; k++) {
        const u = Math.pow((vals[k] - lmin) / (lmax - lmin), 0.45);
        const col = P.viridis(u);
        const lift = (u * 11) % 1 < 0.12 ? 42 : 0;
        const p = 4 * k;
        img.data[p] = Math.min(255, col[0] + lift);
        img.data[p + 1] = Math.min(255, col[1] + lift);
        img.data[p + 2] = Math.min(255, col[2] + lift);
        img.data[p + 3] = 235;
      }
      octx.putImageData(img, 0, 0);
    })();

    const cur = () => traj[traj.length - 1];

    function resetTraj() {
      traj = [start.slice()];
    }

    function stepOnce() {
      if (done()) return;
      const p = cur();
      const g = G.grad(p[0], p[1]);
      traj.push([p[0] - eta * g[0], p[1] - eta * g[1]]);
    }

    function done() {
      if (traj.length > MAX_ITERS) return true;
      const p = cur();
      const g = G.grad(p[0], p[1]);
      return Math.hypot(g[0], g[1]) < GRAD_TOL;
    }

    function clipPlot(ctx, fr) {
      ctx.save();
      ctx.beginPath();
      ctx.rect(fr.X(fr.xmin), fr.Y(fr.ymax), fr.iw, fr.ih);
      ctx.clip();
    }

    /* Pontos que derivam para fora da janela são truncados para o path não degenerar. */
    const clamp = (v) => (isFinite(v) ? Math.max(-60, Math.min(60, v)) : 60);

    function drawSurface() {
      const th = P.theme();
      const { ctx, w, h } = P.setup(cvSurf);
      P.clear(ctx, w, h);
      const fr = P.frame(ctx, w, h, G.T0MIN, G.T0MAX, G.T1MIN, G.T1MAX);
      frSurf = fr;
      ctx.imageSmoothingEnabled = true;
      ctx.drawImage(surf, fr.X(fr.xmin), fr.Y(fr.ymax), fr.iw, fr.ih);
      P.axes(ctx, fr, { xlabel: 'θ_0', ylabel: 'θ_1', xticks: [-10, 0, 10], yticks: [2.5, 12.5, 22.5] });

      clipPlot(ctx, fr);
      const xsT = traj.map((p) => clamp(p[0]));
      const ysT = traj.map((p) => clamp(p[1]));
      P.line(ctx, fr, xsT, ysT, th.cyan, { width: 1.8, alpha: 0.9 });
      P.scatter(ctx, fr, xsT.map((x, i) => [x, ysT[i]]), th.cyan, { r: 2.2, alpha: 0.9 });
      ctx.restore();

      P.scatter(ctx, fr, [G.best], th.green, { r: 4, alpha: 1 });
      P.label(ctx, fr.X(G.best[0]) + 8, fr.Y(G.best[1]) + 4, 'mínimo global', th.green);
      ctx.beginPath();
      ctx.arc(fr.X(start[0]), fr.Y(start[1]), 6, 0, 2 * Math.PI);
      ctx.strokeStyle = th.fg;
      ctx.lineWidth = 2;
      ctx.stroke();
      P.label(ctx, fr.X(start[0]) + 9, fr.Y(start[1]) - 8, 'início', th.fg);
    }

    function drawData() {
      const th = P.theme();
      const { ctx, w, h } = P.setup(cvData);
      P.clear(ctx, w, h);
      const fr = P.frame(ctx, w, h, G.XMIN, G.XMAX, -1.35, 1.35);
      P.axes(ctx, fr, { xlabel: 'x', ylabel: 'y', xticks: [-10, 0, 10], yticks: [-1, 0, 1] });
      P.scatter(ctx, fr, G.xs.map((x, i) => [x, G.ys[i]]), th.orange, { r: 3.4, alpha: 0.85 });
      const bx = [], by = [], cx = [], cy = [];
      const p = cur();
      for (let i = 0; i <= 240; i++) {
        const x = G.XMIN + (G.XMAX - G.XMIN) * i / 240;
        bx.push(x);
        by.push(G.gabor(x, G.best[0], G.best[1]));
        cx.push(x);
        cy.push(G.gabor(x, p[0], p[1]));
      }
      P.line(ctx, fr, bx, by, th.comment, { width: 1.4, dash: [5, 4], alpha: 0.8 });
      P.line(ctx, fr, cx, cy, th.cyan, { width: 2 });
    }

    function updateReadout() {
      const p = cur();
      $('s5-readout').textContent =
        'k = ' + (traj.length - 1) +
        ' · θ = (' + p[0].toFixed(2) + ', ' + p[1].toFixed(2) + ')' +
        ' · L = ' + G.loss(p[0], p[1]).toFixed(3);
    }

    function redraw() {
      drawSurface();
      drawData();
      updateReadout();
    }

    function stopRun() {
      running = false;
      btnRun.textContent = '▶ Rodar';
      DL.stopTicker(tick);
    }

    function tick(dt) {
      acc += dt * 20;
      while (acc >= 1) { acc -= 1; stepOnce(); }
      redraw();
      if (done()) stopRun();
    }

    function setStart(x, y) {
      start = [
        Math.max(G.T0MIN, Math.min(G.T0MAX, x)),
        Math.max(G.T1MIN, Math.min(G.T1MAX, y)),
      ];
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

    slEta.addEventListener('input', () => {
      eta = +slEta.value;
      $('s5-eta-val').textContent = eta.toFixed(1);
    });

    resetTraj();
    redraw();
    P.onRedraw(redraw);
    P.observeResize(cvSurf, redraw);
    P.observeResize(cvData, redraw);
  }

  DL.sections.push({ name: 's5-gabor-landscape', init });
})();
