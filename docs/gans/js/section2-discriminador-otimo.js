/* Seção 2: discriminador ótimo D*(x) = P/(P + P*), mistura M = ½(P + P*)
   e divergência de Jensen-Shannon entre P = N(7,1) fixa e P* = N(μ*, σ*²). */
(function () {
  'use strict';
  window.DL = window.DL || {};
  DL.sections = DL.sections || [];

  const XMIN = -7, XMAX = 19, NGRID = 600;
  const MU_REAL = 7, LN2 = Math.LN2;
  const MU_LO = -4, MU_HI = 18;

  function init() {
    const P = DL.plot;
    const U = DL.utils;
    const $ = (id) => document.getElementById(id);

    const cvDist = $('s2-dist'), cvJS = $('s2-js');
    const slMu = $('s2-mu'), slSigma = $('s2-sigma');
    const btnReset = $('s2-reset');

    let mu = +slMu.value, sigma = +slSigma.value;
    let frDist = null, dragging = false;
    let jsCurve = null; /* cache: D_JS(μ*) com σ* atual */

    const DX = (XMAX - XMIN) / NGRID;
    const xs = new Float64Array(NGRID);
    const pReal = new Float64Array(NGRID);
    for (let i = 0; i < NGRID; i++) {
      xs[i] = XMIN + (i + 0.5) * DX;
      pReal[i] = U.gaussPdf(xs[i], MU_REAL, 1);
    }

    /* devolve { klStar, klReal, js } por integração numérica */
    function divergences(muG, sigmaG) {
      const s2 = sigmaG * sigmaG;
      let klStar = 0, klReal = 0;
      for (let i = 0; i < NGRID; i++) {
        const p = pReal[i];
        const q = U.gaussPdf(xs[i], muG, s2);
        const m = 0.5 * (p + q);
        if (q > 1e-300 && m > 1e-300) klStar += q * Math.log(q / m) * DX;
        if (p > 1e-300 && m > 1e-300) klReal += p * Math.log(p / m) * DX;
      }
      return { klStar, klReal, js: 0.5 * (klStar + klReal) };
    }

    function rebuildCurve() {
      const n = 120;
      jsCurve = { mus: [], vals: [] };
      for (let i = 0; i <= n; i++) {
        const m = MU_LO + (MU_HI - MU_LO) * i / n;
        jsCurve.mus.push(m);
        jsCurve.vals.push(divergences(m, sigma).js);
      }
    }

    function drawDist() {
      const th = P.theme();
      const { ctx, w, h } = P.setup(cvDist);
      P.clear(ctx, w, h);
      const s2 = sigma * sigma;
      const peak = Math.max(0.45, U.gaussPdf(mu, mu, s2));
      const YMAX = peak * 1.2;
      const fr = P.frame(ctx, w, h, XMIN, XMAX, 0, YMAX);
      frDist = fr;
      P.axes(ctx, fr, { xlabel: 'x', ylabel: 'densidade', xticks: [-4, 0, 7, 12, 18], yticks: [0, 0.2, 0.4] });

      const n = 260, gx = [], gp = [], gq = [], gm = [], gd = [];
      for (let i = 0; i <= n; i++) {
        const x = XMIN + (XMAX - XMIN) * i / n;
        const p = U.gaussPdf(x, MU_REAL, 1);
        const q = U.gaussPdf(x, mu, s2);
        gx.push(x);
        gp.push(p);
        gq.push(q);
        gm.push(0.5 * (p + q));
        gd.push((p + q > 1e-300 ? p / (p + q) : 0.5) * YMAX);
      }
      P.line(ctx, fr, gx, gm, th.comment, { width: 1.4, dash: [4, 4] });
      P.line(ctx, fr, gx, gp, th.cyan, { width: 2 });
      P.line(ctx, fr, gx, gq, th.orange, { width: 2 });
      P.line(ctx, fr, gx, gd, th.purple, { width: 2 });

      P.label(ctx, fr.X(XMAX) - 4, fr.Y(YMAX) + 12, '1', th.purple, 'right');
      P.label(ctx, fr.X(XMAX) - 4, fr.Y(YMAX / 2) + 4, '0.5', th.purple, 'right');
      P.label(ctx, fr.X(XMAX) - 4, fr.Y(0) - 4, '0', th.purple, 'right');

      /* marcador arrastável de μ* */
      ctx.beginPath();
      ctx.arc(fr.X(mu), fr.Y(0), 6, 0, 2 * Math.PI);
      ctx.fillStyle = th.orange;
      ctx.fill();
      ctx.strokeStyle = th.fg;
      ctx.lineWidth = 1.5;
      ctx.stroke();
      P.label(ctx, fr.X(mu) + 8, fr.Y(0) - 8, 'μ*', th.orange);

      P.label(ctx, fr.X(XMIN) + 8, fr.Y(YMAX) + 14, 'P(x) real', th.cyan);
      P.label(ctx, fr.X(XMIN) + 8, fr.Y(YMAX) + 28, 'P*(x) gerada', th.orange);
      P.label(ctx, fr.X(XMIN) + 8, fr.Y(YMAX) + 42, 'M(x) mistura', th.comment);
      P.label(ctx, fr.X(XMIN) + 8, fr.Y(YMAX) + 56, 'D*(x)', th.purple);
    }

    function drawJS() {
      const th = P.theme();
      const { ctx, w, h } = P.setup(cvJS);
      P.clear(ctx, w, h);
      const fr = P.frame(ctx, w, h, MU_LO, MU_HI, 0, 0.78);
      P.axes(ctx, fr, { xlabel: 'μ*', ylabel: 'D_{JS} (nats)', xticks: [-4, 0, 7, 12, 18], yticks: [0, 0.35, 0.69] });
      P.line(ctx, fr, [MU_LO, MU_HI], [LN2, LN2], th.green, { width: 1.2, dash: [5, 4] });
      P.label(ctx, fr.X(MU_HI) - 4, fr.Y(LN2) - 6, 'ln 2', th.green, 'right');
      P.line(ctx, fr, jsCurve.mus, jsCurve.vals, th.purple, { width: 2 });
      const cur = divergences(mu, sigma).js;
      P.scatter(ctx, fr, [[mu, cur]], th.orange, { r: 4.5, alpha: 1 });
    }

    function updateReadout() {
      const d = divergences(mu, sigma);
      const h = 0.05;
      const grad = (divergences(mu + h, sigma).js - divergences(mu - h, sigma).js) / (2 * h);
      $('s2-readout').textContent =
        'D_KL(P*‖M) = ' + d.klStar.toFixed(3) +
        ' · D_KL(P‖M) = ' + d.klReal.toFixed(3) +
        ' · D_JS = ' + d.js.toFixed(3) + ' nats' +
        ' · |∂D_JS/∂μ*| = ' + Math.abs(grad).toExponential(1);
    }

    function redraw() {
      drawDist();
      drawJS();
      updateReadout();
    }

    function setMu(v) {
      mu = Math.max(MU_LO, Math.min(MU_HI, v));
      slMu.value = mu;
      $('s2-mu-val').textContent = mu.toFixed(1);
      redraw();
    }

    cvDist.addEventListener('pointerdown', (e) => {
      if (!frDist) return;
      dragging = true;
      cvDist.setPointerCapture(e.pointerId);
      const r = cvDist.getBoundingClientRect();
      setMu(frDist.invX(e.clientX - r.left));
    });
    cvDist.addEventListener('pointermove', (e) => {
      if (!dragging || !frDist) return;
      const r = cvDist.getBoundingClientRect();
      setMu(frDist.invX(e.clientX - r.left));
    });
    cvDist.addEventListener('pointerup', () => { dragging = false; });

    slMu.addEventListener('input', () => {
      mu = +slMu.value;
      $('s2-mu-val').textContent = mu.toFixed(1);
      redraw();
    });
    slSigma.addEventListener('input', () => {
      sigma = +slSigma.value;
      $('s2-sigma-val').textContent = sigma.toFixed(2);
      rebuildCurve();
      redraw();
    });
    btnReset.addEventListener('click', () => {
      mu = 2; sigma = 1;
      slMu.value = mu; slSigma.value = sigma;
      $('s2-mu-val').textContent = mu.toFixed(1);
      $('s2-sigma-val').textContent = sigma.toFixed(2);
      rebuildCurve();
      redraw();
    });

    rebuildCurve();
    redraw();
    P.onRedraw(redraw);
    P.observeResize(cvDist, redraw);
    P.observeResize(cvJS, redraw);
  }

  DL.sections.push({ name: 's2-discriminador-otimo', init });
})();
