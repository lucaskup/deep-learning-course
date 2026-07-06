/* Seção 3: colapso de moda. P(x) bimodal (modas em −1.5 e 4, σ² = 0.2,
   como na figura dos slides), gerador unimodal P* = N(μ*, σ*²) treinado
   por descida de gradiente sobre D_JS. Mostra os termos de qualidade
   ½·KL(P*‖M) e cobertura ½·KL(P‖M) e seus integrandos. */
(function () {
  'use strict';
  window.DL = window.DL || {};
  DL.sections = DL.sections || [];

  const XMIN = -5, XMAX = 8.5, NGRID = 600;
  const M1 = -1.5, M2 = 4, S2_DATA = 0.2, LN2 = Math.LN2;
  const MU_LO = -4, MU_HI = 7;
  const MAX_STEPS = 400;

  function init() {
    const P = DL.plot;
    const U = DL.utils;
    const $ = (id) => document.getElementById(id);

    const cvDist = $('s3-dist'), cvJS = $('s3-js'), cvTerms = $('s3-terms');
    const slMu = $('s3-mu'), slSigma = $('s3-sigma'), slW = $('s3-w');
    const btnStep = $('s3-step'), btnRun = $('s3-run'), btnReset = $('s3-reset');

    let mu = +slMu.value, sigma = +slSigma.value, wMode2 = +slW.value;
    let running = false, accT = 0, steps = 0;
    let frDist = null, dragging = false;
    let jsCurve = null;

    const DX = (XMAX - XMIN) / NGRID;
    const xs = new Float64Array(NGRID);
    const pData = new Float64Array(NGRID);
    for (let i = 0; i < NGRID; i++) xs[i] = XMIN + (i + 0.5) * DX;

    function rebuildData() {
      for (let i = 0; i < NGRID; i++) {
        pData[i] = (1 - wMode2) * U.gaussPdf(xs[i], M1, S2_DATA)
                 + wMode2 * U.gaussPdf(xs[i], M2, S2_DATA);
      }
    }

    const pdfData = (x) => (1 - wMode2) * U.gaussPdf(x, M1, S2_DATA)
                         + wMode2 * U.gaussPdf(x, M2, S2_DATA);

    /* { quality, coverage, js } com os fatores ½ já incluídos */
    function divergences(muG, sigmaG) {
      const s2 = sigmaG * sigmaG;
      let q = 0, c = 0;
      for (let i = 0; i < NGRID; i++) {
        const p = pData[i];
        const g = U.gaussPdf(xs[i], muG, s2);
        const m = 0.5 * (p + g);
        if (g > 1e-300 && m > 1e-300) q += g * Math.log(g / m) * DX;
        if (p > 1e-300 && m > 1e-300) c += p * Math.log(p / m) * DX;
      }
      return { quality: 0.5 * q, coverage: 0.5 * c, js: 0.5 * (q + c) };
    }

    function rebuildCurve() {
      const n = 130;
      jsCurve = { mus: [], vals: [] };
      for (let i = 0; i <= n; i++) {
        const m = MU_LO + (MU_HI - MU_LO) * i / n;
        jsCurve.mus.push(m);
        jsCurve.vals.push(divergences(m, sigma).js);
      }
    }

    /* treina apenas μ* (σ* fica no valor do slider): é o gerador de um
       parâmetro da aula, descendo o gradiente da própria D_JS */
    function trainStep() {
      if (steps >= MAX_STEPS) return false;
      const h = 0.01;
      const gMu = (divergences(mu + h, sigma).js - divergences(mu - h, sigma).js) / (2 * h);
      mu = Math.max(MU_LO, Math.min(MU_HI, mu - 0.8 * gMu));
      steps++;
      syncSliders();
      return Math.abs(gMu) > 2e-3;
    }

    function syncSliders() {
      slMu.value = mu;
      slSigma.value = sigma;
      $('s3-mu-val').textContent = mu.toFixed(2);
      $('s3-sigma-val').textContent = sigma.toFixed(2);
    }

    /* curva preenchida até y = 0 (integrando com sinal) */
    function fillCurve(ctx, fr, gx, gy, color) {
      const rc = P.rgb(color);
      ctx.beginPath();
      ctx.moveTo(fr.X(gx[0]), fr.Y(0));
      for (let i = 0; i < gx.length; i++) ctx.lineTo(fr.X(gx[i]), fr.Y(gy[i]));
      ctx.lineTo(fr.X(gx[gx.length - 1]), fr.Y(0));
      ctx.closePath();
      ctx.fillStyle = 'rgba(' + rc[0] + ',' + rc[1] + ',' + rc[2] + ',0.22)';
      ctx.fill();
      P.line(ctx, fr, gx, gy, color, { width: 1.8 });
    }

    function drawDist() {
      const th = P.theme();
      const { ctx, w, h } = P.setup(cvDist);
      P.clear(ctx, w, h);
      const s2 = sigma * sigma;
      const peak = Math.max(pdfData(M1), pdfData(M2), U.gaussPdf(mu, mu, s2));
      const YMAX = peak * 1.2;
      const fr = P.frame(ctx, w, h, XMIN, XMAX, 0, YMAX);
      frDist = fr;
      P.axes(ctx, fr, { xlabel: 'x', ylabel: 'densidade', xticks: [-4, -1.5, 0, 4, 8], yticks: [0, 0.4, 0.8] });

      const n = 280, gx = [], gp = [], gq = [], gm = [];
      for (let i = 0; i <= n; i++) {
        const x = XMIN + (XMAX - XMIN) * i / n;
        const p = pdfData(x);
        const q = U.gaussPdf(x, mu, s2);
        gx.push(x);
        gp.push(p);
        gq.push(q);
        gm.push(0.5 * (p + q));
      }
      P.line(ctx, fr, gx, gm, th.comment, { width: 1.4, dash: [4, 4] });
      P.line(ctx, fr, gx, gp, th.cyan, { width: 2 });
      P.line(ctx, fr, gx, gq, th.orange, { width: 2 });

      ctx.beginPath();
      ctx.arc(fr.X(mu), fr.Y(0), 6, 0, 2 * Math.PI);
      ctx.fillStyle = th.orange;
      ctx.fill();
      ctx.strokeStyle = th.fg;
      ctx.lineWidth = 1.5;
      ctx.stroke();
      P.label(ctx, fr.X(mu) + 8, fr.Y(0) - 8, 'μ*', th.orange);

      P.label(ctx, fr.X(XMIN) + 8, fr.Y(YMAX) + 14, 'P(x) real (bimodal)', th.cyan);
      P.label(ctx, fr.X(XMIN) + 8, fr.Y(YMAX) + 28, 'P*(x) gerada', th.orange);
      P.label(ctx, fr.X(XMIN) + 8, fr.Y(YMAX) + 42, 'M(x)', th.comment);
    }

    function drawJS() {
      const th = P.theme();
      const { ctx, w, h } = P.setup(cvJS);
      P.clear(ctx, w, h);
      const fr = P.frame(ctx, w, h, MU_LO, MU_HI, 0, 0.78);
      P.axes(ctx, fr, { xlabel: 'μ*', ylabel: 'D_{JS} (nats)', xticks: [-4, -1.5, 0, 4, 7], yticks: [0, 0.35, 0.69] });
      P.line(ctx, fr, [MU_LO, MU_HI], [LN2, LN2], th.green, { width: 1.2, dash: [5, 4] });
      P.label(ctx, fr.X(MU_HI) - 4, fr.Y(LN2) - 6, 'ln 2', th.green, 'right');
      P.line(ctx, fr, jsCurve.mus, jsCurve.vals, th.purple, { width: 2 });
      const cur = divergences(mu, sigma).js;
      P.scatter(ctx, fr, [[mu, cur]], th.orange, { r: 4.5, alpha: 1 });
    }

    function drawTerms() {
      const th = P.theme();
      const { ctx, w, h } = P.setup(cvTerms);
      P.clear(ctx, w, h);
      const s2 = sigma * sigma;
      const n = 280, gx = [], gq = [], gc = [];
      let lo = -0.02, hi = 0.02;
      for (let i = 0; i <= n; i++) {
        const x = XMIN + (XMAX - XMIN) * i / n;
        const p = pdfData(x);
        const q = U.gaussPdf(x, mu, s2);
        const m = 0.5 * (p + q);
        const iq = (q > 1e-300 && m > 1e-300) ? 0.5 * q * Math.log(q / m) : 0;
        const ic = (p > 1e-300 && m > 1e-300) ? 0.5 * p * Math.log(p / m) : 0;
        gx.push(x);
        gq.push(iq);
        gc.push(ic);
        lo = Math.min(lo, iq, ic);
        hi = Math.max(hi, iq, ic);
      }
      const fr = P.frame(ctx, w, h, XMIN, XMAX, lo * 1.15, hi * 1.15);
      P.axes(ctx, fr, { xlabel: 'x', ylabel: 'integrando', xticks: [-4, -1.5, 0, 4, 8], yticks: [0] });
      P.line(ctx, fr, [XMIN, XMAX], [0, 0], th.line, { width: 1 });
      fillCurve(ctx, fr, gx, gc, th.cyan);
      fillCurve(ctx, fr, gx, gq, th.orange);
      P.label(ctx, fr.X(XMIN) + 8, fr.Y(fr.ymax) + 14, 'qualidade ½·P*·ln(P*/M)', th.orange);
      P.label(ctx, fr.X(XMIN) + 8, fr.Y(fr.ymax) + 28, 'cobertura ½·P·ln(P/M)', th.cyan);
    }

    function updateReadout() {
      const d = divergences(mu, sigma);
      $('s3-readout').textContent =
        'passo = ' + steps +
        ' · qualidade = ' + d.quality.toFixed(3) +
        ' · cobertura = ' + d.coverage.toFixed(3) +
        ' · D_JS = ' + d.js.toFixed(3) + ' nats';
    }

    function redraw() {
      drawDist();
      drawJS();
      drawTerms();
      updateReadout();
    }

    function stopRun() {
      running = false;
      btnRun.textContent = '▶ Treinar gerador';
      DL.stopTicker(tick);
    }

    function tick(dt) {
      accT += dt * 12;
      let alive = true;
      while (accT >= 1) { accT -= 1; alive = trainStep(); }
      redraw();
      if (!alive || steps >= MAX_STEPS) stopRun();
    }

    function setMu(v) {
      mu = Math.max(MU_LO, Math.min(MU_HI, v));
      syncSliders();
      redraw();
    }

    cvDist.addEventListener('pointerdown', (e) => {
      if (!frDist) return;
      dragging = true;
      cvDist.setPointerCapture(e.pointerId);
      stopRun();
      const r = cvDist.getBoundingClientRect();
      setMu(frDist.invX(e.clientX - r.left));
    });
    cvDist.addEventListener('pointermove', (e) => {
      if (!dragging || !frDist) return;
      const r = cvDist.getBoundingClientRect();
      setMu(frDist.invX(e.clientX - r.left));
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
      mu = 1; sigma = 0.5; steps = 0;
      syncSliders();
      rebuildCurve();
      redraw();
    });

    slMu.addEventListener('input', () => {
      mu = +slMu.value;
      $('s3-mu-val').textContent = mu.toFixed(2);
      redraw();
    });
    slSigma.addEventListener('input', () => {
      sigma = +slSigma.value;
      $('s3-sigma-val').textContent = sigma.toFixed(2);
      rebuildCurve();
      redraw();
    });
    slW.addEventListener('input', () => {
      wMode2 = +slW.value;
      $('s3-w-val').textContent = wMode2.toFixed(2);
      rebuildData();
      rebuildCurve();
      redraw();
    });

    rebuildData();
    rebuildCurve();
    redraw();
    P.onRedraw(redraw);
    P.observeResize(cvDist, redraw);
    P.observeResize(cvJS, redraw);
    P.observeResize(cvTerms, redraw);
  }

  DL.sections.push({ name: 's3-mode-collapse', init });
})();
