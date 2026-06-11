/* Seção 1: autoencoder com gargalo treinado ao vivo em dados 2D.
   Compara latente 1D (variedade) com latente 2D (sem gargalo) e usa uma
   sonda arrastável para mostrar o score de anomalia s(x). */
(function () {
  'use strict';
  window.DL = window.DL || {};
  DL.sections = DL.sections || [];

  const XR = 2.7, YR = 2.4, LR = 0.01, BATCH = 32, MAX_STEPS = 12000;

  function init() {
    const P = DL.plot, M = DL.model, U = DL.utils;
    const $ = (id) => document.getElementById(id);

    const cvData = $('s1-data'), cvLoss = $('s1-loss');
    const btnRun = $('s1-run'), btnStep = $('s1-step');
    const btnReset = $('s1-reset'), btnSample = $('s1-resample');
    const tab1 = $('s1-tab-1d'), tab2 = $('s1-tab-2d');

    let seed = 7;
    let data, rng, m1, m2, k, hist1, hist2;
    let mode = '1d';
    let probe = [0, 0.2];
    let running = false;
    let frData = null, dragging = false;

    function buildModels() {
      rng = U.mulberry32(seed + 101);
      m1 = { enc: M.mlp([2, 32, 16, 1], rng), dec: M.mlp([1, 16, 32, 2], rng) };
      m2 = { enc: M.mlp([2, 32, 16, 2], rng), dec: M.mlp([2, 16, 32, 2], rng) };
      k = 0;
      hist1 = [];
      hist2 = [];
    }

    function resetAll() {
      data = M.makeData(220, seed);
      buildModels();
    }

    function trainSteps(n) {
      for (let i = 0; i < n && k < MAX_STEPS; i++) {
        const l1 = M.aeStep(m1.enc, m1.dec, data, BATCH, LR, rng, 0);
        const l2 = M.aeStep(m2.enc, m2.dec, data, BATCH, LR, rng, 0);
        k++;
        if (k % 10 === 0) {
          hist1.push([k, Math.log10(Math.max(l1, 1e-6))]);
          hist2.push([k, Math.log10(Math.max(l2, 1e-6))]);
        }
      }
    }

    const model = () => (mode === '1d' ? m1 : m2);
    const reconOf = (m, x) => M.decode(m.dec, M.encode(m.enc, x));
    function sErr(x) {
      const xh = reconOf(model(), x);
      return (xh[0] - x[0]) ** 2 + (xh[1] - x[1]) ** 2;
    }

    function clipPlot(ctx, fr) {
      ctx.save();
      ctx.beginPath();
      ctx.rect(fr.X(fr.xmin), fr.Y(fr.ymax), fr.iw, fr.ih);
      ctx.clip();
    }

    function drawData() {
      const th = P.theme();
      const { ctx, w, h } = P.setup(cvData);
      P.clear(ctx, w, h);
      const fr = P.frame(ctx, w, h, -XR, XR, -YR, YR);
      frData = fr;
      P.axes(ctx, fr, { xlabel: 'x_1', ylabel: 'x_2', xticks: [-2, 0, 2], yticks: [-2, 0, 2] });
      clipPlot(ctx, fr);

      /* dados de treino */
      P.scatter(ctx, fr, data, th.comment, { r: 2.4, alpha: 0.5 });

      const m = model();
      /* reconstruções de um subconjunto: segmento x → x̂ */
      ctx.globalAlpha = 0.45;
      for (let i = 0; i < Math.min(90, data.length); i++) {
        const x = data[i];
        const xh = reconOf(m, x);
        P.line(ctx, fr, [x[0], xh[0]], [x[1], xh[1]], th.cyan, { width: 1, alpha: 0.4 });
        P.scatter(ctx, fr, [[xh[0], xh[1]]], th.cyan, { r: 1.8, alpha: 0.85 });
      }
      ctx.globalAlpha = 1;

      /* variedade aprendida: imagem do decoder ao varrer z (apenas 1D) */
      if (mode === '1d') {
        let zmin = Infinity, zmax = -Infinity;
        for (const x of data) {
          const z = M.encode(m.enc, x)[0];
          if (z < zmin) zmin = z;
          if (z > zmax) zmax = z;
        }
        const pad = 0.04 * (zmax - zmin || 1);
        zmin -= pad; zmax += pad;
        const xs = [], ys = [];
        for (let i = 0; i <= 140; i++) {
          const p = M.decode(m.dec, [zmin + (i / 140) * (zmax - zmin)]);
          xs.push(p[0]); ys.push(p[1]);
        }
        P.line(ctx, fr, xs, ys, th.orange, { width: 2, alpha: 0.9 });
      }

      /* sonda arrastável e sua reconstrução */
      const xhP = reconOf(m, probe);
      P.line(ctx, fr, [probe[0], xhP[0]], [probe[1], xhP[1]], th.purple, { width: 1.4, dash: [4, 3] });
      P.scatter(ctx, fr, [[xhP[0], xhP[1]]], th.green, { r: 4, alpha: 1 });
      ctx.restore();

      ctx.beginPath();
      ctx.arc(fr.X(probe[0]), fr.Y(probe[1]), 6, 0, 2 * Math.PI);
      ctx.strokeStyle = th.purple;
      ctx.lineWidth = 2.2;
      ctx.stroke();
      P.label(ctx, fr.X(probe[0]) + 9, fr.Y(probe[1]) - 8, 'x (sonda)', th.purple);

      P.label(ctx, fr.X(-XR) + 8, fr.Y(YR) + 14, 'dados', th.comment);
      P.label(ctx, fr.X(-XR) + 8, fr.Y(YR) + 28, 'x̂ = f(g(x,φ),θ)', th.cyan);
      if (mode === '1d') P.label(ctx, fr.X(-XR) + 8, fr.Y(YR) + 42, 'variedade f(z,θ)', th.orange);
      P.label(ctx, fr.X(-XR) + 8, fr.Y(YR) + (mode === '1d' ? 56 : 42), 'x̂ da sonda', th.green);
    }

    function drawLoss() {
      const th = P.theme();
      const { ctx, w, h } = P.setup(cvLoss);
      P.clear(ctx, w, h);
      const n = Math.max(500, k);
      const fr = P.frame(ctx, w, h, 0, n, -4, 0.8);
      P.axes(ctx, fr, {
        xlabel: 'passo', ylabel: 'log_{10} J',
        xticks: [0, Math.round(n / 2), n], yticks: [-4, -2, 0],
      });
      clipPlot(ctx, fr);
      if (hist1.length > 1) {
        P.line(ctx, fr, hist1.map((p) => p[0]), hist1.map((p) => p[1]), th.cyan, { width: 1.7 });
      }
      if (hist2.length > 1) {
        P.line(ctx, fr, hist2.map((p) => p[0]), hist2.map((p) => p[1]), th.pink, { width: 1.7 });
      }
      ctx.restore();
      P.label(ctx, fr.X(0) + 8, fr.Y(0.8) + 14, 'latente 1D', th.cyan);
      P.label(ctx, fr.X(0) + 8, fr.Y(0.8) + 28, 'latente 2D', th.pink);
    }

    function updateReadout() {
      const j = hist1.length || hist2.length
        ? Math.pow(10, (mode === '1d' ? hist1 : hist2)[(mode === '1d' ? hist1 : hist2).length - 1][1])
        : NaN;
      $('s1-readout').textContent =
        'passo ' + k +
        ' · J = ' + (isFinite(j) ? j.toFixed(4) : '…') +
        ' · s(sonda) = ' + sErr(probe).toFixed(3);
    }

    function redraw() {
      drawData();
      drawLoss();
      updateReadout();
    }

    function stopRun() {
      running = false;
      btnRun.textContent = '▶ Treinar';
      DL.stopTicker(tick);
    }

    function tick() {
      trainSteps(10);
      redraw();
      if (k >= MAX_STEPS) stopRun();
    }

    cvData.addEventListener('pointerdown', (e) => {
      if (!frData) return;
      dragging = true;
      cvData.setPointerCapture(e.pointerId);
      const r = cvData.getBoundingClientRect();
      setProbe(frData.invX(e.clientX - r.left), frData.invY(e.clientY - r.top));
    });
    cvData.addEventListener('pointermove', (e) => {
      if (!dragging || !frData) return;
      const r = cvData.getBoundingClientRect();
      setProbe(frData.invX(e.clientX - r.left), frData.invY(e.clientY - r.top));
    });
    cvData.addEventListener('pointerup', () => { dragging = false; });

    function setProbe(x, y) {
      probe = [
        Math.max(-XR + 0.05, Math.min(XR - 0.05, x)),
        Math.max(-YR + 0.05, Math.min(YR - 0.05, y)),
      ];
      redraw();
    }

    function setMode(m) {
      mode = m;
      tab1.classList.toggle('active', m === '1d');
      tab2.classList.toggle('active', m === '2d');
      redraw();
    }
    tab1.addEventListener('click', () => setMode('1d'));
    tab2.addEventListener('click', () => setMode('2d'));

    btnRun.addEventListener('click', () => {
      if (running) { stopRun(); return; }
      if (k >= MAX_STEPS) buildModels();
      running = true;
      btnRun.textContent = '⏸ Pausar';
      DL.startTicker(tick);
    });
    btnStep.addEventListener('click', () => { trainSteps(1000); redraw(); });
    btnReset.addEventListener('click', () => { stopRun(); buildModels(); redraw(); });
    btnSample.addEventListener('click', () => {
      stopRun();
      seed = seed * 7 + 13;
      resetAll();
      redraw();
    });

    resetAll();
    redraw();
    P.onRedraw(redraw);
    P.observeResize(cvData, redraw);
    P.observeResize(cvLoss, redraw);
  }

  DL.sections.push({ name: 's1-bottleneck', init });
})();
