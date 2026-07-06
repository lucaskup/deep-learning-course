/* Seção 4: o ELBO como reconstrução + KL, com o peso β ajustável durante
   o treino. β = 0 recupera um AE puro; β grande empurra q(z|x,φ) para o
   prior e induz posterior collapse (KL → 0, μ → 0, σ → 1). */
(function () {
  'use strict';
  window.DL = window.DL || {};
  DL.sections = DL.sections || [];

  const XR = 2.7, YR = 2.4;
  const LR = 0.008, BATCH = 32, RECON_W = 6, MAX_STEPS = 40000;

  function init() {
    const P = DL.plot, M = DL.model, U = DL.utils;
    const $ = (id) => document.getElementById(id);

    const cvLat = $('s4-latent'), cvData = $('s4-data'), cvCurves = $('s4-curves');
    const slBeta = $('s4-beta');
    const btnRun = $('s4-run'), btnReset = $('s4-reset'), btnSample = $('s4-resample');

    let seed = 19;
    let beta = +slBeta.value;
    let data, rng, enc, dec, k, hist, events;
    let running = false;

    function buildModels() {
      rng = U.mulberry32(seed + 71);
      enc = M.mlp([2, 32, 16, 2], rng);
      dec = M.mlp([1, 16, 32, 2], rng);
      k = 0;
      hist = [];
      events = [{ k: 0, beta }];
    }

    function resetAll() {
      data = M.makeData(220, seed);
      buildModels();
    }

    function trainSteps(n) {
      for (let i = 0; i < n && k < MAX_STEPS; i++) {
        const r = M.vaeStep(enc, dec, data, BATCH, LR, beta, RECON_W, rng);
        k++;
        if (k % 20 === 0) hist.push({ k, recon: r.recon, kl: r.kl });
      }
    }

    function clipPlot(ctx, fr) {
      ctx.save();
      ctx.beginPath();
      ctx.rect(fr.X(fr.xmin), fr.Y(fr.ymax), fr.iw, fr.ih);
      ctx.clip();
    }

    function drawLatent(post) {
      const th = P.theme();
      const { ctx, w, h } = P.setup(cvLat);
      P.clear(ctx, w, h);
      /* janela adapta-se ao espalhamento dos μ (β = 0 espalha muito) */
      let zlim = 3.5;
      for (const e of post) zlim = Math.max(zlim, Math.abs(e.mu) + 2 * e.sig);
      zlim = Math.min(12, zlim);

      const grid = [], agg = [];
      let peak = 0;
      for (let i = 0; i <= 150; i++) {
        const z = -zlim + (i / 150) * 2 * zlim;
        let q = 0;
        for (const e of post) q += U.gaussPdf(z, e.mu, e.sig * e.sig);
        q /= post.length;
        grid.push(z);
        agg.push(q);
        if (q > peak) peak = q;
      }
      const ymax = Math.max(0.7, 1.15 * peak);
      const fr = P.frame(ctx, w, h, -zlim, zlim, 0, ymax);
      P.axes(ctx, fr, {
        xlabel: 'z', ylabel: 'densidade',
        xticks: [-Math.round(zlim), 0, Math.round(zlim)], yticks: [],
      });
      clipPlot(ctx, fr);

      /* q(z|x,φ) individuais (finas) e marcas de μ(x) */
      for (let i = 0; i < post.length; i += 6) {
        const e = post[i];
        const xs = [], ys = [];
        for (let j = 0; j <= 80; j++) {
          const z = -zlim + (j / 80) * 2 * zlim;
          xs.push(z);
          ys.push(U.gaussPdf(z, e.mu, e.sig * e.sig));
        }
        P.line(ctx, fr, xs, ys, th.red, { width: 1, alpha: 0.18 });
      }
      for (let i = 0; i < post.length; i += 3) {
        P.line(ctx, fr, [post[i].mu, post[i].mu], [0, 0.035 * ymax], th.cyan, { width: 1, alpha: 0.55 });
      }
      P.line(ctx, fr, grid, agg, th.orange, { width: 2.2 });
      const pxs = [], pys = [];
      for (let i = 0; i <= 150; i++) {
        const z = -zlim + (i / 150) * 2 * zlim;
        pxs.push(z);
        pys.push(U.gaussPdf(z, 0, 1));
      }
      P.line(ctx, fr, pxs, pys, th.green, { width: 1.6, dash: [4, 3] });
      ctx.restore();

      P.label(ctx, fr.X(-zlim) + 8, fr.Y(ymax) + 14, 'posterior agregado', th.orange);
      P.label(ctx, fr.X(-zlim) + 8, fr.Y(ymax) + 28, 'q(z|x,φ)', th.red);
      P.label(ctx, fr.X(-zlim) + 8, fr.Y(ymax) + 42, 'prior N(0,1)', th.green);
      P.label(ctx, fr.X(-zlim) + 8, fr.Y(ymax) + 56, 'marcas: μ(x)', th.cyan);
    }

    function drawData(post) {
      const th = P.theme();
      const { ctx, w, h } = P.setup(cvData);
      P.clear(ctx, w, h);
      const fr = P.frame(ctx, w, h, -XR, XR, -YR, YR);
      P.axes(ctx, fr, { xlabel: 'x_1', ylabel: 'x_2', xticks: [-2, 0, 2], yticks: [-2, 0, 2] });
      clipPlot(ctx, fr);
      P.scatter(ctx, fr, data, th.comment, { r: 2.2, alpha: 0.4 });

      /* reconstruções pela média do posterior: x̂ = f(μ(x), θ) */
      for (let i = 0; i < Math.min(90, data.length); i++) {
        const x = data[i];
        const xh = M.decode(dec, [post[i].mu]);
        P.line(ctx, fr, [x[0], xh[0]], [x[1], xh[1]], th.cyan, { width: 1, alpha: 0.35 });
        P.scatter(ctx, fr, [[xh[0], xh[1]]], th.cyan, { r: 1.9, alpha: 0.85 });
      }

      /* o que o prior gera: f(z, θ) para z em [−2.5, 2.5] */
      const xs = [], ys = [];
      for (let i = 0; i <= 120; i++) {
        const p = M.decode(dec, [-2.5 + (i / 120) * 5]);
        xs.push(p[0]); ys.push(p[1]);
      }
      P.line(ctx, fr, xs, ys, th.orange, { width: 2, alpha: 0.85 });
      ctx.restore();

      P.label(ctx, fr.X(-XR) + 8, fr.Y(YR) + 14, 'x̂ = f(μ(x),θ)', th.cyan);
      P.label(ctx, fr.X(-XR) + 8, fr.Y(YR) + 28, 'f(z,θ), z ∈ [−2.5, 2.5]', th.orange);
    }

    function drawCurves() {
      const th = P.theme();
      const { ctx, w, h } = P.setup(cvCurves);
      P.clear(ctx, w, h);
      const n = Math.max(500, k);
      let ymax = 2;
      for (const p of hist) ymax = Math.max(ymax, p.recon, p.kl);
      ymax *= 1.1;
      const fr = P.frame(ctx, w, h, 0, n, 0, ymax);
      P.axes(ctx, fr, {
        xlabel: 'passo', ylabel: 'valor',
        xticks: [0, Math.round(n / 2), n],
        yticks: [0, Math.round(ymax / 2), Math.round(ymax)],
      });
      clipPlot(ctx, fr);
      if (hist.length > 1) {
        P.line(ctx, fr, hist.map((p) => p.k), hist.map((p) => p.recon), th.cyan, { width: 1.8 });
        P.line(ctx, fr, hist.map((p) => p.k), hist.map((p) => p.kl), th.red, { width: 1.8 });
      }
      /* marcas de mudança de β durante o treino */
      for (const ev of events) {
        if (ev.k === 0) continue;
        P.line(ctx, fr, [ev.k, ev.k], [0, ymax], th.purple, { width: 1, dash: [3, 3], alpha: 0.8 });
        P.label(ctx, fr.X(ev.k) + 4, fr.Y(ymax) + 12, 'β=' + ev.beta, th.purple);
      }
      ctx.restore();
      P.label(ctx, fr.X(0) + 8, fr.Y(ymax) + 14, 'reconstrução', th.cyan);
      P.label(ctx, fr.X(0) + 8, fr.Y(ymax) + 28, 'KL', th.red);
    }

    function updateReadout() {
      const last = hist[hist.length - 1];
      $('s4-beta-val').textContent = beta.toFixed(2);
      $('s4-readout').textContent = last
        ? 'passo ' + k + ' · recon = ' + last.recon.toFixed(2) + ' · KL = ' + last.kl.toFixed(2) +
          ' · perda = recon + β·KL = ' + (last.recon + beta * last.kl).toFixed(2)
        : 'passo 0 · treine para ver as curvas';
    }

    function redraw() {
      const post = data.map((x) => M.vaeEncode(enc, x));
      drawLatent(post);
      drawData(post);
      drawCurves();
      updateReadout();
    }

    function stopRun() {
      running = false;
      btnRun.textContent = '▶ Treinar';
      DL.stopTicker(tick);
    }

    function tick() {
      trainSteps(15);
      redraw();
      if (k >= MAX_STEPS) stopRun();
    }

    slBeta.addEventListener('input', () => {
      beta = +slBeta.value;
      const last = events[events.length - 1];
      if (k > 0 && k - last.k > 100) events.push({ k, beta });
      else last.beta = beta;
      updateReadout();
    });
    btnRun.addEventListener('click', () => {
      if (running) { stopRun(); return; }
      if (k >= MAX_STEPS) buildModels();
      running = true;
      btnRun.textContent = '⏸ Pausar';
      DL.startTicker(tick);
    });
    btnReset.addEventListener('click', () => { stopRun(); buildModels(); redraw(); });
    btnSample.addEventListener('click', () => {
      stopRun();
      seed = seed * 3 + 11;
      resetAll();
      redraw();
    });

    resetAll();
    redraw();
    P.onRedraw(redraw);
    P.observeResize(cvLat, redraw);
    P.observeResize(cvData, redraw);
    P.observeResize(cvCurves, redraw);
  }

  DL.sections.push({ name: 's4-elbo-beta', init });
})();
