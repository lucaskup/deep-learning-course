/* Seção 2: espaço latente do AE vs do VAE, treinados ao vivo nos mesmos
   dados. O AE deixa buracos no latente; no VAE as q(z|x,φ) gaussianas
   se sobrepõem e o posterior agregado cobre o prior N(0,1). */
(function () {
  'use strict';
  window.DL = window.DL || {};
  DL.sections = DL.sections || [];

  const XR = 2.7, YR = 2.4, ZR = 3.2;
  const LR_AE = 0.01, LR_VAE = 0.008, BATCH = 32, RECON_W = 6, MAX_STEPS = 12000;

  function init() {
    const P = DL.plot, M = DL.model, U = DL.utils;
    const $ = (id) => document.getElementById(id);

    const cvAE = $('s2-lat-ae'), cvVAE = $('s2-lat-vae'), cvData = $('s2-data');
    const btnRun = $('s2-run'), btnReset = $('s2-reset'), btnSample = $('s2-resample');
    const slZ = $('s2-z');

    let seed = 11;
    let data, rng, ae, vae, k;
    let zStar = +slZ.value;
    let running = false;
    let lossAE = NaN, lossVAE = null;
    let frAE = null, frVAE = null, dragLat = null;

    function buildModels() {
      rng = U.mulberry32(seed + 31);
      ae = { enc: M.mlp([2, 32, 16, 1], rng), dec: M.mlp([1, 16, 32, 2], rng) };
      vae = { enc: M.mlp([2, 32, 16, 2], rng), dec: M.mlp([1, 16, 32, 2], rng) };
      k = 0;
      lossAE = NaN;
      lossVAE = null;
    }

    function resetAll() {
      data = M.makeData(220, seed);
      buildModels();
    }

    function trainSteps(n) {
      for (let i = 0; i < n && k < MAX_STEPS; i++) {
        lossAE = M.aeStep(ae.enc, ae.dec, data, BATCH, LR_AE, rng, 0);
        lossVAE = M.vaeStep(vae.enc, vae.dec, data, BATCH, LR_VAE, 1, RECON_W, rng);
        k++;
      }
    }

    /* Estatísticas do latente do AE: usadas para padronizar a exibição
       e para "ajustar uma gaussiana" sobre os códigos. */
    function aeStats() {
      let m = 0, s = 0;
      const zs = data.map((x) => M.encode(ae.enc, x)[0]);
      for (const z of zs) m += z;
      m /= zs.length;
      for (const z of zs) s += (z - m) ** 2;
      s = Math.sqrt(s / zs.length) || 1e-3;
      return { zs, mean: m, std: Math.max(s, 1e-3) };
    }

    function clipPlot(ctx, fr) {
      ctx.save();
      ctx.beginPath();
      ctx.rect(fr.X(fr.xmin), fr.Y(fr.ymax), fr.iw, fr.ih);
      ctx.clip();
    }

    function gaussCurve(ctx, fr, mu, sig, color, opts) {
      const xs = [], ys = [];
      for (let i = 0; i <= 160; i++) {
        const z = fr.xmin + (i / 160) * (fr.xmax - fr.xmin);
        xs.push(z);
        ys.push(U.gaussPdf(z, mu, sig * sig));
      }
      P.line(ctx, fr, xs, ys, color, opts);
    }

    function zMarker(ctx, fr, color) {
      P.line(ctx, fr, [zStar, zStar], [fr.ymin, fr.ymax], color, { width: 1.6, dash: [5, 4] });
      P.label(ctx, fr.X(zStar) + 5, fr.Y(fr.ymax) + 14, 'z*', color);
    }

    function drawLatAE(stats) {
      const th = P.theme();
      const { ctx, w, h } = P.setup(cvAE);
      P.clear(ctx, w, h);
      const zsStd = stats.zs.map((z) => (z - stats.mean) / stats.std);
      const fr = P.frame(ctx, w, h, -ZR, ZR, 0, 0.95);
      frAE = fr;
      P.axes(ctx, fr, { xlabel: 'z (padronizado)', ylabel: 'densidade', xticks: [-3, 0, 3], yticks: [] });
      clipPlot(ctx, fr);
      P.histogram(ctx, fr, zsStd, 44, th.cyan, { alpha: 0.6 });
      gaussCurve(ctx, fr, 0, 1, th.comment, { width: 1.4, dash: [4, 3] });
      zMarker(ctx, fr, th.purple);
      ctx.restore();
      P.label(ctx, fr.X(-ZR) + 8, fr.Y(0.95) + 14, 'códigos z = g(x,φ)', th.cyan);
      P.label(ctx, fr.X(-ZR) + 8, fr.Y(0.95) + 28, 'gaussiana ajustada', th.comment);
    }

    function drawLatVAE(post) {
      const th = P.theme();
      const { ctx, w, h } = P.setup(cvVAE);
      P.clear(ctx, w, h);
      /* posterior agregado: média das q(z|x,φ) sobre o dataset */
      const grid = [], agg = [];
      let peak = 0;
      for (let i = 0; i <= 160; i++) {
        const z = -ZR + (i / 160) * 2 * ZR;
        let q = 0;
        for (const e of post) q += U.gaussPdf(z, e.mu, e.sig * e.sig);
        q /= post.length;
        grid.push(z);
        agg.push(q);
        if (q > peak) peak = q;
      }
      const ymax = Math.max(0.8, 1.15 * peak);
      const fr = P.frame(ctx, w, h, -ZR, ZR, 0, ymax);
      frVAE = fr;
      P.axes(ctx, fr, { xlabel: 'z', ylabel: 'densidade', xticks: [-3, 0, 3], yticks: [] });
      clipPlot(ctx, fr);
      /* algumas q(z|x) individuais, finas */
      for (let i = 0; i < post.length; i += 9) {
        gaussCurve(ctx, fr, post[i].mu, post[i].sig, th.red, { width: 1, alpha: 0.22 });
      }
      P.line(ctx, fr, grid, agg, th.orange, { width: 2.2 });
      gaussCurve(ctx, fr, 0, 1, th.green, { width: 1.6, dash: [4, 3] });
      zMarker(ctx, fr, th.purple);
      ctx.restore();
      P.label(ctx, fr.X(-ZR) + 8, fr.Y(ymax) + 14, 'posterior agregado', th.orange);
      P.label(ctx, fr.X(-ZR) + 8, fr.Y(ymax) + 28, 'q(z|x,φ) individuais', th.red);
      P.label(ctx, fr.X(-ZR) + 8, fr.Y(ymax) + 42, 'prior p(z) = N(0,1)', th.green);
    }

    function drawData(stats) {
      const th = P.theme();
      const { ctx, w, h } = P.setup(cvData);
      P.clear(ctx, w, h);
      const fr = P.frame(ctx, w, h, -XR, XR, -YR, YR);
      P.axes(ctx, fr, { xlabel: 'x_1', ylabel: 'x_2', xticks: [-2, 0, 2], yticks: [-2, 0, 2] });
      clipPlot(ctx, fr);
      P.scatter(ctx, fr, data, th.comment, { r: 2.2, alpha: 0.4 });

      /* imagem do decoder ao varrer z em [−3, 3] */
      const curve = (dec, map, color) => {
        const xs = [], ys = [];
        for (let i = 0; i <= 120; i++) {
          const z = -3 + (i / 120) * 6;
          const p = M.decode(dec, [map(z)]);
          xs.push(p[0]); ys.push(p[1]);
        }
        P.line(ctx, fr, xs, ys, color, { width: 1.2, alpha: 0.4 });
      };
      const mapAE = (z) => stats.mean + stats.std * z;
      curve(ae.dec, mapAE, th.cyan);
      curve(vae.dec, (z) => z, th.orange);

      /* gerações: z ~ N(0,1) decodificado por cada modelo */
      const rngG = U.mulberry32(99);
      const genAE = [], genVAE = [];
      for (let i = 0; i < 70; i++) {
        const z = U.randn(rngG);
        const a = M.decode(ae.dec, [mapAE(z)]);
        const v = M.decode(vae.dec, [z]);
        genAE.push([a[0], a[1]]);
        genVAE.push([v[0], v[1]]);
      }
      P.scatter(ctx, fr, genAE, th.cyan, { r: 2.6, alpha: 0.85 });
      P.scatter(ctx, fr, genVAE, th.orange, { r: 2.6, alpha: 0.85 });

      /* ponto decodificado em z* */
      const pa = M.decode(ae.dec, [mapAE(zStar)]);
      const pv = M.decode(vae.dec, [zStar]);
      ctx.restore();
      for (const [p, c, t] of [[pa, th.cyan, 'f_{AE}(z*)'], [pv, th.orange, 'f_{VAE}(z*)']]) {
        ctx.beginPath();
        ctx.arc(fr.X(p[0]), fr.Y(p[1]), 6, 0, 2 * Math.PI);
        ctx.fillStyle = c;
        ctx.fill();
        ctx.strokeStyle = th.fg;
        ctx.lineWidth = 1.6;
        ctx.stroke();
        P.label(ctx, fr.X(p[0]) + 9, fr.Y(p[1]) + 4, t, c);
      }
      P.label(ctx, fr.X(-XR) + 8, fr.Y(YR) + 14, 'gerações do AE', th.cyan);
      P.label(ctx, fr.X(-XR) + 8, fr.Y(YR) + 28, 'gerações do VAE', th.orange);
    }

    function updateReadout() {
      $('s2-z-val').textContent = zStar.toFixed(2);
      const v = lossVAE;
      $('s2-readout').textContent =
        'passo ' + k +
        ' · J AE = ' + (isFinite(lossAE) ? lossAE.toFixed(3) : '…') +
        (v ? ' · VAE: recon = ' + v.recon.toFixed(2) + ', KL = ' + v.kl.toFixed(2) : '');
    }

    function redraw() {
      const stats = aeStats();
      const post = data.map((x) => M.vaeEncode(vae.enc, x));
      drawLatAE(stats);
      drawLatVAE(post);
      drawData(stats);
      updateReadout();
    }

    function stopRun() {
      running = false;
      btnRun.textContent = '▶ Treinar';
      DL.stopTicker(tick);
    }

    function tick() {
      trainSteps(8);
      redraw();
      if (k >= MAX_STEPS) stopRun();
    }

    /* arrastar z* diretamente nos painéis do latente */
    function bindLatDrag(cv, getFr) {
      cv.addEventListener('pointerdown', (e) => {
        dragLat = cv;
        cv.setPointerCapture(e.pointerId);
        moveLat(e, cv, getFr);
      });
      cv.addEventListener('pointermove', (e) => {
        if (dragLat === cv) moveLat(e, cv, getFr);
      });
      cv.addEventListener('pointerup', () => { dragLat = null; });
    }
    function moveLat(e, cv, getFr) {
      const fr = getFr();
      if (!fr) return;
      const r = cv.getBoundingClientRect();
      zStar = Math.max(-3, Math.min(3, fr.invX(e.clientX - r.left)));
      slZ.value = zStar;
      redraw();
    }
    bindLatDrag(cvAE, () => frAE);
    bindLatDrag(cvVAE, () => frVAE);

    slZ.addEventListener('input', () => { zStar = +slZ.value; redraw(); });
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
      seed = seed * 5 + 17;
      resetAll();
      redraw();
    });

    resetAll();
    redraw();
    P.onRedraw(redraw);
    P.observeResize(cvAE, redraw);
    P.observeResize(cvVAE, redraw);
    P.observeResize(cvData, redraw);
  }

  DL.sections.push({ name: 's2-latent-space', init });
})();
