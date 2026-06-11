/* Seção 2: composição de K camadas inversíveis em 1D, f = f_K ∘ … ∘ f_1,
   com a densidade se deformando passo a passo e |f′| = Π|f_k′|. */
(function () {
  'use strict';
  window.DL = window.DL || {};
  DL.sections = DL.sections || [];

  const K_MAX = 8, ZMIN = -3.5, ZMAX = 3.5, NGRID = 500;
  const XLIM = 4.5;

  function init() {
    const P = DL.plot;
    const U = DL.utils;
    const $ = (id) => document.getElementById(id);

    const cvDen = $('s2-density'), cvMap = $('s2-map');
    const slK = $('s2-k');

    let K = +slK.value;
    let seed = 7;
    let layers = [];
    /* stages[k] = { xs, ps } : posição e densidade após k camadas, para a grade de z */
    let stages = [], zgrid = [];

    function makeLayers() {
      const rng = U.mulberry32(seed);
      layers = [];
      for (let k = 0; k < K_MAX; k++) {
        const s = 0.45 + 0.55 * rng();
        layers.push({
          s,
          alpha: s * (-0.7 + 2.3 * rng()),
          mu: -1.6 + 3.2 * rng(),
        });
      }
    }

    /* Propaga a grade inteira camada a camada, acumulando o produto dos jacobianos. */
    function recompute() {
      zgrid = [];
      for (let i = 0; i < NGRID; i++) zgrid.push(ZMIN + (ZMAX - ZMIN) * i / (NGRID - 1));
      const x = zgrid.slice();
      const d = zgrid.map(() => 1);
      stages = [{ xs: x.slice(), ps: zgrid.map((z, i) => U.gaussPdf(z, 0, 1) / d[i]) }];
      for (let k = 0; k < K_MAX; k++) {
        const L = layers[k];
        for (let i = 0; i < NGRID; i++) {
          const th = Math.tanh((x[i] - L.mu) / L.s);
          d[i] *= 1 + (L.alpha / L.s) * (1 - th * th);
          x[i] = x[i] + L.alpha * th;
        }
        stages.push({ xs: x.slice(), ps: zgrid.map((z, i) => U.gaussPdf(z, 0, 1) / d[i]) });
      }
    }

    function drawDensity() {
      const th = P.theme();
      const { ctx, w, h } = P.setup(cvDen);
      P.clear(ctx, w, h);
      let ymax = 0.45;
      for (const v of stages[K].ps) if (v > ymax) ymax = v;
      ymax = Math.min(1.6, ymax * 1.12);
      const fr = P.frame(ctx, w, h, -XLIM, XLIM, 0, ymax);
      P.axes(ctx, fr, {
        xlabel: 'x', ylabel: 'p(x)',
        xticks: [-4, -2, 0, 2, 4],
        yticks: [0, +(ymax / 2).toFixed(1)],
      });
      /* Base tracejada, intermediárias fantasma, final em destaque */
      P.line(ctx, fr, stages[0].xs, stages[0].ps, th.cyan, { width: 1.4, dash: [5, 4], alpha: 0.8 });
      for (let k = 1; k < K; k++) {
        P.line(ctx, fr, stages[k].xs, stages[k].ps, th.comment, { width: 1.2, alpha: 0.3 });
      }
      if (K > 0) P.line(ctx, fr, stages[K].xs, stages[K].ps, th.orange, { width: 2.4 });
      P.label(ctx, fr.X(-XLIM) + 8, fr.Y(ymax) + 12, 'base p(z)', th.cyan);
      P.label(ctx, fr.X(-XLIM) + 8, fr.Y(ymax) + 26, 'após K camadas', th.orange);
    }

    function drawMap() {
      const th = P.theme();
      const { ctx, w, h } = P.setup(cvMap);
      P.clear(ctx, w, h);
      const fr = P.frame(ctx, w, h, ZMIN, ZMAX, -XLIM, XLIM);
      P.axes(ctx, fr, { xlabel: 'z', ylabel: 'x = f(z)', xticks: [-3, 0, 3], yticks: [-4, 0, 4] });
      P.line(ctx, fr, [ZMIN, ZMAX], [ZMIN, ZMAX], th.comment, { width: 1, dash: [4, 4], alpha: 0.5 });
      P.line(ctx, fr, zgrid, stages[K].xs, th.orange, { width: 2.2 });
    }

    function updateReadout() {
      /* Jacobiano composto em z = 0: produto dos f_k′ ao longo do caminho. */
      let x = 0, prod = 1;
      for (let k = 0; k < K; k++) {
        const L = layers[k];
        const t = Math.tanh((x - L.mu) / L.s);
        prod *= 1 + (L.alpha / L.s) * (1 - t * t);
        x = x + L.alpha * t;
      }
      $('s2-readout').textContent =
        'K = ' + K + ' · |f′(0)| = Π|f_k′| = ' + prod.toFixed(2) +
        ' · log|f′(0)| = Σ log|f_k′| = ' + Math.log(prod).toFixed(2);
    }

    function redraw() {
      drawDensity();
      drawMap();
      updateReadout();
    }

    function setK(v) {
      K = Math.max(0, Math.min(K_MAX, v));
      slK.value = K;
      $('s2-k-val').textContent = K;
      redraw();
    }

    slK.addEventListener('input', () => setK(+slK.value));
    $('s2-step').addEventListener('click', () => setK(K + 1));
    $('s2-reset').addEventListener('click', () => setK(0));
    $('s2-shuffle').addEventListener('click', () => {
      seed = (seed * 1103515245 + 12345) % 2147483647;
      makeLayers();
      recompute();
      redraw();
    });

    makeLayers();
    recompute();
    redraw();
    P.onRedraw(redraw);
    P.observeResize(cvDen, redraw);
    P.observeResize(cvMap, redraw);
  }

  DL.sections.push({ name: 's2-composition', init });
})();
