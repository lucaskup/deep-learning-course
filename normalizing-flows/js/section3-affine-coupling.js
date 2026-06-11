/* Seção 3: affine coupling em 2D. Metade A é copiada, metade B sofre
   y^B = x^B·exp(s(x^A)) + t(x^A); log|det J| = s(x^A). Interpolação α
   anima da identidade até a transformação completa. */
(function () {
  'use strict';
  window.DL = window.DL || {};
  DL.sections = DL.sections || [];

  const LIM = 3.5, NSAMP = 900;

  function init() {
    const P = DL.plot;
    const U = DL.utils;
    const $ = (id) => document.getElementById(id);

    const cvPlane = $('s3-plane'), cvSt = $('s3-st');
    const slWs = $('s3-ws'), slWt = $('s3-wt'), slAlpha = $('s3-alpha');
    const tabB = $('s3-tab-b'), tabA = $('s3-tab-a');
    const btnAnim = $('s3-anim');

    let ws = +slWs.value, wt = +slWt.value, alpha = +slAlpha.value;
    let act = 1;            /* dimensão transformada (1 = x2); passiva = 1 − act */
    let seed = 3;
    let zsamp = [];
    let animating = false, dir = -1;

    const sFn = (a) => ws * Math.tanh(a);
    const tFn = (a) => wt * 0.5 * (a * a - 1);

    function resample() {
      const rng = U.mulberry32(seed);
      zsamp = [];
      for (let i = 0; i < NSAMP; i++) zsamp.push([U.randn(rng), U.randn(rng)]);
    }

    function fwd(p) {
      const pas = 1 - act;
      const a = p[pas];
      const out = [0, 0];
      out[pas] = a;
      out[act] = p[act] * Math.exp(alpha * sFn(a)) + alpha * tFn(a);
      return out;
    }

    /* Densidade exata via mudança de variável (inversa em forma fechada). */
    function density(x, y) {
      const q = [x, y];
      const pas = 1 - act;
      const a = q[pas];
      const s = alpha * sFn(a), t = alpha * tFn(a);
      const b = (q[act] - t) * Math.exp(-s);
      return U.gaussPdf(a, 0, 1) * U.gaussPdf(b, 0, 1) * Math.exp(-s);
    }

    function drawPlane() {
      const th = P.theme();
      const { ctx, w, h } = P.setup(cvPlane);
      P.clear(ctx, w, h);
      const fr = P.frame(ctx, w, h, -LIM, LIM, -LIM, LIM);
      const pmax = 1 / (2 * Math.PI) * Math.exp(Math.abs(ws) * alpha);
      P.heatmap(ctx, fr, density, 80, 80, (v) => {
        const u = Math.pow(Math.min(1, v / pmax), 0.55);
        const col = P.viridis(0.25 + 0.75 * u);
        return [col[0], col[1], col[2], Math.round(225 * u)];
      });
      P.axes(ctx, fr, { xlabel: 'x_1', ylabel: 'x_2', xticks: [-3, 0, 3], yticks: [-3, 0, 3] });

      /* Grade deformada: linhas constantes na coordenada passiva são retas,
         linhas constantes na coordenada ativa viram curvas. */
      const pas = 1 - act;
      ctx.save();
      ctx.beginPath();
      ctx.rect(fr.X(-LIM), fr.Y(LIM), fr.iw, fr.ih);
      ctx.clip();
      for (let cv = -3; cv <= 3; cv++) {
        /* coordenada passiva fixa em cv: segmento reto */
        const p1 = [0, 0], p2 = [0, 0];
        p1[pas] = cv; p2[pas] = cv;
        p1[act] = -LIM * Math.exp(alpha * sFn(cv)) + alpha * tFn(cv);
        p2[act] = LIM * Math.exp(alpha * sFn(cv)) + alpha * tFn(cv);
        P.line(ctx, fr, [p1[0], p2[0]], [p1[1], p2[1]], th.comment, { width: 1, alpha: 0.3 });
        /* coordenada ativa fixa em cv: curva sobre a passiva */
        const xs = [], ys = [];
        for (let i = 0; i <= 70; i++) {
          const a = -LIM + 2 * LIM * i / 70;
          const pt = [0, 0];
          pt[pas] = a;
          pt[act] = cv * Math.exp(alpha * sFn(a)) + alpha * tFn(a);
          xs.push(pt[0]); ys.push(pt[1]);
        }
        P.line(ctx, fr, xs, ys, th.comment, { width: 1, alpha: 0.3 });
      }
      /* Amostras: base (fantasma) e transformadas */
      P.scatter(ctx, fr, zsamp, th.cyan, { r: 1.5, alpha: 0.18 });
      P.scatter(ctx, fr, zsamp.map(fwd), th.orange, { r: 2, alpha: 0.65 });
      ctx.restore();
      P.label(ctx, fr.X(-LIM) + 8, fr.Y(LIM) + 14, 'z ~ N(0, I)', th.cyan);
      P.label(ctx, fr.X(-LIM) + 8, fr.Y(LIM) + 28, 'f(z) com α = ' + alpha.toFixed(2), th.orange);
    }

    function drawSt() {
      const th = P.theme();
      const { ctx, w, h } = P.setup(cvSt);
      P.clear(ctx, w, h);
      const fr = P.frame(ctx, w, h, -LIM, LIM, -2.6, 2.6);
      P.axes(ctx, fr, { xlabel: 'x^A', xticks: [-3, 0, 3], yticks: [-2, 0, 2] });
      P.line(ctx, fr, [-LIM, LIM], [0, 0], th.comment, { width: 1, dash: [4, 4], alpha: 0.5 });
      const xs = [], ss = [], ts = [];
      for (let i = 0; i <= 200; i++) {
        const a = -LIM + 2 * LIM * i / 200;
        xs.push(a);
        ss.push(alpha * sFn(a));
        ts.push(alpha * tFn(a));
      }
      P.line(ctx, fr, xs, ss, th.orange, { width: 2.2 });
      P.line(ctx, fr, xs, ts, th.green, { width: 2.2 });
      P.label(ctx, fr.X(-LIM) + 8, fr.Y(2.6) + 14, 'α·s(x^A)  (log-escala)', th.orange);
      P.label(ctx, fr.X(-LIM) + 8, fr.Y(2.6) + 28, 'α·t(x^A)  (translação)', th.green);
    }

    function updateReadout() {
      let mn = Infinity, mx = -Infinity, mean = 0;
      const pas = 1 - act;
      for (const p of zsamp) {
        const s = alpha * sFn(p[pas]);
        if (s < mn) mn = s;
        if (s > mx) mx = s;
        mean += s;
      }
      mean /= zsamp.length;
      $('s3-readout').textContent =
        'log|det J| = α·s(x^A): médio = ' + mean.toFixed(2) +
        ' · mín = ' + mn.toFixed(2) + ' · máx = ' + mx.toFixed(2);
    }

    function redraw() {
      drawPlane();
      drawSt();
      updateReadout();
    }

    function setAlpha(v, syncSlider) {
      alpha = Math.max(0, Math.min(1, v));
      if (syncSlider) slAlpha.value = alpha;
      $('s3-alpha-val').textContent = alpha.toFixed(2);
    }

    function stopAnim() {
      animating = false;
      btnAnim.textContent = '▶ Animar';
      DL.stopTicker(tick);
    }

    function tick(dt) {
      let v = alpha + dir * dt * 0.55;
      if (v >= 1) { v = 1; dir = -1; }
      if (v <= 0) { v = 0; dir = 1; }
      setAlpha(v, true);
      redraw();
    }

    btnAnim.addEventListener('click', () => {
      if (animating) { stopAnim(); return; }
      animating = true;
      dir = alpha >= 1 ? -1 : 1;
      btnAnim.textContent = '⏸ Pausar';
      DL.startTicker(tick);
    });

    slAlpha.addEventListener('input', () => { stopAnim(); setAlpha(+slAlpha.value, false); redraw(); });
    slWs.addEventListener('input', () => {
      ws = +slWs.value;
      $('s3-ws-val').textContent = ws.toFixed(2);
      redraw();
    });
    slWt.addEventListener('input', () => {
      wt = +slWt.value;
      $('s3-wt-val').textContent = wt.toFixed(2);
      redraw();
    });

    function setMode(newAct) {
      act = newAct;
      tabB.classList.toggle('active', act === 1);
      tabA.classList.toggle('active', act === 0);
      redraw();
    }
    tabB.addEventListener('click', () => setMode(1));
    tabA.addEventListener('click', () => setMode(0));

    $('s3-resample').addEventListener('click', () => {
      seed = (seed * 1103515245 + 12345) % 2147483647;
      resample();
      redraw();
    });

    resample();
    redraw();
    P.onRedraw(redraw);
    P.observeResize(cvPlane, redraw);
    P.observeResize(cvSt, redraw);
  }

  DL.sections.push({ name: 's3-affine-coupling', init });
})();
