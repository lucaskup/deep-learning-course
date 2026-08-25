/* Seção 4: o modelo de Gabor dos slides ajustado na mão aos dados,
   ŷ = sin(θ₀ + 0.06·θ₁x)·exp(−(θ₀ + 0.06·θ₁x)²/32). Define DL.gaborDemo
   (dataset fixo, loss, gradiente e mínimo global), reutilizado na seção 5. */
(function () {
  'use strict';
  window.DL = window.DL || {};
  DL.sections = DL.sections || [];

  /* Domínio dos parâmetros (o mesmo da figura dos slides) e dataset fixo:
     30 pontos de um Gabor verdadeiro em (0, 16.6) com ruído gaussiano. */
  const T0MIN = -10, T0MAX = 10, T1MIN = 2.5, T1MAX = 22.5;
  const N = 30, NOISE = 0.28, TRUE0 = 0, TRUE1 = 16.6;
  const XMIN = -15, XMAX = 15;

  const gabor = (x, t0, t1) => {
    const a = t0 + 0.06 * t1 * x;
    return Math.sin(a) * Math.exp(-a * a / 32);
  };

  const rng = DL.utils.mulberry32(7);
  const xs = [], ys = [];
  for (let i = 0; i < N; i++) {
    const x = XMIN + (XMAX - XMIN) * rng();
    xs.push(x);
    ys.push(gabor(x, TRUE0, TRUE1) + NOISE * DL.utils.randn(rng));
  }

  function loss(t0, t1) {
    let s = 0;
    for (let i = 0; i < N; i++) {
      const d = gabor(xs[i], t0, t1) - ys[i];
      s += d * d;
    }
    return s / N;
  }

  /* ∇L analítico: dŷ/da = e^{−a²/32}(cos a − (a/16)·sin a), com
     ∂a/∂θ₀ = 1 e ∂a/∂θ₁ = 0.06x. */
  function grad(t0, t1) {
    let g0 = 0, g1 = 0;
    for (let i = 0; i < N; i++) {
      const a = t0 + 0.06 * t1 * xs[i];
      const e = Math.exp(-a * a / 32);
      const d = Math.sin(a) * e - ys[i];
      const dyda = e * (Math.cos(a) - (a / 16) * Math.sin(a));
      g0 += 2 * d * dyda;
      g1 += 2 * d * dyda * 0.06 * xs[i];
    }
    return [g0 / N, g1 / N];
  }

  /* Mínimo global numérico numa grade fina do domínio. */
  let best = [0, 0], bestL = Infinity;
  for (let i = 0; i <= 300; i++) {
    for (let j = 0; j <= 300; j++) {
      const t0 = T0MIN + (T0MAX - T0MIN) * i / 300;
      const t1 = T1MIN + (T1MAX - T1MIN) * j / 300;
      const v = loss(t0, t1);
      if (v < bestL) { bestL = v; best = [t0, t1]; }
    }
  }

  DL.gaborDemo = { T0MIN, T0MAX, T1MIN, T1MAX, XMIN, XMAX, xs, ys, gabor, loss, grad, best, bestL };

  function init() {
    const P = DL.plot;
    const $ = (id) => document.getElementById(id);

    const cvData = $('s4-data'), cvSlice = $('s4-slice');
    const slT0 = $('s4-t0'), slT1 = $('s4-t1');
    const btnReset = $('s4-reset');

    let t0 = +slT0.value, t1 = +slT1.value;

    function drawData() {
      const th = P.theme();
      const { ctx, w, h } = P.setup(cvData);
      P.clear(ctx, w, h);
      const fr = P.frame(ctx, w, h, XMIN, XMAX, -1.35, 1.35);
      P.axes(ctx, fr, { xlabel: 'x', ylabel: 'y', xticks: [-10, 0, 10], yticks: [-1, 0, 1] });
      P.scatter(ctx, fr, xs.map((x, i) => [x, ys[i]]), th.orange, { r: 3.4, alpha: 0.85 });
      const cx = [], cy = [];
      for (let i = 0; i <= 240; i++) {
        const x = XMIN + (XMAX - XMIN) * i / 240;
        cx.push(x);
        cy.push(gabor(x, t0, t1));
      }
      P.line(ctx, fr, cx, cy, th.cyan, { width: 2 });
    }

    function drawSlice() {
      const th = P.theme();
      const { ctx, w, h } = P.setup(cvSlice);
      P.clear(ctx, w, h);
      const fr = P.frame(ctx, w, h, T1MIN, T1MAX, 0, 1.05);
      P.axes(ctx, fr, { xlabel: 'θ_1', ylabel: 'L', xticks: [2.5, 12.5, 22.5], yticks: [0, 0.5, 1] });
      const sx = [], sy = [];
      for (let i = 0; i <= 200; i++) {
        const v = T1MIN + (T1MAX - T1MIN) * i / 200;
        sx.push(v);
        sy.push(loss(t0, v));
      }
      /* Loss do mínimo global como referência do quão baixo dá para chegar. */
      P.line(ctx, fr, [T1MIN, T1MAX], [bestL, bestL], th.green, { width: 1.2, dash: [5, 4] });
      P.label(ctx, fr.X(T1MAX) - 4, fr.Y(bestL) - 6, 'mínimo global', th.green, 'right');
      P.line(ctx, fr, sx, sy, th.purple, { width: 1.8 });
      const cur = loss(t0, t1);
      ctx.beginPath();
      ctx.arc(fr.X(t1), fr.Y(cur), 5, 0, 2 * Math.PI);
      ctx.fillStyle = th.pink;
      ctx.fill();
      ctx.strokeStyle = th.fg;
      ctx.lineWidth = 1.5;
      ctx.stroke();
    }

    function updateReadout() {
      $('s4-readout').textContent =
        'L(θ₀, θ₁) = ' + loss(t0, t1).toFixed(3) + ' · mínimo global ≈ ' + bestL.toFixed(3);
    }

    function redraw() {
      drawData();
      drawSlice();
      updateReadout();
    }

    function syncLabels() {
      $('s4-t0-val').textContent = t0.toFixed(2);
      $('s4-t1-val').textContent = t1.toFixed(2);
    }

    slT0.addEventListener('input', () => { t0 = +slT0.value; syncLabels(); redraw(); });
    slT1.addEventListener('input', () => { t1 = +slT1.value; syncLabels(); redraw(); });
    btnReset.addEventListener('click', () => {
      slT0.value = '-4';
      slT1.value = '18';
      t0 = -4; t1 = 18;
      syncLabels();
      redraw();
    });

    syncLabels();
    redraw();
    P.onRedraw(redraw);
    P.observeResize(cvData, redraw);
    P.observeResize(cvSlice, redraw);
  }

  DL.sections.push({ name: 's4-gabor-fit', init });
})();
