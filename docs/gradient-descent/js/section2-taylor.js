/* Seção 2: aproximação de Taylor de grau n em torno de θ0.
   Mesma função não convexa da seção 3: f(θ) = (θ⁴ − 3θ² + θ)/4.
   Como f é um polinômio de grau 4, a série com n = 4 é exata. O zoom na
   janela [θ0−r, θ0+r] mostra que o erro máximo cai com r elevado a n+1:
   para passos pequenos, a primeira ordem domina. */
(function () {
  'use strict';
  window.DL = window.DL || {};
  DL.sections = DL.sections || [];

  const XMIN = -2.5, XMAX = 2.5, YMIN = -1.3, YMAX = 6;

  function f(x) { return (x * x * x * x - 3 * x * x + x) / 4; }
  function df(x) { return x * x * x - 1.5 * x + 0.25; }
  function d2f(x) { return 3 * x * x - 1.5; }

  function init() {
    const P = DL.plot;
    const $ = (id) => document.getElementById(id);

    const cvFn = $('s2-fn'), cvZoom = $('s2-zoom');
    const slR = $('s2-r'), slDeg = $('s2-deg');

    /* Slider em escala log: r ∈ [0.02, 2] */
    const sliderToR = (v) => 0.02 * Math.pow(100, v / 100);

    let theta0 = 0.8;
    let r = sliderToR(+slR.value);
    let deg = +slDeg.value;
    let frFn = null, dragging = false;

    /* Coeficientes de Taylor em θ0: c_j = f⁽ʲ⁾(θ0)/j!.
       Derivadas de ordem 3 e 4: f‴(θ) = 6θ e f⁗(θ) = 6, então
       c3 = 6θ0/3! = θ0 e c4 = 6/4! = 0.25 (o coeficiente líder de f). */
    const coefs = () => [f(theta0), df(theta0), d2f(theta0) / 2, theta0, 0.25];

    function tn(x) {
      const c = coefs();
      const d = x - theta0;
      let acc = 0;
      for (let j = deg; j >= 0; j--) acc = acc * d + c[j];
      return acc;
    }

    function clipPlot(ctx, fr) {
      ctx.save();
      ctx.beginPath();
      ctx.rect(fr.X(fr.xmin), fr.Y(fr.ymax), fr.iw, fr.ih);
      ctx.clip();
    }

    function sampled(fn, lo, hi) {
      const xs = [], ys = [];
      const step = (hi - lo) / 220;
      for (let x = lo; x <= hi + 1e-9; x += step) { xs.push(x); ys.push(fn(x)); }
      return [xs, ys];
    }

    function maxErr() {
      let m = 0;
      for (let i = 0; i <= 200; i++) {
        const x = theta0 - r + (2 * r) * i / 200;
        m = Math.max(m, Math.abs(f(x) - tn(x)));
      }
      return m;
    }

    function drawCurves(ctx, fr, th) {
      let c = sampled(f, fr.xmin, fr.xmax);
      P.line(ctx, fr, c[0], c[1], th.purple, { width: 2.2 });
      c = sampled(tn, fr.xmin, fr.xmax);
      P.line(ctx, fr, c[0], c[1], th.cyan, { width: 1.8 });
      /* Ponto de expansão θ0 */
      ctx.beginPath();
      ctx.arc(fr.X(theta0), fr.Y(f(theta0)), 6.5, 0, 2 * Math.PI);
      ctx.fillStyle = th.pink;
      ctx.fill();
      ctx.lineWidth = 1.5;
      ctx.strokeStyle = th.bg;
      ctx.stroke();
    }

    function drawFn() {
      const th = P.theme();
      const { ctx, w, h } = P.setup(cvFn);
      P.clear(ctx, w, h);
      const fr = P.frame(ctx, w, h, XMIN, XMAX, YMIN, YMAX);
      frFn = fr;
      P.axes(ctx, fr, { xlabel: 'θ', ylabel: 'f(θ)', xticks: [-2, -1, 0, 1, 2], yticks: [0, 2, 4, 6] });

      clipPlot(ctx, fr);
      /* Faixa da janela de zoom */
      const bx0 = fr.X(Math.max(fr.xmin, theta0 - r));
      const bx1 = fr.X(Math.min(fr.xmax, theta0 + r));
      const [lr, lg, lb] = P.rgb(th.line);
      ctx.fillStyle = 'rgba(' + lr + ',' + lg + ',' + lb + ',0.28)';
      ctx.fillRect(bx0, fr.Y(fr.ymax), bx1 - bx0, fr.ih);

      drawCurves(ctx, fr, th);
      ctx.restore();

      P.label(ctx, fr.X(XMIN) + 8, fr.Y(YMAX) + 14, 'f(θ)', th.purple);
      P.label(ctx, fr.X(XMIN) + 8, fr.Y(YMAX) + 28, 'grau ' + deg, th.cyan);
      P.label(ctx, fr.X(theta0), fr.Y(YMIN) - 8, 'θ_0', th.pink, 'center');
    }

    function drawZoom() {
      const th = P.theme();
      const { ctx, w, h } = P.setup(cvZoom);
      P.clear(ctx, w, h);
      const lo = theta0 - r, hi = theta0 + r;
      /* Faixa vertical: f e a aproximação na janela, com margem. */
      let ymin = Infinity, ymax = -Infinity;
      for (let i = 0; i <= 60; i++) {
        const x = lo + (hi - lo) * i / 60;
        for (const v of [f(x), tn(x)]) {
          ymin = Math.min(ymin, v);
          ymax = Math.max(ymax, v);
        }
      }
      const m = Math.max(1e-4, (ymax - ymin) * 0.12);
      const fr = P.frame(ctx, w, h, lo, hi, ymin - m, ymax + m);
      P.axes(ctx, fr, { xlabel: 'θ', xticks: [theta0] });

      clipPlot(ctx, fr);
      drawCurves(ctx, fr, th);
      ctx.restore();
    }

    function updateReadout() {
      const e = maxErr();
      const fmtE = (v) => (v === 0 ? '0 (exata)' : v < 1e-3 ? v.toExponential(1) : v.toFixed(3));
      $('s2-readout').textContent =
        'θ0 = ' + theta0.toFixed(2) +
        ' · grau ' + deg +
        ' · erro máx na janela = ' + fmtE(e);
    }

    function redraw() {
      drawFn();
      drawZoom();
      updateReadout();
    }

    function setTheta0(x) {
      theta0 = Math.min(XMAX - 0.2, Math.max(XMIN + 0.2, x));
      redraw();
    }

    cvFn.addEventListener('pointerdown', (e) => {
      if (!frFn) return;
      dragging = true;
      cvFn.setPointerCapture(e.pointerId);
      const rect = cvFn.getBoundingClientRect();
      setTheta0(frFn.invX(e.clientX - rect.left));
    });
    cvFn.addEventListener('pointermove', (e) => {
      if (!dragging || !frFn) return;
      const rect = cvFn.getBoundingClientRect();
      setTheta0(frFn.invX(e.clientX - rect.left));
    });
    cvFn.addEventListener('pointerup', () => { dragging = false; });

    slDeg.addEventListener('input', () => {
      deg = +slDeg.value;
      $('s2-deg-val').textContent = String(deg);
      redraw();
    });
    slR.addEventListener('input', () => {
      r = sliderToR(+slR.value);
      $('s2-r-val').textContent = r.toFixed(2);
      redraw();
    });

    $('s2-deg-val').textContent = String(deg);
    $('s2-r-val').textContent = r.toFixed(2);
    redraw();
    P.onRedraw(redraw);
    P.observeResize(cvFn, redraw);
    P.observeResize(cvZoom, redraw);
  }

  DL.sections.push({ name: 's2-taylor', init });
})();
