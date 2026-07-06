/* Seção 1: neurônio do Perceptron e a geometria da fronteira de decisão. */
(function () {
  'use strict';
  window.DL = window.DL || {};
  DL.sections = DL.sections || [];

  const R = 3.5, NPTS = 30;

  function init() {
    const U = DL.utils, P = DL.plot;
    const $ = (id) => document.getElementById(id);

    const cv = $('s1-plane');
    const sl = { t1: $('s1-t1'), t2: $('s1-t2'), t0: $('s1-t0') };
    const btnResample = $('s1-resample');
    const eq = $('s1-eq');

    let seed = 42;
    let pts = [];

    function regenerate() {
      const rng = U.mulberry32(seed);
      pts = [];
      for (let i = 0; i < NPTS; i++) {
        pts.push([(rng() * 2 - 1) * 3, (rng() * 2 - 1) * 3]);
      }
    }

    function theta() { return [+sl.t0.value, +sl.t1.value, +sl.t2.value]; }
    function predict(th, x, y) { return th[0] + th[1] * x + th[2] * y >= 0 ? 1 : -1; }

    function fmtNum(v) { return (v < 0 ? '−' : '') + Math.abs(v).toFixed(1); }
    function term(v, name) { return (v < 0 ? ' − ' : ' + ') + Math.abs(v).toFixed(1) + '·' + name; }

    function arrow(ctx, fr, x0, y0, x1, y1, color, width, alpha) {
      const px0 = fr.X(x0), py0 = fr.Y(y0), px1 = fr.X(x1), py1 = fr.Y(y1);
      const ang = Math.atan2(py1 - py0, px1 - px0);
      const hs = 5 + 2 * width;
      ctx.save();
      ctx.globalAlpha = alpha != null ? alpha : 1;
      ctx.strokeStyle = color;
      ctx.fillStyle = color;
      ctx.lineWidth = width;
      ctx.beginPath();
      ctx.moveTo(px0, py0);
      ctx.lineTo(px1, py1);
      ctx.stroke();
      ctx.beginPath();
      ctx.moveTo(px1, py1);
      ctx.lineTo(px1 - hs * Math.cos(ang - 0.45), py1 - hs * Math.sin(ang - 0.45));
      ctx.lineTo(px1 - hs * Math.cos(ang + 0.45), py1 - hs * Math.sin(ang + 0.45));
      ctx.closePath();
      ctx.fill();
      ctx.restore();
    }

    function redraw() {
      const th = P.theme();
      const [t0, t1, t2] = theta();
      $('s1-t0-val').textContent = fmtNum(t0);
      $('s1-t1-val').textContent = fmtNum(t1);
      $('s1-t2-val').textContent = fmtNum(t2);
      eq.textContent = 'ŷ = sinal(' + fmtNum(t0) + term(t1, 'x₁') + term(t2, 'x₂') + ')';

      const { ctx, w, h } = P.setup(cv);
      P.clear(ctx, w, h);
      const fr = P.frame(ctx, w, h, -R, R, -R, R);

      /* Semiplanos: ciano para classe +1, laranja para −1 */
      const cp = P.rgb(th.cyan), cn = P.rgb(th.orange);
      P.heatmap(ctx, fr, (x, y) => t0 + t1 * x + t2 * y, 60, 60,
        (v) => (v >= 0 ? [cp[0], cp[1], cp[2], 30] : [cn[0], cn[1], cn[2], 30]));
      P.axes(ctx, fr, { xlabel: 'x₁', ylabel: 'x₂', xticks: [-3, 0, 3], yticks: [-3, 0, 3] });

      /* Fronteira, marcador de ângulo reto e vetor de pesos (recortados na área do gráfico) */
      const n2 = t1 * t1 + t2 * t2;
      if (n2 > 1e-6) {
        ctx.save();
        ctx.beginPath();
        ctx.rect(fr.X(fr.xmin), fr.Y(fr.ymax), fr.iw, fr.ih);
        ctx.clip();
        const nrm = Math.sqrt(n2);
        const fx = -t0 * t1 / n2, fy = -t0 * t2 / n2;  /* pé da reta: ponto mais próximo da origem */
        const ux = t1 / nrm, uy = t2 / nrm;            /* direção normal (vetor de pesos) */
        const vx = -uy, vy = ux;                       /* direção da reta */
        const L = 3 * R;
        P.line(ctx, fr, [fx - L * vx, fx + L * vx], [fy - L * vy, fy + L * vy], th.fg, { width: 1.8 });
        const s = 0.22;
        P.line(ctx, fr,
          [fx + s * vx, fx + s * vx + s * ux, fx + s * ux],
          [fy + s * vy, fy + s * vy + s * uy, fy + s * uy], th.fg, { width: 1 });
        arrow(ctx, fr, fx, fy, fx + 1.4 * ux, fy + 1.4 * uy, th.green, 2.4);
        P.label(ctx, fr.X(fx + 1.75 * ux), fr.Y(fy + 1.75 * uy) + 4, '(θ₁, θ₂)', th.green, 'center');
        ctx.restore();
      }

      /* Pontos coloridos pela classe prevista */
      const pos = pts.filter((p) => predict([t0, t1, t2], p[0], p[1]) === 1);
      const neg = pts.filter((p) => predict([t0, t1, t2], p[0], p[1]) === -1);
      P.scatter(ctx, fr, pos, th.cyan, { r: 4, alpha: 0.9 });
      P.scatter(ctx, fr, neg, th.orange, { r: 4, alpha: 0.9 });
      P.label(ctx, fr.X(-R) + 8, 16, 'ŷ = +1', th.cyan);
      P.label(ctx, fr.X(-R) + 8, 30, 'ŷ = −1', th.orange);
    }

    for (const k of ['t0', 't1', 't2']) sl[k].addEventListener('input', redraw);
    btnResample.addEventListener('click', () => {
      seed = (seed * 1664525 + 1013904223) >>> 0;
      regenerate();
      redraw();
    });

    regenerate();
    redraw();
    P.onRedraw(redraw);
    P.observeResize(cv, redraw);
  }

  DL.sections.push({ name: 's1-neuron', init });
})();
