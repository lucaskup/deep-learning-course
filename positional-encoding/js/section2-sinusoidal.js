/* Seção 2: PE sinusoidal. Heatmap posição × dimensão e curvas sin/cos
   do par de frequência selecionado (ω_k = 10000^(−2k/d)). */
(function () {
  'use strict';
  window.DL = window.DL || {};
  DL.sections = DL.sections || [];

  function omega(k, d) { return Math.pow(10000, -2 * k / d); }

  function pe(i, dim, d) {
    const k = dim >> 1;
    const w = omega(k, d);
    return dim % 2 === 0 ? Math.sin(i * w) : Math.cos(i * w);
  }

  function init() {
    const P = DL.plot;
    const $ = (id) => document.getElementById(id);

    const cvHeat = $('s2-heat'), cvCurves = $('s2-curves');
    const slD = $('s2-d'), slL = $('s2-len'), slK = $('s2-k');

    let d = +slD.value, L = +slL.value, k = +slK.value;

    function drawHeat() {
      const th = P.theme();
      const { ctx, w, h } = P.setup(cvHeat);
      P.clear(ctx, w, h);
      const fr = P.frame(ctx, w, h, 0, L, 0, d);
      const oc = P.rgb(th.orange), cc = P.rgb(th.cyan);
      P.heatmap(ctx, fr, (x, y) => {
        const i = Math.min(L - 1, Math.max(0, Math.floor(x)));
        const dim = Math.min(d - 1, Math.max(0, Math.floor(y)));
        return pe(i, dim, d);
      }, L, d, (v) => {
        const a = Math.round(10 + 220 * Math.abs(v));
        return v >= 0 ? [oc[0], oc[1], oc[2], a] : [cc[0], cc[1], cc[2], a];
      });
      P.axes(ctx, fr, {
        xlabel: 'posição i', ylabel: 'dimensão',
        xticks: [0, Math.round(L / 2), L],
        yticks: [0, Math.round(d / 2), d],
      });
      /* Destaca o par de dimensões (2k, 2k+1) selecionado. */
      ctx.strokeStyle = th.green;
      ctx.lineWidth = 2;
      ctx.strokeRect(fr.X(0), fr.Y(2 * k + 2), fr.iw, fr.Y(2 * k) - fr.Y(2 * k + 2));
      P.label(ctx, fr.X(L) - 6, fr.Y(2 * k + 2) - 4, 'dims ' + 2 * k + ', ' + (2 * k + 1), th.green, 'right');
      P.label(ctx, fr.X(0) + 8, fr.Y(d) + 14, '+1', th.orange);
      P.label(ctx, fr.X(0) + 34, fr.Y(d) + 14, '−1', th.cyan);
    }

    function drawCurves() {
      const th = P.theme();
      const { ctx, w, h } = P.setup(cvCurves);
      P.clear(ctx, w, h);
      const fr = P.frame(ctx, w, h, 0, L, -1.25, 1.25);
      P.axes(ctx, fr, {
        xlabel: 'posição i', ylabel: 'PE',
        xticks: [0, Math.round(L / 2), L],
        yticks: [-1, 0, 1],
      });
      const wk = omega(k, d);
      const n = 360;
      const xs = [], ysin = [], ycos = [];
      for (let t = 0; t <= n; t++) {
        const i = (t / n) * L;
        xs.push(i);
        ysin.push(Math.sin(i * wk));
        ycos.push(Math.cos(i * wk));
      }
      P.line(ctx, fr, [0, L], [0, 0], th.line, { width: 1, dash: [4, 4] });
      P.line(ctx, fr, xs, ysin, th.cyan, { width: 1.8 });
      P.line(ctx, fr, xs, ycos, th.orange, { width: 1.8 });
      /* Posições são inteiras: marca os valores discretos quando cabem. */
      if (L <= 48) {
        const ps = [], pc = [];
        for (let i = 0; i < L; i++) {
          ps.push([i, Math.sin(i * wk)]);
          pc.push([i, Math.cos(i * wk)]);
        }
        P.scatter(ctx, fr, ps, th.cyan, { r: 2.2, alpha: 0.9 });
        P.scatter(ctx, fr, pc, th.orange, { r: 2.2, alpha: 0.9 });
      }
      P.label(ctx, fr.X(0) + 8, fr.Y(fr.ymax) + 12, 'sin · dim ' + 2 * k, th.cyan);
      P.label(ctx, fr.X(0) + 8, fr.Y(fr.ymax) + 26, 'cos · dim ' + (2 * k + 1), th.orange);
    }

    function updateReadout() {
      const wk = omega(k, d);
      const period = 2 * Math.PI / wk;
      $('s2-readout').textContent =
        'ω_' + k + ' = ' + wk.toPrecision(3) + ' · período = ' +
        (period >= 1000 ? period.toExponential(2) : period.toFixed(1)) + ' posições';
    }

    function redraw() {
      drawHeat();
      drawCurves();
      updateReadout();
    }

    function syncK() {
      const kmax = d / 2 - 1;
      slK.max = kmax;
      if (k > kmax) k = kmax;
      slK.value = k;
      $('s2-k-val').textContent = k;
    }

    slD.addEventListener('input', () => {
      d = +slD.value;
      $('s2-d-val').textContent = d;
      syncK();
      redraw();
    });
    slL.addEventListener('input', () => {
      L = +slL.value;
      $('s2-len-val').textContent = L;
      redraw();
    });
    slK.addEventListener('input', () => {
      k = +slK.value;
      $('s2-k-val').textContent = k;
      redraw();
    });

    syncK();
    redraw();
    P.onRedraw(redraw);
    P.observeResize(cvHeat, redraw);
    P.observeResize(cvCurves, redraw);
  }

  DL.sections.push({ name: 's2-sinusoidal', init });
})();
