/* Seção 4: atenção como média ponderada dos values.
   c = Σ aᵢ·vᵢ com a = softmax(γ·s): combinação convexa, c fica no fecho
   convexo dos values; γ → 0 dá a média simples, γ grande dá quase one-hot. */
(function () {
  'use strict';
  window.DL = window.DL || {};
  DL.sections = DL.sections || [];

  const M = 6;
  const RANGE = 2.4, CLAMP = 2.1;

  function init() {
    const P = DL.plot, U = DL.utils;
    const $ = (id) => document.getElementById(id);

    const cvPlane = $('s4-plane'), cvW = $('s4-weights');
    const slG = $('s4-gamma');
    const fmt = (v, d) => v.toFixed(d == null ? 2 : d).replace('.', ',').replace('-', '−');

    let seed = 5;
    let gamma = +slG.value;
    let pts = [], scores = [];
    let frPlane = null, dragging = null;

    function regen() {
      const rng = U.mulberry32(seed);
      pts = [];
      scores = [];
      for (let i = 0; i < M; i++) {
        pts.push([(rng() * 2 - 1) * 1.7, (rng() * 2 - 1) * 1.7]);
        scores.push(Math.round(U.randn(rng) * 10) / 10);
      }
    }

    function softmax(s) {
      const mx = Math.max(...s);
      const e = s.map((v) => Math.exp(v - mx));
      const t = e.reduce((a, b) => a + b, 0);
      return e.map((v) => v / t);
    }

    function compute() {
      const a = softmax(scores.map((s) => gamma * s));
      const c = [0, 1].map((d) => pts.reduce((acc, p, i) => acc + a[i] * p[d], 0));
      const mean = [0, 1].map((d) => pts.reduce((acc, p) => acc + p[d], 0) / M);
      return { a, c, mean };
    }

    /* fecho convexo (cadeia monótona de Andrew) */
    function hull(points) {
      const p = points.slice().sort((u, v) => u[0] - v[0] || u[1] - v[1]);
      const cross = (o, a, b) => (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0]);
      const lower = [], upper = [];
      for (const q of p) {
        while (lower.length >= 2 && cross(lower[lower.length - 2], lower[lower.length - 1], q) <= 0) lower.pop();
        lower.push(q);
      }
      for (const q of p.slice().reverse()) {
        while (upper.length >= 2 && cross(upper[upper.length - 2], upper[upper.length - 1], q) <= 0) upper.pop();
        upper.push(q);
      }
      return lower.slice(0, -1).concat(upper.slice(0, -1));
    }

    function drawPlane() {
      const th = P.theme();
      const { ctx, w, h } = P.setup(cvPlane);
      P.clear(ctx, w, h);
      const fr = P.frame(ctx, w, h, -RANGE, RANGE, -RANGE, RANGE, { l: 10, r: 10, t: 10, b: 10 });
      frPlane = fr;
      const { a, c, mean } = compute();

      /* eixos pela origem */
      ctx.strokeStyle = th.line;
      ctx.lineWidth = 1;
      ctx.beginPath();
      ctx.moveTo(fr.X(-RANGE), fr.Y(0));
      ctx.lineTo(fr.X(RANGE), fr.Y(0));
      ctx.moveTo(fr.X(0), fr.Y(-RANGE));
      ctx.lineTo(fr.X(0), fr.Y(RANGE));
      ctx.stroke();

      /* fecho convexo dos values */
      const hu = hull(pts);
      ctx.strokeStyle = th.comment;
      ctx.setLineDash([5, 4]);
      ctx.beginPath();
      hu.forEach((p, i) => {
        if (i === 0) ctx.moveTo(fr.X(p[0]), fr.Y(p[1]));
        else ctx.lineTo(fr.X(p[0]), fr.Y(p[1]));
      });
      ctx.closePath();
      ctx.stroke();
      ctx.setLineDash([]);

      /* linhas vᵢ → c com opacidade proporcional ao peso */
      for (let i = 0; i < M; i++) {
        ctx.strokeStyle = th.cyan;
        ctx.globalAlpha = 0.15 + 0.75 * a[i];
        ctx.lineWidth = 1.2;
        ctx.beginPath();
        ctx.moveTo(fr.X(pts[i][0]), fr.Y(pts[i][1]));
        ctx.lineTo(fr.X(c[0]), fr.Y(c[1]));
        ctx.stroke();
        ctx.globalAlpha = 1;
      }

      /* values: raio acompanha o peso */
      for (let i = 0; i < M; i++) {
        ctx.fillStyle = th.cyan;
        ctx.globalAlpha = 0.85;
        ctx.beginPath();
        ctx.arc(fr.X(pts[i][0]), fr.Y(pts[i][1]), 4 + 16 * a[i], 0, 2 * Math.PI);
        ctx.fill();
        ctx.globalAlpha = 1;
        P.mathText(ctx, 'v_' + (i + 1), fr.X(pts[i][0]) + 8, fr.Y(pts[i][1]) - 8, th.fg, 'left', 11);
      }

      /* média simples (caso γ = 0) */
      ctx.strokeStyle = th.green;
      ctx.lineWidth = 2;
      ctx.beginPath();
      ctx.moveTo(fr.X(mean[0]) - 5, fr.Y(mean[1]) - 5);
      ctx.lineTo(fr.X(mean[0]) + 5, fr.Y(mean[1]) + 5);
      ctx.moveTo(fr.X(mean[0]) - 5, fr.Y(mean[1]) + 5);
      ctx.lineTo(fr.X(mean[0]) + 5, fr.Y(mean[1]) - 5);
      ctx.stroke();
      P.label(ctx, fr.X(mean[0]) + 8, fr.Y(mean[1]) + 12, 'média simples', th.green);

      /* saída c */
      ctx.fillStyle = th.purple;
      ctx.beginPath();
      ctx.arc(fr.X(c[0]), fr.Y(c[1]), 6, 0, 2 * Math.PI);
      ctx.fill();
      P.mathText(ctx, 'c', fr.X(c[0]) + 9, fr.Y(c[1]) + 4, th.purple, 'left', 13);
    }

    function drawWeights() {
      const th = P.theme();
      const { ctx, w, h } = P.setup(cvW);
      P.clear(ctx, w, h);
      const { a } = compute();
      const fr = P.frame(ctx, w, h, 0, M, 0, 1.1, { l: 34, r: 8, t: 8, b: 40 });
      P.axes(ctx, fr, { ylabel: 'a_i', yticks: [0, 0.5, 1] });
      const cw = fr.iw / M;

      ctx.fillStyle = th.green;
      a.forEach((v, i) => {
        ctx.fillRect(fr.X(i) + cw * 0.2, fr.Y(v), cw * 0.6, fr.Y(0) - fr.Y(v));
      });
      /* linha do caso uniforme 1/m */
      ctx.strokeStyle = th.comment;
      ctx.setLineDash([5, 4]);
      ctx.beginPath();
      ctx.moveTo(fr.X(0), fr.Y(1 / M));
      ctx.lineTo(fr.X(M), fr.Y(1 / M));
      ctx.stroke();
      ctx.setLineDash([]);
      P.mathText(ctx, '1/m', fr.X(M) - 4, fr.Y(1 / M) - 5, th.comment, 'right', 10);

      ctx.font = '10px sans-serif';
      ctx.textAlign = 'center';
      a.forEach((v, i) => {
        const cx = fr.X(i) + cw / 2;
        ctx.fillStyle = th.fg;
        ctx.fillText(fmt(v), cx, fr.Y(v) - 4);
      });
      for (let i = 0; i < M; i++) {
        const cx = fr.X(i) + cw / 2;
        P.mathText(ctx, 'v_' + (i + 1), cx, fr.Y(0) + 16, th.fg, 'center', 11);
        ctx.fillStyle = th.pink;
        ctx.font = '10px sans-serif';
        ctx.textAlign = 'center';
        ctx.fillText('s=' + fmt(scores[i], 1), cx, fr.Y(0) + 30);
      }
    }

    function updateReadout() {
      const { a, c } = compute();
      const mx = Math.max(...a);
      $('s4-readout').textContent =
        'γ = ' + fmt(gamma, 2) + ' · maior peso = ' + fmt(mx) +
        ' · c ≈ (' + fmt(c[0]) + '; ' + fmt(c[1]) + ')';
    }

    function redraw() {
      drawPlane();
      drawWeights();
      updateReadout();
    }

    /* ── arraste dos values ─────────────────────────────── */
    function toData(e) {
      const r = cvPlane.getBoundingClientRect();
      return [frPlane.invX(e.clientX - r.left), frPlane.invY(e.clientY - r.top)];
    }

    cvPlane.addEventListener('pointerdown', (e) => {
      if (!frPlane) return;
      const r = cvPlane.getBoundingClientRect();
      const px = e.clientX - r.left, py = e.clientY - r.top;
      let best = null, bestD = 20;
      pts.forEach((p, i) => {
        const dd = Math.hypot(px - frPlane.X(p[0]), py - frPlane.Y(p[1]));
        if (dd < bestD) { bestD = dd; best = i; }
      });
      if (best === null) return;
      dragging = best;
      cvPlane.setPointerCapture(e.pointerId);
    });
    cvPlane.addEventListener('pointermove', (e) => {
      if (dragging === null) return;
      const [x, y] = toData(e);
      pts[dragging] = [
        Math.max(-CLAMP, Math.min(CLAMP, x)),
        Math.max(-CLAMP, Math.min(CLAMP, y)),
      ];
      redraw();
    });
    const release = () => { dragging = null; };
    cvPlane.addEventListener('pointerup', release);
    cvPlane.addEventListener('pointercancel', release);

    slG.addEventListener('input', () => {
      gamma = +slG.value;
      $('s4-gamma-val').textContent = gamma.toFixed(1);
      redraw();
    });
    $('s4-reset').addEventListener('click', () => {
      gamma = 1;
      slG.value = '1';
      $('s4-gamma-val').textContent = '1.0';
      regen();
      redraw();
    });
    $('s4-rand').addEventListener('click', () => {
      seed = (seed * 131 + 23) % 100000;
      regen();
      redraw();
    });

    regen();
    redraw();
    P.onRedraw(redraw);
    P.observeResize(cvPlane, redraw);
    P.observeResize(cvW, redraw);
  }

  DL.sections.push({ name: 's4-weighted-average', init });
})();
