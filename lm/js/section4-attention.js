/* Seção 4: atenção interativa (vetores 2D arrastáveis) e matriz de alinhamento. */
(function () {
  'use strict';
  window.DM = window.DM || {};
  DM.sections = DM.sections || [];

  const TOKENS = ['Je', 'vois', 'un', 'chat', 'sur', '<eos>'];
  const PRESET_S = [[1, 0], [0, 1], [0.5, -1], [-1, 1], [-1, 2], [0, -1]];
  const PRESET_H = [2, 1];
  const RANGE = 2.7, CLAMP = 2.6;

  function init() {
    const M = DM.model, P = DM.plot, U = DM.utils;
    const $ = (id) => document.getElementById(id);
    const f = M.fmtBR;
    const cvPlane = $('s4-plane'), cvBars = $('s4-bars');

    let S = PRESET_S.map((v) => [...v]);
    let H = [...PRESET_H];
    let seed = 7;
    let frPlane = null;
    let dragging = null; // índice 0..5 para s_k, 'h' para h_t

    function compute() {
      const scores = S.map((s) => H[0] * s[0] + H[1] * s[1]);
      const a = M.softmax(scores);
      const c = [0, 1].map((d) => S.reduce((acc, s, k) => acc + a[k] * s[d], 0));
      return { scores, a, c };
    }

    function drawPlane() {
      const th = P.theme();
      const { ctx, w, h } = P.setup(cvPlane);
      P.clear(ctx, w, h);
      const fr = P.frame(ctx, w, h, -RANGE, RANGE, -RANGE, RANGE, { l: 10, r: 10, t: 10, b: 10 });
      frPlane = fr;
      // eixos pela origem
      ctx.strokeStyle = th.line;
      ctx.lineWidth = 1;
      ctx.beginPath();
      ctx.moveTo(fr.X(-RANGE), fr.Y(0)); ctx.lineTo(fr.X(RANGE), fr.Y(0));
      ctx.moveTo(fr.X(0), fr.Y(-RANGE)); ctx.lineTo(fr.X(0), fr.Y(RANGE));
      ctx.stroke();
      for (const v of [-2, -1, 1, 2]) {
        P.label(ctx, fr.X(v), fr.Y(0) + 12, String(v), th.comment, 'center');
        P.label(ctx, fr.X(0) - 6, fr.Y(v) + 4, String(v), th.comment, 'right');
      }
      const { a, c } = compute();
      // s_k: opacidade acompanha o peso de atenção
      S.forEach((s, k) => {
        ctx.globalAlpha = 0.45 + 0.55 * (a[k] / Math.max(...a));
        P.arrow(ctx, fr, 0, 0, s[0], s[1], th.green, { width: 2 });
        ctx.globalAlpha = 1;
        const off = s[1] >= 0 ? -8 : 14;
        P.label(ctx, fr.X(s[0]), fr.Y(s[1]) + off, TOKENS[k], th.green, 'center');
      });
      // h_t
      P.arrow(ctx, fr, 0, 0, H[0], H[1], th.pink, { width: 3, head: 11 });
      P.label(ctx, fr.X(H[0]), fr.Y(H[1]) - 10, 'h_t', th.pink, 'center');
      // contexto
      P.arrow(ctx, fr, 0, 0, c[0], c[1], th.purple, { width: 2.4, dash: [5, 4] });
      P.label(ctx, fr.X(c[0]) + 8, fr.Y(c[1]) + 4, 'c', th.purple);
    }

    function drawBars() {
      const th = P.theme();
      const { ctx, w, h } = P.setup(cvBars);
      P.clear(ctx, w, h);
      const { scores, a } = compute();
      const fr = P.frame(ctx, w, h, 0, 1, 0, Math.max(0.5, Math.max(...a) * 1.25), { l: 30, r: 8, t: 8, b: 42 });
      P.axes(ctx, fr, { yticks: [0, 0.5] });
      P.bars(ctx, fr, TOKENS, a, th.purple, {
        values: a.map((v) => f(v, 2)),
        sublabels: scores.map((s) => 'score ' + f(s, 1)),
        sublabelColor: th.pink,
      });
    }

    function drawReadout() {
      const { scores, a, c } = compute();
      $('s4-readout').innerHTML =
        'scores = h<sub>t</sub><sup>⊤</sup>s<sub>k</sub> = (' + scores.map((s) => f(s, 1)).join('; ') + ')<br />' +
        'a = softmax(scores) = [' + a.map((v) => f(v, 2)).join('; ') + ']<br />' +
        'c = Σ a<sub>k</sub>·s<sub>k</sub> ≈ (<strong>' + f(c[0], 2) + '; ' + f(c[1], 2) + '</strong>)';
    }

    function redraw() { drawPlane(); drawBars(); drawReadout(); }

    /* ── arraste das pontas ───────────────────────────────── */
    function toData(e) {
      const rect = cvPlane.getBoundingClientRect();
      const px = e.clientX - rect.left, py = e.clientY - rect.top;
      const x = frPlane.xmin + (px - frPlane.pad.l) / frPlane.iw * (frPlane.xmax - frPlane.xmin);
      const y = frPlane.ymax - (py - frPlane.pad.t) / frPlane.ih * (frPlane.ymax - frPlane.ymin);
      return [x, y];
    }

    cvPlane.addEventListener('pointerdown', (e) => {
      if (!frPlane) return;
      const rect = cvPlane.getBoundingClientRect();
      const px = e.clientX - rect.left, py = e.clientY - rect.top;
      const distTo = (v) => Math.hypot(px - frPlane.X(v[0]), py - frPlane.Y(v[1]));
      let best = null, bestD = 18;
      S.forEach((s, k) => { const d = distTo(s); if (d < bestD) { bestD = d; best = k; } });
      if (distTo(H) < bestD) best = 'h';
      if (best === null) return;
      dragging = best;
      cvPlane.setPointerCapture(e.pointerId);
    });
    cvPlane.addEventListener('pointermove', (e) => {
      if (dragging === null) return;
      const [x, y] = toData(e);
      const cx = Math.max(-CLAMP, Math.min(CLAMP, x));
      const cy = Math.max(-CLAMP, Math.min(CLAMP, y));
      if (dragging === 'h') { H[0] = cx; H[1] = cy; }
      else { S[dragging][0] = cx; S[dragging][1] = cy; }
      redraw();
    });
    const release = () => { dragging = null; };
    cvPlane.addEventListener('pointerup', release);
    cvPlane.addEventListener('pointercancel', release);

    $('s4-reset').addEventListener('click', () => {
      S = PRESET_S.map((v) => [...v]);
      H = [...PRESET_H];
      redraw();
    });
    $('s4-random').addEventListener('click', () => {
      seed = (seed * 1664525 + 1013904223) >>> 0;
      const rng = U.mulberry32(seed);
      S = S.map(() => [(rng() * 2 - 1) * 2, (rng() * 2 - 1) * 2]);
      H = [(rng() * 2 - 1) * 2, (rng() * 2 - 1) * 2];
      redraw();
    });

    redraw();
    P.onRedraw(redraw);
    P.observeResize(cvPlane, redraw);
  }

  DM.sections.push({ name: 's4-attention', init });
})();
