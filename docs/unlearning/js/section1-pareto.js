/* Seção 1: trade-off Forget Quality vs Model Utility.
   Fronteira ilustrativa Utility = 1 − ForgetQuality². A Model Utility global é a
   média harmônica das sub-métricas, H = 3 / Σ(1/sᵢ), que colapsa se uma delas cai. */
(function () {
  'use strict';
  window.DL = window.DL || {};
  DL.sections = DL.sections || [];

  const fm = (v, d) => v.toFixed(d == null ? 2 : d).replace('.', ',');

  /* Marcadores de referência fixos no espaço (Forget, Utility). */
  const MARKERS = [
    { x: 0.05, y: 0.95, name: 'sem unlearning', key: 'green' },
    { x: 0.95, y: 0.05, name: 'modelo trivial', key: 'red' },
    { x: 0.70, y: 0.51, name: 'ideal', key: 'orange' },
  ];

  function init() {
    if (!document.getElementById('s1-pareto')) return;
    const P = DL.plot;
    const $ = (id) => document.getElementById(id);

    const cvPar = $('s1-pareto'), cvHarm = $('s1-harm');
    const slS = [$('s1-s1'), $('s1-s2'), $('s1-s3')];
    const btnReset = $('s1-reset');

    /* Ponto de operação inicial sobre a fronteira. */
    let op = { x: 0.55, y: 1 - 0.55 * 0.55 };
    let frPar = null, dragging = false;

    function frontier(x) { return 1 - x * x; }

    function clampOp(x, y) {
      x = Math.max(0, Math.min(1, x));
      const ymax = frontier(x);
      y = Math.max(0, Math.min(ymax, y));
      return { x: x, y: y };
    }

    function harmonic() {
      const s = slS.map((sl) => +sl.value);
      let inv = 0;
      for (const v of s) inv += 1 / Math.max(1e-6, v);
      return { s: s, H: 3 / inv, mean: (s[0] + s[1] + s[2]) / 3 };
    }

    /* ── Painel da fronteira de Pareto ── */
    function drawPareto() {
      const th = P.theme();
      const { ctx, w, h } = P.setup(cvPar);
      P.clear(ctx, w, h);
      const fr = P.frame(ctx, w, h, 0, 1, 0, 1);
      frPar = fr;
      P.axes(ctx, fr, {
        xlabel: 'Forget Quality', ylabel: 'Model Utility',
        xticks: [0, 0.5, 1], yticks: [0, 0.5, 1],
      });

      /* Região inviável (acima da fronteira) levemente sombreada. */
      const xs = [], ys = [];
      for (let i = 0; i <= 100; i++) { const x = i / 100; xs.push(x); ys.push(frontier(x)); }
      ctx.save();
      ctx.beginPath();
      ctx.moveTo(fr.X(0), fr.Y(frontier(0)));
      for (let i = 0; i <= 100; i++) ctx.lineTo(fr.X(xs[i]), fr.Y(ys[i]));
      ctx.lineTo(fr.X(1), fr.Y(1));
      ctx.lineTo(fr.X(0), fr.Y(1));
      ctx.closePath();
      ctx.fillStyle = th.red;
      ctx.globalAlpha = 0.07;
      ctx.fill();
      ctx.globalAlpha = 1;
      ctx.restore();

      P.line(ctx, fr, xs, ys, th.purple, { width: 2.2 });
      P.label(ctx, fr.X(0.05), fr.Y(0.2), 'inviável', th.red);
      P.mathText(ctx, 'Utility = 1 − ForgetQuality²', fr.X(0.50), fr.Y(0.70), th.purple, 'left', 11);

      /* Marcadores de referência. */
      for (const m of MARKERS) {
        const c = th[m.key];
        P.scatter(ctx, fr, [[m.x, m.y]], c, { r: 5, alpha: 1 });
        const al = m.x > 0.7 ? 'right' : 'left';
        const off = m.x > 0.7 ? -8 : 8;
        P.mathText(ctx, m.name, fr.X(m.x) + off, fr.Y(m.y) - 7, c, al, 11);
      }

      /* Ponto de operação arrastável. */
      ctx.beginPath();
      ctx.arc(fr.X(op.x), fr.Y(op.y), 7, 0, 2 * Math.PI);
      ctx.fillStyle = th.pink;
      ctx.fill();
      ctx.lineWidth = 1.5;
      ctx.strokeStyle = th.bg;
      ctx.stroke();
    }

    /* ── Painel da média harmônica ── */
    function drawHarm() {
      const th = P.theme();
      const { ctx, w, h } = P.setup(cvHarm);
      P.clear(ctx, w, h);
      const fr = P.frame(ctx, w, h, 0, 4, 0, 1, { l: 30, r: 10, t: 14, b: 30 });
      P.axes(ctx, fr, { yticks: [0, 0.5, 1] });

      const r = harmonic();
      const bars = [
        { cx: 0.5, v: r.s[0], c: th.cyan, label: 's_1' },
        { cx: 1.5, v: r.s[1], c: th.cyan, label: 's_2' },
        { cx: 2.5, v: r.s[2], c: th.cyan, label: 's_3' },
        { cx: 3.5, v: r.H, c: th.pink, label: 'H' },
      ];
      for (const b of bars) {
        const x0 = fr.X(b.cx - 0.34), x1 = fr.X(b.cx + 0.34);
        const y0 = fr.Y(0), y1 = fr.Y(b.v);
        ctx.fillStyle = b.c;
        ctx.globalAlpha = b.label === 'H' ? 0.9 : 0.55;
        ctx.fillRect(x0, y1, x1 - x0, y0 - y1);
        ctx.globalAlpha = 1;
        ctx.strokeStyle = b.c;
        ctx.lineWidth = 1.4;
        ctx.strokeRect(x0, y1, x1 - x0, y0 - y1);
        P.mathText(ctx, fm(b.v), fr.X(b.cx), y1 - 5, b.c, 'center', 11);
        P.mathText(ctx, b.label, fr.X(b.cx), fr.Y(0) + 16, th.comment, 'center', 12);
      }

      /* Linha tracejada na média aritmética, para contraste com a harmônica. */
      const ym = fr.Y(r.mean);
      ctx.strokeStyle = th.comment;
      ctx.setLineDash([4, 4]);
      ctx.lineWidth = 1.2;
      ctx.beginPath();
      ctx.moveTo(fr.X(0), ym);
      ctx.lineTo(fr.X(4), ym);
      ctx.stroke();
      ctx.setLineDash([]);
      P.mathText(ctx, 'média aritmética = ' + fm(r.mean), fr.X(0.05), ym - 5, th.comment, 'left', 10);
    }

    function updateReadout() {
      const r = harmonic();
      $('s1-readout').textContent =
        'operação: Forget = ' + fm(op.x) + ' · Utility = ' + fm(op.y) +
        ' · H = ' + fm(r.H) + ' (média aritmética = ' + fm(r.mean) + ')';
      for (let i = 0; i < 3; i++) $('s1-s' + (i + 1) + '-val').textContent = fm(+slS[i].value);
    }

    function redraw() { drawPareto(); drawHarm(); updateReadout(); }

    cvPar.addEventListener('pointerdown', (e) => {
      if (!frPar) return;
      dragging = true;
      cvPar.setPointerCapture(e.pointerId);
      const r = cvPar.getBoundingClientRect();
      op = clampOp(frPar.invX(e.clientX - r.left), frPar.invY(e.clientY - r.top));
      redraw();
    });
    cvPar.addEventListener('pointermove', (e) => {
      if (!dragging || !frPar) return;
      const r = cvPar.getBoundingClientRect();
      op = clampOp(frPar.invX(e.clientX - r.left), frPar.invY(e.clientY - r.top));
      redraw();
    });
    cvPar.addEventListener('pointerup', () => { dragging = false; });

    for (const sl of slS) sl.addEventListener('input', redraw);
    btnReset.addEventListener('click', () => {
      slS[0].value = 0.8; slS[1].value = 0.8; slS[2].value = 0.8;
      op = clampOp(0.55, frontier(0.55));
      redraw();
    });

    redraw();
    P.onRedraw(redraw);
    P.observeResize(cvPar, redraw);
    P.observeResize(cvHarm, redraw);
  }

  DL.sections.push({ name: 's1-pareto', init });
})();
