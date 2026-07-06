/* Seção 2: MLP 2-2-1 com gradientes analíticos calculados ao vivo.
   Notação da aula: θᵢⱼ⁽ˡ⁾ conecta a entrada j à saída i, δ⁽ˡ⁾ = ∂J/∂z⁽ˡ⁾. */
(function () {
  'use strict';
  window.DL = window.DL || {};
  DL.sections = DL.sections || [];

  const ETA = 0.5;
  const R = 24;   // raio dos nós

  function sigmoid(z) { return 1 / (1 + Math.exp(-z)); }

  function fnum(v, d) {
    d = d == null ? 4 : d;
    let s = v.toFixed(d);
    if (parseFloat(s) === 0) s = (0).toFixed(d);
    return s.replace('-', '−');
  }

  /* Gradientes podem ficar minúsculos: notação compacta quando preciso. */
  function fgrad(v) {
    if (v !== 0 && Math.abs(v) < 0.0005) return v.toExponential(1).replace('-', '−');
    return fnum(v, 4);
  }

  function init() {
    const U = DL.utils, P = DL.plot;
    const $ = (id) => document.getElementById(id);

    const cv = $('s2-canvas');
    const tabAct = $('s2-tab-act'), tabGrad = $('s2-tab-grad');
    const btnResample = $('s2-resample'), btnReset = $('s2-reset'), btnStep = $('s2-gdstep');

    let seed = 7;
    let W1 = null;   // W1[i][j]: peso da camada 1, entrada j → unidade oculta i
    let W2 = null;   // W2[j]: peso da camada 2, oculta j → saída
    let view = 'act';
    let lastMsg = '';

    function sampleWeights() {
      const rng = U.mulberry32(seed);
      const r = () => Math.round((rng() * 2 - 1) * 120) / 100;   // uniforme em [−1.2, 1.2]
      W1 = [[r(), r()], [r(), r()]];
      W2 = [r(), r()];
    }

    /* Forward + backward analíticos (sem bias, sigmoide nas duas camadas). */
    function compute() {
      const x = [+$('s2-x1').value, +$('s2-x2').value];
      const y = +$('s2-y').value;
      const z1 = [W1[0][0] * x[0] + W1[0][1] * x[1], W1[1][0] * x[0] + W1[1][1] * x[1]];
      const a1 = [sigmoid(z1[0]), sigmoid(z1[1])];
      const z2 = W2[0] * a1[0] + W2[1] * a1[1];
      const a2 = sigmoid(z2);
      const J = 0.5 * (y - a2) * (y - a2);
      const delta2 = (a2 - y) * a2 * (1 - a2);
      const gW2 = [delta2 * a1[0], delta2 * a1[1]];
      const delta1 = [
        delta2 * W2[0] * a1[0] * (1 - a1[0]),
        delta2 * W2[1] * a1[1] * (1 - a1[1]),
      ];
      const gW1 = [
        [delta1[0] * x[0], delta1[0] * x[1]],
        [delta1[1] * x[0], delta1[1] * x[1]],
      ];
      return { x, y, z1, a1, z2, a2, J, delta2, gW2, delta1, gW1 };
    }

    /* Posições fracionárias dos nós. */
    const pos = {
      x1: { fx: 0.10, fy: 0.28 }, x2: { fx: 0.10, fy: 0.72 },
      h1: { fx: 0.46, fy: 0.24 }, h2: { fx: 0.46, fy: 0.76 },
      o:  { fx: 0.82, fy: 0.50 },
    };
    /* labelT: fração ao longo da aresta onde fica o rótulo (evita colisão no cruzamento). */
    const conns = [
      { from: 'x1', to: 'h1', w: () => W1[0][0], g: (S) => S.gW1[0][0], name: 'θ_{11}⁽¹⁾', labelT: 0.45 },
      { from: 'x2', to: 'h1', w: () => W1[0][1], g: (S) => S.gW1[0][1], name: 'θ_{12}⁽¹⁾', labelT: 0.78 },
      { from: 'x1', to: 'h2', w: () => W1[1][0], g: (S) => S.gW1[1][0], name: 'θ_{21}⁽¹⁾', labelT: 0.78 },
      { from: 'x2', to: 'h2', w: () => W1[1][1], g: (S) => S.gW1[1][1], name: 'θ_{22}⁽¹⁾', labelT: 0.45 },
      { from: 'h1', to: 'o',  w: () => W2[0],    g: (S) => S.gW2[0],    name: 'θ_1⁽²⁾',    labelT: 0.5 },
      { from: 'h2', to: 'o',  w: () => W2[1],    g: (S) => S.gW2[1],    name: 'θ_2⁽²⁾',    labelT: 0.5 },
    ];

    function redraw() {
      const th = P.theme();
      const { ctx, w, h } = P.setup(cv);
      P.clear(ctx, w, h);
      const S = compute();
      const px = (id) => ({ x: pos[id].fx * w, y: pos[id].fy * h });
      const grad = view === 'grad';

      /* normalização da espessura pelo maior valor absoluto da vista atual */
      let vmax = 0;
      for (const c of conns) vmax = Math.max(vmax, Math.abs(grad ? c.g(S) : c.w()));
      if (vmax < 1e-12) vmax = 1;

      /* arestas */
      for (const c of conns) {
        const a = px(c.from), b = px(c.to);
        const dx = b.x - a.x, dy = b.y - a.y;
        const len = Math.hypot(dx, dy) || 1;
        const ux = dx / len, uy = dy / len;
        const p1 = { x: a.x + ux * (R + 3), y: a.y + uy * (R + 3) };
        const p2 = { x: b.x - ux * (R + 3), y: b.y - uy * (R + 3) };
        const v = grad ? c.g(S) : c.w();
        const color = grad ? (v >= 0 ? th.purple : th.pink) : (v >= 0 ? th.cyan : th.orange);
        ctx.strokeStyle = color;
        ctx.lineWidth = 1 + 3.5 * Math.abs(v) / vmax;
        ctx.beginPath();
        ctx.moveTo(p1.x, p1.y);
        ctx.lineTo(p2.x, p2.y);
        ctx.stroke();
        ctx.fillStyle = color;
        ctx.beginPath();
        ctx.moveTo(p2.x, p2.y);
        ctx.lineTo(p2.x - 8 * ux - 4 * uy, p2.y - 8 * uy + 4 * ux);
        ctx.lineTo(p2.x - 8 * ux + 4 * uy, p2.y - 8 * uy - 4 * ux);
        ctx.closePath();
        ctx.fill();
        /* rótulo da aresta */
        const lx = p1.x + (p2.x - p1.x) * c.labelT;
        const ly = p1.y + (p2.y - p1.y) * c.labelT;
        let ox = -uy, oy = ux;
        if (oy > 0) { ox = -ox; oy = -oy; }
        const text = grad
          ? '∂J/∂' + c.name + ' = ' + fgrad(v)
          : c.name + ' = ' + fnum(v, 2);
        P.mathText(ctx, text, lx + ox * 12, ly + oy * 12, color, 'center', 10.5);
      }

      /* nós */
      const circles = [
        { id: 'x1', name: 'x_1', below: grad ? null : 'x_1 = ' + fnum(S.x[0], 1), belowColor: th.cyan },
        { id: 'x2', name: 'x_2', below: grad ? null : 'x_2 = ' + fnum(S.x[1], 1), belowColor: th.cyan },
        {
          id: 'h1', name: 'a_1⁽¹⁾',
          below: grad ? 'δ_1⁽¹⁾ = ' + fgrad(S.delta1[0]) : 'a_1⁽¹⁾ = ' + fnum(S.a1[0]),
          belowColor: grad ? th.pink : th.green,
        },
        {
          id: 'h2', name: 'a_2⁽¹⁾',
          below: grad ? 'δ_2⁽¹⁾ = ' + fgrad(S.delta1[1]) : 'a_2⁽¹⁾ = ' + fnum(S.a1[1]),
          belowColor: grad ? th.pink : th.green,
        },
        {
          id: 'o', name: 'ŷ',
          below: grad ? 'δ⁽²⁾ = ' + fgrad(S.delta2) : 'ŷ = ' + fnum(S.a2),
          belowColor: grad ? th.pink : th.green,
        },
      ];
      for (const n of circles) {
        const c = px(n.id);
        ctx.beginPath();
        ctx.arc(c.x, c.y, R, 0, 2 * Math.PI);
        ctx.fillStyle = th.card;
        ctx.fill();
        ctx.strokeStyle = th.line;
        ctx.lineWidth = 1.4;
        ctx.stroke();
        P.mathText(ctx, n.name, c.x, c.y + 4, th.fg, 'center', 12);
        if (n.below) P.mathText(ctx, n.below, c.x, c.y + R + 16, n.belowColor, 'center', 11);
      }

      /* legenda da vista atual */
      const legend = grad
        ? 'roxo: ∂J/∂θ ≥ 0 · rosa: ∂J/∂θ < 0 · espessura ∝ |∂J/∂θ|'
        : 'ciano: θ ≥ 0 · laranja: θ < 0 · espessura ∝ |θ|';
      P.mathText(ctx, legend, w - 8, 16, th.comment, 'right', 10.5);
    }

    function updateStatus() {
      const S = compute();
      $('s2-loss').textContent =
        'J = ' + fnum(S.J) + ' · ŷ = ' + fnum(S.a2) + ' · y = ' + fnum(S.y, 2);
      $('s2-last').textContent = lastMsg;
    }

    function update() { updateStatus(); redraw(); }

    /* ── controles ── */

    function setView(v) {
      view = v;
      tabAct.classList.toggle('active', v === 'act');
      tabGrad.classList.toggle('active', v === 'grad');
      redraw();
    }
    tabAct.addEventListener('click', () => setView('act'));
    tabGrad.addEventListener('click', () => setView('grad'));

    for (const id of ['s2-x1', 's2-x2', 's2-y']) {
      $(id).addEventListener('input', () => {
        $('s2-x1-val').textContent = fnum(+$('s2-x1').value, 1);
        $('s2-x2-val').textContent = fnum(+$('s2-x2').value, 1);
        $('s2-y-val').textContent = fnum(+$('s2-y').value, 2);
        update();
      });
    }

    btnResample.addEventListener('click', () => {
      seed = (seed * 1664525 + 1013904223) >>> 0;
      sampleWeights();
      lastMsg = '';
      update();
    });

    btnReset.addEventListener('click', () => {
      sampleWeights();
      lastMsg = '';
      update();
    });

    btnStep.addEventListener('click', () => {
      const S = compute();
      const before = S.J;
      for (let i = 0; i < 2; i++) {
        for (let j = 0; j < 2; j++) W1[i][j] -= ETA * S.gW1[i][j];
        W2[i] -= ETA * S.gW2[i];
      }
      const after = compute().J;
      lastMsg = 'passo aplicado: J ' + fnum(before) + ' → ' + fnum(after);
      update();
    });

    sampleWeights();
    update();
    P.onRedraw(redraw);
    P.observeResize(cv, redraw);
  }

  DL.sections.push({ name: 's2-mlp', init });
})();
