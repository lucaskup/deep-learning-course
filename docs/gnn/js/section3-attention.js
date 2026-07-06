/* Seção 3: coeficientes de atenção em GAT.
   Esquerda: estrela v + 3 vizinhos com espessura de aresta proporcional
   a alpha. Direita: plano dos embeddings projetados Wh, com pontos
   arrastáveis e a direção de atenção controlada pelo slider theta.
   Score: e_u = LeakyReLU((1,0)·Wh_v + (cos t, sin t)·Wh_u), o preset
   theta = 0 reproduz o exemplo numérico dos slides. */
(function () {
  'use strict';
  window.DL = window.DL || {};
  DL.sections = DL.sections || [];

  const PRESET = {
    v: { x: 1, y: 0 },
    u: [{ x: 1, y: 0 }, { x: 0, y: 1 }, { x: 0.5, y: 0.5 }],
  };
  const state = {
    v: { ...PRESET.v },
    u: PRESET.u.map((p) => ({ ...p })),
    theta: 0,
  };
  /* Posições fixas do desenho da estrela (não são os embeddings). */
  const STAR = [{ x: -1.25, y: 0.95 }, { x: 1.25, y: 0.95 }, { x: 0, y: -1.45 }];

  function leakyRelu(x) { return x >= 0 ? x : 0.2 * x; }

  function alphas() {
    const th = state.theta * Math.PI / 180;
    const scores = state.u.map((u) =>
      leakyRelu(state.v.x + Math.cos(th) * u.x + Math.sin(th) * u.y));
    const m = Math.max(...scores);
    const ex = scores.map((s) => Math.exp(s - m));
    const Z = ex.reduce((a, b) => a + b, 0);
    return ex.map((e) => e / Z);
  }

  function init() {
    const cvG = document.getElementById('s3-graph');
    const cvE = document.getElementById('s3-embed');
    const slider = document.getElementById('s3-theta');
    const thetaVal = document.getElementById('s3-theta-val');
    const readout = document.getElementById('s3-readout');

    const EMB = { xmin: -0.6, xmax: 1.6, ymin: -0.6, ymax: 1.6 };
    function embFrame(w, h) {
      return DL.plot.frame(null, w, h, EMB.xmin, EMB.xmax, EMB.ymin, EMB.ymax);
    }
    function neighborColors(t) { return [t.cyan, t.orange, t.green]; }

    function drawGraph() {
      const { ctx, w, h } = DL.plot.setup(cvG);
      const t = DL.plot.theme();
      DL.plot.clear(ctx, w, h);
      const fr = DL.plot.frame(ctx, w, h, -2, 2, -2, 1.7,
        { l: 8, r: 8, t: 8, b: 8 });
      const al = alphas();
      const cols = neighborColors(t);

      for (let i = 0; i < 3; i++) {
        const sx = fr.X(0), sy = fr.Y(0);
        const ex = fr.X(STAR[i].x), ey = fr.Y(STAR[i].y);
        ctx.strokeStyle = cols[i];
        ctx.lineWidth = 1 + 14 * al[i];
        ctx.globalAlpha = 0.45 + 0.55 * al[i];
        ctx.beginPath();
        ctx.moveTo(sx, sy);
        ctx.lineTo(ex, ey);
        ctx.stroke();
        ctx.globalAlpha = 1;
        /* Rótulo do alpha no meio da aresta, deslocado para fora. */
        const mx = (sx + ex) / 2, my = (sy + ey) / 2;
        ctx.fillStyle = cols[i];
        ctx.font = 'bold 12px sans-serif';
        ctx.textAlign = 'center';
        ctx.fillText(al[i].toFixed(2), mx + (STAR[i].x < 0 ? -26 : 26),
          my + (STAR[i].y < 0 ? 14 : 0));
      }
      /* Nós. */
      const nodes = [{ x: 0, y: 0, lab: 'v', col: t.pink }]
        .concat(STAR.map((p, i) => ({ x: p.x, y: p.y, lab: 'u' + (i + 1), col: neighborColors(t)[i] })));
      for (const nd of nodes) {
        ctx.fillStyle = nd.col;
        ctx.beginPath();
        ctx.arc(fr.X(nd.x), fr.Y(nd.y), nd.lab === 'v' ? 14 : 12, 0, 2 * Math.PI);
        ctx.fill();
        ctx.fillStyle = t.bg;
        ctx.font = 'bold 11px sans-serif';
        ctx.textAlign = 'center';
        ctx.textBaseline = 'middle';
        ctx.fillText(nd.lab, fr.X(nd.x), fr.Y(nd.y) + 0.5);
        ctx.textBaseline = 'alphabetic';
      }
    }

    function drawEmbed() {
      const { ctx, w, h } = DL.plot.setup(cvE);
      const t = DL.plot.theme();
      DL.plot.clear(ctx, w, h);
      const fr = embFrame(w, h);
      DL.plot.axes(ctx, fr, { xticks: [0, 0.5, 1, 1.5], yticks: [0, 0.5, 1, 1.5] });
      /* Linhas de grade no zero. */
      ctx.strokeStyle = t.line;
      ctx.lineWidth = 1;
      ctx.setLineDash([3, 3]);
      ctx.beginPath();
      ctx.moveTo(fr.X(0), fr.Y(EMB.ymin)); ctx.lineTo(fr.X(0), fr.Y(EMB.ymax));
      ctx.moveTo(fr.X(EMB.xmin), fr.Y(0)); ctx.lineTo(fr.X(EMB.xmax), fr.Y(0));
      ctx.stroke();
      ctx.setLineDash([]);

      /* Direção de atenção (parte que multiplica Wh_u). */
      const th = state.theta * Math.PI / 180;
      const ax = 0.85 * Math.cos(th), ay = 0.85 * Math.sin(th);
      ctx.strokeStyle = t.pink;
      ctx.lineWidth = 2;
      ctx.beginPath();
      ctx.moveTo(fr.X(0), fr.Y(0));
      ctx.lineTo(fr.X(ax), fr.Y(ay));
      ctx.stroke();
      const ang = Math.atan2(fr.Y(ay) - fr.Y(0), fr.X(ax) - fr.X(0));
      ctx.fillStyle = t.pink;
      ctx.beginPath();
      ctx.moveTo(fr.X(ax), fr.Y(ay));
      ctx.lineTo(fr.X(ax) - 9 * Math.cos(ang - 0.4), fr.Y(ay) - 9 * Math.sin(ang - 0.4));
      ctx.lineTo(fr.X(ax) - 9 * Math.cos(ang + 0.4), fr.Y(ay) - 9 * Math.sin(ang + 0.4));
      ctx.closePath();
      ctx.fill();
      DL.plot.label(ctx, fr.X(ax * 0.5), fr.Y(ay * 0.5) - 10, 'direção de a', t.pink, 'center');

      /* Pontos: vizinhos (círculos) e v (losango). */
      const cols = neighborColors(t);
      state.u.forEach((u, i) => {
        ctx.fillStyle = cols[i];
        ctx.beginPath();
        ctx.arc(fr.X(u.x), fr.Y(u.y), 7, 0, 2 * Math.PI);
        ctx.fill();
        DL.plot.label(ctx, fr.X(u.x) + 9, fr.Y(u.y) + 4, 'Wh_{u' + (i + 1) + '}', cols[i]);
      });
      ctx.fillStyle = t.pink;
      const vx = fr.X(state.v.x), vy = fr.Y(state.v.y);
      ctx.beginPath();
      ctx.moveTo(vx, vy - 8); ctx.lineTo(vx + 8, vy);
      ctx.lineTo(vx, vy + 8); ctx.lineTo(vx - 8, vy);
      ctx.closePath();
      ctx.fill();
      /* Rótulo à esquerda: no preset, Wh_v coincide com Wh_{u1}. */
      DL.plot.label(ctx, vx - 11, vy + 4, 'Wh_v', t.pink, 'right');
    }

    function redraw() {
      drawGraph();
      drawEmbed();
      const al = alphas();
      readout.textContent = 'α = (' + al.map((a) => a.toFixed(2)).join(', ') + ')';
    }

    /* Arrasto dos pontos no plano de embeddings. */
    let dragging = null;
    cvE.addEventListener('pointerdown', (ev) => {
      const rect = cvE.getBoundingClientRect();
      const px = ev.clientX - rect.left, py = ev.clientY - rect.top;
      const { w, h } = DL.plot.setup(cvE);
      const fr = embFrame(w, h);
      const cands = state.u.map((u, i) => ({ p: u, d: Math.hypot(fr.X(u.x) - px, fr.Y(u.y) - py) }))
        .concat([{ p: state.v, d: Math.hypot(fr.X(state.v.x) - px, fr.Y(state.v.y) - py) }]);
      cands.sort((a, b) => a.d - b.d);
      if (cands[0].d < 16) {
        dragging = cands[0].p;
        cvE.setPointerCapture(ev.pointerId);
      }
    });
    cvE.addEventListener('pointermove', (ev) => {
      if (!dragging) return;
      const rect = cvE.getBoundingClientRect();
      const { w, h } = DL.plot.setup(cvE);
      const fr = embFrame(w, h);
      dragging.x = Math.max(EMB.xmin, Math.min(EMB.xmax, fr.invX(ev.clientX - rect.left)));
      dragging.y = Math.max(EMB.ymin, Math.min(EMB.ymax, fr.invY(ev.clientY - rect.top)));
      redraw();
    });
    cvE.addEventListener('pointerup', () => { dragging = null; });

    slider.addEventListener('input', () => {
      state.theta = parseInt(slider.value, 10);
      thetaVal.textContent = slider.value + '°';
      redraw();
    });
    document.getElementById('s3-preset').addEventListener('click', () => {
      state.v = { ...PRESET.v };
      state.u = PRESET.u.map((p) => ({ ...p }));
      state.theta = 0;
      slider.value = '0';
      thetaVal.textContent = '0°';
      redraw();
    });

    DL.plot.observeResize(cvG, redraw);
    DL.plot.observeResize(cvE, redraw);
    DL.plot.onRedraw(redraw);
    redraw();
  }

  DL.sections.push({ name: 'gat-attention', init });
})();
