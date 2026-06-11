/* Seção 3: debiasing de word embeddings por projeção.
   Recorte 2D: eixo x é a direção de gênero g, eixo y o conteúdo neutro.
   O slider lambda interpola w' = w − lambda*(w·g)g nas palavras neutras. */
(function () {
  'use strict';
  window.DL = window.DL || {};
  DL.sections = DL.sections || [];

  /* Palavras definicionais (mantêm o gênero) e profissões (deveriam ser neutras). */
  const DEFINITIONAL = [
    { w: 'he', x: 1.0, y: 0.15 },
    { w: 'she', x: -1.0, y: 0.15 },
    { w: 'king', x: 0.85, y: 0.75 },
    { w: 'queen', x: -0.85, y: 0.75 },
  ];
  const PROFESSIONS = [
    { w: 'programmer', x: 0.55, y: 1.45 },
    { w: 'engineer', x: 0.62, y: 1.15 },
    { w: 'scientist', x: 0.35, y: 1.75 },
    { w: 'doctor', x: 0.22, y: 1.55 },
    { w: 'teacher', x: -0.28, y: 1.5 },
    { w: 'librarian', x: -0.35, y: 1.35 },
    { w: 'secretary', x: -0.45, y: 1.05 },
    { w: 'nurse', x: -0.6, y: 1.25 },
    { w: 'homemaker', x: -0.68, y: 0.95 },
  ];

  const state = { lambda: 0 };

  /* Posição atual de uma profissão: componente em g escalada por (1 − lambda). */
  function pos(p) {
    return { x: p.x * (1 - state.lambda), y: p.y };
  }

  function cosSim(u, v) {
    const dot = u.x * v.x + u.y * v.y;
    const nu = Math.hypot(u.x, u.y), nv = Math.hypot(v.x, v.y);
    return nu > 1e-9 && nv > 1e-9 ? dot / (nu * nv) : 0;
  }

  function init() {
    const cvE = document.getElementById('s3-embed');
    const cvB = document.getElementById('s3-bias');
    const slider = document.getElementById('s3-lambda');
    const sliderVal = document.getElementById('s3-lambda-val');
    const readout = document.getElementById('s3-readout');

    function drawEmbed() {
      const { ctx, w, h } = DL.plot.setup(cvE);
      const t = DL.plot.theme();
      DL.plot.clear(ctx, w, h);
      const fr = DL.plot.frame(ctx, w, h, -1.35, 1.35, -0.15, 2.05);

      /* Eixos: direção de gênero (x) e componente neutra (y). */
      ctx.strokeStyle = t.line;
      ctx.lineWidth = 1;
      ctx.beginPath();
      ctx.moveTo(fr.X(-1.3), fr.Y(0));
      ctx.lineTo(fr.X(1.3), fr.Y(0));
      ctx.moveTo(fr.X(0), fr.Y(-0.1));
      ctx.lineTo(fr.X(0), fr.Y(2.0));
      ctx.stroke();
      DL.plot.label(ctx, fr.X(1.28), fr.Y(0) + 14, 'direção de gênero g →', t.comment, 'right');
      DL.plot.label(ctx, fr.X(0) + 6, fr.Y(1.98), 'componente neutra', t.comment, 'left');

      /* Setas de deslocamento e pontos das profissões. */
      for (const p of PROFESSIONS) {
        const q = pos(p);
        if (state.lambda > 0.02) {
          ctx.strokeStyle = t.comment;
          ctx.globalAlpha = 0.5;
          ctx.setLineDash([3, 3]);
          ctx.beginPath();
          ctx.moveTo(fr.X(p.x), fr.Y(p.y));
          ctx.lineTo(fr.X(q.x), fr.Y(q.y));
          ctx.stroke();
          ctx.setLineDash([]);
          ctx.globalAlpha = 1;
        }
        ctx.fillStyle = q.x > 0.05 ? t.cyan : (q.x < -0.05 ? t.orange : t.green);
        ctx.beginPath();
        ctx.arc(fr.X(q.x), fr.Y(q.y), 4, 0, 2 * Math.PI);
        ctx.fill();
        DL.plot.label(ctx, fr.X(q.x), fr.Y(q.y) - 8, p.w, t.fg, 'center');
      }

      /* Palavras definicionais como losangos (não são projetadas). */
      for (const d of DEFINITIONAL) {
        ctx.fillStyle = t.purple;
        ctx.beginPath();
        ctx.moveTo(fr.X(d.x), fr.Y(d.y) - 5);
        ctx.lineTo(fr.X(d.x) + 5, fr.Y(d.y));
        ctx.lineTo(fr.X(d.x), fr.Y(d.y) + 5);
        ctx.lineTo(fr.X(d.x) - 5, fr.Y(d.y));
        ctx.closePath();
        ctx.fill();
        DL.plot.label(ctx, fr.X(d.x), fr.Y(d.y) - 8, d.w, t.purple, 'center');
      }
    }

    function drawBias() {
      const { ctx, w, h } = DL.plot.setup(cvB);
      const t = DL.plot.theme();
      DL.plot.clear(ctx, w, h);

      const sorted = PROFESSIONS.slice().sort((a, b) => b.x - a.x);
      const padL = 78, padR = 12, padT = 8, rowH = (h - padT - 8) / sorted.length;
      const mid = padL + (w - padL - padR) / 2;
      const scale = (w - padL - padR) / 2 / 0.8;

      ctx.strokeStyle = t.line;
      ctx.beginPath();
      ctx.moveTo(mid, padT);
      ctx.lineTo(mid, h - 8);
      ctx.stroke();

      sorted.forEach((p, i) => {
        const v = p.x * (1 - state.lambda);
        const y = padT + i * rowH;
        ctx.fillStyle = t.comment;
        ctx.font = '11px sans-serif';
        ctx.textAlign = 'right';
        ctx.fillText(p.w, padL - 6, y + rowH * 0.65);
        ctx.fillStyle = v >= 0 ? t.cyan : t.orange;
        const bw = Math.abs(v) * scale;
        ctx.fillRect(v >= 0 ? mid : mid - bw, y + rowH * 0.22, bw, rowH * 0.56);
      });
      DL.plot.label(ctx, mid, h - 0.5, 'w·g = 0', t.comment, 'center');
    }

    function redraw() {
      drawEmbed();
      drawBias();
      /* Analogia enviesada: alinhamento de (programmer − homemaker) com (he − she). */
      const pr = pos(PROFESSIONS[0]), hm = pos(PROFESSIONS[8]);
      const diffProf = { x: pr.x - hm.x, y: pr.y - hm.y };
      const diffDef = { x: DEFINITIONAL[0].x - DEFINITIONAL[1].x, y: DEFINITIONAL[0].y - DEFINITIONAL[1].y };
      readout.textContent = 'cos(programmer − homemaker, he − she) = ' + cosSim(diffProf, diffDef).toFixed(2);
    }

    slider.addEventListener('input', () => {
      state.lambda = parseFloat(slider.value);
      sliderVal.textContent = state.lambda.toFixed(2);
      redraw();
    });
    document.getElementById('s3-debias').addEventListener('click', () => {
      state.lambda = 1;
      slider.value = '1';
      sliderVal.textContent = '1.00';
      redraw();
    });
    document.getElementById('s3-reset').addEventListener('click', () => {
      state.lambda = 0;
      slider.value = '0';
      sliderVal.textContent = '0.00';
      redraw();
    });

    DL.plot.observeResize(cvE, redraw);
    DL.plot.observeResize(cvB, redraw);
    DL.plot.onRedraw(redraw);
    redraw();
  }

  DL.sections.push({ name: 'embeddings', init });
})();
