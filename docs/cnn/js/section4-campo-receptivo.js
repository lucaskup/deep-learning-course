/* Seção 4: crescimento do campo receptivo em uma pilha de convoluções 1D
   (válidas, sem padding), com a recursão RF_l = RF_{l−1} + (K_l − 1)·∏ S_i. */
(function () {
  'use strict';
  window.DL = window.DL || {};
  DL.sections = DL.sections || [];

  const N0 = 15;
  const LAYER_NAMES = ['x', 'H_1', 'H_2', 'H_3'];
  const SUB = ['₀', '₁', '₂', '₃'];

  function init() {
    const P = DL.plot;
    const G = DL.grid;
    const $ = (id) => document.getElementById(id);

    const cvNet = $('s4-net'), cvChart = $('s4-chart');
    const slK = [null, $('s4-k1'), $('s4-k2'), $('s4-k3')];
    const slS = [null, $('s4-s1'), $('s4-s2'), $('s4-s3')];

    let K = [0, 3, 3, 3], S = [0, 1, 1, 1];
    let sizes = [N0, 0, 0, 0];
    let sel = null;               /* {l, u}: camada 1..3 e índice da unidade */

    function recompute() {
      for (let l = 1; l <= 3; l++) {
        K[l] = +slK[l].value;
        S[l] = +slS[l].value;
        const prev = sizes[l - 1];
        sizes[l] = prev >= 1 ? Math.floor((prev - K[l]) / S[l]) + 1 : 0;
        if (sizes[l] < 0) sizes[l] = 0;
      }
      /* Seleção padrão: unidade central da camada válida mais profunda */
      let l = sel ? sel.l : 3;
      while (l >= 1 && sizes[l] < 1) l--;
      if (l < 1) { sel = null; return; }
      const u = sel ? Math.min(sel.u, sizes[l] - 1) : Math.floor((sizes[l] - 1) / 2);
      sel = { l, u: Math.max(0, u) };
    }

    function rfValues() {
      const rf = [1];
      let prod = 1;
      for (let l = 1; l <= 3; l++) {
        rf.push(rf[l - 1] + (K[l] - 1) * prod);
        prod *= S[l];
      }
      return rf;
    }

    /* Faixas [a, b] do campo receptivo da unidade selecionada em cada camada. */
    function rfRanges() {
      if (!sel) return null;
      const ranges = [null, null, null, null];
      ranges[sel.l] = [sel.u, sel.u];
      for (let l = sel.l; l >= 1; l--) {
        const [a, b] = ranges[l];
        ranges[l - 1] = [a * S[l], b * S[l] + K[l] - 1];
      }
      return ranges;
    }

    function geometry(w, h) {
      const cs = Math.min((w - 90) / N0, (h - 26) / 6.7);
      const gap = cs * 0.9;
      const total = 4 * cs + 3 * gap;
      const oy = (h - total) / 2;
      const rowY = (l) => oy + (3 - l) * (cs + gap);   /* l = 0 embaixo */
      const rowX = (l) => (w - sizes[l] * cs) / 2;
      return { cs, rowY, rowX };
    }

    function drawNet() {
      const th = P.theme();
      const { ctx, w, h } = P.setup(cvNet);
      P.clear(ctx, w, h);
      const { cs, rowY, rowX } = geometry(w, h);
      const ranges = rfRanges();

      /* Cones entre camadas, desenhados antes das células */
      if (ranges) {
        ctx.fillStyle = th.comment;
        ctx.globalAlpha = 0.18;
        for (let l = sel.l; l >= 1; l--) {
          const [a1, b1] = ranges[l];
          const [a0, b0] = ranges[l - 1];
          ctx.beginPath();
          ctx.moveTo(rowX(l) + a1 * cs, rowY(l) + cs);
          ctx.lineTo(rowX(l) + (b1 + 1) * cs, rowY(l) + cs);
          ctx.lineTo(rowX(l - 1) + (b0 + 1) * cs, rowY(l - 1));
          ctx.lineTo(rowX(l - 1) + a0 * cs, rowY(l - 1));
          ctx.closePath();
          ctx.fill();
        }
        ctx.globalAlpha = 1;
      }

      for (let l = 0; l <= 3; l++) {
        const y = rowY(l);
        P.mathText(ctx, LAYER_NAMES[l], rowX(0) - 14, y + cs / 2 + 4, th.comment, 'right', 12);
        if (sizes[l] < 1) {
          ctx.fillStyle = th.pink;
          ctx.font = '11px sans-serif';
          ctx.textAlign = 'center';
          ctx.fillText('camada vazia (reduza K ou S)', w / 2, y + cs / 2 + 4);
          continue;
        }
        const x0 = rowX(l);
        for (let u = 0; u < sizes[l]; u++) {
          const inRange = ranges && ranges[l] && u >= ranges[l][0] && u <= ranges[l][1];
          const isSel = sel && sel.l === l && sel.u === u;
          ctx.fillStyle = isSel ? th.orange : inRange ? G.mix(th.card, th.orange, 0.45) : th.card;
          ctx.fillRect(x0 + u * cs, y, cs, cs);
        }
        ctx.strokeStyle = th.line;
        ctx.lineWidth = 1;
        for (let u = 0; u < sizes[l]; u++) {
          ctx.strokeRect(x0 + u * cs, y, cs, cs);
        }
        if (sel && sel.l === l) {
          ctx.strokeStyle = th.orange;
          ctx.lineWidth = 2.5;
          ctx.strokeRect(x0 + sel.u * cs, y, cs, cs);
        }
      }
    }

    function drawChart() {
      const th = P.theme();
      const { ctx, w, h } = P.setup(cvChart);
      P.clear(ctx, w, h);
      const rf = rfValues();
      const ymax = Math.max(4, ...rf) + 1;
      const fr = P.frame(ctx, w, h, -0.2, 3.2, 0, ymax);
      P.axes(ctx, fr, {
        xlabel: 'camada l', ylabel: 'RF_l',
        xticks: [0, 1, 2, 3],
        yticks: [1, Math.round(ymax / 2), ymax - 1],
      });
      P.line(ctx, fr, [0, 1, 2, 3], rf, th.cyan, { width: 1.8 });
      P.scatter(ctx, fr, rf.map((v, l) => [l, v]), th.cyan, { r: 3.5, alpha: 1 });
      for (let l = 0; l <= 3; l++) {
        P.label(ctx, fr.X(l) + 6, fr.Y(rf[l]) - 6, String(rf[l]), th.fg);
      }
      if (sel) {
        P.scatter(ctx, fr, [[sel.l, rf[sel.l]]], th.orange, { r: 5, alpha: 1 });
      }
    }

    function updateReadout() {
      const rf = rfValues();
      let txt = 'RF₁ = ' + rf[1] + ' · RF₂ = ' + rf[2] + ' · RF₃ = ' + rf[3];
      const ranges = rfRanges();
      if (sel && ranges) {
        const [a, b] = ranges[0];
        const name = 'H' + SUB[sel.l];
        txt += ' · ' + name + '[' + (sel.u + 1) + '] enxerga x[' + (a + 1) + '..' + (b + 1) + '] (' +
          (b - a + 1) + ' posições)';
      }
      $('s4-readout').textContent = txt;
    }

    function redraw() {
      drawNet();
      drawChart();
      updateReadout();
    }

    cvNet.addEventListener('pointerdown', (e) => {
      const r = cvNet.getBoundingClientRect();
      const { w, h } = P.setup(cvNet);
      const { cs, rowY, rowX } = geometry(w, h);
      const px = e.clientX - r.left, py = e.clientY - r.top;
      for (let l = 1; l <= 3; l++) {
        if (sizes[l] < 1) continue;
        const y = rowY(l);
        if (py < y || py > y + cs) continue;
        const u = Math.floor((px - rowX(l)) / cs);
        if (u >= 0 && u < sizes[l]) {
          sel = { l, u };
          redraw();
          return;
        }
      }
    });

    function onSlider() {
      for (let l = 1; l <= 3; l++) {
        $('s4-k' + l + '-val').textContent = slK[l].value;
        $('s4-s' + l + '-val').textContent = slS[l].value;
      }
      recompute();
      redraw();
    }
    for (let l = 1; l <= 3; l++) {
      slK[l].addEventListener('input', onSlider);
      slS[l].addEventListener('input', onSlider);
    }

    recompute();
    redraw();
    P.onRedraw(redraw);
    P.observeResize(cvNet, redraw);
    P.observeResize(cvChart, redraw);
  }

  DL.sections.push({ name: 's4-campo-receptivo', init });
})();
