/* Seção 1: máscara causal vs. atenção bidirecional vs. sliding window.
   Pesos: softmax(QKᵀ/√d_k + M) com scores sorteados, renormalizados por linha. */
(function () {
  'use strict';
  window.DL = window.DL || {};
  DL.sections = DL.sections || [];

  function init() {
    const P = DL.plot;
    const $ = (id) => document.getElementById(id);

    const cvMat = $('s1-matrix'), cvRow = $('s1-row');
    const slN = $('s1-n'), slW = $('s1-w');
    const tabs = { bi: $('s1-tab-bi'), causal: $('s1-tab-causal'), swa: $('s1-tab-swa') };

    let n = +slN.value, W = +slW.value, mode = 'causal', seed = 7;
    let scores = [];       /* matriz n×n de scores q_i·k_j (fixos por semente) */
    let selRow = n - 1;    /* linha inspecionada no gráfico de barras */
    let hover = null;      /* {i, j} da célula sob o ponteiro */
    let geom = null;       /* geometria das células (para hover) */

    function allowed(i, j) {
      if (mode === 'bi') return true;
      if (mode === 'causal') return j <= i;
      return j <= i && j >= i - W; /* sliding window causal */
    }

    function resample() {
      const rng = DL.utils.mulberry32(seed);
      scores = [];
      for (let i = 0; i < n; i++) {
        const row = [];
        for (let j = 0; j < n; j++) row.push(DL.utils.randn(rng) * 1.1);
        scores.push(row);
      }
    }

    /* Softmax da linha i restrita às posições visíveis; mascaradas valem 0. */
    function weightsRow(i) {
      let mx = -Infinity;
      for (let j = 0; j < n; j++) if (allowed(i, j)) mx = Math.max(mx, scores[i][j]);
      const w = new Array(n).fill(0);
      let sum = 0;
      for (let j = 0; j < n; j++) {
        if (!allowed(i, j)) continue;
        w[j] = Math.exp(scores[i][j] - mx);
        sum += w[j];
      }
      for (let j = 0; j < n; j++) w[j] /= sum;
      return w;
    }

    function roundRect(ctx, x, y, w, h, r) {
      ctx.beginPath();
      ctx.moveTo(x + r, y);
      ctx.arcTo(x + w, y, x + w, y + h, r);
      ctx.arcTo(x + w, y + h, x, y + h, r);
      ctx.arcTo(x, y + h, x, y, r);
      ctx.arcTo(x, y, x + w, y, r);
      ctx.closePath();
    }

    function drawMatrix() {
      const th = P.theme();
      const { ctx, w, h } = P.setup(cvMat);
      P.clear(ctx, w, h);
      const padL = 40, padT = 30, padR = 10, padB = 26;
      const cell = Math.min((w - padL - padR) / n, (h - padT - padB) / n);
      const x0 = padL + ((w - padL - padR) - cell * n) / 2;
      const y0 = padT;
      geom = { x0, y0, cell };
      const pc = P.rgb(th.purple);

      /* Faixa da linha selecionada */
      ctx.fillStyle = th.cyan;
      ctx.globalAlpha = 0.10;
      ctx.fillRect(x0 - 2, y0 + selRow * cell - 2, cell * n + 4, cell + 4);
      ctx.globalAlpha = 1;

      for (let i = 0; i < n; i++) {
        const wts = weightsRow(i);
        for (let j = 0; j < n; j++) {
          const cx = x0 + j * cell, cy = y0 + i * cell;
          if (allowed(i, j)) {
            ctx.fillStyle = 'rgba(' + pc[0] + ',' + pc[1] + ',' + pc[2] + ',' +
              (0.10 + 0.85 * wts[j]).toFixed(3) + ')';
            roundRect(ctx, cx + 1.5, cy + 1.5, cell - 3, cell - 3, 3);
            ctx.fill();
            if (cell >= 30) {
              ctx.fillStyle = wts[j] > 0.45 ? th.bg : th.fg;
              ctx.font = '10px sans-serif';
              ctx.textAlign = 'center';
              ctx.fillText(wts[j].toFixed(2), cx + cell / 2, cy + cell / 2 + 3.5);
            }
          } else {
            ctx.strokeStyle = th.line;
            ctx.lineWidth = 1;
            roundRect(ctx, cx + 1.5, cy + 1.5, cell - 3, cell - 3, 3);
            ctx.stroke();
            ctx.fillStyle = th.red;
            ctx.globalAlpha = 0.55;
            ctx.font = Math.round(Math.min(13, cell * 0.4)) + 'px sans-serif';
            ctx.textAlign = 'center';
            ctx.fillText('×', cx + cell / 2, cy + cell / 2 + 4);
            ctx.globalAlpha = 1;
          }
        }
      }

      /* Contorno da célula sob o ponteiro */
      if (hover) {
        ctx.strokeStyle = th.orange;
        ctx.lineWidth = 2;
        roundRect(ctx, x0 + hover.j * cell + 1, y0 + hover.i * cell + 1, cell - 2, cell - 2, 3);
        ctx.stroke();
      }

      /* Rótulos t_1..t_n (linhas e colunas) */
      for (let k = 0; k < n; k++) {
        const lc = k === selRow ? th.cyan : th.comment;
        P.mathText(ctx, 't_' + (k + 1), x0 - 6, y0 + k * cell + cell / 2 + 4, lc, 'right');
        P.mathText(ctx, 't_' + (k + 1), x0 + k * cell + cell / 2, y0 - 8, th.comment, 'center');
      }
      P.mathText(ctx, 'consulta (i)', x0 - 28, y0 + cell * n + 16, th.comment, 'left');
      P.mathText(ctx, 'consultado (j) →', x0 + cell * n, y0 - 8, th.comment, 'right');
    }

    function drawRow() {
      const th = P.theme();
      const { ctx, w, h } = P.setup(cvRow);
      P.clear(ctx, w, h);
      const fr = P.frame(ctx, w, h, 0.4, n + 0.6, 0, 1.05);
      P.axes(ctx, fr, { xlabel: 'j', ylabel: 'a_{i,j}', yticks: [0, 0.5, 1] });
      const wts = weightsRow(selRow);
      const bw = Math.min(28, fr.iw / n * 0.6);
      for (let j = 0; j < n; j++) {
        const px = fr.X(j + 1);
        if (allowed(selRow, j)) {
          ctx.fillStyle = th.purple;
          ctx.globalAlpha = 0.85;
          ctx.fillRect(px - bw / 2, fr.Y(wts[j]), bw, fr.Y(0) - fr.Y(wts[j]));
          ctx.globalAlpha = 1;
        } else {
          ctx.fillStyle = th.red;
          ctx.globalAlpha = 0.5;
          ctx.font = '11px sans-serif';
          ctx.textAlign = 'center';
          ctx.fillText('×', px, fr.Y(0) - 5);
          ctx.globalAlpha = 1;
        }
        P.mathText(ctx, 't_' + (j + 1), px, fr.Y(0) + 14, th.comment, 'center');
      }
      P.mathText(ctx, 'linha i = ' + (selRow + 1), fr.X(0.4) + 8, fr.Y(1.05) + 14, th.cyan, 'left');
    }

    function updateReadout() {
      const el = $('s1-readout');
      const name = mode === 'bi' ? 'bidirecional' : (mode === 'causal' ? 'causal' : 'sliding window');
      if (hover) {
        const i = hover.i + 1, j = hover.j + 1;
        if (allowed(hover.i, hover.j)) {
          const wts = weightsRow(hover.i);
          el.textContent = 'célula (' + i + ', ' + j + '): score = ' +
            scores[hover.i][hover.j].toFixed(2) + ' · peso = ' + wts[hover.j].toFixed(2);
        } else {
          const why = mode === 'causal' || hover.j > hover.i ? 'j > i' : 'j < i − W';
          el.textContent = 'célula (' + i + ', ' + j + '): mascarada (' + why + '), peso 0';
        }
      } else {
        el.textContent = 'modo ' + name + ' · linha i = ' + (selRow + 1) + ' selecionada';
      }
    }

    function redraw() {
      drawMatrix();
      drawRow();
      updateReadout();
    }

    function setMode(m) {
      mode = m;
      for (const k in tabs) tabs[k].classList.toggle('active', k === m);
      const wrap = $('s1-w-wrap');
      wrap.classList.toggle('disabled', m !== 'swa');
      slW.disabled = m !== 'swa';
      redraw();
    }

    function cellAt(e) {
      if (!geom) return null;
      const r = cvMat.getBoundingClientRect();
      const px = e.clientX - r.left, py = e.clientY - r.top;
      const j = Math.floor((px - geom.x0) / geom.cell);
      const i = Math.floor((py - geom.y0) / geom.cell);
      if (i < 0 || i >= n || j < 0 || j >= n) return null;
      return { i, j };
    }

    cvMat.addEventListener('pointermove', (e) => { hover = cellAt(e); redraw(); });
    cvMat.addEventListener('pointerleave', () => { hover = null; redraw(); });
    cvMat.addEventListener('pointerdown', (e) => {
      const c = cellAt(e);
      if (c) { selRow = c.i; redraw(); }
    });

    tabs.bi.addEventListener('click', () => setMode('bi'));
    tabs.causal.addEventListener('click', () => setMode('causal'));
    tabs.swa.addEventListener('click', () => setMode('swa'));

    slN.addEventListener('input', () => {
      n = +slN.value;
      $('s1-n-val').textContent = n;
      selRow = Math.min(selRow, n - 1);
      hover = null;
      resample();
      redraw();
    });
    slW.addEventListener('input', () => {
      W = +slW.value;
      $('s1-w-val').textContent = W;
      redraw();
    });
    $('s1-reseed').addEventListener('click', () => {
      seed = (seed * 1103515245 + 12345) >>> 0;
      resample();
      redraw();
    });

    resample();
    redraw();
    P.onRedraw(redraw);
    P.observeResize(cvMat, redraw);
    P.observeResize(cvRow, redraw);
  }

  DL.sections.push({ name: 's1-causal-mask', init });
})();
