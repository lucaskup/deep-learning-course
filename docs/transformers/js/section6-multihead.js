/* Seção 2: self-attention multi-head sobre a frase dos slides, com a matriz
   softmax(QK^T/√d) de cada cabeça em heatmap, máscara causal opcional,
   concatenação das cabeças e projeção final W_O. */
(function () {
  'use strict';
  window.DL = window.DL || {};
  DL.sections = DL.sections || [];

  const TOKENS = ['The', 'animal', "didn't", 'cross', 'the', 'street', 'because', 'it', 'was', 'tired'];
  const M = TOKENS.length, D = 16;

  function init() {
    const P = DL.plot;
    const U = DL.utils;
    const $ = (id) => document.getElementById(id);

    const cvGrid = $('s6-grid'), cvDetail = $('s6-detail');
    const slH = $('s6-heads');
    const tabEnc = $('s6-tab-enc'), tabDec = $('s6-tab-dec');

    let H = Math.pow(2, +slH.value);
    let wseed = 11;
    let causal = false;
    let selHead = 0, selTok = TOKENS.indexOf('it');
    let X, heads, concatOut, finalOut, tiles = [];

    /* ── embeddings fixos: vetor por palavra (minúscula) + positional encoding ── */
    function buildEmbeddings() {
      const rng = U.mulberry32(42);
      const byWord = {};
      for (const t of TOKENS) {
        const w = t.toLowerCase();
        if (!byWord[w]) {
          const v = new Array(D);
          for (let i = 0; i < D; i++) v[i] = U.randn(rng);
          byWord[w] = v;
        }
      }
      /* correferência sintética: "it" compartilha componente com "animal" */
      const it = byWord['it'], an = byWord['animal'];
      for (let i = 0; i < D; i++) it[i] = 0.55 * it[i] + 0.8 * an[i];

      X = [];
      for (let t = 0; t < M; t++) {
        const base = byWord[TOKENS[t].toLowerCase()];
        const x = new Array(D);
        for (let j = 0; j < D; j++) {
          const k = Math.floor(j / 2);
          const ang = t / Math.pow(100, (2 * k) / D);
          const pe = j % 2 === 0 ? Math.sin(ang) : Math.cos(ang);
          x[j] = base[j] + 0.4 * pe;
        }
        /* normaliza para manter os scores em faixa moderada */
        let nrm = 0;
        for (const v of x) nrm += v * v;
        nrm = Math.sqrt(nrm) || 1;
        for (let j = 0; j < D; j++) x[j] = (x[j] / nrm) * Math.sqrt(D) * 0.6;
        X.push(x);
      }
    }

    function randMat(rng, rows, cols, scale) {
      const W = [];
      for (let r = 0; r < rows; r++) {
        const row = new Array(cols);
        for (let c = 0; c < cols; c++) row[c] = U.randn(rng) * scale;
        W.push(row);
      }
      return W;
    }

    function matVec(x, W) {
      const out = new Array(W[0].length).fill(0);
      for (let r = 0; r < x.length; r++) {
        const xr = x[r], Wr = W[r];
        for (let c = 0; c < Wr.length; c++) out[c] += xr * Wr[c];
      }
      return out;
    }

    function softmaxRow(xs) {
      const mx = Math.max(...xs);
      const ex = xs.map((v) => (v === -Infinity ? 0 : Math.exp(v - mx)));
      const Z = ex.reduce((a, b) => a + b, 0) || 1;
      return ex.map((v) => v / Z);
    }

    /* ── forward do multi-head: A_h, saídas por cabeça, concat e W_O ── */
    function recompute() {
      const dh = D / H;
      const rng = U.mulberry32(wseed + H * 97);
      const scale = 1 / Math.sqrt(D);
      heads = [];
      for (let h = 0; h < H; h++) {
        const Wq = randMat(rng, D, dh, scale);
        const Wk = randMat(rng, D, dh, scale);
        const Wv = randMat(rng, D, dh, scale);
        const Q = X.map((x) => matVec(x, Wq));
        const K = X.map((x) => matVec(x, Wk));
        const V = X.map((x) => matVec(x, Wv));
        const A = [];
        for (let t = 0; t < M; t++) {
          const sc = new Array(M);
          for (let i = 0; i < M; i++) {
            let s = 0;
            for (let j = 0; j < dh; j++) s += Q[t][j] * K[i][j];
            /* scores grandes: amplifica para padrões mais nítidos no heatmap */
            s = (s / Math.sqrt(dh)) * 3;
            sc[i] = causal && i > t ? -Infinity : s;
          }
          A.push(softmaxRow(sc));
        }
        const C = [];
        for (let t = 0; t < M; t++) {
          const c = new Array(dh).fill(0);
          for (let i = 0; i < M; i++) {
            for (let j = 0; j < dh; j++) c[j] += A[t][i] * V[i][j];
          }
          C.push(c);
        }
        heads.push({ A, C });
      }
      const WO = randMat(rng, D, D, scale);
      concatOut = [];
      for (const head of heads) concatOut.push(...head.C[selTok]);
      finalOut = matVec(concatOut, WO);
    }

    /* ── grade de heatmaps (um por cabeça) ── */
    function drawGrid() {
      const th = P.theme();
      const { ctx, w, h } = P.setup(cvGrid);
      P.clear(ctx, w, h);
      const pc = P.rgb(th.purple);
      const cols = H <= 2 ? H : 4;
      const rows = Math.ceil(H / cols);
      const titleH = 16, gap = 10;
      const size = Math.min((w - gap * (cols + 1)) / cols, (h - (titleH + gap) * rows - gap) / rows);
      const totW = cols * size + (cols - 1) * gap;
      const x0all = (w - totW) / 2;
      tiles = [];
      ctx.font = '11px sans-serif';
      for (let hd = 0; hd < H; hd++) {
        const cx = x0all + (hd % cols) * (size + gap);
        const cy = gap + titleH + Math.floor(hd / cols) * (size + titleH + gap);
        tiles.push({ h: hd, x0: cx, y0: cy, size });
        ctx.fillStyle = hd === selHead ? th.cyan : th.comment;
        ctx.textAlign = 'center';
        ctx.fillText('cabeça ' + (hd + 1), cx + size / 2, cy - 4);
        const cell = size / M;
        const A = heads[hd].A;
        for (let t = 0; t < M; t++) {
          for (let i = 0; i < M; i++) {
            const a = A[t][i];
            ctx.fillStyle = 'rgba(' + pc[0] + ',' + pc[1] + ',' + pc[2] + ',' + (0.06 + 0.94 * a) + ')';
            ctx.fillRect(cx + i * cell, cy + t * cell, cell + 0.5, cell + 0.5);
          }
        }
        /* rótulos de palavra quando há espaço (H pequeno) */
        if (cell >= 18) {
          ctx.fillStyle = th.comment;
          ctx.font = '9px sans-serif';
          ctx.textAlign = 'right';
          for (let t = 0; t < M; t++) ctx.fillText(TOKENS[t], cx - 3, cy + t * cell + cell / 2 + 3);
          ctx.font = '11px sans-serif';
        }
        ctx.strokeStyle = hd === selHead ? th.cyan : th.line;
        ctx.lineWidth = hd === selHead ? 2 : 1;
        ctx.strokeRect(cx, cy, size, size);
        if (hd === selHead) {
          ctx.strokeStyle = th.green;
          ctx.lineWidth = 1.6;
          ctx.strokeRect(cx, cy + selTok * cell, size, cell);
        }
      }
    }

    /* ── detalhe: linha da atenção selecionada + concat → W_O ── */
    function diverging(th, v, vmax) {
      const c = P.rgb(v >= 0 ? th.cyan : th.orange);
      const a = Math.min(1, Math.abs(v) / (vmax || 1));
      return 'rgba(' + c[0] + ',' + c[1] + ',' + c[2] + ',' + (0.08 + 0.92 * a) + ')';
    }

    function drawDetail() {
      const th = P.theme();
      const { ctx, w, h } = P.setup(cvDetail);
      P.clear(ctx, w, h);
      const A = heads[selHead].A[selTok];

      /* barras dos pesos a_i^{(t)} */
      const barTop = 22, barH = h * 0.42;
      const padL = 34, padR = 10;
      const bw = (w - padL - padR) / M;
      const maxA = Math.max(...A, 1e-9);
      ctx.font = '10px sans-serif';
      ctx.textAlign = 'center';
      const best = A.indexOf(maxA);
      for (let i = 0; i < M; i++) {
        const x = padL + i * bw;
        const bh = (A[i] / 1.0) * barH;
        ctx.fillStyle = i === best ? th.green : th.purple;
        ctx.globalAlpha = 0.9;
        ctx.fillRect(x + bw * 0.15, barTop + barH - bh, bw * 0.7, bh);
        ctx.globalAlpha = 1;
        ctx.fillStyle = th.comment;
        ctx.save();
        ctx.translate(x + bw * 0.5, barTop + barH + 9);
        ctx.rotate(-0.5);
        ctx.fillText(TOKENS[i], 0, 0);
        ctx.restore();
      }
      ctx.strokeStyle = th.line;
      ctx.beginPath();
      ctx.moveTo(padL, barTop + barH);
      ctx.lineTo(w - padR, barTop + barH);
      ctx.stroke();
      ctx.fillStyle = th.fg;
      ctx.textAlign = 'left';
      P.mathText(ctx, 'a_i de "' + TOKENS[selTok] + '" na cabeça ' + (selHead + 1), padL, 14, th.fg);

      /* strips: concat das cabeças e saída após W_O */
      const stripY = barTop + barH + 38;
      const stripH = 20, cell = (w - padL - padR) / D;
      const vmax = Math.max(...concatOut.map(Math.abs), ...finalOut.map(Math.abs), 1e-9);
      ctx.fillStyle = th.comment;
      ctx.font = '10px sans-serif';
      ctx.fillText('concat(head_1, …, head_' + H + ')  ·  ' + H + ' × ' + (D / H) + ' dims', padL, stripY - 4);
      for (let j = 0; j < D; j++) {
        ctx.fillStyle = diverging(th, concatOut[j], vmax);
        ctx.fillRect(padL + j * cell, stripY, cell + 0.5, stripH);
      }
      ctx.strokeStyle = th.line;
      ctx.strokeRect(padL, stripY, w - padL - padR, stripH);
      /* separadores entre cabeças */
      const dh = D / H;
      ctx.strokeStyle = th.fg;
      for (let hd = 1; hd < H; hd++) {
        ctx.beginPath();
        ctx.moveTo(padL + hd * dh * cell, stripY);
        ctx.lineTo(padL + hd * dh * cell, stripY + stripH);
        ctx.stroke();
      }
      const y2 = stripY + stripH + 26;
      ctx.fillStyle = th.comment;
      ctx.fillText('× W_O  →  saída do multi-head (D = ' + D + ')', padL, y2 - 4);
      for (let j = 0; j < D; j++) {
        ctx.fillStyle = diverging(th, finalOut[j], vmax);
        ctx.fillRect(padL + j * cell, y2, cell + 0.5, stripH);
      }
      ctx.strokeStyle = th.line;
      ctx.strokeRect(padL, y2, w - padL - padR, stripH);
    }

    function updateReadout() {
      const A = heads[selHead].A[selTok];
      let best = 0;
      for (let i = 1; i < M; i++) if (A[i] > A[best]) best = i;
      $('s6-readout').textContent =
        'cabeça ' + (selHead + 1) + ' · query "' + TOKENS[selTok] + '" · maior peso: "' +
        TOKENS[best] + '" (' + A[best].toFixed(2) + ')';
    }

    function redraw() {
      drawGrid();
      drawDetail();
      updateReadout();
    }

    cvGrid.addEventListener('pointerdown', (e) => {
      const r = cvGrid.getBoundingClientRect();
      const px = e.clientX - r.left, py = e.clientY - r.top;
      for (const t of tiles) {
        if (px >= t.x0 && px <= t.x0 + t.size && py >= t.y0 && py <= t.y0 + t.size) {
          selHead = t.h;
          selTok = Math.max(0, Math.min(M - 1, Math.floor((py - t.y0) / (t.size / M))));
          recompute();
          redraw();
          return;
        }
      }
    });

    slH.addEventListener('input', () => {
      H = Math.pow(2, +slH.value);
      $('s6-heads-val').textContent = H;
      if (selHead >= H) selHead = H - 1;
      recompute();
      redraw();
    });
    function setCausal(c) {
      causal = c;
      tabEnc.classList.toggle('active', !c);
      tabDec.classList.toggle('active', c);
      recompute();
      redraw();
    }
    tabEnc.addEventListener('click', () => setCausal(false));
    tabDec.addEventListener('click', () => setCausal(true));
    $('s6-resample').addEventListener('click', () => {
      wseed = (wseed * 1664525 + 1013904223) >>> 0;
      recompute();
      redraw();
    });

    buildEmbeddings();
    recompute();
    redraw();
    P.onRedraw(redraw);
    P.observeResize(cvGrid, redraw);
    P.observeResize(cvDetail, redraw);
  }

  DL.sections.push({ name: 's6-multihead', init });
})();
