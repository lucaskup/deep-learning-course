/* Seção 3: InfoNCE e temperatura.
   N pares imagem/texto (com semente). S = ûV̂ᵀ; softmax por linha e por coluna
   sobre S/τ. Perdas L_img→txt, L_txt→img e L_CLIP = ½(soma). Barras mostram a
   probabilidade da diagonal afiando conforme τ cai. */
(function () {
  'use strict';
  window.DL = window.DL || {};
  DL.sections = DL.sections || [];

  const N = 5, SEED = 13;
  const tauOf = (v) => 0.01 * Math.pow(100, v / 100);   // v∈[0,100] → τ∈[0.01,1]

  function init() {
    if (!document.getElementById('s3-row')) return;
    const P = DL.plot;
    const $ = (id) => document.getElementById(id);
    const cvRow = $('s3-row'), cvCol = $('s3-col'), cvBar = $('s3-bar');
    const slTau = $('s3-tau'), slAlign = $('s3-align');

    let tau = tauOf(+slTau.value);
    let align = +slAlign.value / 100;

    /* Geometria fixa (com semente): ângulos das imagens e deslocamento dos textos. */
    const imgA = [], off = [];
    (function () {
      const rng = DL.utils.mulberry32(SEED);
      for (let i = 0; i < N; i++) {
        imgA.push(-Math.PI / 2 + i * (2 * Math.PI / N) + 0.2 * (rng() - 0.5));
        off.push((rng() - 0.5) * 2.4);            // deslocamento do texto em relação à imagem
      }
    })();

    function txtAngle(j) { return imgA[j] + (1 - align) * off[j]; }
    function S(i, j) { return Math.cos(imgA[i] - txtAngle(j)); }

    /* Softmax por linha: p_row[i][j] sobre j. Softmax por coluna: p_col[i][j] sobre i. */
    function rowSoftmax() {
      const M = [];
      for (let i = 0; i < N; i++) {
        let mx = -Infinity;
        for (let j = 0; j < N; j++) mx = Math.max(mx, S(i, j) / tau);
        let sum = 0; const e = [];
        for (let j = 0; j < N; j++) { const v = Math.exp(S(i, j) / tau - mx); e.push(v); sum += v; }
        M.push(e.map((v) => v / sum));
      }
      return M;
    }
    function colSoftmax() {
      const M = [];
      for (let i = 0; i < N; i++) M.push(new Array(N).fill(0));
      for (let j = 0; j < N; j++) {
        let mx = -Infinity;
        for (let i = 0; i < N; i++) mx = Math.max(mx, S(i, j) / tau);
        let sum = 0; const e = [];
        for (let i = 0; i < N; i++) { const v = Math.exp(S(i, j) / tau - mx); e.push(v); sum += v; }
        for (let i = 0; i < N; i++) M[i][j] = e[i] / sum;
      }
      return M;
    }

    function drawProbHeat(cv, M) {
      const th = P.theme();
      const { ctx, w, h } = P.setup(cv);
      P.clear(ctx, w, h);
      const padL = 16, padT = 10, padR = 10, padB = 10;
      const gw = w - padL - padR, gh = h - padT - padB;
      const cell = Math.min(gw, gh) / N;
      const ox = padL + (gw - cell * N) / 2;
      const oy = padT + (gh - cell * N) / 2;
      for (let i = 0; i < N; i++) {
        for (let j = 0; j < N; j++) {
          const p = M[i][j];
          const col = P.viridis(p);
          const x = ox + j * cell, y = oy + i * cell;
          ctx.fillStyle = 'rgb(' + col[0] + ',' + col[1] + ',' + col[2] + ')';
          ctx.fillRect(x, y, cell - 1, cell - 1);
          ctx.fillStyle = p > 0.55 ? '#101010' : '#f0f0f0';
          ctx.font = (cell > 50 ? 11 : 9) + 'px sans-serif';
          ctx.textAlign = 'center';
          ctx.fillText(p.toFixed(2), x + cell / 2, y + cell / 2 + 4);
          if (i === j) {
            ctx.strokeStyle = th.green; ctx.lineWidth = 2.5;
            ctx.strokeRect(x + 1.5, y + 1.5, cell - 4, cell - 4);
          }
        }
      }
    }

    function drawBars(rowM) {
      const th = P.theme();
      const { ctx, w, h } = P.setup(cvBar);
      P.clear(ctx, w, h);
      const padL = 30, padT = 16, padR = 12, padB = 26;
      const iw = w - padL - padR, ih = h - padT - padB;
      const base = padT + ih;
      /* Eixo y 0..1. */
      ctx.strokeStyle = th.line; ctx.lineWidth = 1;
      ctx.beginPath(); ctx.moveTo(padL, padT); ctx.lineTo(padL, base); ctx.lineTo(padL + iw, base); ctx.stroke();
      ctx.fillStyle = th.comment; ctx.font = '10px sans-serif'; ctx.textAlign = 'right';
      for (const yv of [0, 0.5, 1]) {
        const yy = base - yv * ih;
        ctx.fillText(yv.toFixed(1), padL - 4, yy + 3);
        ctx.strokeStyle = th.line; ctx.globalAlpha = 0.5;
        ctx.beginPath(); ctx.moveTo(padL, yy); ctx.lineTo(padL + iw, yy); ctx.stroke();
        ctx.globalAlpha = 1;
      }
      /* Linha alvo p = 1. */
      ctx.strokeStyle = th.green; ctx.setLineDash([4, 3]); ctx.lineWidth = 1.2;
      ctx.beginPath(); ctx.moveTo(padL, padT); ctx.lineTo(padL + iw, padT); ctx.stroke();
      ctx.setLineDash([]);
      const bw = iw / N;
      for (let i = 0; i < N; i++) {
        const p = rowM[i][i];
        const bx = padL + i * bw + bw * 0.2;
        const bwid = bw * 0.6;
        const bh = p * ih;
        ctx.fillStyle = th.green; ctx.globalAlpha = 0.85;
        ctx.fillRect(bx, base - bh, bwid, bh);
        ctx.globalAlpha = 1;
        ctx.fillStyle = th.comment; ctx.font = '10px sans-serif'; ctx.textAlign = 'center';
        P.mathText(ctx, 'p_{' + (i + 1) + (i + 1) + '}', bx + bwid / 2, base + 14, th.comment, 'center', 10);
      }
    }

    function losses(rowM, colM) {
      let li = 0, lt = 0;
      for (let i = 0; i < N; i++) { li += -Math.log(Math.max(1e-12, rowM[i][i])); lt += -Math.log(Math.max(1e-12, colM[i][i])); }
      li /= N; lt /= N;
      return { li, lt, clip: 0.5 * (li + lt) };
    }

    function redraw() {
      const rowM = rowSoftmax(), colM = colSoftmax();
      drawProbHeat(cvRow, rowM);
      drawProbHeat(cvCol, colM);
      drawBars(rowM);
      const L = losses(rowM, colM);
      $('s3-readout').textContent =
        'L_img→txt = ' + L.li.toFixed(3) +
        ' · L_txt→img = ' + L.lt.toFixed(3) +
        ' · L_CLIP = ' + L.clip.toFixed(3);
    }

    slTau.addEventListener('input', () => {
      tau = tauOf(+slTau.value);
      $('s3-tau-val').textContent = tau.toFixed(2);
      redraw();
    });
    slAlign.addEventListener('input', () => {
      align = +slAlign.value / 100;
      $('s3-align-val').textContent = align.toFixed(2);
      redraw();
    });

    $('s3-tau-val').textContent = tau.toFixed(2);
    $('s3-align-val').textContent = align.toFixed(2);
    redraw();
    P.onRedraw(redraw);
    P.observeResize(cvRow, redraw);
    P.observeResize(cvCol, redraw);
    P.observeResize(cvBar, redraw);
  }

  DL.sections.push({ name: 's3-infonce', init });
})();
