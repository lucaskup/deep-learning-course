/* Seção 6: uma CNN completa em miniatura, com todos os valores visíveis.
   Duas linhas de canvas: (1) entrada → conv1 → ReLU → max pooling e
   (2) pooling → conv2 → ReLU → flatten → classificador linear → sigmoide.
   Hover em qualquer célula mostra a janela lida e a conta completa; o botão
   de passo percorre as posições da convolução; a rede treina por SGD no
   navegador com backprop implementado à mão (js/tinycnn.js). */
(function () {
  'use strict';
  window.DL = window.DL || {};
  DL.sections = DL.sections || [];

  function init() {
    const P = DL.plot, G = DL.grid, T = DL.tinycnn;
    const $ = (id) => document.getElementById(id);
    const N = T.N, M = T.M, F1 = T.F1, F2 = T.F2, C = T.C;

    const cv1 = $('s6-row1'), cv2 = $('s6-row2'), cv3 = $('s6-row3'), cvLoss = $('s6-loss');
    const slLr = $('s6-lr');

    /* ── Estado ─────────────────────────────────────────────────────── */
    const dataRng = DL.utils.mulberry32(2026);
    const train = T.makeDataset(240, dataRng);
    const test = T.makeDataset(80, dataRng);
    const BATCH = 16, STEPS_PER_FRAME = 3, MAX_EPOCHS = 300;
    const NSTEPS = F1 * N * N + F2 * M * M; /* posições da conv1 + conv2 */

    let seedW = 11;
    let model = T.initModel(DL.utils.mulberry32(seedW));
    let X, label;               /* imagem atual e rótulo (null se pintada) */
    let cache;                  /* forward da imagem atual */
    let hover = null;           /* célula sob o mouse */
    let stepIdx = null;         /* null = tudo visível; k = até a posição k */
    let animating = false, animClock = 0;
    let training = false, epoch = 0, stepInEpoch = 0, order = [], epochLoss = 0;
    const hist = { loss: [], acc: [] };
    let painting = false, paintVal = 1;
    let hits1 = [], hits2 = [], hits3 = [];  /* caixas de acerto do mouse por linha */

    function newImage() {
      const ex = T.makeExample(dataRng, Math.floor(dataRng() * 2));
      X = ex.X; label = ex.y;
      recompute();
    }
    function recompute() { cache = T.forward(model, X); }

    /* ── Utilidades de desenho ──────────────────────────────────────── */
    const fmt = (v) => (v < 0 ? '−' : '') + Math.abs(v).toFixed(2);
    const fmtS = (v) => (v < 0 ? '−' : '') + Math.abs(v).toFixed(2);

    function maxAbs3(a) {
      let m = 1e-6;
      for (const g of a) for (const row of g) for (const v of row) m = Math.max(m, Math.abs(v));
      return m;
    }
    function maxAbs4(W) {
      let m = 1e-6;
      for (const f of W) for (const c of f) for (const row of c) for (const v of row) m = Math.max(m, Math.abs(v));
      return m;
    }

    /* Grade n×n de valores. mode: 'act' (cyan) ou 'w' (divergente).
       visible(i,j) decide se a célula já foi "calculada" no passo a passo. */
    function drawGrid(ctx, th, x0, y0, cs, mat, mode, vmax, visible) {
      const n = mat.length;
      for (let i = 0; i < n; i++) {
        for (let j = 0; j < n; j++) {
          const v = mat[i][j];
          const vis = !visible || visible(i, j);
          let fill, text;
          if (!vis) { fill = th.card; text = null; }
          else if (mode === 'act') {
            const t = Math.min(1, v / vmax) * 0.85;
            fill = G.mix(th.card, th.cyan, t);
            text = t > 0.5 ? th.bg : th.fg;
          } else {
            fill = G.diverge(v, vmax, th);
            text = Math.abs(v) / vmax > 0.6 ? th.bg : th.fg;
          }
          ctx.fillStyle = fill;
          ctx.fillRect(x0 + j * cs, y0 + i * cs, cs, cs);
          if (text) {
            ctx.fillStyle = text;
            ctx.font = Math.max(8, Math.min(12, cs * 0.33)) + 'px sans-serif';
            ctx.textAlign = 'center';
            ctx.textBaseline = 'middle';
            ctx.fillText(fmt(v), x0 + j * cs + cs / 2, y0 + i * cs + cs / 2);
            ctx.textBaseline = 'alphabetic';
          }
        }
      }
      ctx.strokeStyle = th.line;
      ctx.lineWidth = 1;
      ctx.beginPath();
      for (let k = 0; k <= n; k++) {
        ctx.moveTo(x0, y0 + k * cs); ctx.lineTo(x0 + n * cs, y0 + k * cs);
        ctx.moveTo(x0 + k * cs, y0); ctx.lineTo(x0 + k * cs, y0 + n * cs);
      }
      ctx.stroke();
    }

    function box(ctx, x0, y0, cs, i, j, di, dj, color, width) {
      ctx.strokeStyle = color;
      ctx.lineWidth = width || 2.5;
      ctx.strokeRect(x0 + j * cs, y0 + i * cs, cs * dj, cs * di);
    }

    /* Janela 3×3 centrada em (i, j) recortada aos limites da grade. */
    function window3(ctx, x0, y0, cs, n, i, j, color) {
      const i0 = Math.max(0, i - 1), j0 = Math.max(0, j - 1);
      const i1 = Math.min(n - 1, i + 1), j1 = Math.min(n - 1, j + 1);
      box(ctx, x0, y0, cs, i0, j0, i1 - i0 + 1, j1 - j0 + 1, color, 2.5);
    }

    function title(ctx, th, x, y, text, color) {
      P.mathText(ctx, text, x, y, color || th.comment, 'left', 11);
    }

    function arrow(ctx, th, x1, x2, y, lines) {
      ctx.strokeStyle = th.comment;
      ctx.fillStyle = th.comment;
      ctx.lineWidth = 1.4;
      ctx.beginPath();
      ctx.moveTo(x1, y); ctx.lineTo(x2 - 6, y);
      ctx.stroke();
      ctx.beginPath();
      ctx.moveTo(x2, y); ctx.lineTo(x2 - 7, y - 4); ctx.lineTo(x2 - 7, y + 4); ctx.closePath();
      ctx.fill();
      const cx = (x1 + x2) / 2;
      for (let k = 0; k < lines.length; k++) {
        P.mathText(ctx, lines[k], cx, y - 8 - (lines.length - 1 - k) * 12, th.comment, 'center', 10);
      }
    }

    /* Passo a passo: qual célula está sendo calculada agora. */
    function stepCell() {
      if (stepIdx === null) return null;
      if (stepIdx < F1 * N * N) {
        const f = Math.floor(stepIdx / (N * N)), r = stepIdx % (N * N);
        return { kind: 'a1', f, i: Math.floor(r / N), j: r % N };
      }
      const k = stepIdx - F1 * N * N;
      const f = Math.floor(k / (M * M)), r = k % (M * M);
      return { kind: 'a2', f, i: Math.floor(r / M), j: r % M };
    }
    const a1Visible = (f) => (i, j) => stepIdx === null || f * N * N + i * N + j <= stepIdx;
    const p1Visible = () => stepIdx === null || stepIdx >= F1 * N * N - 1;
    const a2Visible = (f) => (i, j) => stepIdx === null || F1 * N * N + f * M * M + i * M + j <= stepIdx;
    const outVisible = () => stepIdx === null || stepIdx >= NSTEPS - 1;

    /* Célula em destaque: hover tem prioridade sobre o passo a passo. */
    function focus() { return hover || stepCell(); }

    /* ── Linha 1: x → conv1 → ReLU → pooling ─────────────────────────── */
    function drawRow1() {
      const th = P.theme();
      const { ctx, w, h } = P.setup(cv1);
      P.clear(ctx, w, h);
      hits1 = [];

      const cs = Math.max(14, Math.min((w - 16) / 28.6, (h - 34) / 12.9));
      const aw = 2.4 * cs, gapV = 0.9 * cs, top = 26;
      const xX = 8, xW = xX + 6 * cs + aw, xA = xW + 6.3 * cs + aw, xP = xA + 6 * cs + aw;
      const yG = [top, top + 6 * cs + gapV];      /* topo dos mapas 6×6 */
      const yF = yG.map((y) => y + 1.5 * cs);      /* topo dos kernels 3×3 */
      const yP = yG.map((y) => y + 1.5 * cs);      /* topo dos mapas 3×3 */

      const vmaxX = 1, vmaxA = maxAbs3(cache.a1), vmaxW = maxAbs4(model.W1);
      const fc = focus();

      /* Entrada */
      for (let c = 0; c < C; c++) {
        title(ctx, th, xX, yG[c] - 7, 'x_' + c + '  (canal ' + c + ')');
        drawGrid(ctx, th, xX, yG[c], cs, X[c], 'act', vmaxX);
        hits1.push({ kind: 'x', c, x0: xX, y0: yG[c], cs, n: N });
      }
      title(ctx, th, xX, h - 6, 'clique ou arraste para pintar', th.comment);

      arrow(ctx, th, xX + 6 * cs + 4, xW - 4, yG[0] + 6 * cs + gapV / 2, ['∗', 'conv 3×3', 'pad 1']);

      /* Filtros da conv1: filtro f alinhado ao mapa a1_f */
      for (let f = 0; f < F1; f++) {
        title(ctx, th, xW, yF[f] - 7, 'w¹_' + f + '   b¹_' + f + ' = ' + fmt(model.b1[f]));
        for (let c = 0; c < C; c++) {
          const x0 = xW + c * 3.3 * cs;
          drawGrid(ctx, th, x0, yF[f], cs, model.W1[f][c], 'w', vmaxW);
          P.mathText(ctx, 'canal ' + c, x0 + 1.5 * cs, yF[f] + 3 * cs + 12, th.comment, 'center', 10);
          hits1.push({ kind: 'w1', f, c, x0, y0: yF[f], cs, n: 3 });
        }
      }

      arrow(ctx, th, xW + 6.3 * cs + 4, xA - 4, yG[0] + 6 * cs + gapV / 2, ['+ b', 'ReLU']);

      /* Mapas de ativação a1 */
      for (let f = 0; f < F1; f++) {
        title(ctx, th, xA, yG[f] - 7, 'a¹_' + f + ' = ReLU(z¹_' + f + ')');
        drawGrid(ctx, th, xA, yG[f], cs, cache.a1[f], 'act', vmaxA, a1Visible(f));
        hits1.push({ kind: 'a1', f, x0: xA, y0: yG[f], cs, n: N });
        /* janelas 2×2 do pooling em linha mais forte */
        ctx.strokeStyle = th.comment;
        ctx.lineWidth = 1.4;
        ctx.beginPath();
        for (let k = 0; k <= N; k += 2) {
          ctx.moveTo(xA, yG[f] + k * cs); ctx.lineTo(xA + N * cs, yG[f] + k * cs);
          ctx.moveTo(xA + k * cs, yG[f]); ctx.lineTo(xA + k * cs, yG[f] + N * cs);
        }
        ctx.stroke();
      }

      arrow(ctx, th, xA + 6 * cs + 4, xP - 4, yG[0] + 6 * cs + gapV / 2, ['max pool', '2×2']);

      /* Saída do pooling */
      for (let f = 0; f < F1; f++) {
        title(ctx, th, xP, yP[f] - 7, 'p¹_' + f);
        drawGrid(ctx, th, xP, yP[f], cs, cache.p1[f], 'act', vmaxA, p1Visible);
        hits1.push({ kind: 'p1', f, x0: xP, y0: yP[f], cs, n: M });
      }

      /* Destaques */
      if (fc && fc.kind === 'a1') {
        for (let c = 0; c < C; c++) window3(ctx, xX, yG[c], cs, N, fc.i, fc.j, th.orange);
        for (let c = 0; c < C; c++) box(ctx, xW + c * 3.3 * cs, yF[fc.f], cs, 0, 0, 3, 3, th.orange, 2.5);
        box(ctx, xA, yG[fc.f], cs, fc.i, fc.j, 1, 1, th.orange, 3);
      } else if (fc && fc.kind === 'p1') {
        box(ctx, xA, yG[fc.f], cs, 2 * fc.i, 2 * fc.j, 2, 2, th.orange, 2.5);
        const [bi, bj] = cache.arg[fc.f][fc.i][fc.j];
        box(ctx, xA, yG[fc.f], cs, bi, bj, 1, 1, th.pink, 2.5);
        box(ctx, xP, yP[fc.f], cs, fc.i, fc.j, 1, 1, th.orange, 3);
      } else if (fc && fc.kind === 'x') {
        box(ctx, xX, yG[fc.c], cs, fc.i, fc.j, 1, 1, th.orange, 3);
      } else if (fc && fc.kind === 'w1') {
        box(ctx, xW + fc.c * 3.3 * cs, yF[fc.f], cs, fc.i, fc.j, 1, 1, th.orange, 3);
      } else if (fc && fc.kind === 'a2') {
        /* a conv2 lê p1: destaca as janelas no pooling desta linha também */
        for (let c = 0; c < F1; c++) window3(ctx, xP, yP[c], cs, M, fc.i, fc.j, th.orange);
      }
    }

    /* ── Linha 2: pooling → conv2 → ReLU → flatten → linear → σ ─────── */
    function drawRow2() {
      const th = P.theme();
      const { ctx, w, h } = P.setup(cv2);
      P.clear(ctx, w, h);
      hits2 = [];

      const cs = Math.max(14, Math.min((w - 16) / 19.6, (h - 34) / 8.4));
      const aw = 2.4 * cs, gapV = 1.4 * cs, top = 26;
      const xP = 8, xW = xP + 3 * cs + aw, xA = xW + 6.3 * cs + aw;
      const yR = [top, top + 3 * cs + gapV];
      const vmaxP = maxAbs3(cache.p1), vmaxA = maxAbs3(cache.a2), vmaxW = maxAbs4(model.W2);
      const fc = focus();

      for (let f = 0; f < F1; f++) {
        title(ctx, th, xP, yR[f] - 7, 'p¹_' + f);
        drawGrid(ctx, th, xP, yR[f], cs, cache.p1[f], 'act', vmaxP, p1Visible);
        hits2.push({ kind: 'p1', f, x0: xP, y0: yR[f], cs, n: M });
      }

      arrow(ctx, th, xP + 3 * cs + 4, xW - 4, yR[0] + 3 * cs + gapV / 2, ['∗', 'conv 3×3', 'pad 1']);

      for (let f = 0; f < F2; f++) {
        title(ctx, th, xW, yR[f] - 7, 'w²_' + f + '   b²_' + f + ' = ' + fmt(model.b2[f]));
        for (let c = 0; c < F1; c++) {
          const x0 = xW + c * 3.3 * cs;
          drawGrid(ctx, th, x0, yR[f], cs, model.W2[f][c], 'w', vmaxW);
          P.mathText(ctx, 'mapa ' + c, x0 + 1.5 * cs, yR[f] + 3 * cs + 12, th.comment, 'center', 10);
          hits2.push({ kind: 'w2', f, c, x0, y0: yR[f], cs, n: 3 });
        }
      }

      arrow(ctx, th, xW + 6.3 * cs + 4, xA - 4, yR[0] + 3 * cs + gapV / 2, ['+ b', 'ReLU']);

      for (let f = 0; f < F2; f++) {
        title(ctx, th, xA, yR[f] - 7, 'a²_' + f + ' = ReLU(z²_' + f + ')');
        drawGrid(ctx, th, xA, yR[f], cs, cache.a2[f], 'act', vmaxA, a2Visible(f));
        hits2.push({ kind: 'a2', f, x0: xA, y0: yR[f], cs, n: M });
      }

      arrow(ctx, th, xA + 3 * cs + 4, xA + 3 * cs + aw - 4, yR[0] + 3 * cs + gapV / 2, ['flatten', 'etapa 3']);

      /* Destaques */
      if (fc && fc.kind === 'a2') {
        for (let c = 0; c < F1; c++) window3(ctx, xP, yR[c], cs, M, fc.i, fc.j, th.orange);
        for (let c = 0; c < F1; c++) box(ctx, xW + c * 3.3 * cs, yR[fc.f], cs, 0, 0, 3, 3, th.orange, 2.5);
        box(ctx, xA, yR[fc.f], cs, fc.i, fc.j, 1, 1, th.orange, 3);
      } else if (fc && fc.kind === 'p1') {
        box(ctx, xP, yR[fc.f], cs, fc.i, fc.j, 1, 1, th.orange, 3);
      } else if (fc && fc.kind === 'w2') {
        box(ctx, xW + fc.c * 3.3 * cs, yR[fc.f], cs, fc.i, fc.j, 1, 1, th.orange, 3);
      } else if (fc && (fc.kind === 'wfc' || fc.kind === 'v')) {
        const f = Math.floor(fc.k / (M * M)), r = fc.k % (M * M);
        box(ctx, xA, yR[f], cs, Math.floor(r / M), r % M, 1, 1, th.orange, 3);
      }
    }

    /* ── Linha 3: flatten → produto escalar → logit → σ ─────────────── */
    function drawRow3() {
      const th = P.theme();
      const { ctx, w, h } = P.setup(cv3);
      P.clear(ctx, w, h);
      hits3 = [];

      const D = T.D;
      const cs = Math.max(14, Math.min((w - 16) / 34.2, (h - 56) / 2.2));
      const aw = 2.4 * cs, sq = 1.6 * cs;
      const xV = 8 + 1.3 * cs;                 /* rótulos v_k / w_k à esquerda */
      const yV = 24, yW = yV + cs + 6, yK = yW + cs + 12;
      const xL = xV + D * cs + aw, xS = xL + sq + aw, xT = xS + sq + 0.6 * cs;
      const yc = (yV + yW + cs) / 2;           /* centro vertical das duas linhas */
      const vmaxA = maxAbs3(cache.a2);
      const vmaxFc = Math.max(1e-6, ...model.wfc.map(Math.abs));
      const fc = focus();
      const outVis = outVisible();
      const vVis = (k) => a2Visible(Math.floor(k / (M * M)))(Math.floor((k % (M * M)) / M), k % M);

      /* cabeçalho: de qual mapa veio cada bloco de 9 */
      for (let f = 0; f < F2; f++) {
        P.mathText(ctx, 'de a²_' + f + '  (k = ' + (f * 9) + ' … ' + (f * 9 + 8) + ')', xV + (f * 9 + 4.5) * cs, yV - 7, th.comment, 'center', 10);
      }
      ctx.strokeStyle = th.comment; ctx.lineWidth = 1.4;
      ctx.beginPath(); ctx.moveTo(xV + 9 * cs, yV - 3); ctx.lineTo(xV + 9 * cs, yW + cs + 3); ctx.stroke();

      P.mathText(ctx, 'v_k', xV - 6, yV + cs / 2 + 4, th.fg, 'right', 11);
      P.mathText(ctx, 'w_k', xV - 6, yW + cs / 2 + 4, th.fg, 'right', 11);
      P.mathText(ctx, 'k', xV - 6, yK + 4, th.comment, 'right', 10);

      /* drawGrid trabalha com matrizes n×n: desenha célula a célula aqui */
      for (let k = 0; k < D; k++) {
        drawGrid(ctx, th, xV + k * cs, yV, cs, [[cache.v[k]]], 'act', vmaxA, () => vVis(k));
        drawGrid(ctx, th, xV + k * cs, yW, cs, [[model.wfc[k]]], 'w', vmaxFc);
        P.mathText(ctx, String(k), xV + k * cs + cs / 2, yK + 4, th.comment, 'center', 9);
      }
      hits3.push({ kind: 'v', x0: xV, y0: yV, cs, n: 1, cols: D });
      hits3.push({ kind: 'wfc', x0: xV, y0: yW, cs, n: 1, cols: D });

      /* Σ w·v + b → logit */
      arrow(ctx, th, xV + D * cs + 4, xL - 4, yc, ['Σ_k w_k·v_k', '+ b = ' + fmt(model.bfc)]);
      const yS = yc - sq / 2;
      P.mathText(ctx, 'logit', xL + sq / 2, yS - 7, th.comment, 'center', 10);
      ctx.fillStyle = outVis ? G.diverge(cache.logit, 4, th) : th.card;
      ctx.fillRect(xL, yS, sq, sq);
      ctx.strokeStyle = th.line; ctx.lineWidth = 1; ctx.strokeRect(xL, yS, sq, sq);
      if (outVis) {
        ctx.fillStyle = Math.abs(cache.logit) / 4 > 0.6 ? th.bg : th.fg;
        ctx.font = Math.max(9, Math.min(13, sq * 0.3)) + 'px sans-serif';
        ctx.textAlign = 'center'; ctx.textBaseline = 'middle';
        ctx.fillText(fmt(cache.logit), xL + sq / 2, yS + sq / 2);
        ctx.textBaseline = 'alphabetic';
      }
      hits3.push({ kind: 'logit', x0: xL, y0: yS, cs: sq, n: 1, cols: 1 });

      /* σ → p */
      arrow(ctx, th, xL + sq + 4, xS - 4, yc, ['σ']);
      P.mathText(ctx, 'p = σ(logit)', xS + sq / 2, yS - 7, th.comment, 'center', 10);
      const pr = cache.prob;
      ctx.fillStyle = !outVis ? th.card : pr > 0.5 ? G.mix(th.card, th.orange, (pr - 0.5) * 1.7) : G.mix(th.card, th.cyan, (0.5 - pr) * 1.7);
      ctx.fillRect(xS, yS, sq, sq);
      ctx.strokeStyle = th.line; ctx.strokeRect(xS, yS, sq, sq);
      if (outVis) {
        ctx.fillStyle = Math.abs(pr - 0.5) > 0.3 ? th.bg : th.fg;
        ctx.font = Math.max(9, Math.min(13, sq * 0.3)) + 'px sans-serif';
        ctx.textAlign = 'center'; ctx.textBaseline = 'middle';
        ctx.fillText(fmt(pr), xS + sq / 2, yS + sq / 2);
        ctx.textBaseline = 'alphabetic';
      }
      hits3.push({ kind: 'prob', x0: xS, y0: yS, cs: sq, n: 1, cols: 1 });

      /* Decisão */
      const pred = pr > 0.5 ? 1 : 0;
      const name = (k) => (k === 1 ? 'horizontal (1)' : 'vertical (0)');
      const lh = 15;
      let y2 = yc - lh;
      P.mathText(ctx, 'p > 0.5 ⇒ predição: ' + (outVis ? name(pred) : '?'), xT, y2, th.fg, 'left', 12);
      y2 += lh;
      if (label === null) {
        P.mathText(ctx, 'rótulo: imagem pintada, sem rótulo', xT, y2, th.comment, 'left', 11);
      } else {
        const ok = pred === label;
        P.mathText(ctx, 'rótulo: ' + name(label) + (outVis ? (ok ? '  ✓' : '  ✗') : ''), xT, y2,
          outVis ? (ok ? th.green : th.pink) : th.comment, 'left', 12);
        if (outVis) {
          y2 += lh;
          P.mathText(ctx, 'perda BCE = ' + fmt(T.bceLoss(cache.logit, label)), xT, y2, th.comment, 'left', 11);
        }
      }

      /* Destaques */
      if (fc && (fc.kind === 'wfc' || fc.kind === 'v' || fc.kind === 'a2')) {
        const k = fc.kind === 'a2' ? fc.f * M * M + fc.i * M + fc.j : fc.k;
        box(ctx, xV + k * cs, yV, cs, 0, 0, 1, 1, th.orange, 3);
        box(ctx, xV + k * cs, yW, cs, 0, 0, 1, 1, th.orange, 3);
      } else if (fc && fc.kind === 'logit') {
        box(ctx, xV, yV, cs, 0, 0, 1, D, th.orange, 2.5);
        box(ctx, xV, yW, cs, 0, 0, 1, D, th.orange, 2.5);
        ctx.strokeStyle = th.orange; ctx.lineWidth = 3; ctx.strokeRect(xL, yS, sq, sq);
      } else if (fc && fc.kind === 'prob') {
        ctx.strokeStyle = th.orange; ctx.lineWidth = 3; ctx.strokeRect(xL, yS, sq, sq); ctx.strokeRect(xS, yS, sq, sq);
      }
    }

    /* ── Gráfico de treino ──────────────────────────────────────────── */
    function drawLoss() {
      const th = P.theme();
      const { ctx, w, h } = P.setup(cvLoss);
      P.clear(ctx, w, h);
      const n = hist.loss.length;
      const xmax = Math.max(20, n);
      let ymax = 1;
      for (const v of hist.loss) ymax = Math.max(ymax, v);
      ymax = Math.ceil(ymax * 5) / 5;
      const fr = P.frame(ctx, w, h, 0, xmax, 0, ymax);
      const yt = [];
      for (let v = 0; v <= ymax + 1e-9; v += 0.2) yt.push(Math.round(v * 10) / 10);
      const xt = [0, 0.25, 0.5, 0.75, 1].map((t) => Math.round(t * xmax));
      P.axes(ctx, fr, { xlabel: 'época', xticks: xt, yticks: yt });
      if (n > 0) {
        const xs = hist.loss.map((_, k) => k + 1);
        P.line(ctx, fr, xs, hist.loss, th.purple, { width: 1.8 });
        P.line(ctx, fr, xs, hist.acc.map((a) => a * ymax), th.green, { width: 1.8 });
      }
      P.label(ctx, fr.X(fr.xmin) + 8, fr.Y(fr.ymax) + 14, 'perda BCE (treino)', th.purple);
      P.label(ctx, fr.X(fr.xmin) + 8, fr.Y(fr.ymax) + 28, 'acurácia (teste, escala 0 a 1)', th.green);
    }

    /* ── Leitor de cálculo ──────────────────────────────────────────── */
    const sub = (s) => '<sub>' + s + '</sub>';
    const sup = (s) => '<sup>' + s + '</sup>';
    const F = (s) => '<span class="formula">' + s + '</span>';

    function convHtml(layerName, inName, input, W, b, z, a, fc) {
      const terms = T.convTerms(input, W, fc.f, fc.i, fc.j);
      const parts = [];
      let cur = -1;
      for (const t of terms) {
        if (t.c !== cur) { cur = t.c; parts.push('<br>&nbsp;&nbsp;' + inName + sub(t.c) + ': '); }
        const xs = t.inside ? fmtS(t.x) : '0<span title="fora da imagem (padding)">ₚ</span>';
        parts.push((t.q === 0 && t.a === 0 ? '' : ' + ') + '(' + fmtS(t.w) + ')·' + xs);
      }
      const zv = z[fc.f][fc.i][fc.j], av = a[fc.f][fc.i][fc.j];
      const zn = 'z' + sup('(' + layerName + ')') + sub(fc.f) + '[' + fc.i + ',' + fc.j + ']';
      const an = 'a' + sup('(' + layerName + ')') + sub(fc.f) + '[' + fc.i + ',' + fc.j + ']';
      return F(zn) + ' = ' + F('b' + sup('(' + layerName + ')') + sub(fc.f)) + ' + Σ' + sub('c') + ' Σ' + sub('m,n') +
        ' ' + F('w' + sup('(' + layerName + ')') + sub(fc.f + ',c') + '[m,n] · ' + inName + sub('c') + '[' + fc.i + '+m−1, ' + fc.j + '+n−1]') +
        '<br>= ' + fmtS(b[fc.f]) + parts.join('') +
        '<br>= <b>' + fmtS(zv) + '</b>, e ' + F(an) + ' = ReLU(' + fmtS(zv) + ') = <b>' + fmtS(av) + '</b>' +
        ' &nbsp;·&nbsp; 0ₚ marca posições fora da imagem, preenchidas pelo padding.';
    }

    function calcHtml() {
      let fc = focus();
      if (!fc) {
        return 'Passe o mouse sobre qualquer célula para ver a conta que a produziu, ou use ⏭ Passo para percorrer a rede posição a posição.';
      }
      if (fc.kind === 'x') {
        return F('x' + sub(fc.c) + '[' + fc.i + ',' + fc.j + ']') + ' = <b>' + fmtS(X[fc.c][fc.i][fc.j]) + '</b>. Clique para alternar entre 0 e 1 e criar sua própria imagem.';
      }
      if (fc.kind === 'w1' || fc.kind === 'w2') {
        const W = fc.kind === 'w1' ? model.W1 : model.W2;
        const L = fc.kind === 'w1' ? '1' : '2';
        const npos = fc.kind === 'w1' ? N * N : M * M;
        return F('w' + sup('(' + L + ')') + sub(fc.f + ',' + fc.c) + '[' + fc.i + ',' + fc.j + ']') + ' = <b>' + fmtS(W[fc.f][fc.c][fc.i][fc.j]) +
          '</b>: peso do filtro ' + fc.f + ' sobre o canal ' + fc.c + ' da entrada, compartilhado pelas ' + npos + ' posições do mapa de saída.';
      }
      if (fc.kind === 'a1') return convHtml('1', 'x', X, model.W1, model.b1, cache.z1, cache.a1, fc);
      if (fc.kind === 'a2') return convHtml('2', 'p' + sup('(1)'), cache.p1, model.W2, model.b2, cache.z2, cache.a2, fc);
      if (fc.kind === 'p1') {
        const vals = [];
        for (let di = 0; di < 2; di++) for (let dj = 0; dj < 2; dj++) vals.push(fmtS(cache.a1[fc.f][2 * fc.i + di][2 * fc.j + dj]));
        return F('p' + sup('(1)') + sub(fc.f) + '[' + fc.i + ',' + fc.j + ']') + ' = max(' + vals.join(', ') + ') = <b>' + fmtS(cache.p1[fc.f][fc.i][fc.j]) +
          '</b>. A janela 2×2 laranja em ' + F('a' + sup('(1)') + sub(fc.f)) + ' encolhe para um único valor; a célula rosa é o máximo, a única que recebe gradiente no backward.';
      }
      if (fc.kind === 'v') {
        const k = fc.k, f = Math.floor(k / (M * M)), r = k % (M * M);
        return F('v' + sub(k)) + ' = ' + F('a' + sup('(2)') + sub(f) + '[' + Math.floor(r / M) + ',' + (r % M) + ']') + ' = <b>' + fmtS(cache.v[k]) +
          '</b>. O flatten só reorganiza: a posição no vetor é ' + F('k = f·9 + i·3 + j') + ' = ' + f + '·9 + ' + Math.floor(r / M) + '·3 + ' + (r % M) + ' = ' + k + '.';
      }
      if (fc.kind === 'logit') {
        const parts = [];
        for (let k = 0; k < T.D; k++) parts.push((k ? ' + ' : '') + '(' + fmtS(model.wfc[k]) + ')·' + fmtS(cache.v[k]));
        return F('logit') + ' = ' + F('b + Σ' + sub('k') + ' w' + sub('k') + '·v' + sub('k')) + '<br>= ' + fmtS(model.bfc) + ' + ' + parts.join('') +
          '<br>= <b>' + fmtS(cache.logit) + '</b>. Um único neurônio linear sobre as 18 ativações achatadas.';
      }
      if (fc.kind === 'prob') {
        return F('p') + ' = σ(logit) = 1 / (1 + e' + sup('−logit') + ') = 1 / (1 + e' + sup('−(' + fmtS(cache.logit) + ')') + ') = <b>' + fmtS(cache.prob) +
          '</b>, probabilidade da classe 1 (traço horizontal). A decisão usa o limiar 0.5.';
      }
      if (fc.kind === 'wfc') {
        const k = fc.k, f = Math.floor(k / (M * M)), r = k % (M * M);
        fc = { f, i: Math.floor(r / M), j: r % M };
        const wv = model.wfc[k], vv = cache.v[k];
        return F('w' + sub(k) + '·v' + sub(k)) + ' = (' + fmtS(wv) + ')·' + fmtS(vv) + ' = <b>' + fmtS(wv * vv) + '</b>, contribuição de ' +
          F('a' + sup('(2)') + sub(fc.f) + '[' + fc.i + ',' + fc.j + ']') + ' para o logit. Somando as 18 contribuições e ' + F('b') + ' = ' + fmtS(model.bfc) +
          ': logit = <b>' + fmtS(cache.logit) + '</b>, p = σ(logit) = <b>' + fmtS(cache.prob) + '</b>.';
      }
      return '';
    }

    function updateReadouts() {
      $('s6-calc').innerHTML = calcHtml();
      const pred = cache.prob > 0.5 ? 1 : 0;
      const name = (k) => (k === 1 ? 'horizontal (1)' : 'vertical (0)');
      $('s6-img-readout').textContent = (label === null ? 'imagem pintada, sem rótulo' : 'rótulo: ' + name(label)) +
        ' · p(horizontal) = ' + fmt(cache.prob) + ' → predição: ' + name(pred) + (label === null ? '' : (pred === label ? ' ✓' : ' ✗'));
      const last = hist.loss.length - 1;
      $('s6-train-readout').textContent = 'época ' + epoch + ' · η = ' + (+slLr.value).toFixed(2) +
        (last >= 0 ? ' · perda ' + hist.loss[last].toFixed(3) + ' · acurácia (teste) ' + (100 * hist.acc[last]).toFixed(0) + '%' : ' · rede não treinada') +
        ' · 95 parâmetros';
      const sc = stepCell();
      $('s6-step-readout').textContent = stepIdx === null ? 'todas as posições calculadas' :
        'posição ' + (stepIdx + 1) + ' de ' + NSTEPS + ' · ' + (sc.kind === 'a1' ? 'conv1' : 'conv2') + ', filtro ' + sc.f + ', célula [' + sc.i + ',' + sc.j + ']';
    }

    function redraw() {
      drawRow1();
      drawRow2();
      drawRow3();
      drawLoss();
      updateReadouts();
    }

    /* ── Mouse: hover e pintura ─────────────────────────────────────── */
    function hitTest(hits, cv, e) {
      const r = cv.getBoundingClientRect();
      const px = e.clientX - r.left, py = e.clientY - r.top;
      for (const hb of hits) {
        const j = Math.floor((px - hb.x0) / hb.cs), i = Math.floor((py - hb.y0) / hb.cs);
        const cols = hb.cols || hb.n;
        if (i >= 0 && i < hb.n && j >= 0 && j < cols) return Object.assign({ i, j, k: j }, hb);
      }
      return null;
    }

    cv1.addEventListener('pointermove', (e) => {
      const hb = hitTest(hits1, cv1, e);
      if (painting && hb && hb.kind === 'x') {
        if (X[hb.c][hb.i][hb.j] !== paintVal) {
          X[hb.c][hb.i][hb.j] = paintVal; label = null; recompute();
        }
      }
      hover = hb ? { kind: hb.kind, f: hb.f, c: hb.c, i: hb.i, j: hb.j } : null;
      redraw();
    });
    cv1.addEventListener('pointerleave', () => { hover = null; painting = false; redraw(); });
    cv1.addEventListener('pointerdown', (e) => {
      const hb = hitTest(hits1, cv1, e);
      if (!hb || hb.kind !== 'x') return;
      cv1.setPointerCapture(e.pointerId);
      painting = true;
      paintVal = X[hb.c][hb.i][hb.j] > 0.5 ? 0 : 1;
      X[hb.c][hb.i][hb.j] = paintVal; label = null; recompute();
      redraw();
    });
    cv1.addEventListener('pointerup', () => { painting = false; });

    cv2.addEventListener('pointermove', (e) => {
      const hb = hitTest(hits2, cv2, e);
      hover = hb ? { kind: hb.kind, f: hb.f, c: hb.c, i: hb.i, j: hb.j } : null;
      redraw();
    });
    cv2.addEventListener('pointerleave', () => { hover = null; redraw(); });

    cv3.addEventListener('pointermove', (e) => {
      const hb = hitTest(hits3, cv3, e);
      hover = hb ? { kind: hb.kind, k: hb.k } : null;
      redraw();
    });
    cv3.addEventListener('pointerleave', () => { hover = null; redraw(); });

    /* ── Imagem ─────────────────────────────────────────────────────── */
    $('s6-new').addEventListener('click', () => { newImage(); redraw(); });
    $('s6-clear').addEventListener('click', () => {
      X = T.zeros3(C, N); label = null; recompute(); redraw();
    });

    /* ── Passo a passo ──────────────────────────────────────────────── */
    function stopAnim() {
      if (animating) { animating = false; DL.stopTicker(animTick); $('s6-anim').textContent = '▶ Animar cálculo'; }
    }
    function advance() {
      stepIdx = stepIdx === null ? 0 : stepIdx + 1;
      if (stepIdx >= NSTEPS) { stepIdx = null; stopAnim(); }
    }
    function animTick(dt) {
      animClock += dt;
      const period = stepIdx !== null && stepIdx >= F1 * N * N ? 0.45 : 0.16;
      if (animClock >= period) { animClock = 0; advance(); redraw(); }
    }
    $('s6-step').addEventListener('click', () => { stopAnim(); advance(); redraw(); });
    $('s6-anim').addEventListener('click', () => {
      if (animating) { stopAnim(); }
      else { animating = true; animClock = 0; if (stepIdx === null) stepIdx = -1; DL.startTicker(animTick); $('s6-anim').textContent = '⏸ Pausar cálculo'; }
      redraw();
    });
    $('s6-showall').addEventListener('click', () => { stopAnim(); stepIdx = null; redraw(); });

    /* ── Treinamento ────────────────────────────────────────────────── */
    function shuffleOrder() {
      order = train.map((_, k) => k);
      for (let k = order.length - 1; k > 0; k--) {
        const r = Math.floor(dataRng() * (k + 1));
        [order[k], order[r]] = [order[r], order[k]];
      }
    }
    function trainTick() {
      const lr = +slLr.value;
      const stepsPerEpoch = Math.ceil(train.length / BATCH);
      for (let s = 0; s < STEPS_PER_FRAME; s++) {
        if (stepInEpoch === 0) { shuffleOrder(); epochLoss = 0; }
        const batch = order.slice(stepInEpoch * BATCH, (stepInEpoch + 1) * BATCH).map((k) => train[k]);
        epochLoss += T.sgdStep(model, batch, lr);
        stepInEpoch++;
        if (stepInEpoch >= stepsPerEpoch) {
          stepInEpoch = 0; epoch++;
          hist.loss.push(epochLoss / stepsPerEpoch);
          hist.acc.push(T.accuracy(model, test));
          if (epoch >= MAX_EPOCHS) { stopTraining(); break; }
        }
      }
      recompute();
      redraw();
    }
    function startTraining() {
      if (training) return;
      training = true;
      DL.startTicker(trainTick);
      $('s6-train').textContent = '⏸ Pausar treino';
    }
    function stopTraining() {
      if (!training) return;
      training = false;
      DL.stopTicker(trainTick);
      $('s6-train').textContent = '▶ Treinar';
    }
    $('s6-train').addEventListener('click', () => { if (training) stopTraining(); else startTraining(); });
    $('s6-reset').addEventListener('click', () => {
      stopTraining();
      seedW++;
      model = T.initModel(DL.utils.mulberry32(seedW));
      epoch = 0; stepInEpoch = 0; hist.loss.length = 0; hist.acc.length = 0;
      recompute(); redraw();
    });
    slLr.addEventListener('input', () => { $('s6-lr-val').textContent = (+slLr.value).toFixed(2); updateReadouts(); });

    /* ── Início ─────────────────────────────────────────────────────── */
    newImage();
    $('s6-lr-val').textContent = (+slLr.value).toFixed(2);
    redraw();
    P.onRedraw(redraw);
    P.observeResize(cv1, redraw);
    P.observeResize(cv2, redraw);
    P.observeResize(cv3, redraw);
    P.observeResize(cvLoss, redraw);
  }

  DL.sections.push({ name: 's6-cnn-completa', init });
})();
