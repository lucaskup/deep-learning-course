/* Seção 2: a matriz de atenção softmax(QKᵀ/√d) como heatmap interativo.
   Embeddings com features interpretáveis (artigo, substantivo, verbo, animado, conector)
   e W_q/W_k "treinados" à mão para imitar uma cabeça de correferência. */
(function () {
  'use strict';
  window.DL = window.DL || {};
  DL.sections = DL.sections || [];

  const TOKENS = ['o', 'animal', 'não', 'cruzou', 'a', 'rua', 'porque', 'ele', 'estava', 'cansado'];
  /* features: [artigo, substantivo, verbo, animado, conector/advérbio] */
  const FEAT = [
    [1, 0, 0, 0, 0],      /* o       */
    [0, 1, 0, 1, 0],      /* animal  */
    [0, 0, 0, 0, 1],      /* não     */
    [0, 0, 1, 0.4, 0],    /* cruzou  */
    [1, 0, 0, 0, 0],      /* a       */
    [0, 1, 0, 0, 0],      /* rua     */
    [0, 0, 0, 0, 1],      /* porque  */
    [0.6, 0, 0, 1, 0],    /* ele     */
    [0, 0, 1, 0, 0],      /* estava  */
    [0, 0, 0.4, 0.8, 0],  /* cansado */
  ];
  /* linhas por feature, colunas no espaço d = 3 de query/key */
  const WQ_TRAINED = [
    [0, 1.36, 0],     /* artigos procuram substantivos */
    [0, 0, 1.36],     /* substantivos procuram verbos */
    [0.85, 1.36, 0],  /* verbos procuram o sujeito */
    [2.04, 0.68, 0],  /* referenciais procuram animados */
    [0, 0, 1.53],     /* conectores procuram verbos */
  ];
  const WK_TRAINED = [
    [0, 0.17, 0],
    [0.34, 1.7, 0],   /* substantivos se oferecem como substantivos */
    [0, 0, 1.7],      /* verbos se oferecem como verbos */
    [2.04, 0.34, 0],  /* animados se oferecem como animados */
    [0, 0, 0.34],
  ];
  const N = TOKENS.length, D = 3, SQD = Math.sqrt(D);

  function init() {
    const P = DL.plot, U = DL.utils;
    const $ = (id) => document.getElementById(id);

    const cvMat = $('s2-matrix'), cvRow = $('s2-row');
    const tabTrained = $('s2-tab-trained'), tabRandom = $('s2-tab-random');
    const fmt = (v) => v.toFixed(2).replace('.', ',');

    let mode = 'trained';
    let seed = 7;
    let selRow = 7;                /* "ele" */
    let hover = null;              /* [linha, coluna] sob o cursor */
    let A = null;                  /* matriz de atenção N×N */
    let geo = null;                /* geometria do heatmap para o mouse */

    /* ── modelo ─────────────────────────────────────────── */
    function matmul(Aa, B) {
      return Aa.map((row) => B[0].map((_, j) => row.reduce((s, v, k) => s + v * B[k][j], 0)));
    }

    function softmaxRow(s) {
      const mx = Math.max(...s);
      const e = s.map((v) => Math.exp(v - mx));
      const z = e.reduce((a, b) => a + b, 0);
      return e.map((v) => v / z);
    }

    function randomW(rng) {
      return FEAT[0].map(() => [0, 0, 0].map(() => U.randn(rng) * 0.6));
    }

    function recompute() {
      let Wq = WQ_TRAINED, Wk = WK_TRAINED;
      if (mode === 'random') {
        const rng = U.mulberry32(seed);
        Wq = randomW(rng);
        Wk = randomW(rng);
      }
      const Q = matmul(FEAT, Wq);
      const K = matmul(FEAT, Wk);
      A = Q.map((q) => softmaxRow(K.map((k) =>
        (q[0] * k[0] + q[1] * k[1] + q[2] * k[2]) / SQD)));
    }

    /* ── heatmap ────────────────────────────────────────── */
    function drawMatrix() {
      const th = P.theme();
      const { ctx, w, h } = P.setup(cvMat);
      P.clear(ctx, w, h);
      const x0 = 66, y0 = 14;
      const cs = Math.max(8, Math.min((w - x0 - 10) / N, (h - y0 - 62) / N));
      geo = { x0, y0, cs };
      const gc = P.rgb(th.green);

      for (let i = 0; i < N; i++) {
        for (let j = 0; j < N; j++) {
          const v = A[i][j];
          ctx.fillStyle = 'rgba(' + gc[0] + ',' + gc[1] + ',' + gc[2] + ',' + Math.pow(v, 0.7).toFixed(3) + ')';
          ctx.fillRect(x0 + j * cs, y0 + i * cs, cs - 1, cs - 1);
        }
      }

      /* rótulos: linhas (queries) à esquerda, colunas (keys) embaixo */
      ctx.font = '11px sans-serif';
      for (let i = 0; i < N; i++) {
        ctx.fillStyle = i === selRow ? th.pink : th.comment;
        ctx.textAlign = 'right';
        ctx.fillText(TOKENS[i], x0 - 6, y0 + (i + 0.5) * cs + 4);
      }
      for (let j = 0; j < N; j++) {
        ctx.save();
        ctx.translate(x0 + (j + 0.55) * cs, y0 + N * cs + 8);
        ctx.rotate(Math.PI / 3);
        ctx.fillStyle = th.comment;
        ctx.textAlign = 'left';
        ctx.fillText(TOKENS[j], 0, 0);
        ctx.restore();
      }
      P.mathText(ctx, 'queries (t)', 12, y0 + 10, th.comment, 'left', 10);

      /* linha selecionada e célula sob o cursor */
      ctx.strokeStyle = th.pink;
      ctx.lineWidth = 1.6;
      ctx.strokeRect(x0 - 1, y0 + selRow * cs - 1, N * cs + 1, cs + 1);
      if (hover) {
        ctx.strokeStyle = th.fg;
        ctx.lineWidth = 1.6;
        ctx.strokeRect(x0 + hover[1] * cs - 1, y0 + hover[0] * cs - 1, cs + 1, cs + 1);
      }
    }

    function drawRow() {
      const th = P.theme();
      const { ctx, w, h } = P.setup(cvRow);
      P.clear(ctx, w, h);
      const a = A[selRow];
      const left = 34, base = h - 56, top = 34;
      const cw = (w - left - 10) / N;

      P.mathText(ctx, 'a^{(t)} para t = "' + TOKENS[selRow] + '"', left, 18, th.pink, 'left', 12);
      ctx.strokeStyle = th.line;
      ctx.lineWidth = 1;
      ctx.beginPath();
      ctx.moveTo(left, base);
      ctx.lineTo(left + N * cw, base);
      ctx.stroke();

      ctx.fillStyle = th.green;
      a.forEach((v, j) => {
        const bh = v * (base - top);
        ctx.fillRect(left + (j + 0.5) * cw - cw * 0.3, base - bh, cw * 0.6, bh);
      });
      ctx.font = '10px sans-serif';
      a.forEach((v, j) => {
        if (v >= 0.05) {
          ctx.fillStyle = th.fg;
          ctx.textAlign = 'center';
          ctx.fillText(fmt(v), left + (j + 0.5) * cw, base - v * (base - top) - 4);
        }
        ctx.save();
        ctx.translate(left + (j + 0.6) * cw, base + 8);
        ctx.rotate(Math.PI / 3);
        ctx.fillStyle = j === selRow ? th.pink : th.comment;
        ctx.textAlign = 'left';
        ctx.fillText(TOKENS[j], 0, 0);
        ctx.restore();
      });
    }

    function defaultReadout() {
      $('s2-readout').textContent = mode === 'trained'
        ? 'pesos treinados · passe o mouse sobre a matriz'
        : 'pesos aleatórios (seed ' + seed + ') · passe o mouse sobre a matriz';
    }

    function redraw() {
      drawMatrix();
      drawRow();
    }

    /* ── interação ──────────────────────────────────────── */
    function cellAt(e) {
      if (!geo) return null;
      const r = cvMat.getBoundingClientRect();
      const i = Math.floor((e.clientY - r.top - geo.y0) / geo.cs);
      const j = Math.floor((e.clientX - r.left - geo.x0) / geo.cs);
      return (i >= 0 && i < N && j >= 0 && j < N) ? [i, j] : null;
    }

    cvMat.addEventListener('pointermove', (e) => {
      const c = cellAt(e);
      hover = c;
      if (c) {
        $('s2-readout').textContent =
          'a("' + TOKENS[c[0]] + '" → "' + TOKENS[c[1]] + '") = ' + fmt(A[c[0]][c[1]]) +
          ' · a linha "' + TOKENS[c[0]] + '" soma 1';
      } else {
        defaultReadout();
      }
      drawMatrix();
    });
    cvMat.addEventListener('pointerleave', () => {
      hover = null;
      defaultReadout();
      drawMatrix();
    });
    cvMat.addEventListener('pointerdown', (e) => {
      const c = cellAt(e);
      if (!c) return;
      selRow = c[0];
      redraw();
    });

    function setMode(m) {
      mode = m;
      tabTrained.classList.toggle('active', m === 'trained');
      tabRandom.classList.toggle('active', m === 'random');
      recompute();
      defaultReadout();
      redraw();
    }
    tabTrained.addEventListener('click', () => setMode('trained'));
    tabRandom.addEventListener('click', () => setMode('random'));
    $('s2-rand').addEventListener('click', () => {
      seed = (seed * 31 + 5) % 100000;
      setMode('random');
    });

    recompute();
    defaultReadout();
    redraw();
    P.onRedraw(redraw);
    P.observeResize(cvMat, redraw);
    P.observeResize(cvRow, redraw);
  }

  DL.sections.push({ name: 's2-attention-matrix', init });
})();
