/* Seção 1: pipeline QKV passo a passo numa frase curta
   (q_t = x_t W_q, k_t = x_t W_k, v_t = x_t W_v; score = q·k/√d; softmax; soma ponderada). */
(function () {
  'use strict';
  window.DL = window.DL || {};
  DL.sections = DL.sections || [];

  const TOKENS = ['o', 'gato', 'caça', 'o', 'rato'];
  const VOCAB = 4;                       /* "o" repete: mesmo embedding */
  const TOK2VOC = [0, 1, 2, 0, 3];
  const D_EMB = 3, D = 2;
  const SQD = Math.sqrt(D);
  const STAGE_DESC = [
    'etapa 0/4: embeddings x_i de cada token',
    'etapa 1/4: projeções q = xW_q, k = xW_k, v = xW_v',
    'etapa 2/4: scores q_t·k_i/√d contra todas as keys',
    'etapa 3/4: pesos a_i = softmax(scores)',
    'etapa 4/4: saída c = Σ a_i·v_i (soma ponderada dos values)',
  ];

  function init() {
    const P = DL.plot, U = DL.utils;
    const $ = (id) => document.getElementById(id);

    const cvPipe = $('s1-pipeline'), cvAttn = $('s1-attn');
    const tokBtns = TOKENS.map((_, i) => $('s1-tok-' + i));
    const fmt = (v, d) => v.toFixed(d == null ? 2 : d).replace('.', ',').replace('-', '−');

    let seed = 11;
    let t = 1;          /* token de consulta */
    let stage = 0;      /* 0..4 */
    let X, Wq, Wk, Wv, Q, K, V;

    /* ── modelo ─────────────────────────────────────────── */
    function genMat(rng, rows, cols, scale) {
      const m = [];
      for (let r = 0; r < rows; r++) {
        const row = [];
        for (let c = 0; c < cols; c++) row.push(Math.round((rng() * 2 - 1) * scale * 10) / 10);
        m.push(row);
      }
      return m;
    }

    function matmul(A, B) {
      return A.map((row) => B[0].map((_, j) => row.reduce((s, v, k) => s + v * B[k][j], 0)));
    }

    function softmax(s) {
      const mx = Math.max(...s);
      const e = s.map((v) => Math.exp(v - mx));
      const z = e.reduce((a, b) => a + b, 0);
      return e.map((v) => v / z);
    }

    function regen() {
      const rng = U.mulberry32(seed);
      const E = genMat(rng, VOCAB, D_EMB, 2);
      X = TOK2VOC.map((v) => E[v]);
      Wq = genMat(rng, D_EMB, D, 1);
      Wk = genMat(rng, D_EMB, D, 1);
      Wv = genMat(rng, D_EMB, D, 1);
      Q = matmul(X, Wq);
      K = matmul(X, Wk);
      V = matmul(X, Wv);
    }

    function compute() {
      const q = Q[t];
      const scores = K.map((k) => (q[0] * k[0] + q[1] * k[1]) / SQD);
      const a = softmax(scores);
      const c = [0, 1].map((d) => V.reduce((s, v, i) => s + a[i] * v[d], 0));
      return { scores, a, c };
    }

    const maxAbs = (rows) => Math.max(1e-9, ...rows.map((r) => Math.max(...r.map(Math.abs))));

    /* ── primitivas de desenho ──────────────────────────── */
    function barGroup(ctx, th, cx, cy, hh, vals, scale, color, bw, gap) {
      const total = vals.length * bw + (vals.length - 1) * gap;
      ctx.strokeStyle = th.line;
      ctx.lineWidth = 1;
      ctx.beginPath();
      ctx.moveTo(cx - total / 2 - 3, cy);
      ctx.lineTo(cx + total / 2 + 3, cy);
      ctx.stroke();
      ctx.fillStyle = color;
      let x = cx - total / 2;
      for (const v of vals) {
        const bh = (v / scale) * hh;
        ctx.fillRect(x, Math.min(cy, cy - bh), bw, Math.abs(bh));
        x += bw + gap;
      }
    }

    function drawPipeline() {
      const th = P.theme();
      const { ctx, w, h } = P.setup(cvPipe);
      P.clear(ctx, w, h);
      const left = 52;
      const cw = (w - left - 12) / TOKENS.length;
      const rows = [
        { label: 'x_i', data: X, color: th.fg, cy: 84, hh: 36, minStage: 0 },
        { label: 'q_i', data: Q, color: th.pink, cy: 156, hh: 24, minStage: 1 },
        { label: 'k_i', data: K, color: th.orange, cy: 214, hh: 24, minStage: 1 },
        { label: 'v_i', data: V, color: th.cyan, cy: 272, hh: 24, minStage: 1 },
      ];

      /* coluna do token de consulta destacada */
      const pc = P.rgb(th.pink);
      ctx.fillStyle = 'rgba(' + pc[0] + ',' + pc[1] + ',' + pc[2] + ',0.10)';
      ctx.fillRect(left + t * cw + 2, 6, cw - 4, h - 24);

      ctx.font = '12px sans-serif';
      ctx.textAlign = 'center';
      TOKENS.forEach((tok, i) => {
        ctx.fillStyle = i === t ? th.pink : th.fg;
        ctx.fillText(tok, left + (i + 0.5) * cw, 24);
      });
      P.mathText(ctx, 'query (t)', left + (t + 0.5) * cw, 38, th.pink, 'center', 10);

      for (const row of rows) {
        if (stage < row.minStage) {
          P.label(ctx, 8, row.cy, row.label + ' ?', th.comment);
          continue;
        }
        P.label(ctx, 8, row.cy, row.label, row.color);
        const sc = maxAbs(row.data);
        row.data.forEach((vals, i) => {
          barGroup(ctx, th, left + (i + 0.5) * cw, row.cy, row.hh, vals, sc, row.color, 9, 4);
        });
      }
      if (stage >= 1) {
        P.mathText(ctx, 'q = xW_q   k = xW_k   v = xW_v   (d = 2)',
          left + (w - left) / 2, h - 8, th.comment, 'center', 11);
      } else {
        P.mathText(ctx, 'use ⏭ para projetar os embeddings em q, k, v',
          left + (w - left) / 2, h - 8, th.comment, 'center', 11);
      }
    }

    function drawAttn() {
      const th = P.theme();
      const { ctx, w, h } = P.setup(cvAttn);
      P.clear(ctx, w, h);
      const { scores, a, c } = compute();
      const left = 14;
      const cw = (w - left - 10) / TOKENS.length;

      if (stage < 2) {
        P.mathText(ctx, 'avance com ⏭ até os scores', w / 2, h / 2, th.comment, 'center', 12);
        return;
      }

      /* scores */
      P.mathText(ctx, 'scores  q_t·k_i/√d', left, 20, th.orange, 'left', 11);
      const sScale = Math.max(1e-9, ...scores.map(Math.abs));
      scores.forEach((s, i) => {
        barGroup(ctx, th, left + (i + 0.5) * cw, 72, 34, [s], sScale, th.orange, 16, 0);
        ctx.fillStyle = th.comment;
        ctx.font = '10px sans-serif';
        ctx.textAlign = 'center';
        ctx.fillText(fmt(s, 1), left + (i + 0.5) * cw, 122);
        ctx.fillText(TOKENS[i], left + (i + 0.5) * cw, 134);
      });

      /* pesos da softmax */
      if (stage >= 3) {
        P.mathText(ctx, 'pesos  a_i = softmax(scores)', left, 158, th.green, 'left', 11);
        const base = 222;
        ctx.fillStyle = th.green;
        a.forEach((v, i) => {
          const bh = v * 50;
          ctx.fillRect(left + (i + 0.5) * cw - 8, base - bh, 16, bh);
        });
        ctx.fillStyle = th.comment;
        ctx.font = '10px sans-serif';
        ctx.textAlign = 'center';
        a.forEach((v, i) => ctx.fillText(fmt(v), left + (i + 0.5) * cw, base + 12));
        ctx.strokeStyle = th.line;
        ctx.beginPath();
        ctx.moveTo(left, base);
        ctx.lineTo(left + TOKENS.length * cw, base);
        ctx.stroke();
      } else {
        P.mathText(ctx, 'pesos: avance com ⏭', left, 158, th.comment, 'left', 11);
      }

      /* saída c */
      if (stage >= 4) {
        P.mathText(ctx, 'saída  c = Σ a_i·v_i', left, 256, th.purple, 'left', 11);
        const cy = 296, sc = Math.max(maxAbs(V), maxAbs([c]));
        barGroup(ctx, th, left + 60, cy, 22, c, sc, th.purple, 14, 6);
        P.mathText(ctx, 'c ≈ (' + fmt(c[0]) + '; ' + fmt(c[1]) + ')',
          left + 110, cy + 4, th.purple, 'left', 11);
      } else if (stage >= 3) {
        P.mathText(ctx, 'saída: avance com ⏭', left, 256, th.comment, 'left', 11);
      }
    }

    function updateReadout() {
      $('s1-readout').textContent = STAGE_DESC[stage] + ' · t = ' + (t + 1) + ' ("' + TOKENS[t] + '")';
    }

    function redraw() {
      drawPipeline();
      drawAttn();
      updateReadout();
    }

    /* ── interação ──────────────────────────────────────── */
    tokBtns.forEach((btn, i) => {
      btn.addEventListener('click', () => {
        t = i;
        tokBtns.forEach((b, j) => b.classList.toggle('active', j === i));
        redraw();
      });
    });
    $('s1-step').addEventListener('click', () => {
      stage = Math.min(4, stage + 1);
      redraw();
    });
    $('s1-reset').addEventListener('click', () => {
      stage = 0;
      redraw();
    });
    $('s1-rand').addEventListener('click', () => {
      seed = (seed * 1103 + 17) % 100000;
      regen();
      redraw();
    });

    regen();
    redraw();
    P.onRedraw(redraw);
    P.observeResize(cvPipe, redraw);
    P.observeResize(cvAttn, redraw);
  }

  DL.sections.push({ name: 's1-qkv', init });
})();
