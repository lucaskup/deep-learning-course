/* Seção 1: atenção bidirecional (encoder) vs causal (decoder) na mesma frase,
   com máscara de [PAD] compartilhada entre as duas famílias. */
(function () {
  'use strict';
  window.DL = window.DL || {};
  DL.sections = DL.sections || [];

  const BASE = ['[CLS]', 'o', 'gato', 'sentou', 'no', 'tapete', '[SEP]'];
  const D = 8; /* dimensão dos vetores de brinquedo */

  function init() {
    const P = DL.plot;
    const U = DL.utils;
    const $ = (id) => document.getElementById(id);

    const cvEnc = $('s1-enc'), cvDec = $('s1-dec');
    const slPad = $('s1-pad');

    let seed = 7;
    let nPad = +slPad.value;
    let sel = 2; /* linha (consulta) selecionada */
    let enc = null, dec = null, toks = [];

    /* Hash determinístico para semear o embedding de cada token. */
    function hash(str) {
      let h = 2166136261;
      for (let i = 0; i < str.length; i++) {
        h ^= str.charCodeAt(i);
        h = Math.imul(h, 16777619);
      }
      return h >>> 0;
    }

    function randVec(rng, n) {
      const v = new Array(n);
      for (let i = 0; i < n; i++) v[i] = U.randn(rng);
      return v;
    }

    function matVec(W, e) {
      const out = new Array(W.length);
      for (let r = 0; r < W.length; r++) {
        let s = 0;
        for (let k = 0; k < e.length; k++) s += W[r][k] * e[k];
        out[r] = s;
      }
      return out;
    }

    /* Calcula a matriz de atenção softmax(QKᵀ/√D) com a máscara pedida.
       Devolve { A (pesos), M (true = mascarada) }. */
    function attention(causal) {
      const L = toks.length;
      const rgW = U.mulberry32(seed);
      const Wq = [], Wk = [];
      for (let r = 0; r < D; r++) { Wq.push(randVec(rgW, D)); Wk.push(randVec(rgW, D)); }

      const embs = toks.map((t, i) => {
        const re = U.mulberry32(hash(t) ^ 0x9e3779b9);
        const rp = U.mulberry32(1000 + i);
        const e = randVec(re, D), p = randVec(rp, D);
        return e.map((v, k) => v + 0.5 * p[k]);
      });
      const Q = embs.map((e) => matVec(Wq, e));
      const K = embs.map((e) => matVec(Wk, e));

      const A = [], M = [];
      for (let i = 0; i < L; i++) {
        const scores = new Array(L), mrow = new Array(L);
        let mx = -Infinity;
        for (let j = 0; j < L; j++) {
          const masked = toks[j] === '[PAD]' || (causal && j > i);
          mrow[j] = masked;
          if (masked) { scores[j] = -Infinity; continue; }
          let s = 0;
          for (let k = 0; k < D; k++) s += Q[i][k] * K[j][k];
          scores[j] = s / Math.sqrt(D);
          if (scores[j] > mx) mx = scores[j];
        }
        let z = 0;
        const arow = scores.map((s) => {
          const e = isFinite(s) ? Math.exp(s - mx) : 0;
          z += e;
          return e;
        });
        A.push(arow.map((e) => (z > 0 ? e / z : 0)));
        M.push(mrow);
      }
      return { A, M };
    }

    function recompute() {
      toks = BASE.slice();
      for (let i = 0; i < nPad; i++) toks.push('[PAD]');
      if (sel >= toks.length) sel = toks.length - 1;
      enc = attention(false);
      dec = attention(true);
    }

    /* Geometria da grade dentro de um canvas (compartilhada com o hit-test). */
    function grid(w, h) {
      const L = toks.length;
      const ml = 52, mt = 8, mb = 46, mr = 8;
      const cell = Math.max(8, Math.min((w - ml - mr) / L, (h - mt - mb) / L));
      const x0 = ml + (w - ml - mr - cell * L) / 2;
      return { L, cell, x0, y0: mt };
    }

    function drawMatrix(canvas, att, baseColor, title) {
      const th = P.theme();
      const { ctx, w, h } = P.setup(canvas);
      P.clear(ctx, w, h);
      const g = grid(w, h);
      const bc = P.rgb(baseColor);
      const rc = P.rgb(th.red);

      for (let i = 0; i < g.L; i++) {
        for (let j = 0; j < g.L; j++) {
          const x = g.x0 + j * g.cell, y = g.y0 + i * g.cell;
          if (att.M[i][j]) {
            ctx.fillStyle = 'rgba(' + rc[0] + ',' + rc[1] + ',' + rc[2] + ',0.10)';
            ctx.fillRect(x, y, g.cell - 1, g.cell - 1);
          } else {
            const a = 0.06 + 0.9 * att.A[i][j];
            ctx.fillStyle = 'rgba(' + bc[0] + ',' + bc[1] + ',' + bc[2] + ',' + a.toFixed(3) + ')';
            ctx.fillRect(x, y, g.cell - 1, g.cell - 1);
          }
        }
      }

      /* Valores numéricos na linha selecionada (quando a célula comporta). */
      ctx.strokeStyle = th.orange;
      ctx.lineWidth = 2;
      ctx.strokeRect(g.x0 - 1, g.y0 + sel * g.cell - 1, g.cell * g.L + 1, g.cell + 1);
      if (g.cell >= 24) {
        ctx.font = '9px sans-serif';
        ctx.textAlign = 'center';
        for (let j = 0; j < g.L; j++) {
          if (att.M[sel][j]) continue;
          const v = att.A[sel][j];
          ctx.fillStyle = v > 0.35 ? th.fg : th.comment;
          ctx.fillText(v.toFixed(2), g.x0 + j * g.cell + g.cell / 2, g.y0 + sel * g.cell + g.cell / 2 + 3);
        }
      }

      /* Rótulos: linhas (consultas) à esquerda, colunas (chaves) embaixo. */
      ctx.font = '10px sans-serif';
      for (let i = 0; i < g.L; i++) {
        ctx.fillStyle = i === sel ? th.orange : th.comment;
        ctx.textAlign = 'right';
        ctx.fillText(toks[i], g.x0 - 5, g.y0 + i * g.cell + g.cell / 2 + 3);
      }
      for (let j = 0; j < g.L; j++) {
        ctx.save();
        ctx.translate(g.x0 + j * g.cell + g.cell / 2, g.y0 + g.cell * g.L + 10);
        ctx.rotate(-Math.PI / 4);
        ctx.fillStyle = th.comment;
        ctx.textAlign = 'right';
        ctx.fillText(toks[j], 0, 6);
        ctx.restore();
      }
      ctx.fillStyle = th.comment;
      ctx.textAlign = 'left';
      ctx.font = '10px sans-serif';
      ctx.fillText(title, g.x0, h - 4);
    }

    function updateReadout() {
      const vis = (att) => att.M[sel].filter((m) => !m).length;
      $('s1-readout').textContent =
        "consulta '" + toks[sel] + "': encoder vê " + vis(enc) +
        ' tokens · decoder vê ' + vis(dec);
    }

    function redraw() {
      const th = P.theme();
      drawMatrix(cvEnc, enc, th.green, 'softmax(QKᵀ/√d), sem máscara causal');
      drawMatrix(cvDec, dec, th.cyan, 'softmax(QKᵀ/√d), máscara causal');
      updateReadout();
    }

    function pickRow(canvas, e) {
      const r = canvas.getBoundingClientRect();
      const g = grid(r.width, r.height);
      const i = Math.floor((e.clientY - r.top - g.y0) / g.cell);
      if (i >= 0 && i < g.L) {
        sel = i;
        redraw();
      }
    }

    cvEnc.addEventListener('pointerdown', (e) => pickRow(cvEnc, e));
    cvDec.addEventListener('pointerdown', (e) => pickRow(cvDec, e));

    slPad.addEventListener('input', () => {
      nPad = +slPad.value;
      $('s1-pad-val').textContent = nPad;
      recompute();
      redraw();
    });
    $('s1-seed').addEventListener('click', () => {
      seed = (seed + 1) >>> 0;
      recompute();
      redraw();
    });

    recompute();
    redraw();
    P.onRedraw(redraw);
    P.observeResize(cvEnc, redraw);
    P.observeResize(cvDec, redraw);
  }

  DL.sections.push({ name: 's1-attention', init });
})();
