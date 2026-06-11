/* Seção 1: self-attention é invariante a permutação; somar PE(i) quebra a simetria.
   Frase e embeddings d=4 idênticos ao cálculo manual dos slides. */
(function () {
  'use strict';
  window.DL = window.DL || {};
  DL.sections = DL.sections || [];

  const WORDS = ['O', 'gato', 'sentou', 'no', 'tapete'];
  const EMB = [
    [0.5, -0.2, 0.1, 0.8],
    [0.3, 0.6, -0.4, 0.2],
    [-0.1, 0.4, 0.7, -0.3],
    [0.6, -0.5, 0.2, 0.1],
    [-0.2, 0.3, -0.6, 0.5],
  ];
  const N = 5, D = 4;
  /* Frequências do exemplo dos slides: ω₀ = 1, ω₁ = 0.01 (d = 4). */
  const OMEGA = [1.0, 0.01];

  function peVec(i) {
    return [
      Math.sin(i * OMEGA[0]), Math.cos(i * OMEGA[0]),
      Math.sin(i * OMEGA[1]), Math.cos(i * OMEGA[1]),
    ];
  }

  /* Self-attention com q = k = v = x (uma camada, sem projeções). */
  function attention(perm, usePE) {
    const x = [];
    for (let p = 0; p < N; p++) {
      const e = EMB[perm[p]], pe = peVec(p);
      x.push(e.map((v, d) => v + (usePE ? pe[d] : 0)));
    }
    const A = [], out = [];
    for (let p = 0; p < N; p++) {
      const row = [];
      let mx = -Infinity;
      for (let q = 0; q < N; q++) {
        let s = 0;
        for (let d = 0; d < D; d++) s += x[p][d] * x[q][d];
        s /= Math.sqrt(D);
        row.push(s);
        if (s > mx) mx = s;
      }
      let Z = 0;
      for (let q = 0; q < N; q++) { row[q] = Math.exp(row[q] - mx); Z += row[q]; }
      for (let q = 0; q < N; q++) row[q] /= Z;
      A.push(row);
      const o = [0, 0, 0, 0];
      for (let q = 0; q < N; q++) {
        for (let d = 0; d < D; d++) o[d] += row[q] * x[q][d];
      }
      out.push(o);
    }
    return { A, out };
  }

  function init() {
    const P = DL.plot;
    const $ = (id) => document.getElementById(id);

    const cvAttn = $('s1-attn'), cvOut = $('s1-out');
    const tabNoPE = $('s1-tab-nope'), tabPE = $('s1-tab-pe');
    const btnShuffle = $('s1-shuffle'), btnReset = $('s1-reset');

    const rng = DL.utils.mulberry32(7);
    let perm = [0, 1, 2, 3, 4];
    let usePE = false;

    function shuffle() {
      let next;
      do {
        next = perm.slice();
        for (let a = N - 1; a > 0; a--) {
          const b = Math.floor(rng() * (a + 1));
          const t = next[a]; next[a] = next[b]; next[b] = t;
        }
      } while (next.every((w, p) => w === perm[p]));
      perm = next;
    }

    function drawAttn(A) {
      const th = P.theme();
      const { ctx, w, h } = P.setup(cvAttn);
      P.clear(ctx, w, h);
      const padL = 58, padT = 10, padB = 44, padR = 8;
      const cell = Math.max(8, Math.min((w - padL - padR) / N, (h - padT - padB) / N));
      const x0 = padL, y0 = padT;
      const pc = P.rgb(th.purple);
      ctx.font = '10px sans-serif';
      for (let i = 0; i < N; i++) {
        for (let j = 0; j < N; j++) {
          const a = A[i][j];
          ctx.fillStyle = 'rgba(' + pc[0] + ',' + pc[1] + ',' + pc[2] + ',' + (0.08 + 0.85 * a).toFixed(3) + ')';
          ctx.fillRect(x0 + j * cell, y0 + i * cell, cell - 1, cell - 1);
          if (cell > 26) {
            ctx.fillStyle = th.fg;
            ctx.textAlign = 'center';
            ctx.fillText(a.toFixed(2), x0 + (j + 0.5) * cell, y0 + (i + 0.5) * cell + 3);
          }
        }
      }
      /* Rótulos: linhas = query i, colunas = key j, na ordem atual da frase. */
      ctx.fillStyle = th.comment;
      ctx.font = '11px sans-serif';
      for (let i = 0; i < N; i++) {
        ctx.textAlign = 'right';
        ctx.fillText(WORDS[perm[i]], x0 - 5, y0 + (i + 0.5) * cell + 4);
      }
      for (let j = 0; j < N; j++) {
        ctx.save();
        ctx.translate(x0 + (j + 0.5) * cell, y0 + N * cell + 6);
        ctx.rotate(-Math.PI / 5);
        ctx.textAlign = 'right';
        ctx.fillText(WORDS[perm[j]], 0, 8);
        ctx.restore();
      }
      P.label(ctx, x0, y0 + N * cell + 34, 'linha i (query) · coluna j (key)', th.comment);
    }

    function drawOut(out, base) {
      const th = P.theme();
      const { ctx, w, h } = P.setup(cvOut);
      P.clear(ctx, w, h);
      const fr = P.frame(ctx, w, h, 0, N, -2, 2.4);
      P.axes(ctx, fr, { ylabel: 'saída', yticks: [-2, -1, 0, 1, 2] });
      const colors = [th.cyan, th.orange, th.green, th.purple];
      /* Linha do zero */
      P.line(ctx, fr, [0, N], [0, 0], th.line, { width: 1, dash: [4, 4] });
      const bw = 0.16;
      for (let p = 0; p < N; p++) {
        const word = perm[p];
        for (let d = 0; d < D; d++) {
          const cx = p + 0.16 + d * 0.18;
          const v = out[p][d];
          ctx.fillStyle = colors[d];
          ctx.globalAlpha = 0.8;
          const yv = fr.Y(v), y0 = fr.Y(0);
          ctx.fillRect(fr.X(cx), Math.min(yv, y0), fr.X(cx + bw) - fr.X(cx), Math.abs(y0 - yv));
          ctx.globalAlpha = 1;
          /* Traço: saída da MESMA palavra na ordem original. */
          const bv = base[word][d];
          ctx.strokeStyle = th.fg;
          ctx.lineWidth = 2;
          ctx.beginPath();
          ctx.moveTo(fr.X(cx) - 2, fr.Y(bv));
          ctx.lineTo(fr.X(cx + bw) + 2, fr.Y(bv));
          ctx.stroke();
        }
        ctx.fillStyle = th.comment;
        ctx.font = '11px sans-serif';
        ctx.textAlign = 'center';
        ctx.fillText(WORDS[word], fr.X(p + 0.5), fr.Y(fr.ymin) + 14);
      }
      for (let d = 0; d < D; d++) {
        P.label(ctx, fr.X(0) + 8 + d * 52, fr.Y(fr.ymax) + 12, 'dim ' + d, colors[d]);
      }
    }

    function redraw() {
      const base = attention([0, 1, 2, 3, 4], usePE);
      const cur = attention(perm, usePE);
      drawAttn(cur.A);
      drawOut(cur.out, base.out);
      let delta = 0;
      for (let p = 0; p < N; p++) {
        for (let d = 0; d < D; d++) {
          delta = Math.max(delta, Math.abs(cur.out[p][d] - base.out[perm[p]][d]));
        }
      }
      const sentence = perm.map((wd) => WORDS[wd]).join(' ');
      $('s1-readout').textContent =
        '“' + sentence + '” · Δ máx da saída por palavra = ' + delta.toFixed(3);
    }

    function setMode(pe) {
      usePE = pe;
      tabNoPE.classList.toggle('active', !pe);
      tabPE.classList.toggle('active', pe);
      redraw();
    }

    tabNoPE.addEventListener('click', () => setMode(false));
    tabPE.addEventListener('click', () => setMode(true));
    btnShuffle.addEventListener('click', () => { shuffle(); redraw(); });
    btnReset.addEventListener('click', () => { perm = [0, 1, 2, 3, 4]; redraw(); });

    redraw();
    P.onRedraw(redraw);
    P.observeResize(cvAttn, redraw);
    P.observeResize(cvOut, redraw);
  }

  DL.sections.push({ name: 's1-permutation', init });
})();
