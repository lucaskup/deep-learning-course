/* Transformer encoder-decoder de brinquedo: o mesmo modelo do exemplo completo
   dos slides (d = 4, uma cabeça, um bloco de cada lado), com os pesos de
   weights.js. Tudo é recalculado no navegador, então qualquer frase montada
   com o vocabulário pode ser acompanhada passo a passo. */
(function () {
  'use strict';
  window.DL = window.DL || {};

  const data = DL.toyWeights;
  const W = data.weights;
  const D = 4;
  const SQD = Math.sqrt(D);
  const BOS = data.tgtVocab.indexOf('<bos>');
  const EOS = data.tgtVocab.indexOf('<eos>');

  /* ── álgebra ─────────────────────────────────────────── */
  function matmul(A, B) {
    return A.map((row) => B[0].map((_, j) => row.reduce((s, v, k) => s + v * B[k][j], 0)));
  }
  function add(A, B) { return A.map((row, i) => row.map((v, j) => v + B[i][j])); }
  function addBias(A, b) { return A.map((row) => row.map((v, j) => v + b[j])); }
  function relu(A) { return A.map((row) => row.map((v) => Math.max(0, v))); }
  function softmaxRows(S) {
    return S.map((row) => {
      const mx = Math.max(...row);
      const e = row.map((v) => Math.exp(v - mx));
      const z = e.reduce((a, b) => a + b, 0);
      return e.map((v) => v / z);
    });
  }
  function layerNorm(A) {
    const mu = [], sigma = [];
    const out = A.map((row) => {
      const m = row.reduce((a, b) => a + b, 0) / row.length;
      const s = Math.sqrt(row.reduce((a, v) => a + (v - m) * (v - m), 0) / row.length);
      mu.push(m); sigma.push(s);
      return row.map((v) => (v - m) / s);
    });
    return { out, mu, sigma };
  }
  /* Positional encoding sinusoidal com n = 10000. */
  function posEnc(m) {
    const P = [];
    for (let i = 0; i < m; i++) {
      const row = [];
      for (let j = 0; j < D; j++) {
        const ang = i / Math.pow(10000, 2 * Math.floor(j / 2) / D);
        row.push(j % 2 === 0 ? Math.sin(ang) : Math.cos(ang));
      }
      P.push(row);
    }
    return P;
  }

  /* Scaled dot-product attention; causal = true aplica a máscara triangular. */
  function attention(Xq, Xkv, Wq, Wk, Wv, causal) {
    const Q = matmul(Xq, Wq), K = matmul(Xkv, Wk), V = matmul(Xkv, Wv);
    const S = Q.map((q, i) => K.map((k, j) => {
      if (causal && j > i) return -Infinity;
      return q.reduce((s, v, c) => s + v * k[c], 0) / SQD;
    }));
    const A = softmaxRows(S);
    return { Q, K, V, S, A, C: matmul(A, V) };
  }

  function ffn(X, side) {
    const pre = addBias(matmul(X, W[side + '_W1']), W[side + '_b1']);
    const hid = relu(pre);
    return { pre, hid, F: addBias(matmul(hid, W[side + '_W2']), W[side + '_b2']) };
  }

  /* ── encoder e decoder ───────────────────────────────── */
  function encode(srcIdx) {
    const emb = srcIdx.map((t) => W.E_src[t].slice());
    const pe = posEnc(srcIdx.length);
    const X = add(emb, pe);
    const att = attention(X, X, W.enc_Wq, W.enc_Wk, W.enc_Wv, false);
    const sum1 = add(X, att.C);
    const ln1 = layerNorm(sum1);
    const ff = ffn(ln1.out, 'enc');
    const sum2 = add(ln1.out, ff.F);
    const ln2 = layerNorm(sum2);
    return { idx: srcIdx, emb, pe, X, att, sum1, ln1, H: ln1.out, ff, sum2, ln2, M: ln2.out };
  }

  /* tgtIdx: tokens já escritos, começando por <bos>. */
  function decode(enc, tgtIdx) {
    const emb = tgtIdx.map((t) => W.E_tgt[t].slice());
    const pe = posEnc(tgtIdx.length);
    const Y = add(emb, pe);
    const self = attention(Y, Y, W.dec_Wq, W.dec_Wk, W.dec_Wv, true);
    const sum1 = add(Y, self.C);
    const ln1 = layerNorm(sum1);
    const cross = attention(ln1.out, enc.M, W.x_Wq, W.x_Wk, W.x_Wv, false);
    const sum2 = add(ln1.out, cross.C);
    const ln2 = layerNorm(sum2);
    const ff = ffn(ln2.out, 'dec');
    const sum3 = add(ln2.out, ff.F);
    const ln3 = layerNorm(sum3);
    const logits = matmul(ln3.out, W.W_out);
    const probs = softmaxRows(logits);
    return {
      idx: tgtIdx, emb, pe, Y, self, sum1, ln1, D1: ln1.out, cross, sum2, ln2, D2: ln2.out,
      ff, sum3, ln3, D3: ln3.out, logits, probs,
    };
  }

  function argmax(row) { return row.reduce((best, v, i) => (v > row[best] ? i : best), 0); }

  /* Geração gulosa: devolve o encoder e um registro por passo do decoder. */
  function generate(srcIdx, maxSteps) {
    const enc = encode(srcIdx);
    const written = [BOS];
    const steps = [];
    for (let s = 0; s < (maxSteps || 4); s++) {
      const dec = decode(enc, written.slice());
      const probs = dec.probs[dec.probs.length - 1];
      const tok = argmax(probs);
      steps.push({ dec, probs, tok, prob: probs[tok] });
      written.push(tok);
      if (tok === EOS) break;
    }
    return { enc, steps, written };
  }

  /* ── frase compartilhada entre as seções ─────────────── */
  const state = { det: 0, noun: 0, adj: 0 };
  const listeners = [];
  function sentence() { return [data.dets[state.det], data.nouns[state.noun], data.adjs[state.adj]]; }
  function sentenceIdx() { return sentence().map((w) => data.srcVocab.indexOf(w)); }
  function onSentence(cb) { listeners.push(cb); }
  function setSentence(part, value) {
    state[part] = value;
    for (const cb of listeners) cb();
  }

  /* Liga os grupos de abas "<prefixo>-det-0", "<prefixo>-noun-1", … à frase. */
  function bindSentenceTabs(prefix) {
    const parts = ['det', 'noun', 'adj'];
    function sync() {
      for (const part of parts) {
        for (let v = 0; v < 2; v++) {
          const el = document.getElementById(prefix + '-' + part + '-' + v);
          if (el) el.classList.toggle('active', state[part] === v);
        }
      }
    }
    for (const part of parts) {
      for (let v = 0; v < 2; v++) {
        const el = document.getElementById(prefix + '-' + part + '-' + v);
        if (el) el.addEventListener('click', () => setSentence(part, v));
      }
    }
    onSentence(sync);
    sync();
  }

  /* ── mapa de calor de uma matriz de atenção ──────────── */
  function heatmap(canvas, M, rowLabels, colLabels, opts) {
    const P = DL.plot, th = P.theme();
    opts = opts || {};
    const { ctx, w, h } = P.setup(canvas);
    P.clear(ctx, w, h);
    const rows = M.length, cols = colLabels.length;
    const x0 = 64, y0 = 30;
    const maxRows = opts.maxRows || rows;
    const cw = Math.min(92, (w - x0 - 12) / cols);
    const ch = Math.min(46, (h - y0 - 10) / maxRows);
    const base = P.rgb(th.purple);
    ctx.font = '12px monospace';
    ctx.textAlign = 'center';
    for (let j = 0; j < cols; j++) {
      ctx.fillStyle = th.comment;
      ctx.fillText(colLabels[j], x0 + (j + 0.5) * cw, y0 - 8);
    }
    for (let i = 0; i < rows; i++) {
      ctx.fillStyle = th.comment;
      ctx.textAlign = 'right';
      ctx.font = '12px monospace';
      ctx.fillText(rowLabels[i], x0 - 8, y0 + (i + 0.5) * ch + 4);
      for (let j = 0; j < cols; j++) {
        const v = M[i][j];
        ctx.fillStyle = 'rgba(' + base[0] + ',' + base[1] + ',' + base[2] + ',' + (0.06 + 0.94 * v).toFixed(3) + ')';
        ctx.fillRect(x0 + j * cw, y0 + i * ch, cw - 2, ch - 2);
        P.pill(ctx, v.toFixed(2), x0 + (j + 0.5) * cw - 1, y0 + (i + 0.5) * ch + 3, th.fg, 'center', 11);
      }
    }
    if (rows === 0 && opts.empty) P.mathText(ctx, opts.empty, x0, y0 + 24, th.comment, 'left', 12);
  }

  DL.toy = {
    W, D, SQD, BOS, EOS, srcVocab: data.srcVocab, tgtVocab: data.tgtVocab,
    dets: data.dets, nouns: data.nouns, adjs: data.adjs,
    encode, decode, generate, argmax, posEnc,
    state, sentence, sentenceIdx, onSentence, setSentence, bindSentenceTabs, heatmap,
  };
})();
