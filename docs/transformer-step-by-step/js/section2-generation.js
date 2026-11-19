/* Seção 2: geração autoregressiva. Cada clique roda o decoder mais uma vez
   sobre o que já foi escrito e mostra a distribuição do próximo token e a
   linha nova da matriz de cross-attention. */
(function () {
  'use strict';
  window.DL = window.DL || {};
  DL.sections = DL.sections || [];

  function init() {
    const P = DL.plot, T = DL.toy;
    const $ = (id) => document.getElementById(id);
    const cvProbs = $('s2-probs'), cvCross = $('s2-cross');
    const tok = (w) => '<code>' + w.replace('<', '&lt;').replace('>', '&gt;') + '</code>';

    let run = null;
    let k = 0;   /* passos do decoder já executados */

    function drawProbs() {
      const th = P.theme();
      const { ctx, w, h } = P.setup(cvProbs);
      P.clear(ctx, w, h);
      const n = T.tgtVocab.length;
      const left = 16, base = h - 34, top = 30;
      const bw = (w - 2 * left) / n;
      ctx.strokeStyle = th.line;
      ctx.beginPath(); ctx.moveTo(left, base); ctx.lineTo(w - left, base); ctx.stroke();
      const step = k > 0 ? run.steps[k - 1] : null;
      for (let j = 0; j < n; j++) {
        const x = left + j * bw;
        ctx.font = '11px monospace';
        ctx.textAlign = 'center';
        ctx.fillStyle = step && step.tok === j ? th.pink : th.comment;
        ctx.fillText(T.tgtVocab[j], x + bw / 2, base + 16);
        if (!step) continue;
        const v = step.probs[j];
        const bh = v * (base - top);
        ctx.fillStyle = step.tok === j ? th.pink : th.cyan;
        ctx.fillRect(x + bw * 0.18, base - bh, bw * 0.64, bh);
        P.pill(ctx, v.toFixed(2), x + bw / 2, base - bh - 5, th.fg, 'center', 11);
      }
      if (!step) P.mathText(ctx, 'Clique em "Gerar próximo token" para rodar o decoder.', left, top, th.comment, 'left', 12);
    }

    function drawCross() {
      const rows = [], labels = [];
      for (let s = 0; s < k; s++) {
        const A = run.steps[s].dec.cross.A;
        rows.push(A[A.length - 1]);
        labels.push(T.tgtVocab[run.steps[s].tok]);
      }
      T.heatmap(cvCross, rows, labels, T.sentence(), { maxRows: 4, empty: 'Uma linha por token escrito.' });
    }

    function sync() {
      const done = k >= run.steps.length;
      $('s2-step').disabled = done;
      const written = run.written.slice(0, k + 1).map((t) => tok(T.tgtVocab[t])).join(' ');
      let html = 'Origem: ' + T.sentence().map(tok).join(' ') + ' · já escrito: ' + written;
      if (k > 0) {
        const s = run.steps[k - 1];
        html += ' · passo ' + k + ': ' + tok(T.tgtVocab[s.tok]) + ' com probabilidade ' + s.prob.toFixed(2);
      }
      if (done) html += ' · geração encerrada';
      $('s2-readout').innerHTML = html;
      drawProbs();
      drawCross();
    }

    function recompute() {
      run = T.generate(T.sentenceIdx(), 4);
      k = 0;
      sync();
    }

    $('s2-step').addEventListener('click', () => { if (k < run.steps.length) { k++; sync(); } });
    $('s2-reset').addEventListener('click', () => { k = 0; sync(); });
    T.bindSentenceTabs('s2');
    T.onSentence(recompute);
    P.observeResize(cvProbs, drawProbs);
    P.observeResize(cvCross, drawCross);
    P.onRedraw(() => { drawProbs(); drawCross(); });
    recompute();
  }

  DL.sections.push({ name: 'generation', init });
})();
