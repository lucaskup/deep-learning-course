/* Seção 3: frases fora do corpus de treino. O aluno escolhe qualquer token do
   vocabulário para cada posição da origem e vê a tradução gulosa, a
   self-attention do encoder e a cross-attention. */
(function () {
  'use strict';
  window.DL = window.DL || {};
  DL.sections = DL.sections || [];

  function init() {
    const P = DL.plot, T = DL.toy;
    const $ = (id) => document.getElementById(id);
    const cvEnc = $('s3-enc'), cvCross = $('s3-cross');
    const sels = [0, 1, 2].map((i) => $('s3-pos-' + i));
    const tok = (w) => '<code>' + w.replace('<', '&lt;').replace('>', '&gt;') + '</code>';
    const DEFAULT = ['O', 'preto', 'gato'];

    let run = null;

    sels.forEach((sel, i) => {
      T.srcVocab.forEach((w, v) => {
        const o = document.createElement('option');
        o.value = v; o.textContent = w;
        sel.appendChild(o);
      });
      sel.value = T.srcVocab.indexOf(DEFAULT[i]);
      sel.addEventListener('change', recompute);
    });

    function words() { return sels.map((s) => T.srcVocab[+s.value]); }

    function inCorpus(w) {
      return T.dets.includes(w[0]) && T.nouns.includes(w[1]) && T.adjs.includes(w[2]);
    }

    function draw() {
      const src = words();
      T.heatmap(cvEnc, run.enc.att.A, src, src);
      const rows = run.steps.map((s) => s.dec.cross.A[s.dec.cross.A.length - 1]);
      T.heatmap(cvCross, rows, run.steps.map((s) => T.tgtVocab[s.tok]), src, { maxRows: 5 });
    }

    function recompute() {
      const src = words();
      run = T.generate(sels.map((s) => +s.value), 5);
      const out = run.steps.map((s) => tok(T.tgtVocab[s.tok]) + ' (' + s.prob.toFixed(2) + ')').join(' ');
      const ended = run.steps[run.steps.length - 1].tok === T.EOS;
      $('s3-readout').innerHTML = 'Origem: ' + src.map(tok).join(' ') +
        (inCorpus(src) ? ' · frase do corpus de treino' : ' · frase fora do corpus de treino') +
        '<br>Saída, com a probabilidade de cada token: ' + out +
        (ended ? '' : ' · o modelo não gerou <code>&lt;eos&gt;</code> em 5 passos');
      draw();
    }

    $('s3-reset').addEventListener('click', () => {
      sels.forEach((sel, i) => { sel.value = T.srcVocab.indexOf(DEFAULT[i]); });
      recompute();
    });
    P.observeResize(cvEnc, draw);
    P.observeResize(cvCross, draw);
    P.onRedraw(draw);
    recompute();
  }

  DL.sections.push({ name: 'free-sentence', init });
})();
