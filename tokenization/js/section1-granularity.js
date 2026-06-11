/* Seção 1: granularidade da tokenização, caractere vs palavra vs subpalavra
   sobre o vocabulário aprendido no corpus recorrente da aula. */
(function () {
  'use strict';
  window.DL = window.DL || {};
  DL.sections = DL.sections || [];

  const K_SUB = 7;            /* merges do BPE, como no exemplo dos slides */
  const DEFAULT_TEXT = 'low lower newest widest lowest';

  function init() {
    const P = DL.plot;
    const B = DL.bpe;
    const $ = (id) => document.getElementById(id);

    const input = $('s1-text');
    const btnReset = $('s1-reset');
    const cvLen = $('s1-len'), cvVocab = $('s1-vocab');

    const wordVocab = new Set(B.CORPUS.map((e) => e.word));
    const baseSet = new Set(B.BASE);
    const vChar = B.BASE.length;                 /* 11 nos slides */
    const vWord = B.CORPUS.length;               /* 4 nos slides */
    const vSub = B.BASE.length + K_SUB;          /* 11 + 7 merges */

    let lens = [0, 0, 0];

    function chip(text, cls, title) {
      const el = document.createElement('span');
      el.className = 'tok-chip' + (cls ? ' ' + cls : '');
      el.textContent = text;
      if (title) el.title = title;
      return el;
    }

    function words() {
      return input.value.trim().toLowerCase().split(/\s+/).filter(Boolean);
    }

    function renderRows() {
      const ws = words();

      /* Caractere: cada caractere é um token; nunca há OOV. */
      const rowC = $('s1-char-row');
      rowC.innerHTML = '';
      let nChar = 0;
      for (const w of ws) {
        for (const c of (w + B.EOW).split('')) {
          nChar += 1;
          rowC.appendChild(chip(c, c === B.EOW ? 'tok-purple' : ''));
        }
      }
      $('s1-char-info').textContent =
        nChar + ' tokens · |V| = ' + vChar + ' no exemplo (≈256 bytes na prática) · 0 OOV';

      /* Palavra: vocabulário fechado nas 4 palavras do corpus. */
      const rowW = $('s1-word-row');
      rowW.innerHTML = '';
      let nOov = 0;
      for (const w of ws) {
        if (wordVocab.has(w)) rowW.appendChild(chip(w, 'tok-cyan'));
        else {
          nOov += 1;
          rowW.appendChild(chip('[UNK]', 'tok-red', '"' + w + '" não está no vocabulário'));
        }
      }
      $('s1-word-info').textContent =
        ws.length + ' tokens · |V| = ' + vWord + ' · ' + nOov +
        (nOov === 1 ? ' palavra vira' : ' palavras viram') + ' [UNK]';

      /* Subpalavra: BPE com os 7 merges aprendidos na seção 2. */
      const rowS = $('s1-sub-row');
      rowS.innerHTML = '';
      let nSub = 0, nUnk = 0;
      for (const w of ws) {
        for (const t of B.encodeWord(w, K_SUB).sym) {
          nSub += 1;
          if (t.length === 1 && !baseSet.has(t)) {
            nUnk += 1;
            rowS.appendChild(chip(t, 'tok-red', 'caractere fora do vocabulário base'));
          } else {
            rowS.appendChild(chip(t, t.length > 1 ? 'tok-green' : ''));
          }
        }
      }
      $('s1-sub-info').textContent =
        nSub + ' tokens · |V| = ' + vSub + ' (11 caracteres + ' + K_SUB + ' merges) · ' +
        nUnk + ' caracteres fora do vocabulário base';

      lens = [nChar, ws.length, nSub];
    }

    function bars(canvas, vals, ylab) {
      const th = P.theme();
      const { ctx, w, h } = P.setup(canvas);
      P.clear(ctx, w, h);
      const ymax = Math.max(1, ...vals) * 1.3;
      const fr = P.frame(ctx, w, h, 0, 3, 0, ymax);
      P.axes(ctx, fr, { ylabel: ylab, yticks: [0, Math.round(ymax / 2)] });
      const colors = [th.cyan, th.orange, th.green];
      const labels = ['caractere', 'palavra', 'subpalavra'];
      vals.forEach((v, i) => {
        const x0 = fr.X(i + 0.2), x1 = fr.X(i + 0.8);
        ctx.globalAlpha = 0.75;
        ctx.fillStyle = colors[i];
        ctx.fillRect(x0, fr.Y(v), x1 - x0, fr.Y(0) - fr.Y(v));
        ctx.globalAlpha = 1;
        P.label(ctx, (x0 + x1) / 2, fr.Y(v) - 6, String(v), th.fg, 'center');
        P.label(ctx, (x0 + x1) / 2, fr.Y(0) + 15, labels[i], th.comment, 'center');
      });
    }

    function drawLen() { bars(cvLen, lens, 'tokens'); }
    function drawVocab() { bars(cvVocab, [vChar, vWord, vSub], '|V|'); }

    function redraw() {
      renderRows();
      drawLen();
      drawVocab();
    }

    input.addEventListener('input', redraw);
    btnReset.addEventListener('click', () => {
      input.value = DEFAULT_TEXT;
      redraw();
    });

    input.value = DEFAULT_TEXT;
    redraw();
    P.onRedraw(redraw);
    P.observeResize(cvLen, drawLen);
    P.observeResize(cvVocab, drawVocab);
  }

  DL.sections.push({ name: 's1-granularity', init });
})();
