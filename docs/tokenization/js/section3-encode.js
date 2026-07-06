/* Seção 3: codificação de texto novo com a tabela de merges aprendida na
   seção 2 (aplicados em ordem, como no exemplo "lowest" dos slides). */
(function () {
  'use strict';
  window.DL = window.DL || {};
  DL.sections = DL.sections || [];

  const DEFAULT_TEXT = 'lowest newest low wider';

  function init() {
    const P = DL.plot;
    const B = DL.bpe;
    const $ = (id) => document.getElementById(id);

    const input = $('s3-text');
    const slK = $('s3-k');
    const btnReset = $('s3-reset');
    const baseSet = new Set(B.BASE);
    const maxK = B.BPE.maxK;

    slK.max = String(maxK);
    let k = 7;
    let selected = 0;

    function chip(text, cls, title) {
      const el = document.createElement('span');
      el.className = 'tok-chip' + (cls ? ' ' + cls : '');
      el.textContent = text;
      if (title) el.title = title;
      return el;
    }

    function tokClass(t, idx) {
      if (t.length === 1 && !baseSet.has(t)) return 'tok-red';
      const palette = ['tok-cyan', 'tok-green', 'tok-orange', 'tok-purple', 'tok-pink'];
      return palette[idx % palette.length];
    }

    function words() {
      return input.value.trim().toLowerCase().split(/\s+/).filter(Boolean);
    }

    function renderTokens(ws) {
      const box = $('s3-tokens');
      box.innerHTML = '';
      let nTok = 0, nChar = 0, idx = 0;
      ws.forEach((w, wi) => {
        const group = document.createElement('span');
        group.className = 'word-group' + (wi === selected ? ' sel' : '');
        group.title = 'clique para ver a derivação de "' + w + '"';
        const enc = B.encodeWord(w, k);
        nChar += w.length + 1;
        for (const t of enc.sym) {
          nTok += 1;
          group.appendChild(chip(t, tokClass(t, idx)));
          idx += 1;
        }
        group.addEventListener('click', () => { selected = wi; redraw(); });
        box.appendChild(group);
      });
      const ratio = nTok > 0 ? (nChar / nTok).toFixed(2) : '0';
      $('s3-readout').textContent =
        nChar + ' caracteres (com _) → ' + nTok + ' tokens · compressão ' + ratio + ' caracteres por token';
    }

    function renderDeriv(ws) {
      const box = $('s3-deriv');
      box.innerHTML = '';
      if (ws.length === 0) return;
      if (selected >= ws.length) selected = 0;
      const w = ws[selected];
      const enc = B.encodeWord(w, k);
      const title = document.createElement('p');
      title.className = 'tok-label';
      title.textContent = 'Derivação de "' + w + '" (merges aplicados em ordem):';
      box.appendChild(title);
      enc.steps.forEach((st, si) => {
        const line = document.createElement('div');
        line.className = 'deriv-line';
        const lab = document.createElement('span');
        lab.className = 'deriv-step';
        lab.textContent = st.label;
        line.appendChild(lab);
        const merged = si > 0 ? st.label.slice(st.label.indexOf('(') + 1).replace(/[(), ]/g, '') : null;
        for (const t of st.sym) {
          const unk = t.length === 1 && !baseSet.has(t);
          line.appendChild(chip(t, unk ? 'tok-red' : (merged && t === merged ? 'tok-new' : '')));
        }
        box.appendChild(line);
      });
      if (enc.steps.length === 1 && k > 0) {
        const note = document.createElement('div');
        note.className = 'deriv-line';
        note.textContent = 'Nenhum dos ' + k + ' primeiros merges se aplica: a palavra fica em caracteres.';
        box.appendChild(note);
      }
    }

    function redraw() {
      const ws = words();
      slK.value = String(k);
      $('s3-k-val').textContent = k;
      renderTokens(ws);
      renderDeriv(ws);
    }

    input.addEventListener('input', () => { selected = 0; redraw(); });
    slK.addEventListener('input', () => { k = +slK.value; redraw(); });
    btnReset.addEventListener('click', () => {
      input.value = DEFAULT_TEXT;
      k = 7;
      selected = 0;
      redraw();
    });

    input.value = DEFAULT_TEXT;
    redraw();
    P.onRedraw(redraw);
  }

  DL.sections.push({ name: 's3-encode', init });
})();
