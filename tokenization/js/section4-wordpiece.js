/* Seção 4: WordPiece vs BPE no mesmo corpus, critério de likelihood
   score(x, y) = freq(xy) / (freq(x) · freq(y)) contra a frequência pura. */
(function () {
  'use strict';
  window.DL = window.DL || {};
  DL.sections = DL.sections || [];

  function init() {
    const P = DL.plot;
    const B = DL.bpe;
    const $ = (id) => document.getElementById(id);

    const slK = $('s4-k');
    const btnStep = $('s4-step'), btnReset = $('s4-reset');
    const cvCurve = $('s4-curve');
    const maxK = B.BPE.maxK;

    slK.max = String(maxK);
    let k = 0;

    function chip(text, cls) {
      const el = document.createElement('span');
      el.className = 'tok-chip' + (cls ? ' ' + cls : '');
      el.textContent = text;
      return el;
    }

    function renderScores() {
      const st = B.WP.states[k];
      const box = $('s4-scores');
      box.innerHTML = '';
      if (st.pairs.length === 0) {
        box.textContent = 'Não há mais pares para pontuar.';
        return;
      }
      /* O par que a frequência pura (BPE) escolheria a partir DESTA segmentação. */
      const byCount = st.pairs.slice().sort((p, q) => q.count - p.count || p.first - q.first);
      const freqPick = byCount[0];
      const tb = document.createElement('table');
      tb.className = 'tok-table';
      tb.innerHTML =
        '<tr><th class="mono">par</th><th>freq(xy)</th><th>freq(x)·freq(y)</th><th>score</th><th></th></tr>';
      const shown = st.pairs.slice(0, 8);
      if (!shown.includes(freqPick)) shown.push(freqPick);
      shown.forEach((p, i) => {
        const tr = document.createElement('tr');
        let note = '';
        if (i === 0) { tr.className = 'best'; note = '← WordPiece mescla'; }
        else if (p === freqPick) { tr.className = 'pick-bpe'; note = '← frequência pura mescla'; }
        tr.innerHTML =
          '<td class="mono">(' + p.a + ', ' + p.b + ')</td>' +
          '<td>' + p.count + '</td>' +
          '<td>' + p.fa + ' × ' + p.fb + ' = ' + (p.fa * p.fb) + '</td>' +
          '<td>' + p.score.toFixed(4) + '</td>' +
          '<td>' + note + '</td>';
        tb.appendChild(tr);
      });
      box.appendChild(tb);
    }

    function renderPane(model, paneId, mergesId) {
      const st = model.states[k];
      const lastTok = k > 0 ? model.merges[k - 1].a + model.merges[k - 1].b : null;
      const box = $(paneId);
      box.innerHTML = '';
      for (const s of st.segs) {
        const line = document.createElement('div');
        line.className = 'tok-row';
        for (const t of s.sym) {
          line.appendChild(chip(t, t === lastTok ? 'tok-new' : (t.length > 1 ? 'tok-green' : '')));
        }
        const info = document.createElement('span');
        info.className = 'tok-info';
        info.textContent = '(freq ' + s.freq + ')';
        line.appendChild(info);
        box.appendChild(line);
      }
      $(mergesId).textContent = k > 0
        ? 'merges: ' + model.merges.slice(0, k).map((m) => m.a + m.b).join(', ')
        : 'nenhum merge ainda';
    }

    function drawCurve() {
      const th = P.theme();
      const { ctx, w, h } = P.setup(cvCurve);
      P.clear(ctx, w, h);
      const fr = P.frame(ctx, w, h, 0, maxK, 0, 100);
      P.axes(ctx, fr, { xlabel: 'k (merges)', xticks: [0, 5, 10, 15], yticks: [0, 50, 100] });
      const ks = B.BPE.states.map((_, i) => i);
      P.line(ctx, fr, ks, B.BPE.states.map((s) => s.tokens), th.cyan, { width: 1.8 });
      P.line(ctx, fr, ks, B.WP.states.map((s) => s.tokens), th.orange, { width: 1.8 });
      P.scatter(ctx, fr, [[k, B.BPE.states[k].tokens]], th.cyan, { r: 4, alpha: 1 });
      P.scatter(ctx, fr, [[k, B.WP.states[k].tokens]], th.orange, { r: 4, alpha: 1 });
      P.label(ctx, fr.X(0) + 8, fr.Y(100) + 14, 'BPE (frequência)', th.cyan);
      P.label(ctx, fr.X(0) + 8, fr.Y(100) + 28, 'WordPiece (likelihood)', th.orange);
    }

    function redraw() {
      slK.value = String(k);
      $('s4-k-val').textContent = k;
      renderScores();
      renderPane(B.BPE, 's4-bpe', 's4-bpe-merges');
      renderPane(B.WP, 's4-wp', 's4-wp-merges');
      const nb = k < maxK ? B.BPE.merges[k] : null;
      const nw = k < maxK ? B.WP.merges[k] : null;
      $('s4-readout').textContent = (nb && nw)
        ? 'próximo merge · BPE: (' + nb.a + ', ' + nb.b + ') contagem ' + nb.count +
          ' · WordPiece: (' + nw.a + ', ' + nw.b + ') score ' + nw.score.toFixed(3)
        : 'treinamento completo: os dois vocabulários cobrem o corpus inteiro';
      drawCurve();
    }

    slK.addEventListener('input', () => { k = +slK.value; redraw(); });
    btnStep.addEventListener('click', () => { k = Math.min(maxK, k + 1); redraw(); });
    btnReset.addEventListener('click', () => { k = 0; redraw(); });

    redraw();
    P.onRedraw(redraw);
    P.observeResize(cvCurve, drawCurve);
  }

  DL.sections.push({ name: 's4-wordpiece', init });
})();
