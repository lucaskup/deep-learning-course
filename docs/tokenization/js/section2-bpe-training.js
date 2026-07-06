/* Seção 2: treinamento do BPE passo a passo no corpus dos slides
   (low 5, lower 2, newest 6, widest 3), com contagem de pares,
   merge escolhido, vocabulário crescendo e curva de compressão. */
(function () {
  'use strict';
  window.DL = window.DL || {};
  DL.sections = DL.sections || [];

  function init() {
    const P = DL.plot;
    const B = DL.bpe;
    const $ = (id) => document.getElementById(id);

    const slK = $('s2-k');
    const btnStep = $('s2-step'), btnRun = $('s2-run'), btnReset = $('s2-reset');
    const cvCurve = $('s2-curve');
    const maxK = B.BPE.maxK;
    const nBase = B.BASE.length;

    slK.max = String(maxK);
    let k = 0;
    let running = false, acc = 0;

    function chip(text, cls) {
      const el = document.createElement('span');
      el.className = 'tok-chip' + (cls ? ' ' + cls : '');
      el.textContent = text;
      return el;
    }

    function renderCorpus(st, lastTok) {
      const box = $('s2-corpus');
      box.innerHTML = '';
      const tb = document.createElement('table');
      tb.className = 'tok-table';
      tb.innerHTML = '<tr><th class="mono">palavra</th><th>freq</th><th>segmentação atual</th></tr>';
      for (const s of st.segs) {
        const tr = document.createElement('tr');
        const td1 = document.createElement('td');
        td1.className = 'mono';
        td1.textContent = s.word;
        const td2 = document.createElement('td');
        td2.textContent = s.freq;
        const td3 = document.createElement('td');
        for (const t of s.sym) {
          td3.appendChild(chip(t, t === lastTok ? 'tok-new' : (t.length > 1 ? 'tok-green' : '')));
        }
        tr.append(td1, td2, td3);
        tb.appendChild(tr);
      }
      box.appendChild(tb);
    }

    function renderPairs(st) {
      const box = $('s2-pairs');
      box.innerHTML = '';
      if (st.pairs.length === 0) {
        box.textContent = 'Não há mais pares: cada palavra do corpus virou um único token.';
        return;
      }
      const tb = document.createElement('table');
      tb.className = 'tok-table';
      tb.innerHTML = '<tr><th class="mono">par</th><th>contagem</th></tr>';
      st.pairs.slice(0, 8).forEach((p, i) => {
        const tr = document.createElement('tr');
        if (i === 0) tr.className = 'best';
        const td1 = document.createElement('td');
        td1.className = 'mono';
        td1.textContent = '(' + p.a + ', ' + p.b + ')';
        const td2 = document.createElement('td');
        td2.textContent = p.count + (i === 0 ? '  ← próximo merge' : '');
        tr.append(td1, td2);
        tb.appendChild(tr);
      });
      box.appendChild(tb);
    }

    function renderVocab(st) {
      const box = $('s2-vocab');
      box.innerHTML = '';
      st.vocab.forEach((t, i) => {
        const isNew = k > 0 && i === st.vocab.length - 1;
        box.appendChild(chip(t, isNew ? 'tok-new' : (i >= nBase ? 'tok-green' : '')));
      });
    }

    function renderInfo(st) {
      const last = k > 0 ? B.BPE.merges[k - 1] : null;
      const next = k < maxK ? B.BPE.merges[k] : null;
      let msg = last
        ? 'Último merge: (' + last.a + ', ' + last.b + ') → ' + (last.a + last.b) +
          ', contagem ' + last.count + '. '
        : 'Nenhum merge aplicado ainda: vocabulário só de caracteres. ';
      msg += next
        ? 'Próximo: (' + next.a + ', ' + next.b + ') com contagem ' + next.count + '.'
        : 'Treinamento esgotado: nenhum par restante.';
      $('s2-merge-info').textContent = msg;
      $('s2-readout').textContent =
        'k = ' + k + ' · |V| = ' + st.vocab.length + ' · tokens no corpus = ' + st.tokens;
    }

    function drawCurve() {
      const th = P.theme();
      const { ctx, w, h } = P.setup(cvCurve);
      P.clear(ctx, w, h);
      const fr = P.frame(ctx, w, h, 0, maxK, 0, 100);
      P.axes(ctx, fr, { xlabel: 'k (merges)', xticks: [0, 5, 10, 15], yticks: [0, 50, 100] });
      const ks = B.BPE.states.map((_, i) => i);
      P.line(ctx, fr, ks, B.BPE.states.map((s) => s.tokens), th.cyan, { width: 1.8 });
      P.line(ctx, fr, ks, B.BPE.states.map((s) => s.vocab.length), th.purple, { width: 1.8 });
      const st = B.BPE.states[k];
      P.scatter(ctx, fr, [[k, st.tokens]], th.cyan, { r: 4, alpha: 1 });
      P.scatter(ctx, fr, [[k, st.vocab.length]], th.purple, { r: 4, alpha: 1 });
      P.label(ctx, fr.X(0) + 8, fr.Y(100) + 14, 'tokens no corpus', th.cyan);
      P.label(ctx, fr.X(0) + 8, fr.Y(100) + 28, '|V|', th.purple);
    }

    function redraw() {
      const st = B.BPE.states[k];
      const lastTok = k > 0 ? B.BPE.merges[k - 1].a + B.BPE.merges[k - 1].b : null;
      slK.value = String(k);
      $('s2-k-val').textContent = k;
      renderCorpus(st, lastTok);
      renderPairs(st);
      renderVocab(st);
      renderInfo(st);
      drawCurve();
    }

    function stopRun() {
      running = false;
      btnRun.textContent = '▶ Rodar';
      DL.stopTicker(tick);
    }

    function tick(dt) {
      acc += dt;
      if (acc >= 0.8) {
        acc = 0;
        k = Math.min(maxK, k + 1);
        redraw();
        if (k >= maxK) stopRun();
      }
    }

    slK.addEventListener('input', () => {
      stopRun();
      k = +slK.value;
      redraw();
    });
    btnStep.addEventListener('click', () => {
      stopRun();
      k = Math.min(maxK, k + 1);
      redraw();
    });
    btnRun.addEventListener('click', () => {
      if (running) { stopRun(); return; }
      if (k >= maxK) k = 0;
      running = true;
      acc = 0;
      btnRun.textContent = '⏸ Pausar';
      DL.startTicker(tick);
    });
    btnReset.addEventListener('click', () => {
      stopRun();
      k = 0;
      redraw();
    });

    redraw();
    P.onRedraw(redraw);
    P.observeResize(cvCurve, drawCurve);
  }

  DL.sections.push({ name: 's2-bpe-training', init });
})();
