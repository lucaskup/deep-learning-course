/* Seção 5: Unigram LM — o lattice de segmentações de "lower", a melhor
   segmentação via Viterbi e a poda do vocabulário, no corpus dos slides
   da seção de Unigram: low ×10, er ×10, lower ×1. */
(function () {
  'use strict';
  window.DL = window.DL || {};
  DL.sections = DL.sections || [];

  function init() {
    const P = DL.plot;
    const $ = (id) => document.getElementById(id);

    const WORD = 'lower';
    const WORDS = [['low', 10], ['er', 10], ['lower', 1]];
    const CHARS = ['l', 'o', 'w', 'e', 'r'];
    /* Contagens de substrings do corpus, como no slide (1/6): substrings de
       "low"/"er" valem 11 (10 da própria palavra + 1 de "lower"); substrings
       exclusivas de "lower" valem 1. N = 105. */
    const COUNTS = {
      l: 11, o: 11, w: 11, e: 11, r: 11, lo: 11, ow: 11, low: 11, er: 11,
      we: 1, owe: 1, wer: 1, lowe: 1, ower: 1, lower: 1,
    };
    const ORDER = ['low', 'er', 'lo', 'ow', 'l', 'o', 'w', 'e', 'r',
      'lower', 'lowe', 'ower', 'owe', 'wer', 'we'];
    const N0 = 105;

    const cv = $('s5-lattice');
    const ckRenorm = $('s5-renorm');
    const removed = new Set();

    function activeSum() {
      let s = 0;
      for (const t of ORDER) if (!removed.has(t)) s += COUNTS[t];
      return s;
    }

    function probs() {
      const N = ckRenorm.checked ? activeSum() : N0;
      const p = {};
      for (const t of ORDER) if (!removed.has(t)) p[t] = COUNTS[t] / N;
      return p;
    }

    /* Programação dinâmica do slide (3/6): δ(j) é o log da melhor
       segmentação do prefixo com j caracteres. */
    function viterbi(word, p) {
      const n = word.length;
      const delta = [0];
      const back = [null];
      for (let j = 1; j <= n; j++) {
        let best = -Infinity, arg = null;
        for (let i = 0; i < j; i++) {
          const t = word.slice(i, j);
          if (p[t] !== undefined) {
            const v = delta[i] + Math.log(p[t]);
            if (v > best) { best = v; arg = i; }
          }
        }
        delta.push(best);
        back.push(arg);
      }
      const seg = [];
      let j = n;
      while (j > 0) { const i = back[j]; seg.unshift(word.slice(i, j)); j = i; }
      return { seg, logp: delta[n], delta };
    }

    function corpusLL(p) {
      let L = 0;
      for (const [w, f] of WORDS) L += f * viterbi(w, p).logp;
      return L;
    }

    /* L do vocabulário completo com P por contagem (slide 4/6): −49.63.
       Com renormalização e vocabulário completo o valor é o mesmo. */
    const L_FULL = (() => {
      const p = {};
      for (const t of ORDER) p[t] = COUNTS[t] / N0;
      return corpusLL(p);
    })();

    function chip(text, cls, title) {
      const el = document.createElement('span');
      el.className = 'tok-chip' + (cls ? ' ' + cls : '');
      el.textContent = text;
      if (title) el.title = title;
      return el;
    }

    function renderVocab() {
      const p = probs();
      const box = $('s5-vocab');
      box.innerHTML = '';
      for (const t of ORDER) {
        const isChar = CHARS.includes(t);
        const off = removed.has(t);
        let cls = 'clickable';
        if (off) cls += ' tok-removed';
        else if (isChar) cls += ' tok-cyan';
        else if (COUNTS[t] === 11) cls += ' tok-green';
        const info = off
          ? 'podado (clique para restaurar)'
          : 'count ' + COUNTS[t] + ' · P = ' + p[t].toFixed(3);
        const el = chip(t, cls, isChar ? 'caractere: nunca é podado · ' + info : info);
        if (isChar) el.classList.add('tok-locked');
        el.addEventListener('click', () => {
          if (isChar) return;
          if (removed.has(t)) removed.delete(t); else removed.add(t);
          redraw();
        });
        box.appendChild(el);
      }
    }

    function renderSegs() {
      const p = probs();
      const box = $('s5-segs');
      box.innerHTML = '';
      for (const [w, f] of WORDS) {
        const { seg, logp } = viterbi(w, p);
        const line = document.createElement('div');
        line.className = 'tok-row';
        const lab = document.createElement('span');
        lab.className = 'tok-info';
        lab.textContent = '"' + w + '" ×' + f + ' → ';
        line.appendChild(lab);
        for (const t of seg) line.appendChild(chip(t, t.length > 1 ? 'tok-green' : ''));
        const info = document.createElement('span');
        info.className = 'tok-info';
        info.textContent = 'ln P = ' + logp.toFixed(2);
        line.appendChild(info);
        box.appendChild(line);
      }
    }

    function renderDelta() {
      const p = probs();
      const box = $('s5-viterbi');
      box.innerHTML = '';
      const { delta } = viterbi(WORD, p);
      const tb = document.createElement('table');
      tb.className = 'tok-table';
      tb.innerHTML =
        '<tr><th>j</th><th class="mono">prefixo</th><th class="mono">melhor segmentação</th><th>δ(j)</th></tr>';
      for (let j = 1; j <= WORD.length; j++) {
        const pre = WORD.slice(0, j);
        const { seg } = viterbi(pre, p);
        const tr = document.createElement('tr');
        if (j === WORD.length) tr.className = 'best';
        tr.innerHTML =
          '<td>' + j + '</td>' +
          '<td class="mono">' + pre + '</td>' +
          '<td class="mono">' + seg.join('·') + '</td>' +
          '<td>' + delta[j].toFixed(2) + '</td>';
        tb.appendChild(tr);
      }
      box.appendChild(tb);
    }

    function renderReadout() {
      const L = corpusLL(probs());
      const d = L - L_FULL;
      let txt = 'log-likelihood do corpus: L = ' + L.toFixed(2) +
        ' · vocabulário completo: ' + L_FULL.toFixed(2);
      if (Math.abs(d) > 1e-9) {
        txt += ' · ΔL = ' + (d > 0 ? '+' : '') + d.toFixed(2) +
          (d < 0 ? ' (a poda custou likelihood)' : ' (a poda melhorou a likelihood)');
      } else if (removed.size > 0) {
        txt += ' · ΔL = 0 (tokens podados não faziam falta)';
      }
      $('s5-readout').textContent = txt;
    }

    function drawLattice() {
      const th = P.theme();
      const { ctx, w, h } = P.setup(cv);
      P.clear(ctx, w, h);
      const p = probs();
      const best = viterbi(WORD, p).seg;
      const bestEdges = new Set();
      let pos = 0;
      for (const t of best) { bestEdges.add(pos + ':' + (pos + t.length)); pos += t.length; }

      const n = WORD.length;
      const padX = 46;
      const y0 = h * 0.52;
      const X = (i) => padX + (w - 2 * padX) * (i / n);
      const APEX = { 2: -50, 3: -92, 4: 62, 5: 106 };

      for (let i = 0; i < n; i++) {
        for (let j = i + 1; j <= n; j++) {
          const t = WORD.slice(i, j);
          if (COUNTS[t] === undefined || removed.has(t)) continue;
          const len = j - i;
          const isBest = bestEdges.has(i + ':' + j);
          const rare = COUNTS[t] === 1;
          const color = isBest ? th.green : (rare ? th.comment : th.cyan);
          const x1 = X(i), x2 = X(j), xm = (x1 + x2) / 2;
          ctx.strokeStyle = color;
          ctx.lineWidth = isBest ? 3 : 1.4;
          ctx.setLineDash(rare && !isBest ? [5, 4] : []);
          ctx.beginPath();
          if (len === 1) {
            ctx.moveTo(x1, y0);
            ctx.lineTo(x2, y0);
            P.label(ctx, xm, y0 + 18, t, color, 'center');
          } else {
            const ya = y0 + APEX[len];
            ctx.moveTo(x1, y0);
            ctx.quadraticCurveTo(xm, 2 * ya - y0, x2, y0);
            P.label(ctx, xm, ya + (ya < y0 ? -6 : 16), t, color, 'center');
          }
          ctx.stroke();
          ctx.setLineDash([]);
        }
      }

      for (let i = 0; i <= n; i++) {
        ctx.beginPath();
        ctx.arc(X(i), y0, 9, 0, 2 * Math.PI);
        ctx.fillStyle = th.card;
        ctx.fill();
        ctx.strokeStyle = th.fg;
        ctx.lineWidth = 1.2;
        ctx.stroke();
        P.label(ctx, X(i), y0 + 4, String(i), th.fg, 'center');
      }
    }

    function redraw() {
      renderVocab();
      renderSegs();
      renderDelta();
      renderReadout();
      drawLattice();
    }

    /* Poda todos os tokens (não caracteres) fora das melhores segmentações
       atuais: exatamente os Δ = 0 do slide (5/6). */
    $('s5-prune').addEventListener('click', () => {
      const p = probs();
      const used = new Set();
      for (const [w] of WORDS) for (const t of viterbi(w, p).seg) used.add(t);
      for (const t of ORDER) {
        if (!CHARS.includes(t) && !used.has(t)) removed.add(t);
      }
      redraw();
    });
    $('s5-reset').addEventListener('click', () => {
      removed.clear();
      ckRenorm.checked = false;
      redraw();
    });
    ckRenorm.addEventListener('change', redraw);

    redraw();
    P.onRedraw(redraw);
    P.observeResize(cv, drawLattice);
  }

  DL.sections.push({ name: 's5-unigram', init });
})();
