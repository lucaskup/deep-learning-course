/* Seção 3: perplexidade (preset dos slides + modelo n-gram), BLEU e ROUGE. */
(function () {
  'use strict';
  window.DM = window.DM || {};
  DM.sections = DM.sections || [];

  const esc = (s) => s.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');

  function init() {
    const M = DM.model;
    const $ = (id) => document.getElementById(id);
    const f = M.fmtBR;

    /* ── 3a: perplexidade ───────────────────────────────────── */
    const tabPreset = $('s3a-tab-preset'), tabModel = $('s3a-tab-model');
    let mode = 'preset';

    function pplTable(rows, note) {
      const ps = rows.map((r) => r.p);
      const { logs, L, m, exponent, ppl } = M.perplexity(ps);
      let html = '<table class="metric-table"><tr><th>t</th>';
      rows.forEach((_, i) => { html += `<th>${i + 1}</th>`; });
      html += '</tr><tr><td class="left">token</td>';
      for (const r of rows) {
        const map = r.mappedTo ? ` <span style="color:var(--orange)">→${esc(r.mappedTo)}</span>` : '';
        html += `<td><span class="formula">${esc(r.shown)}</span>${map}</td>`;
      }
      html += '</tr><tr><td class="left">p(y<sub>t</sub> | y<sub>&lt;t</sub>)</td>';
      for (const p of ps) html += `<td>${f(p, p >= 0.01 ? 3 : 5)}</td>`;
      html += '</tr><tr><td class="left">log₂ p</td>';
      for (const lg of logs) html += `<td>${f(lg, 2)}</td>`;
      html += '</tr><tr><td class="left">barra</td>';
      for (const p of ps) {
        const color = p > 0.3 ? 'var(--green)' : p > 0.05 ? 'var(--orange)' : 'var(--pink)';
        html += `<td class="prob-cell"><span class="prob-bar" style="width:${Math.max(3, p * 100)}%; background:${color}"></span></td>`;
      }
      html += '</tr></table>';
      html += `<p class="readout formula formula-lines">` +
        `L = Σ log₂ p = ${f(L, 2)} &nbsp;·&nbsp; −L/m = ${f(exponent, 2)} &nbsp;·&nbsp; ` +
        `<strong>Perplexidade = 2<sup>${f(exponent, 2)}</sup> ≈ ${f(ppl, 3)}</strong></p>`;
      $('s3a-table').innerHTML = html;
      $('s3a-note').textContent = note;
    }

    function renderPreset() {
      /* Exemplo numérico dos slides: p = (0,5; 0,25; 0,5; 0,5) */
      const rows = [0.5, 0.25, 0.5, 0.5].map((p, i) => ({ shown: 'y' + (i + 1), mappedTo: null, p }));
      pplTable(rows,
        'Leitura: em média o modelo se comporta como se estivesse em dúvida entre ' +
        'cerca de 2,4 tokens a cada passo.');
    }

    function renderModel() {
      const toks = M.tokenize($('s3a-input').value);
      if (toks.length === 0) { $('s3a-table').innerHTML = ''; return; }
      const rows = M.scoreSentence(toks);
      pplTable(rows,
        'O modelo n-gram avalia cada token dado o contexto; palavras fora do vocabulário ' +
        'viram <unk> e recebem só a massa de suavização. O <eos> entra como último token ' +
        '(m inclui ele). Compare uma frase do corpus com a mesma frase embaralhada.');
    }

    function setMode(m2) {
      mode = m2;
      tabPreset.classList.toggle('active', mode === 'preset');
      tabModel.classList.toggle('active', mode === 'model');
      $('s3a-input-wrap').classList.toggle('hidden', mode === 'preset');
      if (mode === 'preset') renderPreset(); else renderModel();
    }
    tabPreset.addEventListener('click', () => setMode('preset'));
    tabModel.addEventListener('click', () => setMode('model'));
    $('s3a-eval').addEventListener('click', renderModel);
    $('s3a-input').addEventListener('keydown', (e) => {
      if (e.key === 'Enter') { e.preventDefault(); renderModel(); }
    });

    /* ── 3b: BLEU e ROUGE ──────────────────────────────────── */
    function renderMetrics() {
      const refToks = M.tokenize($('s3b-ref').value);
      const candToks = M.tokenize($('s3b-cand').value);
      if (refToks.length === 0 || candToks.length === 0) {
        $('s3b-bleu').innerHTML = '';
        $('s3b-rouge').innerHTML = '';
        return;
      }

      // BLEU
      const b = M.bleu(refToks, candToks);
      let html = '<h4 class="sub-title" style="margin-top:.6rem">BLEU</h4>';
      html += '<table class="metric-table"><tr><th>n</th><th class="left">n-grams que casam (clipados)</th><th>casados / total</th><th>p<sub>n</sub></th></tr>';
      for (const row of b.rows) {
        let chips = row.details.map((d) => {
          let c = `<span class="ngram-chip">${esc(d.gram)}${d.clip > 1 ? '×' + d.clip : ''}</span>`;
          if (d.count > d.clip) c += `<span class="ngram-chip clipped">${esc(d.gram)}×${d.count - d.clip}</span>`;
          return c;
        }).join('');
        if (!chips) chips = '<span style="color:var(--comment)">nenhum</span>';
        html += `<tr><td>${row.n}</td><td class="left">${chips}</td>` +
          `<td>${row.matched}/${row.total}</td><td>${f(row.pn, 3)}</td></tr>`;
      }
      html += '</table>';
      const bpFormula = b.c > b.r
        ? `c = ${b.c} &gt; r = ${b.r} ⇒ BP = 1`
        : `BP = e<sup>1 − ${b.r}/${b.c}</sup> = ${f(b.bp, 3)}`;
      html += `<p class="readout formula formula-lines">${bpFormula} &nbsp;·&nbsp; `;
      html += b.anyZero
        ? '<strong>BLEU = 0</strong> (algum p<sub>n</sub> = 0, a média geométrica zera)'
        : `<strong>BLEU = ${f(b.bp, 2)} · exp(¼ Σ ln p<sub>n</sub>) ≈ ${f(b.score, 2)}</strong>`;
      html += '</p>';
      $('s3b-bleu').innerHTML = html;

      // ROUGE
      const r1 = M.rougeN(refToks, candToks, 1);
      const r2 = M.rougeN(refToks, candToks, 2);
      const rl = M.rougeL(refToks, candToks);
      let html2 = '<h4 class="sub-title" style="margin-top:.6rem">ROUGE</h4>';
      html2 += '<table class="metric-table"><tr><th>métrica</th><th>em comum</th><th>Recall</th><th>Precisão</th><th>F₁</th></tr>';
      html2 += `<tr><td>ROUGE-1</td><td>${r1.overlap} unigramas</td><td>${r1.overlap}/${r1.refTot} = ${f(r1.R, 2)}</td><td>${r1.overlap}/${r1.candTot} = ${f(r1.P, 2)}</td><td>${f(r1.F1, 2)}</td></tr>`;
      html2 += `<tr><td>ROUGE-2</td><td>${r2.overlap} bigramas</td><td>${r2.overlap}/${r2.refTot} = ${f(r2.R, 2)}</td><td>${r2.overlap}/${r2.candTot} = ${f(r2.P, 2)}</td><td>${f(r2.F1, 2)}</td></tr>`;
      html2 += `<tr><td>ROUGE-L</td><td>LCS = ${rl.length}</td><td>${rl.length}/${refToks.length} = ${f(rl.R, 2)}</td><td>${rl.length}/${candToks.length} = ${f(rl.P, 2)}</td><td>${f(rl.F1, 2)}</td></tr>`;
      html2 += '</table>';
      // frases com a LCS destacada
      const refSet = new Set(rl.refIdx), candSet = new Set(rl.candIdx);
      const mark = (toks, set) => toks.map((t, i) =>
        set.has(i) ? `<mark class="match">${esc(t)}</mark>` : `<mark class="miss">${esc(t)}</mark>`).join(' ');
      html2 += `<p class="readout formula formula-lines">referência: ${mark(refToks, refSet)}<br />` +
        `candidato: ${mark(candToks, candSet)}<br />` +
        `<span style="color:var(--comment)">verde = na LCS (ordem preservada), rosa = fora da LCS</span></p>`;
      $('s3b-rouge').innerHTML = html2;
    }

    $('s3b-preset-bleu').addEventListener('click', () => {
      $('s3b-ref').value = 'the small cat sat on the mat';
      $('s3b-cand').value = 'the big cat sat on the mat';
      renderMetrics();
    });
    $('s3b-preset-rouge').addEventListener('click', () => {
      $('s3b-ref').value = 'the cat was found under the bed';
      $('s3b-cand').value = 'the cat was under the bed';
      renderMetrics();
    });
    $('s3b-ref').addEventListener('input', renderMetrics);
    $('s3b-cand').addEventListener('input', renderMetrics);

    setMode('preset');
    renderMetrics();
  }

  DM.sections.push({ name: 's3-eval', init });
})();
