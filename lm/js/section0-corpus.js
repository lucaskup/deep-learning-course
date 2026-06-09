/* Corpus de treinamento e explorador de contagens do modelo n-gram. */
(function () {
  'use strict';
  window.DM = window.DM || {};
  DM.sections = DM.sections || [];

  const esc = (s) => s.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');

  function init() {
    const M = DM.model;
    const $ = (id) => document.getElementById(id);
    const f = M.fmtBR;

    function renderCorpus(phrase) {
      const list = $('s0-corpus');
      let hits = 0;
      const rx = phrase
        ? new RegExp('(^| )(' + phrase.replace(/[.*+?^${}()|[\]\\]/g, '\\$&') + ')( |$)', 'g')
        : null;
      list.innerHTML = DM.CORPUS.map((s) => {
        if (rx && rx.test(s)) {
          hits++;
          rx.lastIndex = 0;
          return '<div>' + esc(s).replace(rx, '$1<mark class="match">$2</mark>$3') + '</div>';
        }
        return '<div>' + esc(s) + '</div>';
      }).join('');
      $('s0-corpus-caption').textContent =
        `Corpus de treinamento (${DM.CORPUS.length} frases · ${M.vocabSize()} tipos)` +
        (phrase ? ` · ${hits} frases contêm "${phrase}"` : '');
    }

    function query() {
      const ctx = M.tokenize($('s0-ctx').value);
      const cc = M.contextCounts(ctx, 12);
      const showU = cc.u !== M.BOS, showV = cc.v !== M.BOS;
      const histLabel = (showU ? cc.u + ' ' : '') + (showV ? cc.v : '⟨início⟩');

      let html = `<p class="readout formula formula-lines">histórico (u, v) = (${esc(cc.u)}, ${esc(cc.v)})` +
        ` · c(u, v) = ${cc.tTot} · c(v) = ${cc.bTot} · N = ${cc.N}</p>`;
      html += '<table class="metric-table"><tr><th>w</th><th>trigrama<br />c(u,v,w)</th><th>bigrama<br />c(v,w)</th><th>unigrama<br />c(w)</th><th>p(w | u, v)</th></tr>';
      for (const r of cc.rows) {
        html += `<tr><td class="left"><span class="formula">${esc(r.token)}</span></td>` +
          `<td>${r.cTri}${cc.tTot ? '/' + cc.tTot : ''}</td>` +
          `<td>${r.cBi}${cc.bTot ? '/' + cc.bTot : ''}</td>` +
          `<td>${r.cUni}</td><td>${f(r.p, 3)}</td></tr>`;
      }
      html += '</table>';
      html += '<p class="demo-text small">p interpola os três níveis: ' +
        '<span class="formula">p = 0,6·c(u,v,w)/c(u,v) + 0,3·c(v,w)/c(v) + 0,1·(c(w)+0,1)/(N+0,1·|V|)</span>. ' +
        'Quando um histórico não aparece no corpus, o peso dele desce para o nível de baixo. ' +
        'É exatamente essa distribuição que alimenta o playground e o beam search.</p>';
      $('s0-counts').innerHTML = html;

      renderCorpus(showV ? (showU ? cc.u + ' ' + cc.v : cc.v) : null);
    }

    $('s0-query').addEventListener('click', query);
    $('s0-ctx').addEventListener('keydown', (e) => {
      if (e.key === 'Enter') { e.preventDefault(); query(); }
    });

    query();
  }

  DM.sections.push({ name: 's0-corpus', init });
})();
