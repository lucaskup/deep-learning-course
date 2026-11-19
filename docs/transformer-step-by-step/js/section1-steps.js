/* Seção 1: o forward pass do transformer etapa por etapa, com todas as
   matrizes na tela. Passar o mouse (ou tocar) em uma célula mostra a conta que
   a produziu e marca os operandos; o diagrama ao lado destaca o bloco da etapa. */
(function () {
  'use strict';
  window.DL = window.DL || {};
  DL.sections = DL.sections || [];

  function init() {
    const T = DL.toy, W = T.W;
    const $ = (id) => document.getElementById(id);
    const elStage = $('s1-stage'), elExplain = $('s1-explain'), elDiagram = $('s1-diagram');
    const elTitle = $('s1-stage-title'), elReadout = $('s1-readout'), elSteps = $('s1-dec-steps');
    const HINT = 'Passe o mouse ou toque em uma célula para ver a conta que a produziu.';

    let stage = 0;
    let decStep = 1;       /* quantos tokens o decoder já recebeu, contando <bos> */
    let run = null;        /* resultado de T.generate para a frase atual */
    let mats = {};         /* matrizes da etapa em tela, por id */

    /* ── formatação ──────────────────────────────────────── */
    function fmt(v, dec) {
      if (v === -Infinity) return '−∞';
      const s = v.toFixed(dec === undefined ? 2 : dec);
      return (parseFloat(s) === 0 ? s.replace('-', '') : s).replace('-', '−');
    }
    const par = (v, dec) => (v < 0 ? '(' + fmt(v, dec) + ')' : fmt(v, dec));
    const tok = (w) => '<code>' + w.replace('<', '&lt;').replace('>', '&gt;') + '</code>';
    const sym = (html) => ({ sym: html });

    /* ── explicações de célula ───────────────────────────── */
    function exMatmul(aId, bId, bias) {
      return (i, j) => {
        const A = mats[aId].data, B = mats[bId].data;
        const terms = A[i].map((a, k) => par(a) + '·' + par(B[k][j], mats[bId].dec));
        let total = A[i].reduce((s, a, k) => s + a * B[k][j], 0);
        let html = 'linha ' + (i + 1) + ' de ' + mats[aId].name + ' × coluna ' + (j + 1) + ' de ' + mats[bId].name +
          ':<br>' + terms.join(' + ');
        const marks = [[aId, 'row', i], [bId, 'col', j]];
        if (bias) {
          const b = mats[bias].data[0][j];
          html += ' + ' + par(b, 1);
          total += b;
          marks.push([bias, 'cell', 0, j]);
        }
        return { html: html + ' ≈ <b>' + fmt(total) + '</b>', marks };
      };
    }
    function exAdd(aId, bId) {
      return (i, j) => ({
        html: fmt(mats[aId].data[i][j]) + ' + ' + par(mats[bId].data[i][j]) + ' ≈ <b>' +
          fmt(mats[aId].data[i][j] + mats[bId].data[i][j]) + '</b>',
        marks: [[aId, 'cell', i, j], [bId, 'cell', i, j]],
      });
    }
    function exLookup(eId, idx, vocab) {
      return (i) => ({
        html: 'O token ' + tok(vocab[idx[i]]) + ' tem índice ' + (idx[i] + 1) + ': a linha copiada é a linha ' +
          (idx[i] + 1) + ' da matriz de embedding.',
        marks: [[eId, 'row', idx[i]]],
      });
    }
    function exPE(i, j) {
      const k = Math.floor(j / 2);
      const div = Math.pow(10000, 2 * k / T.D);
      const fn = j % 2 === 0 ? 'sin' : 'cos';
      return {
        html: 'posição i = ' + i + ', dimensão j = ' + j + ': ' + fn + '(' + i + ' / ' + div + ') ≈ <b>' +
          fmt(Math[fn](i / div)) + '</b>',
        marks: [],
      };
    }
    function exScore(qId, kId, masked) {
      return (i, j) => {
        if (masked && j > i) {
          return { html: 'Posição ' + (j + 1) + ' vem depois da posição ' + (i + 1) + ': a máscara causal troca o score por −∞ e o peso vira 0.', marks: [] };
        }
        const q = mats[qId].data[i], k = mats[kId].data[j];
        const raw = q.reduce((s, v, c) => s + v * k[c], 0);
        return {
          html: 'linha ' + (i + 1) + ' de Q · linha ' + (j + 1) + ' de K, dividido por √d = 2:<br>(' +
            q.map((v, c) => par(v) + '·' + par(k[c])).join(' + ') + ') / 2 ≈ ' + fmt(raw) + ' / 2 ≈ <b>' + fmt(raw / 2) + '</b>',
          marks: [[qId, 'row', i], [kId, 'row', j]],
        };
      };
    }
    function exSoftmax(sId) {
      return (i, j) => {
        const row = mats[sId].data[i];
        const e = row.map((v) => Math.exp(v));
        const z = e.reduce((a, b) => a + b, 0);
        return {
          html: 'softmax na linha ' + (i + 1) + ': e<sup>' + fmt(row[j]) + '</sup> / (' +
            row.map((v) => 'e<sup>' + fmt(v) + '</sup>').join(' + ') + ') ≈ ' + fmt(e[j], 3) + ' / ' + fmt(z, 3) +
            ' ≈ <b>' + fmt(e[j] / z) + '</b>',
          marks: [[sId, 'row', i]],
        };
      };
    }
    /* LayerNorm em três contas: média da linha, desvio padrão da linha e normalização. */
    function exMean(sId) {
      return (i) => {
        const row = mats[sId].data[i];
        return {
          html: 'média das ' + row.length + ' componentes da linha ' + (i + 1) + ' da soma:<br>(' +
            row.map((v, c) => (c === 0 ? fmt(v) : par(v))).join(' + ') + ') / ' + row.length + ' ≈ <b>' +
            fmt(row.reduce((a, b) => a + b, 0) / row.length) + '</b>',
          marks: [[sId, 'row', i]],
        };
      };
    }
    function exStd(sId, ln) {
      return (i) => {
        const dev = mats[sId].data[i].map((v) => v - ln.mu[i]);
        const variance = ln.sigma[i] * ln.sigma[i];
        return {
          html: 'desvios da linha ' + (i + 1) + ' em relação a μ ≈ ' + fmt(ln.mu[i]) + ': [' + dev.map((v) => fmt(v)).join(', ') + ']<br>' +
            'σ² = (' + dev.map((v) => par(v) + '²').join(' + ') + ') / ' + dev.length + ' ≈ ' + fmt(variance) +
            '<br>σ = √' + fmt(variance) + ' ≈ <b>' + fmt(ln.sigma[i]) + '</b>',
          marks: [[sId, 'row', i], ['mu', 'cell', i, 0]],
        };
      };
    }
    function exLN(sId, ln) {
      return (i, j) => ({
        html: 'componente da soma menos a média da linha, dividido pelo desvio padrão da linha (γ = 1, β = 0):<br>(' +
          fmt(mats[sId].data[i][j]) + ' − ' + par(ln.mu[i]) + ') / ' + fmt(ln.sigma[i]) + ' ≈ <b>' +
          fmt((mats[sId].data[i][j] - ln.mu[i]) / ln.sigma[i]) + '</b>',
        marks: [[sId, 'cell', i, j], ['mu', 'cell', i, 0], ['sigma', 'cell', i, 0]],
      });
    }
    function exRelu(aId, wId, bId) {
      const inner = exMatmul(aId, wId, bId);
      return (i, j) => {
        const r = inner(i, j);
        return { html: r.html.replace(' ≈ <b>', ' ≈ ').replace('</b>', '') + '; ReLU = max(0, ·) = <b>' + fmt(mats.hid.data[i][j]) + '</b>', marks: r.marks };
      };
    }

    /* ── descrição de matriz: { id, name, data, rows, cols, dec, cls, heat, hi, explain } ── */
    const M = (id, name, data, o) => Object.assign({ id, name, data: Array.isArray(data[0]) ? data : [data] }, o || {});

    /* ── etapas ──────────────────────────────────────────── */
    function addNorm(block, title, aId, aName, a, bId, bName, b, ln, outName, rows, text) {
      return {
        title, block,
        text: text + ' A LayerNorm trabalha linha a linha, em três contas: a média <span class="formula">μ</span> das ' +
          '<span class="formula">d = 4</span> componentes, o desvio padrão <span class="formula">σ</span> e a normalização ' +
          '<span class="formula">(x − μ) / σ</span>. Neste modelo <span class="formula">γ = 1</span> e <span class="formula">β = 0</span>.',
        eqs: [
          [sym(aName + ' + ' + bName + ' ='), M(aId, aName, a, { rows }), sym('+'), M(bId, bName, b, {}), sym('='),
            M('sum', 'a soma', add2(a, b), { explain: exAdd(aId, bId) })],
          [sym('μ ='), M('mu', 'μ', ln.mu.map((v) => [v]), { rows, explain: exMean('sum') }),
            sym('σ ='), M('sigma', 'σ', ln.sigma.map((v) => [v]), { explain: exStd('sum', ln) })],
          [sym(outName + ' = (soma − μ) / σ ='), M('out', outName, ln.out, { rows, explain: exLN('sum', ln) })],
        ],
      };
    }
    const add2 = (A, B) => A.map((r, i) => r.map((v, j) => v + B[i][j]));

    function feedForward(block, side, inName, X, ff, rows, text) {
      return {
        title: 'Feed forward', block, text,
        eqs: [
          [sym('ReLU(' + inName + ' W<sub>1</sub> + b<sub>1</sub>) = ReLU('), M('in', inName, X, { rows }),
            M('w1', 'W<sub>1</sub>', W[side + '_W1'], { dec: 1, cls: 'w' }), sym('+'), M('b1', 'b<sub>1</sub>', W[side + '_b1'], { dec: 1, cls: 'w' }),
            sym(') ='), M('hid', 'a camada escondida', ff.hid, { explain: exRelu('in', 'w1', 'b1') })],
          [sym('FFN(' + inName + ') ='), M('hid2', 'a camada escondida', ff.hid, { rows }),
            M('w2', 'W<sub>2</sub>', W[side + '_W2'], { dec: 1, cls: 'w' }), sym('+'), M('b2', 'b<sub>2</sub>', W[side + '_b2'], { dec: 1, cls: 'w' }),
            sym('='), M('F', 'FFN', ff.F, { explain: exMatmul('hid2', 'w2', 'b2') })],
        ],
      };
    }

    function buildStages() {
      const enc = run.enc;
      const src = T.sentence();
      const dec = run.steps[decStep - 1].dec;
      const tgt = dec.idx.map((t) => T.tgtVocab[t]);
      const last = tgt.length - 1;
      const next = run.steps[decStep - 1];
      const srcHi = new Set(enc.idx), tgtHi = new Set(dec.idx);

      return [
        { title: 'Tokens, índices e embeddings', block: 'emb',
          text: 'A frase ' + src.map(tok).join(' ') + ' vira os índices [' + enc.idx.map((t) => t + 1).join(', ') +
            ']. Cada índice seleciona uma linha da matriz de embedding <span class="formula">E</span> da origem.',
          eqs: [
            [sym('E ='), M('E', 'E', W.E_src, { rows: T.srcVocab, dec: 1, cls: 'w', hi: srcHi })],
            [sym('X<sub>emb</sub> ='), M('emb', 'X<sub>emb</sub>', enc.emb, { rows: src, explain: exLookup('E', enc.idx, T.srcVocab) })],
          ] },
        { title: 'Positional encoding', block: 'pe',
          text: 'A posição <span class="formula">i</span> recebe o vetor <span class="formula">[sin(i), cos(i), sin(i/100), cos(i/100)]</span>, somado ao embedding do token que está nela.',
          eqs: [
            [sym('X = X<sub>emb</sub> + P ='), M('emb', 'X<sub>emb</sub>', enc.emb, { rows: src }), sym('+'),
              M('pe', 'P', enc.pe, { explain: exPE }), sym('='), M('X', 'X', enc.X, { explain: exAdd('emb', 'pe') })],
          ] },
        { title: 'Self-attention do encoder: projeções Q, K e V', block: 'att',
          text: 'A mesma matriz <span class="formula">X</span> é multiplicada por três matrizes de pesos. Cada linha de <span class="formula">Q</span>, <span class="formula">K</span> e <span class="formula">V</span> pertence a um token.',
          eqs: [
            [sym('Q = X W<sub>q</sub> ='), M('X', 'X', enc.X, { rows: src }), M('Wq', 'W<sub>q</sub>', W.enc_Wq, { dec: 1, cls: 'w' }), sym('='),
              M('Q', 'Q', enc.att.Q, { cls: 'q', explain: exMatmul('X', 'Wq') })],
            [sym('K = X W<sub>k</sub> ='), M('X2', 'X', enc.X, { rows: src }), M('Wk', 'W<sub>k</sub>', W.enc_Wk, { dec: 1, cls: 'w' }), sym('='),
              M('K', 'K', enc.att.K, { cls: 'k', explain: exMatmul('X2', 'Wk') })],
            [sym('V = X W<sub>v</sub> ='), M('X3', 'X', enc.X, { rows: src }), M('Wv', 'W<sub>v</sub>', W.enc_Wv, { dec: 1, cls: 'w' }), sym('='),
              M('V', 'V', enc.att.V, { cls: 'v', explain: exMatmul('X3', 'Wv') })],
          ] },
        { title: 'Self-attention do encoder: scores e pesos', block: 'att',
          text: 'O score entre dois tokens é o produto escalar da query de um com a key do outro, dividido por <span class="formula">√d = 2</span>. A softmax transforma cada linha de scores em pesos que somam 1.',
          eqs: [
            [sym('Q K<sup>⊤</sup> / √d :'), M('Q', 'Q', enc.att.Q, { rows: src, cls: 'q' }), sym('e'), M('K', 'K', enc.att.K, { rows: src, cls: 'k' }),
              sym('→'), M('S', 'os scores', enc.att.S, { rows: src, cols: src, explain: exScore('Q', 'K') })],
            [sym('A = softmax ='), M('A', 'A', enc.att.A, { rows: src, cols: src, heat: true, explain: exSoftmax('S') })],
          ] },
        { title: 'Self-attention do encoder: contexto', block: 'att',
          text: 'A saída da atenção é uma média dos values, ponderada pelos pesos de cada linha de <span class="formula">A</span>.',
          eqs: [
            [sym('C = A V ='), M('A', 'A', enc.att.A, { rows: src, heat: true }), M('V', 'V', enc.att.V, { cls: 'v' }), sym('='),
              M('C', 'C', enc.att.C, { explain: exMatmul('A', 'V') })],
          ] },
        addNorm('an1', 'Add & Norm depois da atenção', 'a', 'X', enc.X, 'b', 'C', enc.att.C, enc.ln1, 'H', src,
          'A conexão residual soma a entrada da atenção à saída dela. A LayerNorm deixa cada linha com média 0 e variância 1.'),
        feedForward('ff', 'enc', 'H', enc.H, enc.ff, src,
          'O mesmo MLP é aplicado a cada linha de <span class="formula">H</span>, sem misturar tokens.'),
        addNorm('an2', 'Add & Norm: a saída do encoder', 'a', 'H', enc.H, 'b', 'FFN(H)', enc.ff.F, enc.ln2, 'M', src,
          'A matriz <span class="formula">M</span> é a saída do encoder: um vetor contextualizado por token da origem. O decoder a consulta na cross-attention.'),

        { title: 'Entrada do decoder', block: 'demb',
          text: 'O decoder recebe o que já foi escrito: ' + tgt.map(tok).join(' ') + '. Cada token seleciona uma linha da matriz de embedding do alvo e recebe o positional encoding da sua posição.',
          eqs: [
            [sym('E<sup>alvo</sup> ='), M('E', 'E', W.E_tgt, { rows: T.tgtVocab, dec: 1, cls: 'w', hi: tgtHi })],
            [sym('Y ='), M('emb', 'o embedding', dec.emb, { rows: tgt, explain: exLookup('E', dec.idx, T.tgtVocab) }), sym('+'),
              M('pe', 'P', dec.pe, { explain: exPE }), sym('='), M('Y', 'Y', dec.Y, { explain: exAdd('emb', 'pe') })],
          ] },
        { title: 'Masked self-attention: projeções', block: 'dself',
          text: 'O decoder tem as suas próprias matrizes <span class="formula">W<sub>q</sub></span>, <span class="formula">W<sub>k</sub></span> e <span class="formula">W<sub>v</sub></span>, aplicadas a <span class="formula">Y</span>.',
          eqs: [
            [sym('Q = Y W<sub>q</sub> ='), M('Y', 'Y', dec.Y, { rows: tgt }), M('Wq', 'W<sub>q</sub>', W.dec_Wq, { dec: 1, cls: 'w' }), sym('='),
              M('Q', 'Q', dec.self.Q, { cls: 'q', explain: exMatmul('Y', 'Wq') })],
            [sym('K = Y W<sub>k</sub> ='), M('Y2', 'Y', dec.Y, { rows: tgt }), M('Wk', 'W<sub>k</sub>', W.dec_Wk, { dec: 1, cls: 'w' }), sym('='),
              M('K', 'K', dec.self.K, { cls: 'k', explain: exMatmul('Y2', 'Wk') })],
            [sym('V = Y W<sub>v</sub> ='), M('Y3', 'Y', dec.Y, { rows: tgt }), M('Wv', 'W<sub>v</sub>', W.dec_Wv, { dec: 1, cls: 'w' }), sym('='),
              M('V', 'V', dec.self.V, { cls: 'v', explain: exMatmul('Y3', 'Wv') })],
          ] },
        { title: 'Masked self-attention: máscara, pesos e contexto', block: 'dself',
          text: 'A máscara causal troca por −∞ o score de toda posição futura, então cada token só atende a si mesmo e aos anteriores.' +
            (last === 0 ? ' Com um único token, o peso é 1 e o contexto é o próprio value.' : ''),
          eqs: [
            [sym('scores com máscara:'), M('Q', 'Q', dec.self.Q, { rows: tgt, cls: 'q' }), sym('e'), M('K', 'K', dec.self.K, { rows: tgt, cls: 'k' }),
              sym('→'), M('S', 'os scores', dec.self.S, { rows: tgt, cols: tgt, explain: exScore('Q', 'K', true) })],
            [sym('A = softmax ='), M('A', 'A', dec.self.A, { rows: tgt, cols: tgt, heat: true, explain: exSoftmax('S') })],
            [sym('C = A V ='), M('A2', 'A', dec.self.A, { rows: tgt, heat: true }), M('V', 'V', dec.self.V, { cls: 'v' }), sym('='),
              M('C', 'C', dec.self.C, { explain: exMatmul('A2', 'V') })],
          ] },
        addNorm('dan1', 'Add & Norm depois da masked self-attention', 'a', 'Y', dec.Y, 'b', 'C', dec.self.C, dec.ln1, 'D<sub>1</sub>', tgt,
          'Residual e LayerNorm, como no encoder.'),
        { title: 'Cross-attention: projeções', block: 'dx',
          text: 'A query vem do decoder (<span class="formula">D<sub>1</sub></span>). As keys e os values vêm da saída do encoder (<span class="formula">M</span>), então têm uma linha por token da origem.',
          eqs: [
            [sym('Q = D<sub>1</sub> W<sub>q</sub> ='), M('D1', 'D<sub>1</sub>', dec.D1, { rows: tgt }), M('Wq', 'W<sub>q</sub>', W.x_Wq, { dec: 1, cls: 'w' }), sym('='),
              M('Q', 'Q', dec.cross.Q, { cls: 'q', explain: exMatmul('D1', 'Wq') })],
            [sym('K = M W<sub>k</sub> ='), M('M1', 'M', enc.M, { rows: src }), M('Wk', 'W<sub>k</sub>', W.x_Wk, { dec: 1, cls: 'w' }), sym('='),
              M('K', 'K', dec.cross.K, { cls: 'k', explain: exMatmul('M1', 'Wk') })],
            [sym('V = M W<sub>v</sub> ='), M('M2', 'M', enc.M, { rows: src }), M('Wv', 'W<sub>v</sub>', W.x_Wv, { dec: 1, cls: 'w' }), sym('='),
              M('V', 'V', dec.cross.V, { cls: 'v', explain: exMatmul('M2', 'Wv') })],
          ] },
        { title: 'Cross-attention: pesos e contexto', block: 'dx',
          text: 'Cada linha de <span class="formula">A</span> mostra para quais tokens da origem um token do alvo olha. A última linha decide o próximo token: compare a coluna de maior peso com a palavra que será escrita.',
          eqs: [
            [sym('Q K<sup>⊤</sup> / √d :'), M('Q', 'Q', dec.cross.Q, { rows: tgt, cls: 'q' }), sym('e'), M('K', 'K', dec.cross.K, { rows: src, cls: 'k' }),
              sym('→'), M('S', 'os scores', dec.cross.S, { rows: tgt, cols: src, explain: exScore('Q', 'K') })],
            [sym('A = softmax ='), M('A', 'A', dec.cross.A, { rows: tgt, cols: src, heat: true, explain: exSoftmax('S') })],
            [sym('C = A V ='), M('A2', 'A', dec.cross.A, { rows: tgt, heat: true }), M('V', 'V', dec.cross.V, { rows: src, cls: 'v' }), sym('='),
              M('C', 'C', dec.cross.C, { explain: exMatmul('A2', 'V') })],
          ] },
        addNorm('dan2', 'Add & Norm depois da cross-attention', 'a', 'D<sub>1</sub>', dec.D1, 'b', 'C', dec.cross.C, dec.ln2, 'D<sub>2</sub>', tgt,
          'Residual e LayerNorm sobre a saída da cross-attention.'),
        feedForward('dff', 'dec', 'D<sub>2</sub>', dec.D2, dec.ff, tgt,
          'O decoder tem o seu próprio MLP, com pesos diferentes dos do encoder.'),
        addNorm('dan3', 'Add & Norm: a saída do decoder', 'a', 'D<sub>2</sub>', dec.D2, 'b', 'FFN(D<sub>2</sub>)', dec.ff.F, dec.ln3, 'D<sub>3</sub>', tgt,
          'A última linha de <span class="formula">D<sub>3</sub></span> pertence à posição do token que será escrito.'),
        { title: 'Linear e softmax', block: 'out',
          text: 'A camada linear projeta cada linha de <span class="formula">D<sub>3</sub></span> para o tamanho do vocabulário do alvo, e a softmax transforma os logits em probabilidades. A última linha dá o próximo token: ' +
            tok(T.tgtVocab[next.tok]) + ', com probabilidade ' + fmt(next.prob) + '.',
          eqs: [
            [sym('logits = D<sub>3</sub> W<sub>out</sub> ='), M('D3', 'D<sub>3</sub>', dec.D3, { rows: tgt }),
              M('Wout', 'W<sub>out</sub>', W.W_out, { dec: 1, cls: 'w', cols: T.tgtVocab })],
            [sym('='), M('L', 'os logits', dec.logits, { rows: tgt, cols: T.tgtVocab, explain: exMatmul('D3', 'Wout') })],
            [sym('softmax ='), M('P', 'as probabilidades', dec.probs, { rows: tgt, cols: T.tgtVocab, heat: true, hi: new Set([last]), explain: exSoftmax('L') })],
          ] },
      ];
    }

    /* ── renderização ────────────────────────────────────── */
    function matrixEl(m) {
      const wrap = document.createElement('span');
      wrap.className = 'mx' + (m.cls ? ' mx-' + m.cls : '');
      if (m.rows) {
        const labels = document.createElement('span');
        labels.className = 'mx-rows' + (m.cols ? ' with-cols' : '');
        m.rows.forEach((r, i) => {
          const s = document.createElement('span');
          s.textContent = r;
          if (m.hi && m.hi.has(i)) s.className = 'hi';
          labels.appendChild(s);
        });
        wrap.appendChild(labels);
      }
      const table = document.createElement('table');
      if (m.cols) {
        const tr = document.createElement('tr');
        m.cols.forEach((c) => { const th = document.createElement('th'); th.textContent = c; tr.appendChild(th); });
        table.appendChild(tr);
      }
      m.data.forEach((row, i) => {
        const tr = document.createElement('tr');
        if (m.hi && m.hi.has(i)) tr.className = 'hi';
        row.forEach((v, j) => {
          const td = document.createElement('td');
          td.textContent = fmt(v, m.dec);
          td.dataset.m = m.id; td.dataset.i = i; td.dataset.j = j;
          if (m.heat) td.style.background = 'color-mix(in srgb, var(--purple) ' + Math.round(70 * v) + '%, transparent)';
          if (m.explain) {
            td.classList.add('has-explain');
            const show = () => showExplain(m, i, j, td);
            td.addEventListener('mouseenter', show);
            td.addEventListener('click', show);
          }
          tr.appendChild(td);
        });
        table.appendChild(tr);
      });
      const br = document.createElement('span');
      br.className = 'mx-bracket';
      br.appendChild(table);
      wrap.appendChild(br);
      return wrap;
    }

    function showExplain(m, i, j, td) {
      const r = m.explain(i, j);
      elStage.querySelectorAll('.mark, .picked').forEach((el) => el.classList.remove('mark', 'picked'));
      td.classList.add('picked');
      for (const mk of r.marks) {
        let sel = 'td[data-m="' + mk[0] + '"]';
        if (mk[1] === 'row') sel += '[data-i="' + mk[2] + '"]';
        else if (mk[1] === 'col') sel += '[data-j="' + mk[2] + '"]';
        else sel += '[data-i="' + mk[2] + '"][data-j="' + mk[3] + '"]';
        elStage.querySelectorAll(sel).forEach((el) => el.classList.add('mark'));
      }
      elExplain.innerHTML = '<b>' + m.name + '</b>, linha ' + (i + 1) + ', coluna ' + (j + 1) + ': ' + r.html;
    }

    function render() {
      const stages = buildStages();
      stage = Math.max(0, Math.min(stages.length - 1, stage));
      const st = stages[stage];
      mats = {};
      elStage.innerHTML = '';
      const p = document.createElement('p');
      p.className = 'demo-text';
      p.innerHTML = st.text;
      elStage.appendChild(p);
      for (const eq of st.eqs) {
        const row = document.createElement('div');
        row.className = 'eq';
        for (const item of eq) {
          if (item.sym !== undefined) {
            const s = document.createElement('span');
            s.className = 'formula';
            s.innerHTML = item.sym;
            row.appendChild(s);
          } else {
            mats[item.id] = item;
            row.appendChild(matrixEl(item));
          }
        }
        elStage.appendChild(row);
      }
      elTitle.textContent = 'Etapa ' + (stage + 1) + ' de ' + stages.length + ' · ' + st.title;
      elExplain.textContent = HINT;
      $('s1-prev').disabled = stage === 0;
      $('s1-next').disabled = stage === stages.length - 1;
      elDiagram.querySelectorAll('.blk').forEach((g) => g.classList.toggle('active', g.dataset.block === st.block));
      elReadout.innerHTML = 'Origem: ' + T.sentence().map(tok).join(' ') + ' · tradução do modelo: ' +
        run.written.slice(1).map((t) => tok(T.tgtVocab[t])).join(' ');
    }

    function renderDecTabs() {
      elSteps.innerHTML = '';
      run.steps.forEach((s, i) => {
        const b = document.createElement('button');
        b.type = 'button';
        b.className = 'tab' + (i + 1 === decStep ? ' active' : '');
        b.textContent = (i + 1) + ' · ' + s.dec.idx.map((t) => T.tgtVocab[t]).join(' ');
        b.addEventListener('click', () => { decStep = i + 1; renderDecTabs(); render(); });
        elSteps.appendChild(b);
      });
    }

    function recompute() {
      run = T.generate(T.sentenceIdx(), 4);
      decStep = Math.min(decStep, run.steps.length);
      renderDecTabs();
      render();
    }

    /* ── diagrama da arquitetura ─────────────────────────── */
    const BLOCKS = [
      /* id, rótulo, x, y, largura, classe */
      ['emb', 'Input Embedding', 14, 372, 120, 'b-emb'],
      ['pe', '+ Positional Enc.', 14, 336, 120, 'b-pe'],
      ['att', 'Self-Attention', 14, 290, 120, 'b-att'],
      ['an1', 'Add & Norm', 14, 258, 120, 'b-an'],
      ['ff', 'Feed Forward', 14, 212, 120, 'b-ff'],
      ['an2', 'Add & Norm', 14, 180, 120, 'b-an'],
      ['demb', 'Output Emb. + PE', 166, 372, 120, 'b-emb'],
      ['dself', 'Masked Self-Att.', 166, 322, 120, 'b-att'],
      ['dan1', 'Add & Norm', 166, 290, 120, 'b-an'],
      ['dx', 'Cross-Attention', 166, 244, 120, 'b-att'],
      ['dan2', 'Add & Norm', 166, 212, 120, 'b-an'],
      ['dff', 'Feed Forward', 166, 166, 120, 'b-ff'],
      ['dan3', 'Add & Norm', 166, 134, 120, 'b-an'],
      ['out', 'Linear + Softmax', 166, 84, 120, 'b-out'],
    ];
    function buildDiagram() {
      let svg = '<svg viewBox="0 0 300 430" role="img" aria-label="Arquitetura encoder-decoder com o bloco da etapa em destaque">';
      svg += '<path class="wire" d="M74 398 V180 M226 398 V84 M74 180 V156 H150 V257 H166" />';
      svg += '<text class="cap" x="74" y="418">origem</text><text class="cap" x="226" y="418">já escrito</text>';
      svg += '<text class="cap" x="226" y="70">próximo token</text>';
      svg += '<text class="cap" x="74" y="146">Encoder</text><text class="cap" x="226" y="52">Decoder</text>';
      for (const b of BLOCKS) {
        svg += '<g class="blk ' + b[5] + '" data-block="' + b[0] + '"><rect x="' + b[2] + '" y="' + b[3] + '" width="' + b[4] +
          '" height="26" rx="4" /><text x="' + (b[2] + b[4] / 2) + '" y="' + (b[3] + 17) + '">' + b[1].replace('&', '&amp;') + '</text></g>';
      }
      elDiagram.innerHTML = svg + '</svg>';
      elDiagram.querySelectorAll('.blk').forEach((g) => g.addEventListener('click', () => {
        const idx = buildStages().findIndex((s) => s.block === g.dataset.block);
        if (idx >= 0) { stage = idx; render(); }
      }));
    }

    /* ── eventos ─────────────────────────────────────────── */
    $('s1-prev').addEventListener('click', () => { stage--; render(); });
    $('s1-next').addEventListener('click', () => { stage++; render(); });
    $('s1-reset').addEventListener('click', () => { stage = 0; decStep = 1; renderDecTabs(); render(); });
    T.bindSentenceTabs('s1');
    T.onSentence(recompute);
    buildDiagram();
    recompute();
  }

  DL.sections.push({ name: 'steps', init });
})();
