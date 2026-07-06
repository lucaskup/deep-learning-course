/* Seção 2: greedy vs beam search em uma árvore de busca interativa. */
(function () {
  'use strict';
  window.DM = window.DM || {};
  DM.sections = DM.sections || [];

  const EXPAND_K = 3, MAX_DEPTH = 4;

  /* Árvore do exemplo dos slides (probabilidades condicionais). */
  function presetTree() {
    const mk = (token, p, children) => ({ token, p, children: children || [] });
    return {
      token: 'The', p: 1, children: [
        mk('dog', 0.4, [mk('has', 0.9), mk('and', 0.05), mk('runs', 0.05)]),
        mk('nice', 0.5, [mk('woman', 0.4), mk('house', 0.3), mk('guy', 0.3)]),
        mk('car', 0.1, []),
      ],
    };
  }

  function init() {
    const M = DM.model, P = DM.plot;
    const $ = (id) => document.getElementById(id);
    const f = M.fmtBR;
    const canvas = $('s2-tree');

    let mode = 'preset';
    let b = 2, alpha = 0;
    let ngramDepth = 1;
    let root = null;

    const STARTS = ['o gato', 'o cachorro', 'a menina', 'o professor', 'o aluno', 'a rede neural', 'o modelo', 'a turma'];
    const sel = $('s2-start');
    for (const s of STARTS) {
      const opt = document.createElement('option');
      opt.value = s; opt.textContent = s;
      sel.appendChild(opt);
    }

    /* Anota a árvore nível a nível: top-b da fronteira por Σ ln p vira
       'beam', o resto 'pruned'. Nós com <eos> ficam guardados como
       candidatos e não competem mais pelo beam. Só nós 'beam' expandem. */
    function annotate(rootNode, expandFn) {
      rootNode.logp = 0; rootNode.depth = 0; rootNode.status = 'beam'; rootNode.finished = false;
      let frontier = [rootNode];
      for (let lvl = 1; lvl <= MAX_DEPTH; lvl++) {
        const next = [];
        for (const node of frontier) {
          if (node.status !== 'beam' || node.finished) continue;
          if (expandFn && node.children.length === 0 && lvl <= ngramDepth) expandFn(node);
          for (const child of node.children) {
            child.logp = node.logp + Math.log(child.p);
            child.depth = lvl;
            child.finished = child.token === M.EOS || false;
            child.children = child.children || [];
            child.seq = child.seq || [...node.seq, child.token];
            next.push(child);
          }
        }
        if (next.length === 0) break;
        next.sort((a, c) => c.logp - a.logp);
        let kept = 0;
        for (const n of next) {
          if (n.finished) { n.status = 'beam'; continue; }
          n.status = kept < b ? 'beam' : 'pruned';
          if (n.status === 'beam') kept++;
        }
        frontier = next;
      }
      /* caminho greedy: argmax local em cada passo (só entre nós exibidos) */
      let g = rootNode;
      g.greedy = true;
      while (vis(g).length > 0) {
        g = vis(g).reduce((best, c) => (c.p > best.p ? c : best), vis(g)[0]);
        g.greedy = true;
      }
    }

    /* Filhos anotados: na árvore do preset, filhos de nós podados nunca
       são expandidos e não devem aparecer. */
    function vis(node) {
      return node.children.filter((c) => c.status);
    }

    function expandNgram(node) {
      const dist = M.nextDist(node.seq, true).slice(0, EXPAND_K);
      node.children = dist.map((d) => ({
        token: d.token, p: d.p,
        seq: [...node.seq, d.token],
        children: [],
      }));
    }

    function rebuild() {
      if (mode === 'preset') {
        root = presetTree();
        root.seq = ['The'];
        annotate(root, null);
      } else {
        const startToks = M.tokenize(sel.value);
        root = { token: sel.value, p: 1, seq: startToks, children: [] };
        annotate(root, expandNgram);
      }
      draw();
      drawTable();
    }

    /* ── layout e desenho ─────────────────────────────────── */
    function layoutTree() {
      const leaves = { n: 0 };
      let maxDepth = 0;
      (function place(node) {
        maxDepth = Math.max(maxDepth, node.depth);
        const kids = vis(node);
        if (kids.length === 0) {
          node.row = leaves.n++;
        } else {
          for (const c of kids) place(c);
          node.row = kids.reduce((a, c) => a + c.row, 0) / kids.length;
        }
      })(root);
      return { numLeaves: leaves.n, maxDepth: Math.max(1, maxDepth) };
    }

    function pill(ctx, cx, cy, text, opts) {
      const th = P.theme();
      ctx.font = '10px sans-serif';
      const w = ctx.measureText(text).width + 12, h = 16;
      ctx.fillStyle = th.card;
      ctx.strokeStyle = opts.border;
      ctx.lineWidth = opts.greedy ? 2 : 1;
      if (opts.dash) ctx.setLineDash([3, 3]);
      ctx.beginPath();
      if (ctx.roundRect) ctx.roundRect(cx - w / 2, cy - h / 2, w, h, 5);
      else ctx.rect(cx - w / 2, cy - h / 2, w, h);
      ctx.fill();
      ctx.stroke();
      ctx.setLineDash([]);
      ctx.fillStyle = opts.faded ? th.comment : th.fg;
      ctx.textAlign = 'center';
      ctx.fillText(text, cx, cy + 3.5);
      return w;
    }

    function draw() {
      const th = P.theme();
      const { ctx, w, h } = P.setup(canvas);
      P.clear(ctx, w, h);
      const { numLeaves, maxDepth } = layoutTree();
      const padL = 16, padR = 70, padT = 14, padB = 10;
      const colW = (w - padL - padR) / maxDepth;
      const rowH = (h - padT - padB) / Math.max(1, numLeaves);

      (function pos(node) {
        node.px = padL + node.depth * colW + 30;
        node.py = padT + (node.row + 0.5) * rowH;
        for (const c of vis(node)) pos(c);
      })(root);

      // arestas primeiro
      (function edges(node) {
        for (const c of vis(node)) {
          const color = c.greedy ? th.green : c.status === 'beam' ? th.purple : th.comment;
          ctx.strokeStyle = color;
          ctx.lineWidth = c.greedy ? 2.4 : c.status === 'beam' ? 1.8 : 1;
          if (c.status === 'pruned' && !c.greedy) ctx.setLineDash([4, 3]);
          ctx.beginPath();
          ctx.moveTo(node.px + 24, node.py);
          ctx.lineTo(c.px - 26, c.py);
          ctx.stroke();
          ctx.setLineDash([]);
          edges(c);
        }
      })(root);

      // nós
      (function nodes(node) {
        const isRoot = node.depth === 0;
        const text = isRoot ? node.token : `${node.token} ${f(node.p, 2)}`;
        const border = node.greedy ? th.green : node.status === 'beam' ? th.purple : th.line;
        pill(ctx, node.px, node.py, text, {
          border,
          greedy: node.greedy,
          dash: node.status === 'pruned' && !node.greedy,
          faded: node.status === 'pruned' && !node.greedy,
        });
        // candidatos (folhas do beam) ganham o produto acumulado logo abaixo
        if (vis(node).length === 0 && node.status === 'beam' && !isRoot) {
          const prod = Math.exp(node.logp);
          P.label(ctx, node.px, node.py + 19, '∏=' + (prod >= 0.001 ? f(prod, 2) : prod.toExponential(1).replace('.', ',')), th.purple, 'center');
        }
        for (const c of vis(node)) nodes(c);
      })(root);
    }

    /* ── tabela de candidatos com penalidade por comprimento ── */
    function collectCandidates(node, out) {
      if (node.depth > 0 && node.status === 'beam' && (node.finished || vis(node).length === 0)) {
        out.push(node);
      }
      for (const c of vis(node)) collectCandidates(c, out);
      return out;
    }

    function drawTable() {
      const cands = collectCandidates(root, []);
      if (cands.length === 0) { $('s2-table').innerHTML = ''; return; }
      const scored = cands.map((c) => ({
        text: c.seq.join(' ').replace(/</g, '&lt;').replace(/>/g, '&gt;'),
        n: c.depth,
        prod: Math.exp(c.logp),
        logp: c.logp,
        greedy: c.greedy && vis(c).length === 0,
        score: c.logp / Math.pow(c.depth, alpha),
      })).sort((a, c) => c.score - a.score);
      let html = '<table class="metric-table"><tr><th>candidato</th><th>n</th><th>∏ P</th><th>Σ ln P</th><th>score = Σ ln P / n<sup>α</sup></th></tr>';
      scored.forEach((c, i) => {
        const name = (i === 0 ? '🏆 ' : '') + c.text + (c.greedy ? ' <span style="color:var(--green)">(greedy)</span>' : '');
        const prod = c.prod >= 0.001 ? f(c.prod, 3) : c.prod.toExponential(2).replace('.', ',');
        html += `<tr${i === 0 ? ' style="font-weight:700"' : ''}><td class="left">${name}</td>` +
          `<td>${c.n}</td><td>${prod}</td><td>${f(c.logp, 2)}</td><td>${f(c.score, 2)}</td></tr>`;
      });
      html += '</table>';
      if (mode === 'preset') {
        html += '<p class="demo-text small">No exemplo dos slides todos os candidatos têm n = 2, então α não muda a ordem. ' +
          'Troque para o modelo n-gram e expanda até aparecer um candidato com &lt;eos&gt; para ver α reordenar.</p>';
      }
      $('s2-table').innerHTML = html;
    }

    /* ── controles ────────────────────────────────────────── */
    $('s2-tab-preset').addEventListener('click', () => setModeUI('preset'));
    $('s2-tab-ngram').addEventListener('click', () => setModeUI('ngram'));
    function setModeUI(m2) {
      mode = m2;
      $('s2-tab-preset').classList.toggle('active', mode === 'preset');
      $('s2-tab-ngram').classList.toggle('active', mode === 'ngram');
      for (const id of ['s2-start-wrap', 's2-expand', 's2-reset']) {
        $(id).classList.toggle('hidden', mode === 'preset');
      }
      if (mode === 'ngram') ngramDepth = 1;
      rebuild();
    }

    $('s2-b').addEventListener('input', (e) => {
      b = +e.target.value;
      $('s2-b-val').textContent = b;
      rebuild();
    });
    $('s2-alpha').addEventListener('input', (e) => {
      alpha = +e.target.value;
      $('s2-alpha-val').textContent = f(alpha, 2);
      drawTable();
    });
    $('s2-expand').addEventListener('click', () => {
      if (ngramDepth < MAX_DEPTH) { ngramDepth++; rebuild(); }
    });
    $('s2-reset').addEventListener('click', () => { ngramDepth = 1; rebuild(); });
    sel.addEventListener('change', () => { ngramDepth = 1; rebuild(); });

    rebuild();
    P.onRedraw(draw);
    P.observeResize(canvas, draw);
  }

  DM.sections.push({ name: 's2-beam', init });
})();
