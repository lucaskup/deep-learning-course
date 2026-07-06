/* Seção 3: memorização literal medida pelo ROUGE-L (maior subsequência comum).
   LCS por programação dinâmica entre a referência memorizada e a geração do modelo;
   daí precisão, recall e F1, mais um indicador de vazamento acima de um limiar. */
(function () {
  'use strict';
  window.DL = window.DL || {};
  DL.sections = DL.sections || [];

  const REF_TEXT = 'a chave secreta do servidor é zebra verde 1987';
  const DEFAULT_GEN = 'a chave secreta do servidor é banana';
  const LEAK_GEN = 'a chave secreta do servidor é zebra verde 1987';
  const NOLEAK_GEN = 'o tempo está ensolarado hoje na praia';
  const THRESHOLD = 0.5;
  const fm = (v, d) => v.toFixed(d == null ? 2 : d).replace('.', ',');

  function tokenize(s) {
    return s.trim().split(/\s+/).filter((t) => t.length > 0);
  }

  /* LCS por DP, devolvendo o comprimento e os pares de índices casados (i, j). */
  function lcs(a, b) {
    const n = a.length, m = b.length;
    const dp = [];
    for (let i = 0; i <= n; i++) dp.push(new Int32Array(m + 1));
    for (let i = 1; i <= n; i++) {
      for (let j = 1; j <= m; j++) {
        dp[i][j] = a[i - 1] === b[j - 1]
          ? dp[i - 1][j - 1] + 1
          : Math.max(dp[i - 1][j], dp[i][j - 1]);
      }
    }
    const pairs = [];
    let i = n, j = m;
    while (i > 0 && j > 0) {
      if (a[i - 1] === b[j - 1]) { pairs.push([i - 1, j - 1]); i--; j--; }
      else if (dp[i - 1][j] >= dp[i][j - 1]) i--;
      else j--;
    }
    pairs.reverse();
    return { len: dp[n][m], pairs };
  }

  function init() {
    if (!document.getElementById('s3-align')) return;
    const P = DL.plot;
    const $ = (id) => document.getElementById(id);

    const cv = $('s3-align'), inp = $('s3-gen');
    const refTok = tokenize(REF_TEXT);

    $('s3-ref-text').textContent = REF_TEXT;
    inp.value = DEFAULT_GEN;

    function metrics() {
      const genTok = tokenize(inp.value);
      const L = lcs(refTok, genTok);
      const prec = genTok.length ? L.len / genTok.length : 0;
      const rec = refTok.length ? L.len / refTok.length : 0;
      const f1 = (prec + rec) > 0 ? 2 * prec * rec / (prec + rec) : 0;
      return { genTok, L, prec, rec, f1 };
    }

    /* Desenha uma fileira de tokens centrada verticalmente em cy. */
    function drawRow(ctx, tokens, matchedSet, cy, baseColor, hiColor, th) {
      const rect = cv.getBoundingClientRect();
      const w = rect.width;
      const pad = 14;
      const inner = w - 2 * pad;
      const n = Math.max(1, tokens.length);
      const slot = inner / n;
      const centers = [];
      ctx.font = '12px ui-monospace, SFMono-Regular, Menlo, Consolas, monospace';
      for (let k = 0; k < tokens.length; k++) {
        const cx = pad + slot * (k + 0.5);
        centers.push(cx);
        const tw = ctx.measureText(tokens[k]).width;
        const bw = Math.min(slot - 6, tw + 14);
        const bh = 22;
        const matched = matchedSet.has(k);
        ctx.fillStyle = matched ? hiColor : th.card;
        ctx.globalAlpha = matched ? 0.22 : 1;
        roundRect(ctx, cx - bw / 2, cy - bh / 2, bw, bh, 5);
        ctx.fill();
        ctx.globalAlpha = 1;
        ctx.strokeStyle = matched ? hiColor : th.line;
        ctx.lineWidth = matched ? 1.6 : 1;
        roundRect(ctx, cx - bw / 2, cy - bh / 2, bw, bh, 5);
        ctx.stroke();
        ctx.fillStyle = matched ? hiColor : th.fg;
        ctx.textAlign = 'center';
        ctx.textBaseline = 'middle';
        ctx.fillText(tokens[k], cx, cy + 1);
      }
      ctx.textBaseline = 'alphabetic';
      return centers;
    }

    function roundRect(ctx, x, y, w, h, r) {
      ctx.beginPath();
      ctx.moveTo(x + r, y);
      ctx.arcTo(x + w, y, x + w, y + h, r);
      ctx.arcTo(x + w, y + h, x, y + h, r);
      ctx.arcTo(x, y + h, x, y, r);
      ctx.arcTo(x, y, x + w, y, r);
      ctx.closePath();
    }

    function redraw() {
      const th = P.theme();
      const { ctx, w, h } = P.setup(cv);
      P.clear(ctx, w, h);
      const M = metrics();

      const matchRef = new Set(M.L.pairs.map((p) => p[0]));
      const matchGen = new Set(M.L.pairs.map((p) => p[1]));
      const yRef = 44, yGen = h - 44;

      P.mathText(ctx, 'referência θ a esquecer', 14, 22, th.comment, 'left', 11);
      P.mathText(ctx, 'gerado pelo modelo', 14, h - 14, th.comment, 'left', 11);

      const cRef = drawRow(ctx, refTok, matchRef, yRef, th.fg, th.green, th);
      const cGen = drawRow(ctx, M.genTok, matchGen, yGen, th.fg, th.green, th);

      /* Linhas ligando os pares casados (a maior subsequência comum). */
      ctx.strokeStyle = th.green;
      ctx.lineWidth = 1.4;
      ctx.globalAlpha = 0.75;
      for (const [i, j] of M.L.pairs) {
        if (cRef[i] == null || cGen[j] == null) continue;
        ctx.beginPath();
        ctx.moveTo(cRef[i], yRef + 11);
        ctx.lineTo(cGen[j], yGen - 11);
        ctx.stroke();
      }
      ctx.globalAlpha = 1;
    }

    function updateReadout() {
      const M = metrics();
      $('s3-readout').textContent =
        'LCS = ' + M.L.len + ' · precisão = ' + fm(M.prec) +
        ' · recall = ' + fm(M.rec) + ' · F1 = ' + fm(M.f1);
      const badge = $('s3-verdict');
      const leak = M.f1 >= THRESHOLD;
      badge.textContent = leak ? 'vazamento provável' : 'sem vazamento';
      badge.className = 'un-badge ' + (leak ? 'warn' : 'ok');
    }

    function refreshAll() { redraw(); updateReadout(); }

    inp.addEventListener('input', refreshAll);
    $('s3-leak').addEventListener('click', () => { inp.value = LEAK_GEN; refreshAll(); });
    $('s3-noleak').addEventListener('click', () => { inp.value = NOLEAK_GEN; refreshAll(); });

    refreshAll();
    P.onRedraw(refreshAll);
    P.observeResize(cv, refreshAll);
  }

  DL.sections.push({ name: 's3-rougel', init });
})();
