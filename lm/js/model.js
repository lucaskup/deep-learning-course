/* Modelo de linguagem n-gram (contagens), filtros de amostragem e métricas.
   Sem DOM: também roda em Node para os testes. */
(function () {
  'use strict';
  window.DM = window.DM || {};

  const BOS = '<bos>', EOS = '<eos>', UNK = '<unk>';
  const L3 = 0.6, L2 = 0.3, L1 = 0.1, ALPHA = 0.1;

  function tokenize(str) {
    return str.toLowerCase().replace(/[.,!?;:"()]/g, ' ').trim().split(/\s+/).filter(Boolean);
  }

  /* ── treino por contagem: unigrama, bigrama, trigrama ───── */
  const uni = new Map();
  const bi = new Map(), biTot = new Map();
  const tri = new Map(), triTot = new Map();
  let N = 0;
  let vocab = [];

  function bump(map, totMap, key, w) {
    let m = map.get(key);
    if (!m) { m = new Map(); map.set(key, m); }
    m.set(w, (m.get(w) || 0) + 1);
    totMap.set(key, (totMap.get(key) || 0) + 1);
  }

  function train(corpus) {
    for (const sentence of corpus) {
      const toks = [BOS, BOS, ...tokenize(sentence), EOS];
      for (let i = 2; i < toks.length; i++) {
        const w = toks[i];
        uni.set(w, (uni.get(w) || 0) + 1);
        N++;
        bump(bi, biTot, toks[i - 1], w);
        bump(tri, triTot, toks[i - 2] + ' ' + toks[i - 1], w);
      }
    }
    vocab = [...uni.keys(), UNK];
  }

  /* p(w | u, v) interpolada; redistribui λ quando o histórico não existe,
     para que Σ_w p(w) = 1 sempre. */
  function probOf(w, u, v) {
    const triKey = u + ' ' + v;
    const tTot = triTot.get(triKey) || 0;
    const bTot = biTot.get(v) || 0;
    let l3 = L3, l2 = L2, l1 = L1;
    if (tTot === 0) { l2 += l3; l3 = 0; }
    if (bTot === 0) { l1 += l2; l2 = 0; }
    let p = l1 * ((uni.get(w) || 0) + ALPHA) / (N + ALPHA * vocab.length);
    if (l2 > 0) p += l2 * ((bi.get(v)?.get(w) || 0) / bTot);
    if (l3 > 0) p += l3 * ((tri.get(triKey)?.get(w) || 0) / tTot);
    return p;
  }

  function history(ctx) {
    const c = [BOS, BOS, ...ctx];
    return [c[c.length - 2], c[c.length - 1]];
  }

  /* Distribuição completa do próximo token, ordenada por p desc.
     forGeneration: remove <unk> e renormaliza. */
  function nextDist(ctx, forGeneration) {
    const [u, v] = history(ctx);
    let dist = vocab.map((w) => ({ token: w, p: probOf(w, u, v) }));
    if (forGeneration) {
      dist = dist.filter((d) => d.token !== UNK);
      const s = dist.reduce((a, d) => a + d.p, 0);
      for (const d of dist) d.p /= s;
    }
    dist.sort((a, b) => b.p - a.p);
    return dist;
  }

  /* Filtros de decoding, na ordem: temperatura → top-k → top-p.
     Devolve TODOS os tokens anotados para a UI desenhar os excluídos. */
  function filterDist(dist, opts) {
    const tau = opts.tau || 1, topK = opts.topK || Infinity, topP = opts.topP != null ? opts.topP : 1;
    // temperatura sobre os logits ln p: p^(1/τ) renormalizado
    let sTau = 0;
    const out = dist.map((d) => {
      const pt = Math.pow(d.p, 1 / tau);
      sTau += pt;
      return { token: d.token, pRaw: d.p, pTau: pt, pFinal: 0, excluded: false, by: null };
    });
    for (const d of out) d.pTau /= sTau;
    out.sort((a, b) => b.pTau - a.pTau);
    // top-k
    for (let i = 0; i < out.length; i++) {
      if (i >= topK) { out[i].excluded = true; out[i].by = 'topk'; }
    }
    // top-p (núcleo) sobre os sobreviventes do top-k, massa renormalizada
    const surv = out.filter((d) => !d.excluded);
    const survMass = surv.reduce((a, d) => a + d.pTau, 0);
    let cum = 0;
    for (let i = 0; i < surv.length; i++) {
      if (i > 0 && cum >= topP - 1e-12) { surv[i].excluded = true; surv[i].by = 'topp'; }
      cum += surv[i].pTau / survMass;
    }
    // renormaliza os finais
    const kept = out.filter((d) => !d.excluded);
    const keptMass = kept.reduce((a, d) => a + d.pTau, 0);
    for (const d of kept) d.pFinal = d.pTau / keptMass;
    return out;
  }

  function sampleFrom(filtered, rng) {
    const kept = filtered.filter((d) => !d.excluded);
    let r = rng();
    for (const d of kept) {
      r -= d.pFinal;
      if (r <= 0) return d;
    }
    return kept[kept.length - 1];
  }

  function argmaxOf(filtered) {
    return filtered.filter((d) => !d.excluded)[0];
  }

  /* Avaliação de uma frase pelo modelo: linhas com p(y_t | y_<t)
     por token (OOV → <unk>), incluindo <eos> como último token. */
  function scoreSentence(tokens) {
    const mapped = tokens.map((t) => (uni.has(t) ? t : UNK));
    const rows = [];
    for (let i = 0; i <= mapped.length; i++) {
      const target = i < mapped.length ? mapped[i] : EOS;
      const ctx = mapped.slice(0, i);
      const [u, v] = history(ctx);
      rows.push({
        shown: i < mapped.length ? tokens[i] : EOS,
        mappedTo: i < mapped.length && mapped[i] !== tokens[i] ? UNK : null,
        p: probOf(target, u, v),
      });
    }
    return rows;
  }

  /* Perplexidade = 2^(−L/m), L = Σ log2 p. */
  function perplexity(ps) {
    const logs = ps.map((p) => Math.log2(p));
    const L = logs.reduce((a, b) => a + b, 0);
    const m = ps.length;
    return { logs, L, m, exponent: -L / m, ppl: Math.pow(2, -L / m) };
  }

  /* ── métricas de texto (puras, sem modelo) ──────────────── */
  function ngrams(tokens, n) {
    const out = [];
    for (let i = 0; i + n <= tokens.length; i++) out.push(tokens.slice(i, i + n).join(' '));
    return out;
  }

  function countMap(arr) {
    const m = new Map();
    for (const x of arr) m.set(x, (m.get(x) || 0) + 1);
    return m;
  }

  /* BLEU com precisões clipadas e penalidade por brevidade. */
  function bleu(refToks, candToks, Nmax) {
    Nmax = Nmax || 4;
    const rows = [];
    let logSum = 0, anyZero = false;
    for (let n = 1; n <= Nmax; n++) {
      const cand = countMap(ngrams(candToks, n));
      const ref = countMap(ngrams(refToks, n));
      let matched = 0, total = 0;
      const details = [];
      for (const [g, c] of cand) {
        const clip = Math.min(c, ref.get(g) || 0);
        matched += clip;
        total += c;
        if (clip > 0) details.push({ gram: g, count: c, clip });
      }
      const pn = total > 0 ? matched / total : 0;
      if (pn === 0) anyZero = true; else logSum += Math.log(pn);
      rows.push({ n, matched, total, pn, details });
    }
    const c = candToks.length, r = refToks.length;
    const bp = c > r ? 1 : (c === 0 ? 0 : Math.exp(1 - r / c));
    const score = anyZero ? 0 : bp * Math.exp(logSum / Nmax);
    return { rows, c, r, bp, score, anyZero };
  }

  /* ROUGE-N (clipado) e ROUGE-L (LCS com backtrack p/ destaque). */
  function rougeN(refToks, candToks, n) {
    const ref = countMap(ngrams(refToks, n));
    const cand = countMap(ngrams(candToks, n));
    let overlap = 0;
    for (const [g, c] of cand) overlap += Math.min(c, ref.get(g) || 0);
    const refTot = Math.max(0, refToks.length - n + 1);
    const candTot = Math.max(0, candToks.length - n + 1);
    const R = refTot > 0 ? overlap / refTot : 0;
    const P = candTot > 0 ? overlap / candTot : 0;
    const F1 = R + P > 0 ? 2 * P * R / (P + R) : 0;
    return { overlap, refTot, candTot, R, P, F1 };
  }

  function lcs(refToks, candToks) {
    const m = refToks.length, n = candToks.length;
    const L = Array.from({ length: m + 1 }, () => new Array(n + 1).fill(0));
    for (let i = 1; i <= m; i++)
      for (let j = 1; j <= n; j++)
        L[i][j] = refToks[i - 1] === candToks[j - 1]
          ? L[i - 1][j - 1] + 1
          : Math.max(L[i - 1][j], L[i][j - 1]);
    // backtrack: índices casados em cada sequência
    const refIdx = [], candIdx = [];
    let i = m, j = n;
    while (i > 0 && j > 0) {
      if (refToks[i - 1] === candToks[j - 1]) { refIdx.push(i - 1); candIdx.push(j - 1); i--; j--; }
      else if (L[i - 1][j] >= L[i][j - 1]) i--;
      else j--;
    }
    refIdx.reverse(); candIdx.reverse();
    return { length: L[m][n], refIdx, candIdx };
  }

  function rougeL(refToks, candToks) {
    const { length, refIdx, candIdx } = lcs(refToks, candToks);
    const R = refToks.length > 0 ? length / refToks.length : 0;
    const P = candToks.length > 0 ? length / candToks.length : 0;
    const F1 = R + P > 0 ? 2 * P * R / (P + R) : 0;
    return { length, refIdx, candIdx, R, P, F1 };
  }

  /* Contagens cruas que compõem p(w | u, v) para um contexto, para a UI
     mostrar de onde a probabilidade interpolada vem. */
  function contextCounts(ctx, topN) {
    const [u, v] = history(ctx);
    const triKey = u + ' ' + v;
    const tTot = triTot.get(triKey) || 0;
    const bTot = biTot.get(v) || 0;
    const triMap = tri.get(triKey), biMap = bi.get(v);
    const rows = vocab
      .filter((w) => w !== UNK)
      .map((w) => ({
        token: w,
        cTri: triMap?.get(w) || 0,
        cBi: biMap?.get(w) || 0,
        cUni: uni.get(w) || 0,
        p: probOf(w, u, v),
      }))
      .sort((a, b2) => b2.p - a.p)
      .slice(0, topN || 12);
    return { u, v, tTot, bTot, N, rows };
  }

  function softmax(scores) {
    const mx = Math.max(...scores);
    const ex = scores.map((s) => Math.exp(s - mx));
    const sum = ex.reduce((a, b) => a + b, 0);
    return ex.map((e) => e / sum);
  }

  /* Números no padrão pt-BR (vírgula decimal). */
  function fmtBR(x, d) {
    return x.toFixed(d).replace('.', ',');
  }

  train((window.DM && DM.CORPUS) || []);

  DM.model = {
    BOS, EOS, UNK,
    tokenize, nextDist, probOf, filterDist, sampleFrom, argmaxOf,
    scoreSentence, perplexity,
    ngrams, bleu, rougeN, rougeL, lcs, softmax, fmtBR,
    uni, vocabSize: () => vocab.length, contextCounts,
  };
})();
