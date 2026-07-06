/* Helpers da demo de tokenização: corpus recorrente da aula, treinamento
   de BPE (critério de frequência) e de WordPiece (critério de likelihood),
   além da codificação de palavras novas com a tabela de merges aprendida. */
(function () {
  'use strict';
  window.DL = window.DL || {};

  const EOW = '_';

  /* Corpus recorrente dos slides: palavra e frequência. */
  const CORPUS = [
    { word: 'low', freq: 5 },
    { word: 'lower', freq: 2 },
    { word: 'newest', freq: 6 },
    { word: 'widest', freq: 3 },
  ];

  function initialSegs() {
    return CORPUS.map((e) => ({
      word: e.word + EOW,
      freq: e.freq,
      sym: (e.word + EOW).split(''),
    }));
  }

  /* Vocabulário base: caracteres em ordem de primeira aparição no corpus
     (l, o, w, _, e, r, n, s, t, i, d), como nos slides. */
  function baseVocab() {
    const seen = [];
    for (const s of initialSegs()) {
      for (const c of s.sym) if (!seen.includes(c)) seen.push(c);
    }
    return seen;
  }

  /* Conta pares adjacentes; "first" registra a ordem de primeira ocorrência
     na varredura do corpus (usada como desempate, como nos slides). */
  function pairStats(segs) {
    const map = new Map();
    let pos = 0;
    for (const s of segs) {
      for (let i = 0; i < s.sym.length - 1; i++) {
        pos += 1;
        const key = s.sym[i] + '' + s.sym[i + 1];
        const e = map.get(key);
        if (e) e.count += s.freq;
        else map.set(key, { a: s.sym[i], b: s.sym[i + 1], count: s.freq, first: pos });
      }
    }
    return Array.from(map.values());
  }

  /* Frequência de cada símbolo na segmentação atual (usada pelo WordPiece). */
  function symFreqs(segs) {
    const f = new Map();
    for (const s of segs) {
      for (const t of s.sym) f.set(t, (f.get(t) || 0) + s.freq);
    }
    return f;
  }

  function applyMerge(segs, a, b) {
    for (const s of segs) {
      const out = [];
      let i = 0;
      while (i < s.sym.length) {
        if (i < s.sym.length - 1 && s.sym[i] === a && s.sym[i + 1] === b) {
          out.push(a + b);
          i += 2;
        } else {
          out.push(s.sym[i]);
          i += 1;
        }
      }
      s.sym = out;
    }
  }

  function totalTokens(segs) {
    return segs.reduce((acc, s) => acc + s.freq * s.sym.length, 0);
  }

  function copySegs(segs) {
    return segs.map((s) => ({ word: s.word, freq: s.freq, sym: s.sym.slice() }));
  }

  /* Treina até esgotar os pares, gravando o estado completo a cada passo.
     criterion(par, freqs) define a pontuação do merge:
       BPE:       count
       WordPiece: count / (freq(a) · freq(b))
     Desempate: maior contagem, depois primeira ocorrência na varredura. */
  function train(criterion) {
    const states = [];
    const merges = [];
    const segs = initialSegs();
    let vocab = baseVocab();
    for (;;) {
      const freqs = symFreqs(segs);
      const pairs = pairStats(segs);
      for (const p of pairs) {
        p.fa = freqs.get(p.a);
        p.fb = freqs.get(p.b);
        p.score = criterion(p, freqs);
      }
      pairs.sort((p, q) => q.score - p.score || q.count - p.count || p.first - q.first);
      states.push({ segs: copySegs(segs), vocab: vocab.slice(), pairs, tokens: totalTokens(segs) });
      if (pairs.length === 0) break;
      const best = pairs[0];
      merges.push({ a: best.a, b: best.b, count: best.count, score: best.score });
      applyMerge(segs, best.a, best.b);
      vocab = vocab.concat([best.a + best.b]);
    }
    return { states, merges, maxK: merges.length };
  }

  const BPE = train((p) => p.count);
  const WP = train((p, f) => p.count / (f.get(p.a) * f.get(p.b)));

  /* Codifica uma palavra nova aplicando os k primeiros merges do BPE em ordem,
     registrando a derivação passo a passo (como o exemplo "lowest" da aula). */
  function encodeWord(word, k) {
    let sym = (word + EOW).split('');
    const steps = [{ label: 'início', sym: sym.slice() }];
    const n = Math.min(k, BPE.merges.length);
    for (let m = 0; m < n; m++) {
      const a = BPE.merges[m].a, b = BPE.merges[m].b;
      const out = [];
      let i = 0, changed = false;
      while (i < sym.length) {
        if (i < sym.length - 1 && sym[i] === a && sym[i + 1] === b) {
          out.push(a + b);
          i += 2;
          changed = true;
        } else {
          out.push(sym[i]);
          i += 1;
        }
      }
      sym = out;
      if (changed) {
        steps.push({ label: 'merge ' + (m + 1) + ': (' + a + ', ' + b + ')', sym: sym.slice() });
      }
    }
    return { sym, steps };
  }

  DL.bpe = { EOW, CORPUS, BASE: baseVocab(), BPE, WP, encodeWord, initialSegs };
})();
