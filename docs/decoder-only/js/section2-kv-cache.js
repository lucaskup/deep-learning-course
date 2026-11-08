/* Seção 2: geração autoregressiva (prefill + decode) com e sem KV-cache.
   O modelo é um trigrama por contagem com interpolação, embutido na página. */
(function () {
  'use strict';
  window.DL = window.DL || {};
  DL.sections = DL.sections || [];

  const EOS = '⟨eos⟩', BOS = '⟨bos⟩';
  const MAXGEN = 9;

  const CORPUS = [
    'o rato roeu a rolha da garrafa do rei',
    'o rato roeu a roupa do rei de roma',
    'o gato perseguiu o rato no porão da casa',
    'a rainha viu o rato e chamou o gato',
    'o rei guardou a garrafa na adega do castelo',
    'o gato dorme no telhado da casa amarela',
    'a roupa do rei secou ao sol da manhã',
    'o rato fugiu do gato e se escondeu na adega',
    'a garrafa do rei caiu da mesa do salão',
    'o rei de roma visitou o castelo da rainha',
  ];

  /* Modelo: contagens de trigramas, bigramas e unigramas sobre o corpus. */
  function buildModel() {
    const c3 = new Map(), c2 = new Map(), c1 = new Map();
    const ctx3 = new Map(), ctx2 = new Map();
    const vocab = new Set([EOS]);
    const bump = (m, k) => m.set(k, (m.get(k) || 0) + 1);
    for (const s of CORPUS) {
      const toks = [BOS, BOS, ...s.split(/\s+/), EOS];
      for (let t = 2; t < toks.length; t++) {
        const u = toks[t - 2], v = toks[t - 1], w = toks[t];
        if (w !== EOS) vocab.add(w);
        bump(c3, u + ' ' + v + ' ' + w);
        bump(ctx3, u + ' ' + v);
        bump(c2, v + ' ' + w);
        bump(ctx2, v);
        bump(c1, w);
      }
    }
    const words = Array.from(vocab).sort();
    let total = 0;
    for (const v of c1.values()) total += v;

    /* P(w | u, v) interpolando trigrama (0.62), bigrama (0.28) e unigrama (0.10). */
    function dist(u, v) {
      const n3 = ctx3.get(u + ' ' + v) || 0;
      const n2 = ctx2.get(v) || 0;
      const out = [];
      let sum = 0;
      for (const w of words) {
        const p3 = n3 ? (c3.get(u + ' ' + v + ' ' + w) || 0) / n3 : 0;
        const p2 = n2 ? (c2.get(v + ' ' + w) || 0) / n2 : 0;
        const p1 = ((c1.get(w) || 0) + 0.5) / (total + 0.5 * words.length);
        const p = 0.62 * p3 + 0.28 * p2 + 0.10 * p1;
        out.push([w, p]);
        sum += p;
      }
      for (const e of out) e[1] /= sum;
      return out;
    }
    return { dist };
  }

  function init() {
    const P = DL.plot;
    const $ = (id) => document.getElementById(id);

    const model = buildModel();
    const cvCells = $('s2-cells'), cvDist = $('s2-dist'), cvCost = $('s2-cost');
    const selPrompt = $('s2-prompt');
    const tabCache = $('s2-tab-cache'), tabNo = $('s2-tab-nocache');
    const btnStep = $('s2-step'), btnRun = $('s2-run'), btnReset = $('s2-reset');

    let mode = 'cache';
    let seed = 11;
    let rng, tokens, promptLen, finished, lastComputed, dist;
    let running = false, acc = 0;

    function reset() {
      rng = DL.utils.mulberry32(seed);
      tokens = selPrompt.value.split(/\s+/);
      promptLen = tokens.length;
      finished = false;
      /* Prefill: K,V de todo o prompt computados em paralelo. */
      lastComputed = new Set(tokens.map((_, i) => i));
      computeDist();
    }

    function computeDist() {
      const m = tokens.length;
      const u = m >= 2 ? tokens[m - 2] : BOS;
      const v = tokens[m - 1];
      dist = model.dist(u, v).slice().sort((a, b) => b[1] - a[1]);
    }

    function stepOnce() {
      if (finished || tokens.length - promptLen >= MAXGEN) { finished = true; return; }
      /* Amostra x_t ~ P(· | x_<t) */
      let r = rng(), pick = dist[dist.length - 1][0];
      for (const [w, p] of dist) { r -= p; if (r <= 0) { pick = w; break; } }
      tokens.push(pick);
      lastComputed = mode === 'cache'
        ? new Set([tokens.length - 1])
        : new Set(tokens.map((_, i) => i));
      if (pick === EOS || tokens.length - promptLen >= MAXGEN) finished = true;
      else computeDist();
    }

    /* Custo acumulado (pares K,V computados) após t passos de decode. */
    const costCache = (t) => promptLen + t;
    const costNo = (t) => promptLen + t * promptLen + (t * (t + 1)) / 2;

    function roundRect(ctx, x, y, w, h, r) {
      ctx.beginPath();
      ctx.moveTo(x + r, y);
      ctx.arcTo(x + w, y, x + w, y + h, r);
      ctx.arcTo(x + w, y + h, x, y + h, r);
      ctx.arcTo(x, y + h, x, y, r);
      ctx.arcTo(x, y, x + w, y, r);
      ctx.closePath();
    }

    /* Bloco de legenda com fundo semitransparente, para que o texto continue
       legível quando barras ou curvas passam por baixo. items: [[texto, cor]]. */
    function legendBox(ctx, th, x, y, items, align) {
      ctx.font = '11px sans-serif';
      let wmax = 0;
      for (const it of items) wmax = Math.max(wmax, ctx.measureText(it[0].replace(/[_{}]/g, '')).width);
      const bx = align === 'right' ? x - wmax - 6 : x - 6;
      ctx.save();
      ctx.globalAlpha = 0.85;
      ctx.fillStyle = th.card;
      ctx.fillRect(bx, y, wmax + 12, 14 * items.length + 6);
      ctx.restore();
      items.forEach((it, k) => P.label(ctx, x, y + 14 * (k + 1), it[0], it[1], align || 'left'));
    }

    function drawCells() {
      const th = P.theme();
      const { ctx, w, h } = P.setup(cvCells);
      P.clear(ctx, w, h);
      const maxLen = promptLen + MAXGEN;
      const slot = Math.min(76, (w - 50) / maxLen);
      const x0 = 42;
      /* Em canvas estreito (celular) a caixa de cada token fica com menos de
         30 px: a palavra é escrita na vertical numa caixa mais alta. */
      const narrow = slot < 34;
      const boxH = narrow ? 48 : 26, kvH = 24;
      const yTok = 46, yK = yTok + boxH + 30, yV = yK + kvH + 10;

      const t = tokens.length - promptLen;
      let phase = t === 0
        ? 'Prefill: K,V dos ' + promptLen + ' tokens do prompt computados em paralelo'
        : (mode === 'cache'
          ? 'Decode passo ' + t + ': 1 par K,V novo, ' + (tokens.length - 1) + ' reutilizados do cache'
          : 'Decode passo ' + t + ': ' + tokens.length + ' pares K,V recomputados do zero');
      ctx.fillStyle = th.fg;
      ctx.font = '12px sans-serif';
      ctx.textAlign = 'left';
      if (finished) phase += ' · geração encerrada';
      if (ctx.measureText(phase).width > w - 16) {
        /* versão curta do título quando não cabe na largura do canvas */
        phase = t === 0
          ? 'Prefill: ' + promptLen + ' pares K,V em paralelo'
          : (mode === 'cache'
            ? 'Decode ' + t + ': 1 par novo, ' + (tokens.length - 1) + ' do cache'
            : 'Decode ' + t + ': ' + tokens.length + ' pares recomputados');
        if (finished) phase += ' · fim';
      }
      ctx.fillText(phase, 10, 16);

      P.mathText(ctx, 'K_i', x0 - 28, yK + kvH / 2 + 4, th.comment, 'left');
      P.mathText(ctx, 'V_i', x0 - 28, yV + kvH / 2 + 4, th.comment, 'left');

      for (let i = 0; i < tokens.length; i++) {
        const cx = x0 + i * slot;
        const isPrompt = i < promptLen;
        const word = tokens[i];
        /* Token */
        ctx.fillStyle = word === EOS ? th.comment : (isPrompt ? th.cyan : th.pink);
        ctx.globalAlpha = 0.22;
        roundRect(ctx, cx, yTok, slot - 6, boxH, 5);
        ctx.fill();
        ctx.globalAlpha = 1;
        ctx.strokeStyle = word === EOS ? th.comment : (isPrompt ? th.cyan : th.pink);
        ctx.lineWidth = 1.2;
        roundRect(ctx, cx, yTok, slot - 6, boxH, 5);
        ctx.stroke();
        ctx.fillStyle = th.fg;
        if (narrow) {
          ctx.save();
          ctx.translate(cx + (slot - 6) / 2 + 3, yTok + boxH / 2);
          ctx.rotate(-Math.PI / 2);
          ctx.font = (word.length > 7 ? 8 : 9) + 'px sans-serif';
          ctx.textAlign = 'center';
          ctx.fillText(word, 0, 0);
          ctx.restore();
        } else {
          ctx.font = (word.length > 7 ? 9 : 11) + 'px sans-serif';
          ctx.textAlign = 'center';
          ctx.fillText(word, cx + (slot - 6) / 2, yTok + boxH / 2 + 4);
        }
        /* Seta token → K */
        ctx.strokeStyle = th.line;
        ctx.beginPath();
        ctx.moveTo(cx + (slot - 6) / 2, yTok + boxH);
        ctx.lineTo(cx + (slot - 6) / 2, yK - 3);
        ctx.stroke();
        /* Células K e V */
        const hot = lastComputed.has(i);
        for (const y of [yK, yV]) {
          ctx.fillStyle = hot ? th.orange : th.green;
          ctx.globalAlpha = hot ? 0.75 : 0.45;
          roundRect(ctx, cx, y, slot - 6, kvH, 4);
          ctx.fill();
          ctx.globalAlpha = 1;
        }
        if (slot >= 34) {
          ctx.fillStyle = th.bg;
          P.mathText(ctx, 'K_' + (i + 1), cx + (slot - 6) / 2, yK + kvH / 2 + 4, th.fg, 'center', 10);
          P.mathText(ctx, 'V_' + (i + 1), cx + (slot - 6) / 2, yV + kvH / 2 + 4, th.fg, 'center', 10);
        }
      }

      /* Separador prompt | gerado */
      const sx = x0 + promptLen * slot - 3;
      ctx.strokeStyle = th.comment;
      ctx.setLineDash([4, 4]);
      ctx.beginPath();
      ctx.moveTo(sx, yTok - 8);
      ctx.lineTo(sx, yV + kvH + 8);
      ctx.stroke();
      ctx.setLineDash([]);
      ctx.fillStyle = th.comment;
      ctx.font = '10px sans-serif';
      ctx.textAlign = 'left';
      ctx.fillText('prompt | gerado', sx + 4, yTok - 12);

      /* Legenda */
      const ly = h - 14;
      const sw = 14;
      ctx.globalAlpha = 0.6;
      ctx.fillStyle = th.green;
      ctx.fillRect(10, ly - 9, sw, 10);
      ctx.fillStyle = th.orange;
      ctx.fillRect(140, ly - 9, sw, 10);
      ctx.globalAlpha = 1;
      ctx.fillStyle = th.comment;
      ctx.font = '11px sans-serif';
      ctx.fillText('cache (reutilizado)', 28, ly);
      ctx.fillText(mode === 'cache' ? 'computado agora' : 'recomputado agora', 158, ly);
    }

    function drawDist() {
      const th = P.theme();
      const { ctx, w, h } = P.setup(cvDist);
      P.clear(ctx, w, h);
      if (finished) {
        ctx.fillStyle = th.comment;
        ctx.font = '12px sans-serif';
        ctx.textAlign = 'center';
        ctx.fillText('geração encerrada (↺ para recomeçar)', w / 2, h / 2);
        return;
      }
      const top = dist.slice(0, 8);
      const ymax = Math.max(0.3, top[0][1] * 1.15);
      const fr = P.frame(ctx, w, h, 0.4, top.length + 0.6, 0, ymax, { l: 38, r: 10, t: 10, b: 56 });
      P.axes(ctx, fr, { ylabel: 'P', yticks: [0, +(ymax / 2).toFixed(2), +ymax.toFixed(2)] });
      const bw = Math.min(34, fr.iw / top.length * 0.6);
      for (let k = 0; k < top.length; k++) {
        const px = fr.X(k + 1);
        ctx.fillStyle = k === 0 ? th.cyan : th.purple;
        ctx.globalAlpha = 0.85;
        ctx.fillRect(px - bw / 2, fr.Y(top[k][1]), bw, fr.Y(0) - fr.Y(top[k][1]));
        ctx.globalAlpha = 1;
        ctx.save();
        ctx.translate(px, fr.Y(0) + 10);
        ctx.rotate(-Math.PI / 5);
        ctx.fillStyle = th.comment;
        ctx.font = '10px sans-serif';
        ctx.textAlign = 'right';
        ctx.fillText(top[k][0], 0, 6);
        ctx.restore();
      }
    }

    function drawCost() {
      const th = P.theme();
      const { ctx, w, h } = P.setup(cvCost);
      P.clear(ctx, w, h);
      const ymax = costNo(MAXGEN) * 1.08;
      const fr = P.frame(ctx, w, h, 0, MAXGEN, 0, ymax);
      P.axes(ctx, fr, {
        xlabel: 'passo t', ylabel: 'pares K,V',
        xticks: [0, Math.round(MAXGEN / 2), MAXGEN],
        yticks: [0, Math.round(ymax / 2), Math.round(ymax)],
      });
      const ts = [];
      for (let t = 0; t <= MAXGEN; t++) ts.push(t);
      P.line(ctx, fr, ts, ts.map(costNo), th.red, { width: 1.8, dash: [5, 4] });
      P.line(ctx, fr, ts, ts.map(costCache), th.green, { width: 1.8, dash: [5, 4] });
      const t = Math.min(tokens.length - promptLen, MAXGEN);
      const tsNow = ts.slice(0, t + 1);
      P.line(ctx, fr, tsNow, tsNow.map(costNo), th.red, { width: 2.4 });
      P.line(ctx, fr, tsNow, tsNow.map(costCache), th.green, { width: 2.4 });
      const cur = mode === 'cache' ? costCache(t) : costNo(t);
      P.scatter(ctx, fr, [[t, cur]], mode === 'cache' ? th.green : th.red, { r: 4.5, alpha: 1 });
      /* Rótulos fora das curvas: o de O(n²) termina em t = n/2, onde a curva
         ainda está abaixo do valor em 0.72·n; o de O(n) fica acima do fim da reta. */
      legendBox(ctx, th, fr.X(MAXGEN * 0.5), fr.Y(costNo(MAXGEN * 0.72)) - 10, [['sem cache ~ O(n²)', th.red]], 'right');
      legendBox(ctx, th, fr.X(MAXGEN) - 4, fr.Y(costCache(MAXGEN)) - 30, [['com cache ~ O(n)', th.green]], 'right');
    }

    function updateReadout() {
      const t = tokens.length - promptLen;
      const cur = mode === 'cache' ? costCache(t) : costNo(t);
      $('s2-readout').textContent = 'tokens gerados: ' + t +
        ' · pares K,V computados até aqui: ' + cur +
        (mode === 'cache' ? ' (com cache)' : ' (sem cache)');
    }

    function redraw() {
      drawCells();
      drawDist();
      drawCost();
      updateReadout();
    }

    function stopRun() {
      running = false;
      btnRun.textContent = '▶ Gerar tudo';
      DL.stopTicker(tick);
    }

    function tick(dt) {
      acc += dt * 2.5;
      while (acc >= 1) { acc -= 1; stepOnce(); }
      redraw();
      if (finished) stopRun();
    }

    function setMode(m) {
      mode = m;
      tabCache.classList.toggle('active', m === 'cache');
      tabNo.classList.toggle('active', m === 'nocache');
      /* Recolore o que foi "computado agora" sem reiniciar a sequência. */
      const t = tokens.length - promptLen;
      if (t > 0) {
        lastComputed = m === 'cache'
          ? new Set([tokens.length - 1])
          : new Set(tokens.map((_, i) => i));
      }
      redraw();
    }

    tabCache.addEventListener('click', () => setMode('cache'));
    tabNo.addEventListener('click', () => setMode('nocache'));
    selPrompt.addEventListener('change', () => { stopRun(); reset(); redraw(); });
    btnStep.addEventListener('click', () => { stepOnce(); redraw(); });
    btnRun.addEventListener('click', () => {
      if (running) { stopRun(); return; }
      if (finished) reset();
      running = true;
      acc = 0;
      btnRun.textContent = '⏸ Pausar';
      DL.startTicker(tick);
    });
    btnReset.addEventListener('click', () => { stopRun(); reset(); redraw(); });
    $('s2-reseed').addEventListener('click', () => {
      stopRun();
      seed = (seed * 1103515245 + 12345) >>> 0;
      reset();
      redraw();
    });

    reset();
    redraw();
    P.onRedraw(redraw);
    P.observeResize(cvCells, redraw);
    P.observeResize(cvDist, redraw);
    P.observeResize(cvCost, redraw);
  }

  DL.sections.push({ name: 's2-kv-cache', init });
})();
