/* Seção 1: forward e backward passo a passo no grafo computacional.
   Replica o exemplo da aula: θ⁽¹⁾ = −0.5, θ⁽²⁾ = 0.5, x₁ = 1, y = 1, sigmoide. */
(function () {
  'use strict';
  window.DL = window.DL || {};
  DL.sections = DL.sections || [];

  /* Tamanho da caixa de cada nó; encolhe com a largura do canvas (ver redraw). */
  let BOX_W = 96, BOX_H = 42;
  const LAST = 11;                               // passos: 0 (inicial), 1–5 forward, 6–11 backward
  const FWD = ['z1', 'a1', 'z2', 'a2', 'J'];     // ordem de preenchimento do forward
  const DEFAULTS = { x: 1, th1: -0.5, th2: 0.5, y: 1 };

  function sigmoid(z) { return 1 / (1 + Math.exp(-z)); }

  /* Formata com sinal de menos tipográfico e sem "-0.0000". */
  function fnum(v, d) {
    d = d == null ? 4 : d;
    let s = v.toFixed(d);
    if (parseFloat(s) === 0) s = (0).toFixed(d);
    return s.replace('-', '−');
  }

  function init() {
    const P = DL.plot;
    const $ = (id) => document.getElementById(id);

    const cv = $('s1-canvas');
    const btnPrev = $('s1-prev'), btnNext = $('s1-next');
    const btnPlay = $('s1-play'), btnReset = $('s1-reset'), btnPreset = $('s1-preset');
    const sX = $('s1-x'), sTh1 = $('s1-th1'), sTh2 = $('s1-th2'), sY = $('s1-y');

    let step = 0, playing = false, acc = 0;
    let V = null;   // todos os valores do forward e do backward

    /* Posições fracionárias dos nós no canvas. */
    const nodes = {
      x:   { fx: 0.07,  fy: 0.26, name: 'x_1' },
      th1: { fx: 0.07,  fy: 0.74, name: 'θ⁽¹⁾' },
      z1:  { fx: 0.24,  fy: 0.50, name: 'z⁽¹⁾' },
      a1:  { fx: 0.405, fy: 0.50, name: 'a⁽¹⁾' },
      th2: { fx: 0.405, fy: 0.86, name: 'θ⁽²⁾' },
      z2:  { fx: 0.57,  fy: 0.50, name: 'z⁽²⁾' },
      a2:  { fx: 0.735, fy: 0.50, name: 'a⁽²⁾ = ŷ' },
      y:   { fx: 0.735, fy: 0.86, name: 'y' },
      J:   { fx: 0.90,  fy: 0.50, name: 'J' },
    };
    const edges = [
      ['x', 'z1'], ['th1', 'z1'], ['z1', 'a1'], ['a1', 'z2'],
      ['th2', 'z2'], ['z2', 'a2'], ['a2', 'J'], ['y', 'J'],
    ];
    /* Ordem do backward: aresta percorrida e nó que recebe o gradiente acumulado. */
    const bwd = [
      { edge: 'a2-J',   node: 'a2' },
      { edge: 'z2-a2',  node: 'z2' },
      { edge: 'th2-z2', node: 'th2' },
      { edge: 'a1-z2',  node: 'a1' },
      { edge: 'z1-a1',  node: 'z1' },
      { edge: 'th1-z1', node: 'th1' },
    ];

    function compute() {
      const x = +sX.value, th1 = +sTh1.value, th2 = +sTh2.value, y = +sY.value;
      const z1 = x * th1, a1 = sigmoid(z1);
      const z2 = a1 * th2, a2 = sigmoid(z2);
      const J = 0.5 * (y - a2) * (y - a2);
      const dJ_da2 = a2 - y;
      const da2_dz2 = a2 * (1 - a2);
      const delta2 = dJ_da2 * da2_dz2;
      const gth2 = delta2 * a1;
      const dJ_da1 = delta2 * th2;
      const da1_dz1 = a1 * (1 - a1);
      const delta1 = dJ_da1 * da1_dz1;
      const gth1 = delta1 * x;
      V = { x, th1, th2, y, z1, a1, z2, a2, J, dJ_da2, da2_dz2, delta2, gth2, dJ_da1, da1_dz1, delta1, gth1 };
    }

    /* Valor do forward de cada nó (folhas sempre visíveis). */
    function nodeValue(id) {
      return { x: V.x, th1: V.th1, th2: V.th2, y: V.y, z1: V.z1, a1: V.a1, z2: V.z2, a2: V.a2, J: V.J }[id];
    }
    function isLeaf(id) { return id === 'x' || id === 'th1' || id === 'th2' || id === 'y'; }
    function fwdVisible(id) { return isLeaf(id) || step >= FWD.indexOf(id) + 1; }

    /* Gradiente acumulado mostrado embaixo de cada nó (rosa). */
    function nodeGrad(id) {
      return {
        a2:  { label: '∂J/∂a⁽²⁾', val: V.dJ_da2 },
        z2:  { label: 'δ⁽²⁾',      val: V.delta2 },
        th2: { label: '∂J/∂θ⁽²⁾', val: V.gth2 },
        a1:  { label: '∂J/∂a⁽¹⁾', val: V.dJ_da1 },
        z1:  { label: 'δ⁽¹⁾',      val: V.delta1 },
        th1: { label: '∂J/∂θ⁽¹⁾', val: V.gth1 },
      }[id];
    }

    /* Derivada local anotada em cada aresta percorrida no backward (laranja). */
    function edgeLocal(key) {
      return {
        'a2-J':   { sym: 'a⁽²⁾ − y',       val: V.dJ_da2 },
        'z2-a2':  { sym: 'a⁽²⁾(1−a⁽²⁾)',   val: V.da2_dz2 },
        'th2-z2': { sym: 'a⁽¹⁾',           val: V.a1 },
        'a1-z2':  { sym: 'θ⁽²⁾',           val: V.th2 },
        'z1-a1':  { sym: 'a⁽¹⁾(1−a⁽¹⁾)',   val: V.da1_dz1 },
        'th1-z1': { sym: 'x_1',            val: V.x },
      }[key];
    }

    function descFor(s) {
      const f = fnum;
      switch (s) {
        case 0: return 'Estado inicial. As folhas do grafo (x₁, θ⁽¹⁾, θ⁽²⁾ e y) já têm valores. ' +
          'Avance os passos para executar o forward pass e depois o backward pass.';
        case 1: return 'Forward: pré-ativação da camada 1, z⁽¹⁾ = x₁·θ⁽¹⁾ = ' + f(V.z1) + '.';
        case 2: return 'Forward: ativação da camada 1, a⁽¹⁾ = σ(z⁽¹⁾) = ' + f(V.a1) + '.';
        case 3: return 'Forward: pré-ativação da camada 2, z⁽²⁾ = a⁽¹⁾·θ⁽²⁾ = ' + f(V.z2) + '.';
        case 4: return 'Forward: saída da rede, ŷ = a⁽²⁾ = σ(z⁽²⁾) = ' + f(V.a2) + '.';
        case 5: return 'Loss: J(θ) = ½(y − ŷ)² = ' + f(V.J) + '. Fim do forward. ' +
          'Agora o backward pass percorre o grafo de volta, da direita para a esquerda.';
        case 6: return 'Backward: derivada do custo em relação à saída, ∂J/∂a⁽²⁾ = a⁽²⁾ − y. ' +
          'É o primeiro fator de todas as cadeias.';
        case 7: return 'Backward: atravessamos a sigmoide da camada 2. A derivada local é ' +
          '∂a⁽²⁾/∂z⁽²⁾ = a⁽²⁾(1−a⁽²⁾) e o produto acumulado até z⁽²⁾ é o delta da camada, δ⁽²⁾.';
        case 8: return 'Backward: chegamos em θ⁽²⁾. O gradiente é a cadeia inteira até z⁽²⁾ vezes a ' +
          'derivada local ∂z⁽²⁾/∂θ⁽²⁾ = a⁽¹⁾, a ativação da camada anterior.';
        case 9: return 'Backward: o erro também flui para a⁽¹⁾, pela derivada local ' +
          '∂z⁽²⁾/∂a⁽¹⁾ = θ⁽²⁾. É assim que o erro se propaga para as camadas anteriores.';
        case 10: return 'Backward: atravessamos a sigmoide da camada 1. O acumulado em z⁽¹⁾ é δ⁽¹⁾, ' +
          'obtido a partir de δ⁽²⁾, sem recomputar a cadeia desde o início.';
        case 11: return 'Backward: chegamos em θ⁽¹⁾ com ∂z⁽¹⁾/∂θ⁽¹⁾ = x₁. Os dois gradientes estão ' +
          'prontos: o otimizador pode atualizar os pesos com θ ← θ − η·∇J.';
      }
      return '';
    }

    function chainFor(s) {
      const f = fnum;
      const s1 = '<sup>(1)</sup>', s2 = '<sup>(2)</sup>';
      switch (s) {
        case 1: return 'z' + s1 + ' = x<sub>1</sub>·θ' + s1 + ' = (' + f(V.x, 2) + ')·(' + f(V.th1, 2) + ') = ' + f(V.z1);
        case 2: return 'a' + s1 + ' = σ(z' + s1 + ') = σ(' + f(V.z1) + ') = ' + f(V.a1);
        case 3: return 'z' + s2 + ' = a' + s1 + '·θ' + s2 + ' = (' + f(V.a1) + ')·(' + f(V.th2, 2) + ') = ' + f(V.z2);
        case 4: return 'ŷ = a' + s2 + ' = σ(z' + s2 + ') = σ(' + f(V.z2) + ') = ' + f(V.a2);
        case 5: return 'J(θ) = ½(y − ŷ)² = ½(' + f(V.y, 2) + ' − ' + f(V.a2) + ')² = ' + f(V.J);
        case 6: return '∂J/∂a' + s2 + ' = a' + s2 + ' − y = ' + f(V.a2) + ' − ' + f(V.y, 2) + ' = ' + f(V.dJ_da2);
        case 7: return 'δ' + s2 + ' = ∂J/∂a' + s2 + ' · ∂a' + s2 + '/∂z' + s2 +
          ' = (' + f(V.dJ_da2) + ')·(' + f(V.da2_dz2) + ') = ' + f(V.delta2);
        case 8: return '∂J/∂θ' + s2 + ' = ∂J/∂a' + s2 + ' · ∂a' + s2 + '/∂z' + s2 + ' · ∂z' + s2 + '/∂θ' + s2 +
          ' = (' + f(V.dJ_da2) + ')·(' + f(V.da2_dz2) + ')·(' + f(V.a1) + ') = ' + f(V.gth2);
        case 9: return '∂J/∂a' + s1 + ' = δ' + s2 + ' · ∂z' + s2 + '/∂a' + s1 +
          ' = (' + f(V.delta2) + ')·(' + f(V.th2, 2) + ') = ' + f(V.dJ_da1);
        case 10: return 'δ' + s1 + ' = ∂J/∂a' + s1 + ' · ∂a' + s1 + '/∂z' + s1 +
          ' = (' + f(V.dJ_da1) + ')·(' + f(V.da1_dz1) + ') = ' + f(V.delta1);
        case 11: return '∂J/∂θ' + s1 + ' = ∂J/∂a' + s2 + ' · ∂a' + s2 + '/∂z' + s2 + ' · ∂z' + s2 + '/∂a' + s1 +
          ' · ∂a' + s1 + '/∂z' + s1 + ' · ∂z' + s1 + '/∂θ' + s1 +
          ' = (' + f(V.dJ_da2) + ')·(' + f(V.da2_dz2) + ')·(' + f(V.th2, 2) + ')·(' + f(V.da1_dz1) + ')·(' + f(V.x, 2) + ') = ' + f(V.gth1);
      }
      return '';
    }

    /* ── desenho ── */

    function roundRect(ctx, x, y, w, h, r) {
      ctx.beginPath();
      ctx.moveTo(x + r, y);
      ctx.arcTo(x + w, y, x + w, y + h, r);
      ctx.arcTo(x + w, y + h, x, y + h, r);
      ctx.arcTo(x, y + h, x, y, r);
      ctx.arcTo(x, y, x + w, y, r);
      ctx.closePath();
    }

    /* Ponto na borda da caixa do nó, na direção (ux, uy) a partir do centro. */
    function trim(c, ux, uy) {
      const sx = ux === 0 ? Infinity : (BOX_W / 2 + 4) / Math.abs(ux);
      const sy = uy === 0 ? Infinity : (BOX_H / 2 + 4) / Math.abs(uy);
      const t = Math.min(sx, sy);
      return { x: c.x + ux * t, y: c.y + uy * t };
    }

    function redraw() {
      const th = P.theme();
      const { ctx, w, h } = P.setup(cv);
      P.clear(ctx, w, h);
      /* Em canvas estreito (celular) as caixas de 96px se sobrepõem: escala
         caixa e fontes pela largura disponível. */
      BOX_W = Math.min(96, w / 8);
      const sc = BOX_W / 96;
      BOX_H = Math.max(32, Math.round(42 * sc));
      const fsz = (base) => Math.max(8.5, base * sc);
      const pos = (id) => ({ x: nodes[id].fx * w, y: nodes[id].fy * h });
      /* Canvas estreito: os rótulos de gradiente e de derivada local são mais
         largos que o espaço entre nós; viram duas linhas e alternam a altura. */
      const narrow = w < 560;
      let hEdgeIdx = 0;
      /* mais altura no celular: os rótulos em duas linhas precisam de espaço
         entre a linha do meio e as folhas de baixo (sem oscilar: só depende de w) */
      const wantH = narrow ? '400px' : '';
      if (cv.style.height !== wantH) { cv.style.height = wantH; return; }

      const activeEdge = step >= 6 ? bwd[step - 6].edge : null;
      const activeNode = step >= 6 ? bwd[step - 6].node : (step >= 1 ? FWD[step - 1] : null);
      const bwdCount = Math.max(0, step - 5);   // quantas arestas do backward já foram percorridas
      const doneEdges = new Set(bwd.slice(0, bwdCount).map((b) => b.edge));

      /* arestas */
      for (const [from, to] of edges) {
        const key = from + '-' + to;
        const a = pos(from), b = pos(to);
        const dx = b.x - a.x, dy = b.y - a.y;
        const len = Math.hypot(dx, dy) || 1;
        const ux = dx / len, uy = dy / len;
        const p1 = trim(a, ux, uy), p2 = trim(b, -ux, -uy);
        const isActive = key === activeEdge;
        const isDone = doneEdges.has(key);
        ctx.strokeStyle = isActive ? th.purple : isDone ? th.pink : th.line;
        ctx.lineWidth = isActive ? 3 : isDone ? 2 : 1.4;
        ctx.beginPath();
        ctx.moveTo(p1.x, p1.y);
        ctx.lineTo(p2.x, p2.y);
        ctx.stroke();
        /* seta */
        ctx.fillStyle = ctx.strokeStyle;
        ctx.beginPath();
        ctx.moveTo(p2.x, p2.y);
        ctx.lineTo(p2.x - 8 * ux - 4 * uy, p2.y - 8 * uy + 4 * ux);
        ctx.lineTo(p2.x - 8 * ux + 4 * uy, p2.y - 8 * uy - 4 * ux);
        ctx.closePath();
        ctx.fill();
        /* derivada local da aresta (revelada no backward) */
        if (isDone || isActive) {
          const loc = edgeLocal(key);
          if (loc) {
            const mx = (p1.x + p2.x) / 2, my = (p1.y + p2.y) / 2;
            let ox = -uy, oy = ux;
            if (oy > 0) { ox = -ox; oy = -oy; }   // rótulo sempre acima da aresta
            /* no canvas estreito a aresta θ⁽²⁾→z⁽²⁾ leva o rótulo para baixo e à
               direita, longe do gradiente escrito embaixo de a⁽¹⁾ */
            if (narrow && key === 'th2-z2') { ox = -ox; oy = -oy; }
            const horiz = Math.abs(uy) < 1e-6;
            /* no canvas estreito as arestas horizontais são curtas: o rótulo sobe
               acima das caixas, e arestas vizinhas alternam a altura */
            const off = narrow ? (horiz ? BOX_H / 2 + 14 : 22) : 14;
            const lift = narrow && horiz && (hEdgeIdx++ % 2 === 1) ? 24 : 0;
            P.mathText(ctx, loc.sym, mx + ox * off, my + oy * off - 6 - lift, th.orange, 'center', fsz(10));
            P.mathText(ctx, fnum(loc.val), mx + ox * off, my + oy * off + 7 - lift, th.orange, 'center', fsz(10));
          }
        }
      }

      /* nós */
      for (const id in nodes) {
        const c = pos(id);
        const isActive = id === activeNode;
        roundRect(ctx, c.x - BOX_W / 2, c.y - BOX_H / 2, BOX_W, BOX_H, 7);
        ctx.fillStyle = th.card;
        ctx.fill();
        ctx.strokeStyle = isActive ? th.purple : th.line;
        ctx.lineWidth = isActive ? 2.5 : 1.2;
        ctx.stroke();
        if (isActive) {
          ctx.globalAlpha = 0.12;
          ctx.fillStyle = th.purple;
          roundRect(ctx, c.x - BOX_W / 2, c.y - BOX_H / 2, BOX_W, BOX_H, 7);
          ctx.fill();
          ctx.globalAlpha = 1;
        }
        P.mathText(ctx, nodes[id].name, c.x, c.y - 5, th.fg, 'center', fsz(12));
        if (fwdVisible(id)) {
          P.mathText(ctx, fnum(nodeValue(id)), c.x, c.y + 11, isLeaf(id) ? th.cyan : th.green, 'center', fsz(11));
        } else {
          P.mathText(ctx, '?', c.x, c.y + 11, th.comment, 'center', fsz(11));
        }
        /* gradiente acumulado embaixo do nó */
        const gIdx = bwd.findIndex((b) => b.node === id);
        if (gIdx !== -1 && gIdx < bwdCount) {
          const g = nodeGrad(id);
          if (narrow) {
            /* duas linhas; nós da linha do meio alternam a altura para não
               colidir com o vizinho (ficam a ~w/6 de distância) */
            const mid = FWD.indexOf(id);
            const drop = mid !== -1 && mid % 2 === 0 ? 22 : 0;
            const y0 = c.y + BOX_H / 2 + 11 + drop;
            /* a folha θ⁽¹⁾ encosta na borda esquerda: alinha à esquerda da caixa;
               z⁽²⁾ recebe a aresta de θ⁽²⁾ por baixo à esquerda: rótulo à direita */
            const ax = id === 'th1' ? c.x - BOX_W / 2 : id === 'z2' ? c.x - 4 : c.x;
            const al = id === 'th1' || id === 'z2' ? 'left' : 'center';
            P.mathText(ctx, g.label, ax, y0, th.pink, al, fsz(10));
            P.mathText(ctx, fnum(g.val), ax, y0 + 11, th.pink, al, fsz(10));
          } else {
            P.mathText(ctx, g.label + ' = ' + fnum(g.val), c.x, c.y + BOX_H / 2 + 14, th.pink, 'center', fsz(10.5));
          }
        }
      }

      /* legenda das fases */
      const phase = step === 0 ? 'pronto para começar'
        : step <= 5 ? 'forward pass' : 'backward pass';
      P.mathText(ctx, phase, w - 8, 16, step > 5 ? th.pink : th.green, 'right', 11);
    }

    /* ── estado e controles ── */

    function update() {
      compute();
      $('s1-step-label').textContent = 'passo ' + step + '/' + LAST;
      $('s1-desc').textContent = descFor(step);
      $('s1-chain').innerHTML = chainFor(step);
      btnPrev.disabled = step === 0;
      btnNext.disabled = step === LAST;
      redraw();
    }

    function setStep(s) {
      step = Math.max(0, Math.min(LAST, s));
      update();
    }

    function stopPlay() {
      playing = false;
      btnPlay.textContent = '▶ Reproduzir';
      DL.stopTicker(tick);
    }

    function tick(dt) {
      acc += dt;
      if (acc >= 1.3) {
        acc = 0;
        setStep(step + 1);
        if (step >= LAST) stopPlay();
      }
    }

    btnNext.addEventListener('click', () => { stopPlay(); setStep(step + 1); });
    btnPrev.addEventListener('click', () => { stopPlay(); setStep(step - 1); });
    btnReset.addEventListener('click', () => { stopPlay(); setStep(0); });
    btnPlay.addEventListener('click', () => {
      if (playing) { stopPlay(); return; }
      if (step >= LAST) setStep(0);
      playing = true;
      acc = 0;
      btnPlay.textContent = '⏸ Pausar';
      DL.startTicker(tick);
    });

    function syncSliderLabels() {
      $('s1-x-val').textContent = fnum(+sX.value, 1);
      $('s1-th1-val').textContent = fnum(+sTh1.value, 2);
      $('s1-th2-val').textContent = fnum(+sTh2.value, 2);
      $('s1-y-val').textContent = fnum(+sY.value, 2);
    }

    for (const sl of [sX, sTh1, sTh2, sY]) {
      sl.addEventListener('input', () => { syncSliderLabels(); update(); });
    }

    btnPreset.addEventListener('click', () => {
      sX.value = DEFAULTS.x; sTh1.value = DEFAULTS.th1;
      sTh2.value = DEFAULTS.th2; sY.value = DEFAULTS.y;
      syncSliderLabels();
      update();
    });

    syncSliderLabels();
    update();
    P.onRedraw(redraw);
    P.observeResize(cv, redraw);
  }

  DL.sections.push({ name: 's1-graph', init });
})();
