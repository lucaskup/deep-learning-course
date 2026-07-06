/* Seção 2: backpropagation vetorizada passo a passo.
   Replica o exemplo vetorizado da aula: rede 2-3-1 sem bias, sigmoide nas duas
   camadas, batch de 2 exemplos X = [[0.5, 0.1], [0.2, 0.6]], y = [0.6, 0.8],
   θ⁽¹⁾ = [[0.5, 0.2], [0.6, −0.1], [−0.4, −0.3]], θ⁽²⁾ = [0.7, −0.1, 0.2].
   Convenção dos slides: θᵢⱼ⁽ˡ⁾ conecta a entrada j à saída i, logo
   Z⁽¹⁾ = X·θ⁽¹⁾ᵀ, Z⁽²⁾ = A⁽¹⁾·θ⁽²⁾ᵀ, δ⁽ˡ⁾ = ∂J/∂Z⁽ˡ⁾ e ∂J/∂θ⁽ˡ⁾ = δ⁽ˡ⁾ᵀ·A⁽ˡ⁻¹⁾. */
(function () {
  'use strict';
  window.DL = window.DL || {};
  DL.sections = DL.sections || [];

  const LAST = 9;   // passos: 0 (inicial), 1–5 forward, 6–9 backward
  const TH1 = [[0.5, 0.2], [0.6, -0.1], [-0.4, -0.3]];   // θ⁽¹⁾ (3×2), pesos da aula
  const TH2 = [[0.7, -0.1, 0.2]];                         // θ⁽²⁾ (1×3)
  const DEFAULTS = { x11: 0.5, x12: 0.1, x21: 0.2, x22: 0.6, y1: 0.6, y2: 0.8 };

  /* Definição de cada matriz do quadro: rótulo, forma, casas decimais
     (as mesmas precisões impressas nos slides), fase e passo em que aparece. */
  const DEF = {
    X:    { label: 'X',                          rows: 2, cols: 2, dec: 2, kind: 'given', step: 0 },
    y:    { label: 'y',                          rows: 2, cols: 1, dec: 2, kind: 'given', step: 0 },
    T1:   { label: 'θ<sup>(1)</sup>',            rows: 3, cols: 2, dec: 2, kind: 'given', step: 0 },
    T2:   { label: 'θ<sup>(2)</sup>',            rows: 1, cols: 3, dec: 2, kind: 'given', step: 0 },
    Z1:   { label: 'Z<sup>(1)</sup>',            rows: 2, cols: 3, dec: 4, kind: 'fwd',   step: 1 },
    A1:   { label: 'A<sup>(1)</sup>',            rows: 2, cols: 3, dec: 4, kind: 'fwd',   step: 2 },
    Z2:   { label: 'Z<sup>(2)</sup>',            rows: 2, cols: 1, dec: 4, kind: 'fwd',   step: 3 },
    Yhat: { label: 'ŷ = A<sup>(2)</sup>',        rows: 2, cols: 1, dec: 4, kind: 'fwd',   step: 4 },
    J:    { label: 'J',                          rows: 1, cols: 1, dec: 5, kind: 'fwd',   step: 5 },
    D2:   { label: 'δ<sup>(2)</sup>',            rows: 2, cols: 1, dec: 5, kind: 'bwd',   step: 6 },
    D1:   { label: 'δ<sup>(1)</sup>',            rows: 2, cols: 3, dec: 5, kind: 'bwd',   step: 7 },
    GT2:  { label: '∂J/∂θ<sup>(2)</sup>',        rows: 1, cols: 3, dec: 4, kind: 'bwd',   step: 8 },
    GT1:  { label: '∂J/∂θ<sup>(1)</sup>',        rows: 3, cols: 2, dec: 5, kind: 'bwd',   step: 9 },
  };
  const LAYOUT = [
    { title: 'Dados e pesos', items: ['X', 'y', 'T1', 'T2'] },
    { title: 'Forward', items: ['Z1', 'A1', 'Z2', 'Yhat', 'J'] },
    { title: 'Backward', items: ['D2', 'D1', 'GT2', 'GT1'] },
  ];
  /* Matrizes de entrada (borda roxa) e matriz produzida em cada passo. */
  const STEPS = [
    null,
    { out: 'Z1',   inputs: ['X', 'T1'] },
    { out: 'A1',   inputs: ['Z1'] },
    { out: 'Z2',   inputs: ['A1', 'T2'] },
    { out: 'Yhat', inputs: ['Z2'] },
    { out: 'J',    inputs: ['Yhat', 'y'] },
    { out: 'D2',   inputs: ['Yhat', 'y', 'Z2'] },
    { out: 'D1',   inputs: ['D2', 'T2', 'Z1'] },
    { out: 'GT2',  inputs: ['D2', 'A1'] },
    { out: 'GT1',  inputs: ['D1', 'X'] },
  ];

  function sigmoid(z) { return 1 / (1 + Math.exp(-z)); }

  /* Formata com sinal de menos tipográfico e sem "-0.0000". */
  function fnum(v, d) {
    d = d == null ? 4 : d;
    let s = v.toFixed(d);
    if (parseFloat(s) === 0) s = (0).toFixed(d);
    return s.replace('-', '−');
  }

  function init() {
    const $ = (id) => document.getElementById(id);

    const board = $('s2v-matrices');
    const btnPrev = $('s2v-prev'), btnNext = $('s2v-next');
    const btnPlay = $('s2v-play'), btnReset = $('s2v-reset'), btnPreset = $('s2v-preset');
    const sliders = {
      x11: $('s2v-x11'), x12: $('s2v-x12'), x21: $('s2v-x21'),
      x22: $('s2v-x22'), y1: $('s2v-y1'), y2: $('s2v-y2'),
    };

    let step = 0, playing = false, acc = 0;
    let V = null;   // todas as matrizes do forward e do backward

    /* ── matemática (idêntica aos slides, batch de 2 exemplos) ── */

    function compute() {
      const X = [[+sliders.x11.value, +sliders.x12.value],
                 [+sliders.x21.value, +sliders.x22.value]];
      const Y = [[+sliders.y1.value], [+sliders.y2.value]];
      const Z1 = X.map((r) => TH1.map((w) => r[0] * w[0] + r[1] * w[1]));
      const A1 = Z1.map((r) => r.map(sigmoid));
      const Z2 = A1.map((r) => [r[0] * TH2[0][0] + r[1] * TH2[0][1] + r[2] * TH2[0][2]]);
      const Yhat = Z2.map((r) => [sigmoid(r[0])]);
      const J = 0.5 * ((Y[0][0] - Yhat[0][0]) ** 2 + (Y[1][0] - Yhat[1][0]) ** 2);
      const D2 = Yhat.map((r, n) => [(r[0] - Y[n][0]) * r[0] * (1 - r[0])]);
      const D1 = D2.map((d, n) => TH2[0].map((t, k) => d[0] * t * A1[n][k] * (1 - A1[n][k])));
      const GT2 = [TH2[0].map((_, k) => D2[0][0] * A1[0][k] + D2[1][0] * A1[1][k])];
      const GT1 = TH1.map((_, k) => [
        D1[0][k] * X[0][0] + D1[1][k] * X[1][0],
        D1[0][k] * X[0][1] + D1[1][k] * X[1][1],
      ]);
      V = { X, y: Y, T1: TH1, T2: TH2, Z1, A1, Z2, Yhat, J: [[J]], D2, D1, GT2, GT1 };
    }

    /* ── quadro de matrizes (HTML, sem canvas) ── */

    const blocks = {};   // key → { el, cells }

    function buildBoard() {
      for (const group of LAYOUT) {
        const g = document.createElement('div');
        g.className = 'matrix-group';
        const t = document.createElement('p');
        t.className = 'matrix-group-title';
        t.textContent = group.title;
        g.appendChild(t);
        for (const key of group.items) {
          const d = DEF[key];
          const block = document.createElement('div');
          block.className = 'matrix-block';
          block.dataset.kind = d.kind;
          const label = document.createElement('div');
          label.className = 'matrix-label';
          label.innerHTML = '<span class="formula">' + d.label + '</span>' +
            '<span class="matrix-dims">' + d.rows + '×' + d.cols + '</span>';
          block.appendChild(label);
          const m = document.createElement('div');
          m.className = 'matrix';
          m.style.gridTemplateColumns = 'repeat(' + d.cols + ', auto)';
          const cells = [];
          for (let i = 0; i < d.rows * d.cols; i++) {
            const c = document.createElement('span');
            m.appendChild(c);
            cells.push(c);
          }
          block.appendChild(m);
          g.appendChild(block);
          blocks[key] = { el: block, cells };
        }
        board.appendChild(g);
      }
    }

    function renderBoard() {
      const active = STEPS[step];
      for (const key in DEF) {
        const d = DEF[key];
        const b = blocks[key];
        const ready = step >= d.step;
        b.el.classList.toggle('pending', !ready);
        b.el.classList.toggle('active-in', !!active && active.inputs.indexOf(key) !== -1);
        b.el.classList.toggle('active-out', !!active && active.out === key);
        const M = V[key];
        for (let i = 0; i < d.rows; i++) {
          for (let j = 0; j < d.cols; j++) {
            b.cells[i * d.cols + j].textContent = ready ? fnum(M[i][j], d.dec) : '·';
          }
        }
      }
    }

    /* ── descrições e equação do passo atual ── */

    function descFor(s) {
      switch (s) {
        case 0: return 'Estado inicial. O batch X (um exemplo por linha), o alvo y e os pesos ' +
          'θ⁽¹⁾ e θ⁽²⁾ já têm valores. Avance para executar o forward pass vetorizado e depois o backward pass.';
        case 1: return 'Forward: pré-ativações da camada 1 para o batch inteiro em uma única ' +
          'multiplicação de matrizes.';
        case 2: return 'Forward: a sigmoide é aplicada elemento a elemento sobre Z⁽¹⁾.';
        case 3: return 'Forward: pré-ativações da camada de saída, uma linha por exemplo do batch.';
        case 4: return 'Forward: saídas da rede para os 2 exemplos de uma vez.';
        case 5: return 'Loss: J soma o erro quadrático dos 2 exemplos do batch. Fim do forward.';
        case 6: return 'Backward: erro na camada de saída, um delta por exemplo. O produto ⊙ é ' +
          'elemento a elemento (Hadamard).';
        case 7: return 'Backward: o erro volta para a camada 1 multiplicando por θ⁽²⁾, sem recomputar ' +
          'a cadeia desde o início. δ⁽¹⁾ tem um delta por exemplo e por unidade oculta.';
        case 8: return 'Backward: gradiente dos pesos da camada 2. O produto δ⁽²⁾ᵀ·A⁽¹⁾ já soma as ' +
          'contribuições dos 2 exemplos do batch.';
        case 9: return 'Backward: gradiente dos pesos da camada 1. Os dois gradientes estão prontos: ' +
          'o otimizador pode atualizar os pesos com θ ← θ − η·∇J, exatamente como nos slides (η = 0.1).';
      }
      return '';
    }

    function eqFor(s) {
      const f = fnum;
      const s1 = '<sup>(1)</sup>', s2 = '<sup>(2)</sup>';
      const sp = 'σ′';
      switch (s) {
        case 1: return 'Z' + s1 + ' = X·θ<sup>(1)ᵀ</sup> : (2×2)·(2×3) → 2×3<br>' +
          'z<sub>11</sub>' + s1 + ' = x<sub>11</sub>·θ<sub>11</sub>' + s1 + ' + x<sub>12</sub>·θ<sub>12</sub>' + s1 +
          ' = (' + f(V.X[0][0], 2) + ')·(' + f(V.T1[0][0], 2) + ') + (' + f(V.X[0][1], 2) + ')·(' +
          f(V.T1[0][1], 2) + ') = ' + f(V.Z1[0][0]);
        case 2: return 'A' + s1 + ' = σ(Z' + s1 + ') : elemento a elemento em 2×3<br>' +
          'a<sub>11</sub>' + s1 + ' = σ(' + f(V.Z1[0][0]) + ') = ' + f(V.A1[0][0]);
        case 3: return 'Z' + s2 + ' = A' + s1 + '·θ<sup>(2)ᵀ</sup> : (2×3)·(3×1) → 2×1<br>' +
          'z<sub>1</sub>' + s2 + ' = (' + f(V.A1[0][0]) + ')·(' + f(V.T2[0][0], 2) + ') + (' +
          f(V.A1[0][1]) + ')·(' + f(V.T2[0][1], 2) + ') + (' + f(V.A1[0][2]) + ')·(' +
          f(V.T2[0][2], 2) + ') = ' + f(V.Z2[0][0]);
        case 4: return 'ŷ = A' + s2 + ' = σ(Z' + s2 + ') : elemento a elemento em 2×1<br>' +
          'ŷ<sub>1</sub> = σ(' + f(V.Z2[0][0]) + ') = ' + f(V.Yhat[0][0]);
        case 5: return 'J = ½Σ(y − ŷ)² = ½[(' + f(V.y[0][0], 2) + ' − ' + f(V.Yhat[0][0]) + ')² + (' +
          f(V.y[1][0], 2) + ' − ' + f(V.Yhat[1][0]) + ')²] = ' + f(V.J[0][0], 5);
        case 6: return 'δ' + s2 + ' = (ŷ − y) ⊙ ' + sp + '(Z' + s2 + ') : (2×1) ⊙ (2×1), com ' +
          sp + '(z) = σ(z)(1 − σ(z)) = ŷ(1 − ŷ)<br>' +
          'δ<sub>1</sub>' + s2 + ' = (' + f(V.Yhat[0][0]) + ' − ' + f(V.y[0][0], 2) + ')·(' +
          f(V.Yhat[0][0]) + ')·(1 − ' + f(V.Yhat[0][0]) + ') = ' + f(V.D2[0][0], 5);
        case 7: return 'δ' + s1 + ' = (δ' + s2 + '·θ' + s2 + ') ⊙ ' + sp + '(Z' + s1 +
          ') : (2×1)·(1×3) ⊙ (2×3) → 2×3<br>' +
          'δ<sub>11</sub>' + s1 + ' = (' + f(V.D2[0][0], 5) + ')·(' + f(V.T2[0][0], 2) + ')·(' +
          f(V.A1[0][0]) + ')·(1 − ' + f(V.A1[0][0]) + ') = ' + f(V.D1[0][0], 5);
        case 8: return '∂J/∂θ' + s2 + ' = δ<sup>(2)ᵀ</sup>·A' + s1 + ' : (1×2)·(2×3) → 1×3<br>' +
          '∂J/∂θ<sub>1</sub>' + s2 + ' = (' + f(V.D2[0][0], 5) + ')·(' + f(V.A1[0][0]) + ') + (' +
          f(V.D2[1][0], 5) + ')·(' + f(V.A1[1][0]) + ') = ' + f(V.GT2[0][0]);
        case 9: return '∂J/∂θ' + s1 + ' = δ<sup>(1)ᵀ</sup>·X : (3×2)·(2×2) → 3×2<br>' +
          '∂J/∂θ<sub>11</sub>' + s1 + ' = (' + f(V.D1[0][0], 5) + ')·(' + f(V.X[0][0], 2) + ') + (' +
          f(V.D1[1][0], 5) + ')·(' + f(V.X[1][0], 2) + ') = ' + f(V.GT1[0][0], 5);
      }
      return '';
    }

    /* ── estado e controles (mesmo padrão da seção 1) ── */

    function update() {
      compute();
      $('s2v-step-label').textContent = 'passo ' + step + '/' + LAST;
      $('s2v-desc').textContent = descFor(step);
      $('s2v-eq').innerHTML = eqFor(step);
      btnPrev.disabled = step === 0;
      btnNext.disabled = step === LAST;
      renderBoard();
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
      if (acc >= 1.6) {
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
      for (const key in sliders) {
        $('s2v-' + key + '-val').textContent = fnum(+sliders[key].value, key[0] === 'y' ? 2 : 1);
      }
    }

    for (const key in sliders) {
      sliders[key].addEventListener('input', () => { syncSliderLabels(); update(); });
    }

    btnPreset.addEventListener('click', () => {
      for (const key in sliders) sliders[key].value = DEFAULTS[key];
      syncSliderLabels();
      update();
    });

    buildBoard();
    syncSliderLabels();
    update();
  }

  DL.sections.push({ name: 's2v-vectorized', init });
})();
