/* Seção 2: perda de Bradley-Terry do Reward Model, ℒ_RM = −log σ(Δr),
   com o exemplo numérico dos slides e um RM de brinquedo treinado em pares. */
(function () {
  'use strict';
  window.DL = window.DL || {};
  DL.sections = DL.sections || [];

  const NPAIRS = 10, MAX_ITERS = 300;
  const sigmoid = (z) => 1 / (1 + Math.exp(-z));

  function init() {
    const P = DL.plot;
    const $ = (id) => document.getElementById(id);

    const cvBT = $('s2-bt'), cvTrain = $('s2-train');
    const slRw = $('s2-rw'), slRl = $('s2-rl'), slEta = $('s2-eta');
    const btnScenA = $('s2-scenA'), btnScenB = $('s2-scenB');
    const btnStep = $('s2-step'), btnRun = $('s2-run');
    const btnReset = $('s2-reset'), btnSeed = $('s2-seed');

    let rw = +slRw.value, rl = +slRl.value, eta = +slEta.value;
    let seed = 11;
    let pairs, k;
    let running = false, acc = 0;

    function resetPairs() {
      const rng = DL.utils.mulberry32(seed);
      pairs = [];
      for (let i = 0; i < NPAIRS; i++) {
        pairs.push({ w: 1.5 * DL.utils.randn(rng), l: 1.5 * DL.utils.randn(rng) });
      }
      k = 0;
    }

    function meanLoss() {
      let s = 0;
      for (const p of pairs) s += -Math.log(Math.max(1e-12, sigmoid(p.w - p.l)));
      return s / NPAIRS;
    }

    function trainStep() {
      for (const p of pairs) {
        const g = sigmoid(-(p.w - p.l)); /* |∂ℒ/∂Δr| do par */
        p.w += eta * g;
        p.l -= eta * g;
      }
      k++;
    }

    function drawBT() {
      const th = P.theme();
      const { ctx, w, h } = P.setup(cvBT);
      P.clear(ctx, w, h);
      const YMAX = 3.4;
      const fr = P.frame(ctx, w, h, -6, 6, 0, YMAX);
      P.axes(ctx, fr, {
        xlabel: 'Δr = r(y_w) − r(y_l)', ylabel: '',
        xticks: [-6, -3, 0, 3, 6],
        yticks: [0, 1, 2, 3],
      });
      const xs = [], sg = [], ls = [];
      for (let i = 0; i <= 240; i++) {
        const x = -6 + 12 * i / 240;
        xs.push(x);
        sg.push(sigmoid(x));
        ls.push(Math.min(YMAX, -Math.log(Math.max(1e-12, sigmoid(x)))));
      }
      P.line(ctx, fr, xs, sg, th.green, { width: 2 });
      P.line(ctx, fr, xs, ls, th.orange, { width: 2 });
      P.line(ctx, fr, [-6, 6], [1, 1], th.comment, { width: 1, alpha: 0.5, dash: [3, 4] });

      /* marcador do par atual */
      const d = rw - rl;
      const dC = Math.max(-6, Math.min(6, d));
      const sigD = sigmoid(d);
      const lossD = Math.min(YMAX, -Math.log(Math.max(1e-12, sigD)));
      P.line(ctx, fr, [dC, dC], [0, YMAX], th.fg, { width: 1, alpha: 0.6, dash: [4, 4] });
      P.scatter(ctx, fr, [[dC, sigD]], th.green, { r: 5, alpha: 1 });
      P.scatter(ctx, fr, [[dC, lossD]], th.orange, { r: 5, alpha: 1 });
      P.label(ctx, fr.X(-5.7), fr.Y(0.93) - 6, 'σ(Δr): P(y_w ≻ y_l)', th.green);
      P.label(ctx, fr.X(-5.7), fr.Y(2.8), 'ℒ_RM = −log σ(Δr)', th.orange);
      P.label(ctx, fr.X(dC) + 6, fr.Y(YMAX) + 14, 'Δr = ' + d.toFixed(1), th.fg);
    }

    function drawTrain() {
      const th = P.theme();
      const { ctx, w, h } = P.setup(cvTrain);
      P.clear(ctx, w, h);
      const fr = P.frame(ctx, w, h, 0.5, NPAIRS + 0.5, -5, 5);
      P.axes(ctx, fr, {
        xlabel: 'par i', ylabel: 'r_φ',
        xticks: [1, 5, 10],
        yticks: [-4, 0, 4],
      });
      P.line(ctx, fr, [0.5, NPAIRS + 0.5], [0, 0], th.comment, { width: 1, alpha: 0.5, dash: [3, 4] });
      const clip = (v) => Math.max(-5, Math.min(5, v));
      for (let i = 0; i < NPAIRS; i++) {
        const x = i + 1;
        const yw = clip(pairs[i].w), yl = clip(pairs[i].l);
        P.line(ctx, fr, [x, x], [yl, yw], th.comment, { width: 1.2, alpha: 0.7 });
        P.scatter(ctx, fr, [[x, yw]], th.green, { r: 4, alpha: 1 });
        P.scatter(ctx, fr, [[x, yl]], th.red, { r: 4, alpha: 1 });
      }
      P.label(ctx, fr.X(0.7), fr.Y(5) + 14, 'r(y_w)', th.green);
      P.label(ctx, fr.X(0.7), fr.Y(5) + 28, 'r(y_l)', th.red);
    }

    function updateReadout() {
      const d = rw - rl;
      const sigD = sigmoid(d);
      const lossD = -Math.log(Math.max(1e-12, sigD));
      $('s2-readout').textContent =
        'Δr = ' + d.toFixed(1) + ' · σ(Δr) = ' + sigD.toFixed(3) +
        ' · ℒ_RM = ' + lossD.toFixed(3) +
        ' · treino: k = ' + k + ', ℒ média = ' + meanLoss().toFixed(3);
    }

    function redraw() {
      drawBT();
      drawTrain();
      updateReadout();
    }

    function doneTrain() {
      return k >= MAX_ITERS || meanLoss() < 0.02;
    }

    function stopRun() {
      running = false;
      btnRun.textContent = '▶ Treinar RM';
      DL.stopTicker(tick);
    }

    function tick(dt) {
      acc += dt * 10;
      while (acc >= 1 && !doneTrain()) { acc -= 1; trainStep(); }
      redraw();
      if (doneTrain()) stopRun();
    }

    function setScenario(w0, l0) {
      rw = w0; rl = l0;
      slRw.value = String(w0);
      slRl.value = String(l0);
      $('s2-rw-val').textContent = w0.toFixed(1);
      $('s2-rl-val').textContent = l0.toFixed(1);
      redraw();
    }

    btnScenA.addEventListener('click', () => setScenario(2.1, -0.8));
    btnScenB.addEventListener('click', () => setScenario(-0.8, 2.1));

    slRw.addEventListener('input', () => {
      rw = +slRw.value;
      $('s2-rw-val').textContent = rw.toFixed(1);
      redraw();
    });
    slRl.addEventListener('input', () => {
      rl = +slRl.value;
      $('s2-rl-val').textContent = rl.toFixed(1);
      redraw();
    });
    slEta.addEventListener('input', () => {
      eta = +slEta.value;
      $('s2-eta-val').textContent = eta.toFixed(2);
    });

    btnStep.addEventListener('click', () => { trainStep(); redraw(); });
    btnRun.addEventListener('click', () => {
      if (running) { stopRun(); return; }
      if (doneTrain()) resetPairs();
      running = true;
      acc = 0;
      btnRun.textContent = '⏸ Pausar';
      DL.startTicker(tick);
    });
    btnReset.addEventListener('click', () => { stopRun(); resetPairs(); redraw(); });
    btnSeed.addEventListener('click', () => {
      stopRun();
      seed = (seed * 1664525 + 1013904223) >>> 0;
      resetPairs();
      redraw();
    });

    resetPairs();
    redraw();
    P.onRedraw(redraw);
    P.observeResize(cvBT, redraw);
    P.observeResize(cvTrain, redraw);
  }

  DL.sections.push({ name: 's2-reward-model', init });
})();
