/* Seção 3: classificação de sentença sobre h_[CLS] em 2D com cabeça logística,
   comparando feature extraction (encoder congelado) e fine-tuning (os
   embeddings também recebem gradiente). */
(function () {
  'use strict';
  window.DL = window.DL || {};
  DL.sections = DL.sections || [];

  const POS = [
    'que filme maravilhoso',
    'adorei este filme',
    'atuação excelente do elenco',
    'roteiro criativo e emocionante',
    'fotografia linda demais',
    'uma obra prima do cinema',
    'recomendo muito este filme',
    'trilha sonora perfeita',
    'final surpreendente e lindo',
    'o melhor filme do ano',
  ];
  const NEG = [
    'que filme horrível',
    'odiei este filme',
    'atuação fraca do elenco',
    'roteiro confuso e arrastado',
    'fotografia escura e feia',
    'um desastre completo',
    'não recomendo este filme',
    'trilha sonora irritante',
    'final previsível e bobo',
    'o pior filme do ano',
  ];

  const DOMAIN = 3.5, MAX_STEPS = 600, FT_FACTOR = 0.08;

  function init() {
    const P = DL.plot;
    const U = DL.utils;
    const $ = (id) => document.getElementById(id);

    const cvSpace = $('s3-space'), cvLoss = $('s3-loss');
    const slEta = $('s3-eta');
    const btnRun = $('s3-run');

    let seed = 3;
    let eta = +slEta.value;
    let mode = 'frozen'; /* 'frozen' | 'ft' */
    let basePts = [], pts = [], labels = [];
    let sentences = [];
    let wgt = [0, 0], bias = 0;
    let lossHist = [], accHist = [];
    let running = false, acc = 0;
    let sel = -1, frSpace = null;

    const sigmoid = (z) => 1 / (1 + Math.exp(-z));

    function sampleEmbeddings() {
      const rng = U.mulberry32(seed);
      basePts = [];
      labels = [];
      sentences = [];
      for (let i = 0; i < POS.length; i++) {
        basePts.push([-1.0 + 0.8 * U.randn(rng), 0.6 + 0.8 * U.randn(rng)]);
        labels.push(1);
        sentences.push(POS[i]);
      }
      for (let i = 0; i < NEG.length; i++) {
        basePts.push([1.0 + 0.8 * U.randn(rng), -0.5 + 0.8 * U.randn(rng)]);
        labels.push(0);
        sentences.push(NEG[i]);
      }
    }

    function resetTraining() {
      pts = basePts.map((p) => p.slice());
      wgt = [0, 0];
      bias = 0;
      lossHist = [];
      accHist = [];
      pushMetrics();
    }

    function forward(p) {
      return sigmoid(wgt[0] * p[0] + wgt[1] * p[1] + bias);
    }

    function metrics() {
      const n = pts.length;
      let loss = 0, hits = 0;
      for (let i = 0; i < n; i++) {
        const pr = forward(pts[i]);
        loss += -(labels[i] * Math.log(Math.max(pr, 1e-9)) +
          (1 - labels[i]) * Math.log(Math.max(1 - pr, 1e-9)));
        if ((pr > 0.5 ? 1 : 0) === labels[i]) hits++;
      }
      return { loss: loss / n, accuracy: hits / n };
    }

    function pushMetrics() {
      const m = metrics();
      lossHist.push(m.loss);
      accHist.push(m.accuracy);
    }

    function stepOnce() {
      const n = pts.length;
      let gw0 = 0, gw1 = 0, gb = 0;
      const errs = new Array(n);
      for (let i = 0; i < n; i++) {
        const e = forward(pts[i]) - labels[i];
        errs[i] = e;
        gw0 += e * pts[i][0] / n;
        gw1 += e * pts[i][1] / n;
        gb += e / n;
      }
      wgt[0] -= eta * gw0;
      wgt[1] -= eta * gw1;
      bias -= eta * gb;
      if (mode === 'ft') {
        /* Gradiente também flui para os embeddings: ∂ℒᵢ/∂xᵢ = (pᵢ − yᵢ)·w */
        const lim = DOMAIN - 0.15;
        for (let i = 0; i < n; i++) {
          pts[i][0] = Math.max(-lim, Math.min(lim, pts[i][0] - eta * FT_FACTOR * errs[i] * wgt[0]));
          pts[i][1] = Math.max(-lim, Math.min(lim, pts[i][1] - eta * FT_FACTOR * errs[i] * wgt[1]));
        }
      }
      pushMetrics();
    }

    function done() {
      return lossHist.length > MAX_STEPS || lossHist[lossHist.length - 1] < 0.02;
    }

    function drawSpace() {
      const th = P.theme();
      const { ctx, w, h } = P.setup(cvSpace);
      P.clear(ctx, w, h);
      const fr = P.frame(ctx, w, h, -DOMAIN, DOMAIN, -DOMAIN, DOMAIN);
      frSpace = fr;

      const gc = P.rgb(th.green), rc = P.rgb(th.red);
      P.heatmap(ctx, fr, (x, y) => sigmoid(wgt[0] * x + wgt[1] * y + bias), 80, 80, (v) => {
        const c = v > 0.5 ? gc : rc;
        const a = 8 + Math.abs(v - 0.5) * 2 * 70;
        return [c[0], c[1], c[2], a];
      });
      P.axes(ctx, fr, { xlabel: 'h_1', ylabel: 'h_2', xticks: [-3, 0, 3], yticks: [-3, 0, 3] });

      /* Fronteira w·h + b = 0 */
      if (Math.abs(wgt[0]) + Math.abs(wgt[1]) > 1e-6) {
        const xs = [], ys = [];
        if (Math.abs(wgt[1]) >= Math.abs(wgt[0])) {
          for (const x of [-DOMAIN, DOMAIN]) {
            xs.push(x);
            ys.push(-(wgt[0] * x + bias) / wgt[1]);
          }
        } else {
          for (const y of [-DOMAIN, DOMAIN]) {
            ys.push(y);
            xs.push(-(wgt[1] * y + bias) / wgt[0]);
          }
        }
        P.line(ctx, fr, xs, ys, th.fg, { width: 1.4, dash: [5, 4], alpha: 0.8 });
      }

      for (let i = 0; i < pts.length; i++) {
        const color = labels[i] === 1 ? th.green : th.red;
        ctx.beginPath();
        ctx.arc(fr.X(pts[i][0]), fr.Y(pts[i][1]), i === sel ? 6 : 4.2, 0, 2 * Math.PI);
        ctx.fillStyle = color;
        ctx.globalAlpha = 0.9;
        ctx.fill();
        ctx.globalAlpha = 1;
        if (i === sel) {
          ctx.strokeStyle = th.fg;
          ctx.lineWidth = 2;
          ctx.stroke();
        }
      }
      P.label(ctx, fr.X(-DOMAIN) + 8, fr.Y(DOMAIN) + 14, 'positivo', th.green);
      P.label(ctx, fr.X(-DOMAIN) + 8, fr.Y(DOMAIN) + 28, 'negativo', th.red);
    }

    function drawLoss() {
      const th = P.theme();
      const { ctx, w, h } = P.setup(cvLoss);
      P.clear(ctx, w, h);
      const n = Math.max(50, lossHist.length - 1);
      const fr = P.frame(ctx, w, h, 0, n, 0, 1.05);
      P.axes(ctx, fr, {
        xlabel: 'passo', ylabel: 'ℒ / acurácia',
        xticks: [0, Math.round(n / 2), n],
        yticks: [0, 0.5, 1],
      });
      const ks = lossHist.map((_, i) => i);
      P.line(ctx, fr, ks, lossHist.map((v) => Math.min(v, 1.05)), th.cyan, { width: 1.8 });
      P.line(ctx, fr, ks, accHist, th.orange, { width: 1.8 });
    }

    function updateReadout() {
      const m = metrics();
      let txt = 'passo ' + (lossHist.length - 1) +
        ' · ℒ = ' + m.loss.toFixed(3) +
        ' · acurácia = ' + Math.round(m.accuracy * 100) + '%';
      if (sel >= 0) {
        txt += ' · "' + sentences[sel] + '" (p = ' + forward(pts[sel]).toFixed(2) + ')';
      }
      $('s3-readout').textContent = txt;
    }

    function redraw() {
      drawSpace();
      drawLoss();
      updateReadout();
    }

    function stopRun() {
      running = false;
      btnRun.textContent = '▶ Treinar';
      DL.stopTicker(tick);
    }

    function tick(dt) {
      acc += dt * 40;
      while (acc >= 1) { acc -= 1; stepOnce(); }
      redraw();
      if (done()) stopRun();
    }

    cvSpace.addEventListener('pointerdown', (e) => {
      if (!frSpace) return;
      const r = cvSpace.getBoundingClientRect();
      const px = e.clientX - r.left, py = e.clientY - r.top;
      let best = -1, bd = 16;
      for (let i = 0; i < pts.length; i++) {
        const d = Math.hypot(frSpace.X(pts[i][0]) - px, frSpace.Y(pts[i][1]) - py);
        if (d < bd) { bd = d; best = i; }
      }
      sel = best;
      redraw();
    });

    btnRun.addEventListener('click', () => {
      if (running) { stopRun(); return; }
      if (done()) resetTraining();
      running = true;
      acc = 0;
      btnRun.textContent = '⏸ Pausar';
      DL.startTicker(tick);
    });
    $('s3-reset').addEventListener('click', () => {
      stopRun();
      resetTraining();
      redraw();
    });
    $('s3-sample').addEventListener('click', () => {
      stopRun();
      seed = (seed + 1) >>> 0;
      sel = -1;
      sampleEmbeddings();
      resetTraining();
      redraw();
    });
    slEta.addEventListener('input', () => {
      eta = +slEta.value;
      $('s3-eta-val').textContent = eta.toFixed(2);
    });

    const tabFrozen = $('s3-tab-frozen'), tabFt = $('s3-tab-ft');
    function setMode(m) {
      mode = m;
      tabFrozen.classList.toggle('active', m === 'frozen');
      tabFt.classList.toggle('active', m === 'ft');
      stopRun();
      resetTraining();
      redraw();
    }
    tabFrozen.addEventListener('click', () => setMode('frozen'));
    tabFt.addEventListener('click', () => setMode('ft'));

    sampleEmbeddings();
    resetTraining();
    redraw();
    P.onRedraw(redraw);
    P.observeResize(cvSpace, redraw);
    P.observeResize(cvLoss, redraw);
  }

  DL.sections.push({ name: 's3-cls', init });
})();
