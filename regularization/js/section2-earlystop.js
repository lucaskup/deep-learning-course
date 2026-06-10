/* Seção 2: early stopping com um MLP 1→24(tanh)→1 treinado ao vivo por GD. */
(function () {
  'use strict';
  window.DL = window.DL || {};
  DL.sections = DL.sections || [];

  const H = 24, NTRAIN = 14, NVAL = 80, NOISE = 0.35, LR = 0.1;
  const EPOCHS_PER_FRAME = 60, MAX_EPOCHS = 200000;
  const YMIN = -2, YMAX = 2;

  function trueF(x) { return Math.sin(2 * Math.PI * x); }

  /* Rede pura (sem DOM), exposta em DL.esTest para checagem em node/console. */
  function createNet(seed, utils) {
    const U = utils;
    const rng = U.mulberry32(seed);
    const p = {
      w1: new Float64Array(H), b1: new Float64Array(H),
      w2: new Float64Array(H), b2: new Float64Array(1),
    };
    for (let i = 0; i < H; i++) {
      p.w1[i] = U.randn(rng) * 3;
      p.b1[i] = -p.w1[i] * (2 * rng() - 1);   /* espalha os "joelhos" da tanh em [−1,1] */
      p.w2[i] = U.randn(rng) * 0.1;
    }
    p.b2[0] = 0;

    function forward(x) {
      let s = p.b2[0];
      for (let i = 0; i < H; i++) s += p.w2[i] * Math.tanh(p.w1[i] * x + p.b1[i]);
      return s;
    }

    function loss(xs, ys) {
      let s = 0;
      for (let i = 0; i < xs.length; i++) {
        const e = forward(xs[i]) - ys[i];
        s += e * e;
      }
      return s / xs.length;
    }

    /* Uma época de gradiente descendente full-batch sobre o MSE. */
    function epoch(xs, ys, lr) {
      const n = xs.length;
      const gw1 = new Float64Array(H), gb1 = new Float64Array(H);
      const gw2 = new Float64Array(H);
      let gb2 = 0, L = 0;
      for (let k = 0; k < n; k++) {
        const x = xs[k];
        let pred = p.b2[0];
        const a = new Float64Array(H);
        for (let i = 0; i < H; i++) {
          a[i] = Math.tanh(p.w1[i] * x + p.b1[i]);
          pred += p.w2[i] * a[i];
        }
        const err = pred - ys[k];
        L += err * err;
        const dPred = 2 * err / n;
        gb2 += dPred;
        for (let i = 0; i < H; i++) {
          gw2[i] += dPred * a[i];
          const dPre = dPred * p.w2[i] * (1 - a[i] * a[i]);
          gw1[i] += dPre * x;
          gb1[i] += dPre;
        }
      }
      for (let i = 0; i < H; i++) {
        p.w1[i] -= lr * gw1[i];
        p.b1[i] -= lr * gb1[i];
        p.w2[i] -= lr * gw2[i];
      }
      p.b2[0] -= lr * gb2;
      return L / n;
    }

    function snapshot() {
      return { w1: p.w1.slice(), b1: p.b1.slice(), w2: p.w2.slice(), b2: p.b2.slice() };
    }
    function restore(snap) {
      p.w1.set(snap.w1); p.b1.set(snap.b1); p.w2.set(snap.w2); p.b2.set(snap.b2);
    }

    return { p, forward, loss, epoch, snapshot, restore };
  }

  DL.esTest = { createNet, H, trueF };

  function init() {
    const U = DL.utils, P = DL.plot;
    const $ = (id) => document.getElementById(id);

    const cvFit = $('s2-fit'), cvLoss = $('s2-loss');
    const btnTrain = $('s2-train'), btnReset = $('s2-reset');
    const btnRestore = $('s2-restore'), btnResample = $('s2-resample');

    let dataSeed = 7, netSeed = 1234;
    let train = null, val = null;
    let net = null, epochCount = 0;
    let trainHist = [], valHist = [];
    let bestVal = Infinity, bestEpoch = -1, bestSnap = null;
    let training = false;

    function regenerate() {
      const rng = U.mulberry32(dataSeed);
      train = U.noisyCurve(NTRAIN, trueF, NOISE, -1, 1, rng);
      val = U.noisyCurve(NVAL, trueF, NOISE, -1, 1, rng);
    }

    function resetNet() {
      stopTraining();
      net = createNet(netSeed, U);
      epochCount = 0;
      trainHist = []; valHist = [];
      bestVal = Infinity; bestEpoch = -1; bestSnap = null;
      btnRestore.disabled = true;
      updateStatus();
      redraw();
    }

    function fmtLoss(v) {
      if (v == null) return '—';
      return v >= 0.01 ? v.toFixed(3) : v.toExponential(1);
    }

    function updateStatus() {
      $('s2-status').textContent =
        'época: ' + epochCount +
        ' · treino: ' + fmtLoss(trainHist.length ? trainHist[trainHist.length - 1] : null) +
        ' · validação: ' + fmtLoss(valHist.length ? valHist[valHist.length - 1] : null) +
        ' · melhor época: ' + (bestEpoch < 0 ? '—' : bestEpoch);
    }

    function trainTick() {
      for (let s = 0; s < EPOCHS_PER_FRAME && epochCount < MAX_EPOCHS; s++) {
        const tl = net.epoch(train.xs, train.ys, LR);
        const vl = net.loss(val.xs, val.ys);
        epochCount++;
        trainHist.push(tl);
        valHist.push(vl);
        if (vl < bestVal) {
          bestVal = vl;
          bestEpoch = epochCount;
          bestSnap = net.snapshot();
          btnRestore.disabled = false;
        }
      }
      if (epochCount >= MAX_EPOCHS) stopTraining();
      updateStatus();
      redraw();
    }

    function stopTraining() {
      training = false;
      btnTrain.textContent = '▶ Treinar';
      DL.stopTicker(trainTick);
    }

    btnTrain.addEventListener('click', () => {
      if (training) { stopTraining(); return; }
      training = true;
      btnTrain.textContent = '⏸ Pausar';
      DL.startTicker(trainTick);
    });
    btnReset.addEventListener('click', resetNet);
    btnRestore.addEventListener('click', () => {
      if (!bestSnap) return;
      stopTraining();
      net.restore(bestSnap);
      updateStatus();
      redraw();
    });
    btnResample.addEventListener('click', () => {
      dataSeed = (dataSeed * 1664525 + 1013904223) >>> 0;
      netSeed = (netSeed * 1664525 + 1013904223) >>> 0;
      regenerate();
      resetNet();
    });

    function drawFit() {
      const th = P.theme();
      const { ctx, w: cw, h: ch } = P.setup(cvFit);
      P.clear(ctx, cw, ch);
      const fr = P.frame(ctx, cw, ch, -1.08, 1.08, YMIN, YMAX);
      P.axes(ctx, fr, { xlabel: 'x', ylabel: 'y', xticks: [-1, 0, 1], yticks: [-2, 0, 2] });

      ctx.save();
      ctx.beginPath();
      ctx.rect(fr.X(fr.xmin), fr.Y(fr.ymax), fr.iw, fr.ih);
      ctx.clip();
      const xs = [], yTrue = [], yNet = [];
      for (let x = -1.08; x <= 1.081; x += 0.01) {
        xs.push(x);
        yTrue.push(trueF(x));
        yNet.push(net.forward(x));
      }
      P.line(ctx, fr, xs, yTrue, th.comment, { dash: [5, 4], width: 1.4 });
      P.line(ctx, fr, xs, yNet, th.purple, { width: 2.2 });
      ctx.restore();

      P.scatter(ctx, fr, val.xs.map((x, i) => [x, val.ys[i]]), th.orange, { r: 2, alpha: 0.3 });
      P.scatter(ctx, fr, train.xs.map((x, i) => [x, train.ys[i]]), th.cyan, { r: 3.5, alpha: 0.95 });
      P.label(ctx, fr.X(-1.08) + 8, 14, 'MLP (época ' + epochCount + ')', th.purple);
    }

    function drawLoss() {
      const th = P.theme();
      const { ctx, w: cw, h: ch } = P.setup(cvLoss);
      P.clear(ctx, cw, ch);
      const ymin = -2.6, ymax = 0.6;   /* log10 da loss */
      const xmax = Math.max(2000, epochCount);
      const fr = P.frame(ctx, cw, ch, 0, xmax, ymin, ymax);
      P.axes(ctx, fr, { xlabel: 'época', xticks: [0, Math.round(xmax / 2), xmax] });

      for (const v of [0.01, 0.1, 1]) {
        P.line(ctx, fr, [0, xmax], [Math.log10(v), Math.log10(v)], th.line, { dash: [3, 4], width: 1 });
        P.label(ctx, fr.X(0) - 4, fr.Y(Math.log10(v)) + 4, String(v), th.comment, 'right');
      }

      const clamp = (v) => Math.min(ymax, Math.max(ymin, Math.log10(Math.max(v, 1e-12))));
      if (trainHist.length > 1) {
        const stride = Math.max(1, Math.floor(trainHist.length / 600));
        const ex = [], ty = [], vy = [];
        for (let i = 0; i < trainHist.length; i += stride) {
          ex.push(i + 1);
          ty.push(clamp(trainHist[i]));
          vy.push(clamp(valHist[i]));
        }
        P.line(ctx, fr, ex, ty, th.cyan, { width: 1.8 });
        P.line(ctx, fr, ex, vy, th.orange, { width: 1.8 });
      }
      if (bestEpoch > 0) {
        P.line(ctx, fr, [bestEpoch, bestEpoch], [ymin, ymax], th.purple, { dash: [4, 3], width: 1.4 });
        P.label(ctx, fr.X(bestEpoch) + 4, fr.Y(ymax) + 12, 'melhor', th.purple);
      }
      P.label(ctx, fr.X(0) + 8, 14, 'validação', th.orange);
      P.label(ctx, fr.X(0) + 8, 28, 'treino', th.cyan);
    }

    function redraw() { drawFit(); drawLoss(); }

    regenerate();
    resetNet();
    P.onRedraw(redraw);
    P.observeResize(cvFit, drawFit);
    P.observeResize(cvLoss, drawLoss);
  }

  DL.sections.push({ name: 's2-earlystop', init });
})();
