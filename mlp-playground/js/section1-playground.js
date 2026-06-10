/* Seção 1: playground de treinamento de uma MLP 2D (fronteira de decisão ao vivo).
   MLP e backpropagation escritos à mão, otimização por descida de gradiente (SGD)
   em batch completo. Expõe o modelo em DL.mlpPlayground para a seção 2. */
(function () {
  'use strict';
  window.DL = window.DL || {};
  DL.sections = DL.sections || [];

  const N = 300;               // pontos por dataset (batch completo)
  const STEPS_PER_FRAME = 10;  // épocas por quadro de animação
  const CLIP = 5;              // clipping da norma global do gradiente
  const RANGE = 2.8;           // janela do plano de entrada
  const WSEED = 1234;          // semente fixa da inicialização dos pesos

  const ACTS = {
    tanh: { f: Math.tanh, df: (a) => 1 - a * a },
    relu: { f: (z) => (z > 0 ? z : 0), df: (a) => (a > 0 ? 1 : 0) },
    sigmoid: { f: (z) => 1 / (1 + Math.exp(-z)), df: (a) => a * (1 - a) },
  };

  /* ── modelo puro (sem DOM), também usado no gradient check ── */
  function createModel(seed, sizes, actName, utils) {
    const U = utils;
    const act = ACTS[actName];
    const rng = U.mulberry32(seed);
    const layers = [];
    for (let l = 0; l < sizes.length - 1; l++) {
      const nin = sizes[l], nout = sizes[l + 1];
      const W = new Float32Array(nout * nin);
      for (let i = 0; i < W.length; i++) W[i] = 0.5 * U.randn(rng);
      layers.push({ W, b: new Float32Array(nout), nin, nout });
    }
    const L = layers.length;

    /* Forward em batch; guarda as ativações de cada camada para o backward.
       acts[0] = entrada, acts[l+1] = saída da camada l (a última é o logit). */
    function forward(inp, B) {
      const acts = [inp];
      let prev = inp;
      for (let l = 0; l < L; l++) {
        const { W, b, nin, nout } = layers[l];
        const last = l === L - 1;
        const out = new Float32Array(B * nout);
        for (let bi = 0; bi < B; bi++) {
          for (let j = 0; j < nout; j++) {
            let s = b[j];
            for (let k = 0; k < nin; k++) s += W[j * nin + k] * prev[bi * nin + k];
            out[bi * nout + j] = last ? s : act.f(s);
          }
        }
        acts.push(out);
        prev = out;
      }
      return acts;
    }

    /* Loss + acurácia sem atualizar pesos (BCE numericamente estável). */
    function evalAll(inp, ys, B) {
      const logits = forward(inp, B)[L];
      let loss = 0, correct = 0;
      for (let bi = 0; bi < B; bi++) {
        const z = logits[bi], y = ys[bi];
        loss += Math.max(z, 0) - z * y + Math.log(1 + Math.exp(-Math.abs(z)));
        if ((z > 0 ? 1 : 0) === y) correct++;
      }
      return { loss: loss / B, acc: correct / B };
    }

    /* Gradientes de J = média da entropia cruzada binária sobre o batch.
       delta_l = ∂J/∂z^(l); na saída, ∂J/∂z = (σ(z) − y)/B. */
    function computeGrads(inp, ys, B) {
      const acts = forward(inp, B);
      const logits = acts[L];
      const grads = layers.map((l) => ({ W: new Float32Array(l.W.length), b: new Float32Array(l.nout) }));
      let loss = 0, correct = 0;
      let delta = new Float32Array(B);
      for (let bi = 0; bi < B; bi++) {
        const z = logits[bi], y = ys[bi];
        loss += Math.max(z, 0) - z * y + Math.log(1 + Math.exp(-Math.abs(z)));
        if ((z > 0 ? 1 : 0) === y) correct++;
        delta[bi] = (1 / (1 + Math.exp(-z)) - y) / B;
      }
      loss /= B;
      for (let l = L - 1; l >= 0; l--) {
        const { W, nin, nout } = layers[l];
        const aPrev = acts[l], g = grads[l];
        const dPrev = l > 0 ? new Float32Array(B * nin) : null;
        for (let bi = 0; bi < B; bi++) {
          for (let j = 0; j < nout; j++) {
            const d = delta[bi * nout + j];
            g.b[j] += d;
            for (let k = 0; k < nin; k++) {
              g.W[j * nin + k] += d * aPrev[bi * nin + k];
              if (dPrev) dPrev[bi * nin + k] += W[j * nin + k] * d;
            }
          }
        }
        if (dPrev) {
          /* ∂J/∂a^(l−1) vira ∂J/∂z^(l−1) multiplicando por φ′ (avaliada na própria ativação) */
          for (let i = 0; i < dPrev.length; i++) dPrev[i] *= act.df(aPrev[i]);
          delta = dPrev;
        }
      }
      return { loss, acc: correct / B, grads };
    }

    /* SGD com clipping da norma global: θ ← θ − η·∇J */
    function sgdUpdate(grads, lr) {
      let sq = 0;
      for (const g of grads) {
        for (let i = 0; i < g.W.length; i++) sq += g.W[i] * g.W[i];
        for (let j = 0; j < g.b.length; j++) sq += g.b[j] * g.b[j];
      }
      const norm = Math.sqrt(sq);
      const scale = norm > CLIP ? CLIP / norm : 1;
      for (let l = 0; l < L; l++) {
        const { W, b } = layers[l], g = grads[l];
        for (let i = 0; i < W.length; i++) W[i] -= lr * scale * g.W[i];
        for (let j = 0; j < b.length; j++) b[j] -= lr * scale * g.b[j];
      }
    }

    function trainStep(inp, ys, B, lr) {
      const { loss, acc, grads } = computeGrads(inp, ys, B);
      sgdUpdate(grads, lr);
      return { loss, acc };
    }

    /* Predição de um ponto só com buffers reaproveitados (usada no heatmap). */
    const maxW = Math.max.apply(null, sizes);
    const bufA = new Float32Array(maxW), bufB = new Float32Array(maxW);
    function predict1(x, y) {
      bufA[0] = x; bufA[1] = y;
      let cur = bufA, nxt = bufB;
      for (let l = 0; l < L; l++) {
        const { W, b, nin, nout } = layers[l];
        const last = l === L - 1;
        for (let j = 0; j < nout; j++) {
          let s = b[j];
          for (let k = 0; k < nin; k++) s += W[j * nin + k] * cur[k];
          nxt[j] = last ? s : act.f(s);
        }
        const tmp = cur; cur = nxt; nxt = tmp;
      }
      return 1 / (1 + Math.exp(-cur[0]));
    }

    /* Ativações da primeira camada escondida num ponto (usada na seção 2). */
    function hidden1(x, y, out) {
      const { W, b, nin, nout } = layers[0];
      for (let j = 0; j < nout; j++) {
        out[j] = act.f(b[j] + W[j * nin] * x + W[j * nin + 1] * y);
      }
    }

    return { layers, forward, evalAll, computeGrads, trainStep, predict1, hidden1 };
  }

  /* Checagem por diferenças finitas (rodar no console: DL.mlpModel.gradCheck()) */
  function gradCheck(utils, actName) {
    const U = utils || DL.utils;
    const model = createModel(123, [2, 5, 4, 1], actName || 'tanh', U);
    const rng = U.mulberry32(7);
    const B = 6;
    const inp = new Float32Array(B * 2), ys = new Float32Array(B);
    for (let i = 0; i < inp.length; i++) inp[i] = U.randn(rng);
    for (let i = 0; i < B; i++) ys[i] = rng() < 0.5 ? 0 : 1;
    const { grads } = model.computeGrads(inp, ys, B);
    /* eps pequeno o bastante para não cruzar o "joelho" da ReLU,
       grande o bastante para o arredondamento float32 não dominar */
    const eps = 1e-3;
    let maxRel = 0;
    for (let l = 0; l < model.layers.length; l++) {
      for (const key of ['W', 'b']) {
        const arr = model.layers[l][key];
        for (let trial = 0; trial < 8; trial++) {
          const i = Math.floor(rng() * arr.length);
          const orig = arr[i];
          arr[i] = orig + eps;
          const lp = model.computeGrads(inp, ys, B).loss;
          arr[i] = orig - eps;
          const lm = model.computeGrads(inp, ys, B).loss;
          arr[i] = orig;
          const fd = (lp - lm) / (2 * eps);
          const an = grads[l][key][i];
          /* gradientes minúsculos são dominados pelo arredondamento float32 */
          if (Math.abs(fd) + Math.abs(an) < 1e-4) continue;
          const rel = Math.abs(fd - an) / (Math.abs(fd) + Math.abs(an));
          if (rel > maxRel) maxRel = rel;
        }
      }
    }
    return maxRel;
  }

  DL.mlpModel = { createModel, gradCheck, ACTS };

  function init() {
    const U = DL.utils, P = DL.plot;
    const $ = (id) => document.getElementById(id);

    const cvBoundary = $('s1-boundary'), cvLoss = $('s1-loss');
    const btnTrain = $('s1-train'), btnReset = $('s1-reset'), btnResample = $('s1-resample');
    const selArch = $('s1-arch'), selAct = $('s1-act'), sliderLr = $('s1-lr');
    const tabs = {
      moons: $('s1-tab-moons'),
      circles: $('s1-tab-circles'),
      xor: $('s1-tab-xor'),
    };

    const DATASETS = {
      moons: (rng) => U.twoMoons(N, 0.12, rng),
      circles: (rng) => U.circles(N, 0.15, rng),
      xor: (rng) => U.xorData(N, 0.45, rng),
    };

    let seed = 42;
    let dsName = 'moons';
    let data = null, labels = null;
    let inp = null, ys = null;          // batch completo em Float32Array
    let model = null, actName = 'tanh';
    let lr = 0.1;
    let epochs = 0, lossHist = [], lastAcc = null, lastLoss = null;
    let training = false, frameCount = 0;
    const listeners = [];

    /* ── estado compartilhado com a seção 2 ── */
    function notify() { for (const cb of listeners) cb(); }
    DL.mlpPlayground = {
      subscribe(cb) { listeners.push(cb); },
      getState() {
        const squash = actName === 'tanh' ? (v) => v
          : actName === 'sigmoid' ? (v) => 2 * v - 1
            : (v) => Math.tanh(v);   // ReLU: comprime [0, ∞) para [0, 1)
        return { model, actName, h1: model.layers[0].nout, data, labels, squash, range: RANGE };
      },
    };

    function regenData() {
      const rng = U.mulberry32(seed);
      const ds = DATASETS[dsName](rng);
      data = ds.pts; labels = ds.lab;
      inp = new Float32Array(N * 2);
      ys = new Float32Array(N);
      for (let i = 0; i < N; i++) {
        inp[2 * i] = data[i][0];
        inp[2 * i + 1] = data[i][1];
        ys[i] = labels[i];
      }
    }

    function resetModel() {
      stopTraining();
      const sizes = [2].concat(selArch.value.split(',').map(Number)).concat([1]);
      actName = selAct.value;
      model = createModel(WSEED, sizes, actName, U);
      epochs = 0; lossHist = [];
      const ev = model.evalAll(inp, ys, N);
      lastLoss = ev.loss; lastAcc = ev.acc;
      updateStatus();
      drawBoundary(); drawLoss();
      notify();
    }

    function trainTick() {
      let res = null;
      for (let s = 0; s < STEPS_PER_FRAME; s++) {
        res = model.trainStep(inp, ys, N, lr);
        lossHist.push(res.loss);
        epochs++;
      }
      lastLoss = res.loss; lastAcc = res.acc;
      if (!isFinite(res.loss)) {
        stopTraining();
        $('s1-status').textContent = 'a loss divergiu: reduza η e resete os pesos';
        return;
      }
      frameCount++;
      updateStatus();
      drawLoss();
      if (frameCount % 6 === 0) drawBoundary();   // heatmap ~10x por segundo
      if (frameCount % 30 === 0) notify();        // mapas da seção 2, mais raro
    }

    function stopTraining() {
      if (!training) return;
      training = false;
      btnTrain.textContent = '▶ Treinar';
      DL.stopTicker(trainTick);
      drawBoundary();
      notify();
    }

    btnTrain.addEventListener('click', () => {
      if (training) { stopTraining(); return; }
      training = true;
      btnTrain.textContent = '⏸ Pausar';
      DL.startTicker(trainTick);
    });
    btnReset.addEventListener('click', resetModel);

    btnResample.addEventListener('click', () => {
      seed = (seed * 1664525 + 1013904223) >>> 0;
      regenData();
      const ev = model.evalAll(inp, ys, N);
      lastLoss = ev.loss; lastAcc = ev.acc;
      updateStatus();
      drawBoundary();
      notify();
    });

    for (const name of Object.keys(tabs)) {
      tabs[name].addEventListener('click', () => {
        if (dsName === name) return;
        dsName = name;
        for (const k of Object.keys(tabs)) tabs[k].classList.toggle('active', k === name);
        regenData();
        resetModel();   // dataset novo, problema novo: pesos do zero
      });
    }

    selArch.addEventListener('change', resetModel);
    selAct.addEventListener('change', resetModel);

    function lrFromSlider() {
      /* escala log: 0.001 (v = 0) até 0.3 (v = 100) */
      const v = +sliderLr.value / 100;
      return 0.001 * Math.pow(300, v);
    }
    sliderLr.addEventListener('input', () => {
      lr = lrFromSlider();
      $('s1-lr-val').textContent = lr.toFixed(3);
    });
    lr = lrFromSlider();
    $('s1-lr-val').textContent = lr.toFixed(3);

    function updateStatus() {
      $('s1-status').textContent = 'épocas: ' + epochs +
        ' · loss: ' + (lastLoss === null ? '—' : lastLoss.toFixed(3)) +
        ' · acurácia: ' + (lastAcc === null ? '—' : (100 * lastAcc).toFixed(1) + '%');
    }

    function drawBoundary() {
      const th = P.theme();
      const { ctx, w, h } = P.setup(cvBoundary);
      P.clear(ctx, w, h);
      const fr = P.frame(ctx, w, h, -RANGE, RANGE, -RANGE, RANGE);
      const cCyan = P.rgb(th.cyan), cOrange = P.rgb(th.orange);
      P.heatmap(ctx, fr, (x, y) => model.predict1(x, y), 90, 90, (p) => {
        const c = p < 0.5 ? cCyan : cOrange;
        const a = Math.min(1, Math.abs(p - 0.5) * 2) * 90;
        return [c[0], c[1], c[2], a];
      });
      P.axes(ctx, fr, { xlabel: 'x₁', ylabel: 'x₂', xticks: [-2, 0, 2], yticks: [-2, 0, 2] });
      P.scatter(ctx, fr, data.filter((_, i) => labels[i] === 0), th.cyan, { r: 2.5, alpha: 0.85 });
      P.scatter(ctx, fr, data.filter((_, i) => labels[i] === 1), th.orange, { r: 2.5, alpha: 0.85 });
    }

    function drawLoss() {
      const th = P.theme();
      const { ctx, w, h } = P.setup(cvLoss);
      P.clear(ctx, w, h);
      const ymin = Math.log10(0.01), ymax = Math.log10(2);
      const xmax = Math.max(500, lossHist.length);
      const fr = P.frame(ctx, w, h, 0, xmax, ymin, ymax);
      P.axes(ctx, fr, { xlabel: 'épocas', xticks: [0, Math.round(xmax / 2), xmax] });
      const refs = [[0.01, '0.01'], [0.1, '0.1'], [Math.LN2, 'ln 2']];
      for (const [v, txt] of refs) {
        P.line(ctx, fr, [0, xmax], [Math.log10(v), Math.log10(v)], th.line, { dash: [3, 4], width: 1 });
        P.label(ctx, fr.X(0) - 4, fr.Y(Math.log10(v)) + 4, txt, th.comment, 'right');
      }
      if (lossHist.length > 1) {
        const stride = Math.max(1, Math.floor(lossHist.length / 600));
        const xs = [], lys = [];
        for (let i = 0; i < lossHist.length; i += stride) {
          xs.push(i);
          lys.push(Math.log10(Math.max(0.01, lossHist[i])));
        }
        P.line(ctx, fr, xs, lys, th.green, { width: 2 });
      }
    }

    regenData();
    resetModel();
    P.onRedraw(() => { drawBoundary(); drawLoss(); });
    P.observeResize(cvBoundary, drawBoundary);
    P.observeResize(cvLoss, drawLoss);
  }

  DL.sections.push({ name: 's1-playground', init });
})();
