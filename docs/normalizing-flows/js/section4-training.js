/* Seção 4: treino por máxima verossimilhança de um flow real em 2D.
   K camadas de affine coupling alternando a metade transformada; cada
   coupling network é um MLP 1→H→2 que produz (s, t), com s = SCAP·tanh(·).
   Direção de treino: x → z (normalizadora); amostragem: inversas em ordem
   contrária. NLL = ½‖z‖² − Σ s + log(2π), backprop manual + Adam. */
(function () {
  'use strict';
  window.DL = window.DL || {};
  DL.sections = DL.sections || [];

  const H = 16, SCAP = 1.5, B = 96, LR = 0.003;
  const BETA1 = 0.9, BETA2 = 0.999, EPS = 1e-8;
  const LOG2PI = Math.log(2 * Math.PI);
  const NDATA = 600, NZ = 400, LIM = 3, HN = 70;
  const STEPS_PER_TICK = 8, MAX_STEPS = 30000;

  function init() {
    const P = DL.plot;
    const U = DL.utils;
    const $ = (id) => document.getElementById(id);

    const cvPlane = $('s4-plane'), cvNll = $('s4-nll');
    const slK = $('s4-k');
    const btnTrain = $('s4-train'), btnStage = $('s4-stage');
    const tabMoons = $('s4-tab-moons'), tabCircles = $('s4-tab-circles');

    let K = +slK.value;
    let dataset = 'moons';
    let seed = 11;
    let data = [], zBatch = [];
    let layers = [], tensors = [], tAdam = 0;
    let rngT;
    let step = 0, nllHist = [], emaHist = [];
    let running = false;
    let stagePos = K, stageTarget = K, stagesCache = null;
    let heat = new Float64Array(HN * HN), heatMax = 1e-9, lastHeat = -1;

    /* Buffers de cache do batch (realocados quando K muda) */
    let cA, cB, cS, cT, cES, cH;
    function allocCaches() {
      cA = new Float64Array(K * B);
      cB = new Float64Array(K * B);
      cS = new Float64Array(K * B);
      cT = new Float64Array(K * B);
      cES = new Float64Array(K * B);
      cH = new Float64Array(K * B * H);
    }

    function makeData() {
      const rng = U.mulberry32(seed);
      data = dataset === 'moons'
        ? U.twoMoons(NDATA, 0.09, rng).pts
        : U.circles(NDATA, 0.08, rng).pts;
      const rngZ = U.mulberry32(seed + 303);
      zBatch = [];
      for (let i = 0; i < NZ; i++) zBatch.push([U.randn(rngZ), U.randn(rngZ)]);
    }

    function initModel() {
      const rngW = U.mulberry32(seed + 101);
      rngT = U.mulberry32(seed + 202);
      layers = [];
      tensors = [];
      for (let l = 0; l < K; l++) {
        const L = {
          W1: new Float64Array(H), b1: new Float64Array(H),
          W2: new Float64Array(2 * H), b2: new Float64Array(2),
        };
        for (let j = 0; j < H; j++) L.W1[j] = U.randn(rngW);
        for (let j = 0; j < 2 * H; j++) L.W2[j] = 0.1 * U.randn(rngW);
        layers.push(L);
        for (const key of ['W1', 'b1', 'W2', 'b2']) {
          const p = L[key];
          tensors.push({
            p,
            g: new Float64Array(p.length),
            m: new Float64Array(p.length),
            v: new Float64Array(p.length),
          });
        }
      }
      tAdam = 0;
      step = 0;
      nllHist = [];
      emaHist = [];
      stagesCache = null;
      stagePos = K;
      stageTarget = K;
      lastHeat = -1;
      allocCaches();
    }

    /* Coupling network do layer L no escalar a: devolve [s, t]. */
    function netEval(L, a, out) {
      let sr = L.b2[0], tr = L.b2[1];
      for (let j = 0; j < H; j++) {
        const hj = Math.tanh(L.W1[j] * a + L.b1[j]);
        sr += L.W2[j] * hj;
        tr += L.W2[H + j] * hj;
      }
      out[0] = SCAP * Math.tanh(sr);
      out[1] = tr;
    }

    const tmpST = [0, 0];

    /* Direção normalizadora x → z; devolve log p_X(x). */
    function logProb(x, y) {
      let u0 = x, u1 = y, ld = 0;
      for (let l = 0; l < K; l++) {
        const pas = l % 2;
        const a = pas === 0 ? u0 : u1;
        const b = pas === 0 ? u1 : u0;
        netEval(layers[l], a, tmpST);
        const bn = b * Math.exp(tmpST[0]) + tmpST[1];
        if (pas === 0) u1 = bn; else u0 = bn;
        ld += tmpST[0];
      }
      return -0.5 * (u0 * u0 + u1 * u1) - LOG2PI + ld;
    }

    /* Inversa de uma camada (amostragem), in place no par u. */
    function invLayer(l, u) {
      const pas = l % 2, act = 1 - pas;
      netEval(layers[l], u[pas], tmpST);
      u[act] = (u[act] - tmpST[1]) * Math.exp(-tmpST[0]);
    }

    /* Estágios da amostragem: stages[k] = zBatch após k camadas inversas. */
    function getStages() {
      if (stagesCache) return stagesCache;
      const stages = [zBatch.map((p) => p.slice())];
      let cur = stages[0];
      for (let l = K - 1; l >= 0; l--) {
        const nxt = cur.map((p) => {
          const u = p.slice();
          invLayer(l, u);
          return u;
        });
        stages.push(nxt);
        cur = nxt;
      }
      stagesCache = stages;
      return stages;
    }

    function trainStep() {
      let nllSum = 0;
      for (let i = 0; i < B; i++) {
        const pt = data[(rngT() * NDATA) | 0];
        let u0 = pt[0], u1 = pt[1], ld = 0;
        /* forward com caches */
        for (let l = 0; l < K; l++) {
          const L = layers[l];
          const pas = l % 2;
          const a = pas === 0 ? u0 : u1;
          const b = pas === 0 ? u1 : u0;
          const ci = l * B + i, hb = ci * H;
          let sr = L.b2[0], tr = L.b2[1];
          for (let j = 0; j < H; j++) {
            const hj = Math.tanh(L.W1[j] * a + L.b1[j]);
            cH[hb + j] = hj;
            sr += L.W2[j] * hj;
            tr += L.W2[H + j] * hj;
          }
          const s = SCAP * Math.tanh(sr);
          const es = Math.exp(s);
          cA[ci] = a; cB[ci] = b; cS[ci] = s; cT[ci] = tr; cES[ci] = es;
          const bn = b * es + tr;
          if (pas === 0) u1 = bn; else u0 = bn;
          ld += s;
        }
        nllSum += 0.5 * (u0 * u0 + u1 * u1) + LOG2PI - ld;
        /* backward: dNLL/dz = z; −Σs contribui −1 em cada s */
        let g0 = u0, g1 = u1;
        for (let l = K - 1; l >= 0; l--) {
          const L = layers[l];
          const tens = tensors.slice(4 * l, 4 * l + 4);
          const gW1 = tens[0].g, gb1 = tens[1].g, gW2 = tens[2].g, gb2 = tens[3].g;
          const pas = l % 2;
          const ci = l * B + i, hb = ci * H;
          const a = cA[ci], b = cB[ci], s = cS[ci], es = cES[ci];
          const gAct = pas === 0 ? g1 : g0;
          const gPas = pas === 0 ? g0 : g1;
          const gB = gAct * es;
          const gS = gAct * b * es - 1;
          const gT = gAct;
          const gSraw = gS * (SCAP - (s * s) / SCAP);
          let gA = 0;
          for (let j = 0; j < H; j++) {
            const hj = cH[hb + j];
            const gh = (gSraw * L.W2[j] + gT * L.W2[H + j]) * (1 - hj * hj);
            gW1[j] += gh * a;
            gb1[j] += gh;
            gA += gh * L.W1[j];
            gW2[j] += gSraw * hj;
            gW2[H + j] += gT * hj;
          }
          gb2[0] += gSraw;
          gb2[1] += gT;
          if (pas === 0) { g0 = gPas + gA; g1 = gB; }
          else { g1 = gPas + gA; g0 = gB; }
        }
      }
      /* Adam */
      tAdam++;
      const bc1 = 1 - Math.pow(BETA1, tAdam), bc2 = 1 - Math.pow(BETA2, tAdam);
      for (const t of tensors) {
        for (let i = 0; i < t.p.length; i++) {
          const gi = t.g[i] / B;
          t.m[i] = BETA1 * t.m[i] + (1 - BETA1) * gi;
          t.v[i] = BETA2 * t.v[i] + (1 - BETA2) * gi * gi;
          t.p[i] -= LR * (t.m[i] / bc1) / (Math.sqrt(t.v[i] / bc2) + EPS);
          t.g[i] = 0;
        }
      }
      step++;
      const nll = nllSum / B;
      nllHist.push(nll);
      const prev = emaHist.length ? emaHist[emaHist.length - 1] : nll;
      emaHist.push(0.98 * prev + 0.02 * nll);
      stagesCache = null;
    }

    function computeHeat() {
      heatMax = 1e-9;
      const cw = 2 * LIM / HN;
      for (let j = 0; j < HN; j++) {
        const y = LIM - (j + 0.5) * cw;
        for (let i = 0; i < HN; i++) {
          const x = -LIM + (i + 0.5) * cw;
          const p = Math.exp(logProb(x, y));
          heat[j * HN + i] = p;
          if (p > heatMax) heatMax = p;
        }
      }
      lastHeat = step;
    }

    function heatLookup(x, y) {
      let i = Math.floor((x + LIM) / (2 * LIM) * HN);
      let j = Math.floor((LIM - y) / (2 * LIM) * HN);
      i = Math.max(0, Math.min(HN - 1, i));
      j = Math.max(0, Math.min(HN - 1, j));
      return heat[j * HN + i];
    }

    function currentSamples() {
      const stages = getStages();
      const k0 = Math.min(K, Math.floor(stagePos));
      const k1 = Math.min(K, k0 + 1);
      const f = stagePos - k0;
      if (f < 1e-3 || k0 === k1) return stages[k0];
      const s0 = stages[k0], s1 = stages[k1];
      return s0.map((p, i) => [
        p[0] + f * (s1[i][0] - p[0]),
        p[1] + f * (s1[i][1] - p[1]),
      ]);
    }

    function drawPlane() {
      const th = P.theme();
      const { ctx, w, h } = P.setup(cvPlane);
      P.clear(ctx, w, h);
      const fr = P.frame(ctx, w, h, -LIM, LIM, -LIM, LIM);
      P.heatmap(ctx, fr, heatLookup, HN, HN, (v) => {
        const u = Math.pow(Math.min(1, v / heatMax), 0.5);
        const col = P.viridis(0.25 + 0.75 * u);
        return [col[0], col[1], col[2], Math.round(235 * u)];
      });
      P.axes(ctx, fr, { xlabel: 'x_1', ylabel: 'x_2', xticks: [-3, 0, 3], yticks: [-3, 0, 3] });
      ctx.save();
      ctx.beginPath();
      ctx.rect(fr.X(-LIM), fr.Y(LIM), fr.iw, fr.ih);
      ctx.clip();
      P.scatter(ctx, fr, data, th.cyan, { r: 1.8, alpha: 0.3 });
      P.scatter(ctx, fr, currentSamples(), th.orange, { r: 2, alpha: 0.65 });
      ctx.restore();
      P.label(ctx, fr.X(-LIM) + 8, fr.Y(LIM) + 14, 'dados', th.cyan);
      P.label(ctx, fr.X(-LIM) + 8, fr.Y(LIM) + 28, 'amostras do flow', th.orange);
      const k = Math.round(stagePos);
      if (k < K) {
        P.label(ctx, fr.X(LIM) - 8, fr.Y(LIM) + 14,
          'após ' + k + ' de ' + K + ' camadas inversas', th.orange, 'right');
      }
    }

    function drawNll() {
      const th = P.theme();
      const { ctx, w, h } = P.setup(cvNll);
      P.clear(ctx, w, h);
      const n = Math.max(100, nllHist.length);
      let ymin = 1.6, ymax = 3.1;
      if (emaHist.length > 5) {
        let lo = Infinity, hi = -Infinity;
        for (const v of emaHist) if (v < lo) lo = v;
        for (let i = 0; i < Math.min(50, nllHist.length); i++) if (nllHist[i] > hi) hi = nllHist[i];
        ymin = lo - 0.15;
        ymax = Math.min(6, hi + 0.15);
        if (ymax - ymin < 0.5) ymax = ymin + 0.5;
      }
      const fr = P.frame(ctx, w, h, 0, n, ymin, ymax);
      P.axes(ctx, fr, {
        xlabel: 'passo', ylabel: 'NLL',
        xticks: [0, Math.round(n / 2), n],
        yticks: [+(ymin + 0.1).toFixed(1), +((ymin + ymax) / 2).toFixed(1), +(ymax - 0.1).toFixed(1)],
      });
      if (nllHist.length > 1) {
        const stride = Math.max(1, Math.floor(nllHist.length / 600));
        const xs = [], raw = [], ema = [];
        for (let i = 0; i < nllHist.length; i += stride) {
          xs.push(i);
          raw.push(Math.min(ymax, Math.max(ymin, nllHist[i])));
          ema.push(Math.min(ymax, Math.max(ymin, emaHist[i])));
        }
        P.line(ctx, fr, xs, raw, th.cyan, { width: 1, alpha: 0.3 });
        P.line(ctx, fr, xs, ema, th.orange, { width: 2 });
      }
    }

    function updateReadout() {
      const last = emaHist.length ? emaHist[emaHist.length - 1].toFixed(3) : '—';
      $('s4-readout').textContent = 'passo = ' + step + ' · NLL (média móvel) = ' + last;
    }

    function redraw() {
      drawPlane();
      drawNll();
      updateReadout();
    }

    function stopTrain() {
      running = false;
      btnTrain.textContent = '▶ Treinar';
      DL.stopTicker(trainTick);
      computeHeat();
      redraw();
    }

    function trainTick() {
      for (let s = 0; s < STEPS_PER_TICK; s++) trainStep();
      if (step - lastHeat >= 60) computeHeat();
      redraw();
      if (step >= MAX_STEPS) stopTrain();
    }

    function stageTick(dt) {
      stagePos = Math.min(stageTarget, stagePos + dt * 2.2);
      redraw();
      if (stagePos >= stageTarget) DL.stopTicker(stageTick);
    }

    btnTrain.addEventListener('click', () => {
      if (running) { stopTrain(); return; }
      DL.stopTicker(stageTick);
      stagePos = K;
      stageTarget = K;
      running = true;
      btnTrain.textContent = '⏸ Pausar';
      DL.startTicker(trainTick);
    });

    btnStage.addEventListener('click', () => {
      if (running) stopTrain();
      DL.stopTicker(stageTick);
      if (stageTarget >= K) {
        stagePos = 0;
        stageTarget = 0;
        redraw();
      } else {
        stageTarget = Math.min(K, Math.floor(stagePos) + 1);
        DL.startTicker(stageTick);
      }
    });

    $('s4-reset').addEventListener('click', () => {
      if (running) { running = false; btnTrain.textContent = '▶ Treinar'; DL.stopTicker(trainTick); }
      DL.stopTicker(stageTick);
      initModel();
      computeHeat();
      redraw();
    });

    $('s4-seed').addEventListener('click', () => {
      if (running) { running = false; btnTrain.textContent = '▶ Treinar'; DL.stopTicker(trainTick); }
      DL.stopTicker(stageTick);
      seed = (seed * 1103515245 + 12345) % 2147483647;
      makeData();
      initModel();
      computeHeat();
      redraw();
    });

    slK.addEventListener('input', () => {
      if (running) { running = false; btnTrain.textContent = '▶ Treinar'; DL.stopTicker(trainTick); }
      DL.stopTicker(stageTick);
      K = +slK.value;
      $('s4-k-val').textContent = K;
      initModel();
      computeHeat();
      redraw();
    });

    function setDataset(name) {
      if (dataset === name) return;
      if (running) { running = false; btnTrain.textContent = '▶ Treinar'; DL.stopTicker(trainTick); }
      DL.stopTicker(stageTick);
      dataset = name;
      tabMoons.classList.toggle('active', name === 'moons');
      tabCircles.classList.toggle('active', name === 'circles');
      makeData();
      initModel();
      computeHeat();
      redraw();
    }
    tabMoons.addEventListener('click', () => setDataset('moons'));
    tabCircles.addEventListener('click', () => setDataset('circles'));

    makeData();
    initModel();
    computeHeat();
    redraw();
    P.onRedraw(redraw);
    P.observeResize(cvPlane, redraw);
    P.observeResize(cvNll, redraw);
  }

  DL.sections.push({ name: 's4-training', init });
})();
