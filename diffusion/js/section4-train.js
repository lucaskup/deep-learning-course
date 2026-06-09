/* Seção 4: mini diffusion model treinado ao vivo (MLP + Adam escritos à mão). */
(function () {
  'use strict';
  window.DM = window.DM || {};
  DM.sections = DM.sections || [];

  const T = 50;
  const DIN = 13, H = 64, DOUT = 2;      // [z1 z2 | temb(8) | one-hot(2) | flag]
  const NDATA = 2000, BATCH = 128, LR = 2e-3;
  const CFG_DROP = 0.1, CLIP = 5;
  const NPART = 300, STEPS_PER_FRAME = 20;

  /* ── modelo puro (sem DOM), também usado no gradient check ── */
  function createModel(seed, utils) {
    const U = utils;
    const rng = U.mulberry32(seed);
    const he = (n, fanIn) => {
      const a = new Float32Array(n);
      for (let i = 0; i < n; i++) a[i] = U.randn(rng) * Math.sqrt(2 / fanIn);
      return a;
    };
    const p = {
      W1: he(H * DIN, DIN), b1: new Float32Array(H),
      W2: he(H * H, H), b2: new Float32Array(H),
      W3: he(DOUT * H, H), b3: new Float32Array(DOUT),
    };
    const mAdam = {}, vAdam = {};
    for (const k in p) { mAdam[k] = new Float32Array(p[k].length); vAdam[k] = new Float32Array(p[k].length); }
    let adamStep = 0;

    function forward(inp, B) {
      const h1 = new Float32Array(B * H), h2 = new Float32Array(B * H), out = new Float32Array(B * DOUT);
      for (let b = 0; b < B; b++) {
        for (let j = 0; j < H; j++) {
          let s = p.b1[j];
          for (let k = 0; k < DIN; k++) s += p.W1[j * DIN + k] * inp[b * DIN + k];
          h1[b * H + j] = s > 0 ? s : 0;
        }
        for (let j = 0; j < H; j++) {
          let s = p.b2[j];
          for (let k = 0; k < H; k++) s += p.W2[j * H + k] * h1[b * H + k];
          h2[b * H + j] = s > 0 ? s : 0;
        }
        for (let o = 0; o < DOUT; o++) {
          let s = p.b3[o];
          for (let k = 0; k < H; k++) s += p.W3[o * H + k] * h2[b * H + k];
          out[b * DOUT + o] = s;
        }
      }
      return { h1, h2, out };
    }

    /* Gradientes de L = mean (out - target)^2 sobre B*DOUT */
    function computeGrads(inp, target, B) {
      const { h1, h2, out } = forward(inp, B);
      const g = {
        W1: new Float32Array(p.W1.length), b1: new Float32Array(H),
        W2: new Float32Array(p.W2.length), b2: new Float32Array(H),
        W3: new Float32Array(p.W3.length), b3: new Float32Array(DOUT),
      };
      let loss = 0;
      const dOut = new Float32Array(B * DOUT);
      for (let i = 0; i < B * DOUT; i++) {
        const d = out[i] - target[i];
        loss += d * d;
        dOut[i] = 2 * d / (B * DOUT);
      }
      loss /= B * DOUT;
      const dH2 = new Float32Array(B * H), dH1 = new Float32Array(B * H);
      for (let b = 0; b < B; b++) {
        for (let o = 0; o < DOUT; o++) {
          const d = dOut[b * DOUT + o];
          g.b3[o] += d;
          for (let k = 0; k < H; k++) {
            g.W3[o * H + k] += d * h2[b * H + k];
            dH2[b * H + k] += p.W3[o * H + k] * d;
          }
        }
        for (let j = 0; j < H; j++) {
          if (h2[b * H + j] <= 0) { dH2[b * H + j] = 0; continue; }
          const d = dH2[b * H + j];
          g.b2[j] += d;
          for (let k = 0; k < H; k++) {
            g.W2[j * H + k] += d * h1[b * H + k];
            dH1[b * H + k] += p.W2[j * H + k] * d;
          }
        }
        for (let j = 0; j < H; j++) {
          if (h1[b * H + j] <= 0) continue;
          const d = dH1[b * H + j];
          g.b1[j] += d;
          for (let k = 0; k < DIN; k++) g.W1[j * DIN + k] += d * inp[b * DIN + k];
        }
      }
      return { loss, grads: g };
    }

    function adamUpdate(grads, lr) {
      let sq = 0;
      for (const k in grads) for (let i = 0; i < grads[k].length; i++) sq += grads[k][i] * grads[k][i];
      const norm = Math.sqrt(sq);
      const scale = norm > CLIP ? CLIP / norm : 1;
      adamStep++;
      const b1c = 1 - Math.pow(0.9, adamStep), b2c = 1 - Math.pow(0.999, adamStep);
      for (const k in p) {
        const pk = p[k], gk = grads[k], mk = mAdam[k], vk = vAdam[k];
        for (let i = 0; i < pk.length; i++) {
          const gi = gk[i] * scale;
          mk[i] = 0.9 * mk[i] + 0.1 * gi;
          vk[i] = 0.999 * vk[i] + 0.001 * gi * gi;
          pk[i] -= lr * (mk[i] / b1c) / (Math.sqrt(vk[i] / b2c) + 1e-8);
        }
      }
    }

    function trainStep(inp, target, B, lr) {
      const { loss, grads } = computeGrads(inp, target, B);
      adamUpdate(grads, lr);
      return loss;
    }

    return { p, forward, computeGrads, trainStep };
  }

  /* Checagem por diferenças finitas (rodar no console: DM.mlpTest.gradCheck()) */
  function gradCheck(utils) {
    const U = utils || DM.utils;
    const model = createModel(123, U);
    const rng = U.mulberry32(7);
    const B = 3;
    const inp = new Float32Array(B * DIN), target = new Float32Array(B * DOUT);
    for (let i = 0; i < inp.length; i++) inp[i] = U.randn(rng);
    for (let i = 0; i < target.length; i++) target[i] = U.randn(rng);
    const { grads } = model.computeGrads(inp, target, B);
    const eps = 1e-2;
    let maxRel = 0;
    for (const k in model.p) {
      const arr = model.p[k];
      for (let trial = 0; trial < 8; trial++) {
        const i = Math.floor(rng() * arr.length);
        const orig = arr[i];
        arr[i] = orig + eps;
        const lp = model.computeGrads(inp, target, B).loss;
        arr[i] = orig - eps;
        const lm = model.computeGrads(inp, target, B).loss;
        arr[i] = orig;
        const fd = (lp - lm) / (2 * eps);
        const an = grads[k][i];
        /* gradientes minúsculos são dominados pelo arredondamento float32 */
        if (Math.abs(fd) + Math.abs(an) < 1e-4) continue;
        const rel = Math.abs(fd - an) / (Math.abs(fd) + Math.abs(an));
        if (rel > maxRel) maxRel = rel;
      }
    }
    return maxRel;
  }

  DM.mlpTest = { createModel, gradCheck, DIN, H, DOUT };

  /* Embedding de tempo senoidal + condição de classe */
  function fillInput(buf, off, z1, z2, t, cls) {
    buf[off] = z1; buf[off + 1] = z2;
    const tau = t / T;
    for (let k = 0; k < 4; k++) {
      buf[off + 2 + 2 * k] = Math.sin(Math.pow(2, k) * Math.PI * tau);
      buf[off + 3 + 2 * k] = Math.cos(Math.pow(2, k) * Math.PI * tau);
    }
    buf[off + 10] = cls === 0 ? 1 : 0;
    buf[off + 11] = cls === 1 ? 1 : 0;
    buf[off + 12] = cls === -1 ? 0 : 1;
  }

  function init() {
    const U = DM.utils, P = DM.plot;
    const $ = (id) => document.getElementById(id);
    const sched = U.linearSchedule(T, 1e-4, 0.2);

    const cvLoss = $('s4-loss'), cvSamples = $('s4-samples');
    const btnTrain = $('s4-train'), btnReset = $('s4-reset'), btnSample = $('s4-sample');
    const selClass = $('s4-class'), sliderW = $('s4-w');

    const dataRng = U.mulberry32(99);
    const { pts: data, lab: labels } = U.twoMoons(NDATA, 0.08, dataRng);

    let model = null, steps = 0, lossEma = null, lossHist = [];
    let training = false;
    let trainRng = null;

    let particles = null, partT = 0, sampling = false, sampleCls = -1, guidW = 0;

    function resetModel() {
      model = createModel(1234, U);
      trainRng = U.mulberry32(2024);
      steps = 0; lossEma = null; lossHist = [];
      stopTraining(); stopSampling();
      particles = null;
      btnSample.disabled = true;
      updateStatus();
      drawLoss(); drawSamples();
    }

    function trainTick() {
      const inp = new Float32Array(BATCH * DIN);
      const target = new Float32Array(BATCH * DOUT);
      for (let s = 0; s < STEPS_PER_FRAME; s++) {
        for (let b = 0; b < BATCH; b++) {
          const i = Math.floor(trainRng() * NDATA);
          const t = 1 + Math.floor(trainRng() * T);
          const e1 = U.randn(trainRng), e2 = U.randn(trainRng);
          const sa = Math.sqrt(sched.alpha[t]), sn = Math.sqrt(1 - sched.alpha[t]);
          const z1 = sa * data[i][0] + sn * e1;
          const z2 = sa * data[i][1] + sn * e2;
          const cls = trainRng() < CFG_DROP ? -1 : labels[i];
          fillInput(inp, b * DIN, z1, z2, t, cls);
          target[b * DOUT] = e1; target[b * DOUT + 1] = e2;
        }
        const loss = model.trainStep(inp, target, BATCH, LR);
        lossEma = lossEma === null ? loss : 0.98 * lossEma + 0.02 * loss;
        lossHist.push(lossEma);
        steps++;
      }
      if (steps >= 500) btnSample.disabled = false;
      updateStatus();
      drawLoss();
    }

    function stopTraining() {
      training = false;
      btnTrain.textContent = '▶ Treinar';
      DM.stopTicker(trainTick);
    }

    btnTrain.addEventListener('click', () => {
      if (training) { stopTraining(); return; }
      training = true;
      btnTrain.textContent = '⏸ Pausar';
      DM.startTicker(trainTick);
    });
    btnReset.addEventListener('click', resetModel);

    /* Amostragem reversa animada (algoritmo dos slides) */
    function startSampling() {
      sampleCls = +selClass.value;
      guidW = +sliderW.value;
      const rng = U.mulberry32((Math.max(1, steps) * 2654435761) >>> 0);
      particles = new Float32Array(NPART * 2);
      for (let i = 0; i < particles.length; i++) particles[i] = U.randn(rng);
      particles._rng = rng;
      partT = T;
      sampling = true;
      DM.startTicker(sampleTick);
      drawSamples();
    }

    function predictEps(z, t) {
      const inp = new Float32Array(NPART * DIN);
      const eps = new Float32Array(NPART * DOUT);
      if (sampleCls === -1) {
        for (let i = 0; i < NPART; i++) fillInput(inp, i * DIN, z[i * 2], z[i * 2 + 1], t, -1);
        const out = model.forward(inp, NPART).out;
        eps.set(out);
      } else {
        for (let i = 0; i < NPART; i++) fillInput(inp, i * DIN, z[i * 2], z[i * 2 + 1], t, sampleCls);
        const cond = model.forward(inp, NPART).out;
        for (let i = 0; i < NPART; i++) fillInput(inp, i * DIN, z[i * 2], z[i * 2 + 1], t, -1);
        const unc = model.forward(inp, NPART).out;
        for (let i = 0; i < eps.length; i++) eps[i] = (1 + guidW) * cond[i] - guidW * unc[i];
      }
      return eps;
    }

    function sampleTick() {
      if (partT < 1) { stopSampling(); return; }
      const t = partT;
      const b = sched.beta[t], a = sched.alpha[t];
      const c1 = 1 / Math.sqrt(1 - b);
      const c2 = b / (Math.sqrt(1 - a) * Math.sqrt(1 - b));
      const sigma = t > 1 ? Math.sqrt(b) : 0;   // passo final sem ruído
      const eps = predictEps(particles, t);
      const rng = particles._rng;
      for (let i = 0; i < NPART; i++) {
        for (let d = 0; d < 2; d++) {
          const zh = c1 * particles[i * 2 + d] - c2 * eps[i * 2 + d];
          particles[i * 2 + d] = zh + sigma * U.randn(rng);
        }
      }
      partT--;
      drawSamples();
    }

    function stopSampling() {
      sampling = false;
      DM.stopTicker(sampleTick);
      drawSamples();
    }

    btnSample.addEventListener('click', () => { stopSampling(); startSampling(); });
    sliderW.addEventListener('input', () => { $('s4-w-val').textContent = (+sliderW.value).toFixed(1); });

    function updateStatus() {
      $('s4-status').textContent = 'passos: ' + steps +
        ' · loss: ' + (lossEma === null ? '—' : lossEma.toFixed(3));
    }

    function drawLoss() {
      const th = P.theme();
      const { ctx, w, h } = P.setup(cvLoss);
      P.clear(ctx, w, h);
      const ymin = Math.log10(0.05), ymax = Math.log10(2);
      const xmax = Math.max(1000, lossHist.length);
      const fr = P.frame(ctx, w, h, 0, xmax, ymin, ymax);
      P.axes(ctx, fr, { xlabel: 'passos de gradiente', xticks: [0, Math.round(xmax / 2), xmax] });
      const t9 = P.theme();
      for (const v of [0.1, 0.5, 1]) {
        P.line(ctx, fr, [0, xmax], [Math.log10(v), Math.log10(v)], t9.line, { dash: [3, 4], width: 1 });
        P.label(ctx, fr.X(0) - 4, fr.Y(Math.log10(v)) + 4, String(v), th.comment, 'right');
      }
      if (lossHist.length > 1) {
        const stride = Math.max(1, Math.floor(lossHist.length / 600));
        const xs = [], ys = [];
        for (let i = 0; i < lossHist.length; i += stride) {
          xs.push(i);
          ys.push(Math.log10(Math.max(0.05, lossHist[i])));
        }
        P.line(ctx, fr, xs, ys, th.green, { width: 2 });
      }
    }

    function drawSamples() {
      const th = P.theme();
      const { ctx, w, h } = P.setup(cvSamples);
      P.clear(ctx, w, h);
      const fr = P.frame(ctx, w, h, -3, 3, -3, 3);
      P.axes(ctx, fr, { xlabel: 'z₁', ylabel: 'z₂', xticks: [-2, 0, 2], yticks: [-2, 0, 2] });
      /* dados reais ao fundo */
      P.scatter(ctx, fr, data.filter((_, i) => labels[i] === 0), th.cyan, { r: 1.5, alpha: 0.12 });
      P.scatter(ctx, fr, data.filter((_, i) => labels[i] === 1), th.orange, { r: 1.5, alpha: 0.12 });
      if (particles) {
        const col = sampleCls === 0 ? th.cyan : sampleCls === 1 ? th.orange : th.purple;
        const pts = [];
        for (let i = 0; i < NPART; i++) pts.push([particles[i * 2], particles[i * 2 + 1]]);
        P.scatter(ctx, fr, pts, col, { r: 2.5, alpha: 0.85 });
        P.label(ctx, fr.X(-3) + 8, 14,
          partT >= 1 ? 'revertendo… t = ' + partT : 'amostras finais', th.fg);
      } else {
        P.label(ctx, fr.X(-3) + 8, 14, 'treine e clique em Amostrar', th.comment);
      }
    }

    resetModel();
    P.onRedraw(() => { drawLoss(); drawSamples(); });
    P.observeResize(cvLoss, drawLoss);
    P.observeResize(cvSamples, drawSamples);
  }

  DM.sections.push({ name: 's4-train', init });
})();
