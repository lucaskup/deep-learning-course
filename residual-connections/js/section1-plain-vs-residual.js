/* Seção 1: treino ao vivo de rede plain vs residual numa regressão 1D.
   Mesmos pesos iniciais (He), mesmos passos de Adam; só o skip difere.
   Na rede residual a soma da junção é dividida por √2 para manter a
   variância das ativações (o papel que o batch norm cumpre, seção 4). */
(function () {
  'use strict';
  window.DL = window.DL || {};
  DL.sections = DL.sections || [];

  const WIDTH = 8, N = 80, MAX_STEPS = 2500, LR = 0.01;
  const XMIN = -1.5, XMAX = 1.5;
  const INV = Math.SQRT1_2;
  const target = (x) => 0.8 * Math.sin(8 * x);

  function init() {
    const P = DL.plot, U = DL.utils;
    const $ = (id) => document.getElementById(id);

    const cvFit = $('s1-fit'), cvLoss = $('s1-loss');
    const slDepth = $('s1-depth');
    const btnRun = $('s1-run'), btnReset = $('s1-reset'), btnSeed = $('s1-seed');

    let seed = 7;
    let depth = +slDepth.value;
    let data, plainNet, resNet, lossP, lossR, stepCount;
    let running = false, acc = 0;
    const scrA = new Float64Array(WIDTH), scrB = new Float64Array(WIDTH);

    /* Rede: Linear inicial, depois `depth` blocos (ReLU seguida de Linear),
       depois Linear de saída. Parâmetros em listas paralelas para o Adam. */
    function makeNet(rng) {
      const net = { params: [], grads: [], m: [], v: [], t: 0, Ws: [], bs: [], gWs: [], gbs: [] };
      function add(n, std) {
        const a = new Float64Array(n);
        if (std) for (let i = 0; i < n; i++) a[i] = U.randn(rng) * std;
        net.params.push(a);
        net.grads.push(new Float64Array(n));
        net.m.push(new Float64Array(n));
        net.v.push(new Float64Array(n));
        return net.params.length - 1;
      }
      let k = add(WIDTH, Math.sqrt(2));
      net.W0 = net.params[k]; net.gW0 = net.grads[k];
      k = add(WIDTH, 0);
      net.b0 = net.params[k]; net.gb0 = net.grads[k];
      const heW = Math.sqrt(2 / WIDTH);
      for (let l = 0; l < depth; l++) {
        k = add(WIDTH * WIDTH, heW);
        net.Ws.push(net.params[k]); net.gWs.push(net.grads[k]);
        k = add(WIDTH, 0);
        net.bs.push(net.params[k]); net.gbs.push(net.grads[k]);
      }
      k = add(WIDTH, Math.sqrt(1 / WIDTH));
      net.Wout = net.params[k]; net.gWout = net.grads[k];
      k = add(1, 0);
      net.bout = net.params[k]; net.gbout = net.grads[k];
      /* Caches de ativação por camada (batch inteiro). */
      net.H = [];
      for (let l = 0; l <= depth; l++) net.H.push(new Float64Array(N * WIDTH));
      return net;
    }

    function makeData(rng) {
      return U.noisyCurve(N, target, 0.07, XMIN, XMAX, rng);
    }

    function rebuild() {
      data = makeData(U.mulberry32(seed));
      /* Mesma semente para as duas redes: pesos iniciais idênticos. */
      plainNet = makeNet(U.mulberry32(seed + 1));
      resNet = makeNet(U.mulberry32(seed + 1));
      lossP = [];
      lossR = [];
      stepCount = 0;
    }

    /* Avaliação de uma entrada (para desenhar a curva ajustada). */
    function evalNet(net, x, residual) {
      let h = scrA, hn = scrB;
      for (let i = 0; i < WIDTH; i++) h[i] = net.W0[i] * x + net.b0[i];
      for (let l = 0; l < depth; l++) {
        const W = net.Ws[l], b = net.bs[l];
        for (let i = 0; i < WIDTH; i++) {
          let s = b[i];
          for (let j = 0; j < WIDTH; j++) {
            const a = h[j];
            if (a > 0) s += W[i * WIDTH + j] * a;
          }
          hn[i] = residual ? (h[i] + s) * INV : s;
        }
        const tmp = h; h = hn; hn = tmp;
      }
      let y = net.bout[0];
      for (let j = 0; j < WIDTH; j++) y += net.Wout[j] * h[j];
      return y;
    }

    /* Um passo de treino: forward + backprop no batch inteiro, depois Adam. */
    function trainStep(net, residual) {
      for (const g of net.grads) g.fill(0);
      const H = net.H;
      let mse = 0;
      let dh = scrA, dhPrev = scrB;
      for (let n = 0; n < N; n++) {
        const x = data.xs[n], t = data.ys[n], base = n * WIDTH;
        for (let i = 0; i < WIDTH; i++) H[0][base + i] = net.W0[i] * x + net.b0[i];
        for (let l = 0; l < depth; l++) {
          const W = net.Ws[l], b = net.bs[l], hp = H[l], hn = H[l + 1];
          for (let i = 0; i < WIDTH; i++) {
            let s = b[i];
            for (let j = 0; j < WIDTH; j++) {
              const a = hp[base + j];
              if (a > 0) s += W[i * WIDTH + j] * a;
            }
            hn[base + i] = residual ? (hp[base + i] + s) * INV : s;
          }
        }
        const hl = H[depth];
        let y = net.bout[0];
        for (let j = 0; j < WIDTH; j++) y += net.Wout[j] * hl[base + j];
        const err = y - t;
        mse += err * err;
        const dy = 2 * err / N;
        net.gbout[0] += dy;
        for (let j = 0; j < WIDTH; j++) {
          net.gWout[j] += dy * hl[base + j];
          dh[j] = net.Wout[j] * dy;
        }
        const inv = residual ? INV : 1;
        for (let l = depth - 1; l >= 0; l--) {
          const W = net.Ws[l], gW = net.gWs[l], gb = net.gbs[l], hp = H[l];
          for (let i = 0; i < WIDTH; i++) {
            const dz = dh[i] * inv;
            gb[i] += dz;
            for (let j = 0; j < WIDTH; j++) {
              const a = hp[base + j];
              if (a > 0) gW[i * WIDTH + j] += dz * a;
            }
          }
          for (let j = 0; j < WIDTH; j++) {
            let s = 0;
            if (hp[base + j] > 0) {
              for (let i = 0; i < WIDTH; i++) s += W[i * WIDTH + j] * dh[i];
            }
            dhPrev[j] = (s + (residual ? dh[j] : 0)) * inv;
          }
          const tmp = dh; dh = dhPrev; dhPrev = tmp;
        }
        for (let i = 0; i < WIDTH; i++) {
          net.gW0[i] += dh[i] * x;
          net.gb0[i] += dh[i];
        }
      }
      /* Adam */
      net.t++;
      const b1 = 0.9, b2 = 0.999, eps = 1e-8;
      const c1 = 1 - Math.pow(b1, net.t), c2 = 1 - Math.pow(b2, net.t);
      for (let k = 0; k < net.params.length; k++) {
        const p = net.params[k], g = net.grads[k], m = net.m[k], v = net.v[k];
        for (let i = 0; i < p.length; i++) {
          m[i] = b1 * m[i] + (1 - b1) * g[i];
          v[i] = b2 * v[i] + (1 - b2) * g[i] * g[i];
          p[i] -= LR * (m[i] / c1) / (Math.sqrt(v[i] / c2) + eps);
        }
      }
      return mse / N;
    }

    function stepBoth() {
      lossP.push(trainStep(plainNet, false));
      lossR.push(trainStep(resNet, true));
      stepCount++;
    }

    function clipPlot(ctx, fr) {
      ctx.save();
      ctx.beginPath();
      ctx.rect(fr.X(fr.xmin), fr.Y(fr.ymax), fr.iw, fr.ih);
      ctx.clip();
    }

    function drawFit() {
      const th = P.theme();
      const { ctx, w, h } = P.setup(cvFit);
      P.clear(ctx, w, h);
      const fr = P.frame(ctx, w, h, XMIN - 0.1, XMAX + 0.1, -1.7, 1.7);
      P.axes(ctx, fr, { xlabel: 'x', ylabel: 'y', xticks: [-1, 0, 1], yticks: [-1, 0, 1] });
      clipPlot(ctx, fr);
      const M = 140;
      const gx = [], gt = [], gp = [], gr = [];
      for (let i = 0; i < M; i++) {
        const x = XMIN + (XMAX - XMIN) * i / (M - 1);
        gx.push(x);
        gt.push(target(x));
        gp.push(evalNet(plainNet, x, false));
        gr.push(evalNet(resNet, x, true));
      }
      P.scatter(ctx, fr, data.xs.map((x, i) => [x, data.ys[i]]), th.comment, { r: 2.2, alpha: 0.55 });
      P.line(ctx, fr, gx, gt, th.green, { width: 1.4, dash: [5, 4], alpha: 0.8 });
      P.line(ctx, fr, gx, gp, th.cyan, { width: 2 });
      P.line(ctx, fr, gx, gr, th.orange, { width: 2 });
      ctx.restore();
      P.label(ctx, fr.X(fr.xmin) + 8, fr.Y(fr.ymax) + 14, 'plain', th.cyan);
      P.label(ctx, fr.X(fr.xmin) + 8, fr.Y(fr.ymax) + 28, 'residual', th.orange);
      P.label(ctx, fr.X(fr.xmin) + 8, fr.Y(fr.ymax) + 42, 'alvo', th.green);
    }

    function drawLoss() {
      const th = P.theme();
      const { ctx, w, h } = P.setup(cvLoss);
      P.clear(ctx, w, h);
      const n = Math.max(50, lossP.length);
      const logv = (v) => Math.log10(Math.max(v, 1e-6));
      let lo = -2.2, hi = 0.5;
      for (const v of lossP) { const u = logv(v); if (u < lo) lo = u; if (u > hi) hi = u; }
      for (const v of lossR) { const u = logv(v); if (u < lo) lo = u; if (u > hi) hi = u; }
      const fr = P.frame(ctx, w, h, 0, n, lo - 0.3, hi + 0.3);
      P.axes(ctx, fr, {
        xlabel: 'passo', ylabel: 'log₁₀ MSE',
        xticks: [0, Math.round(n / 2), n],
        yticks: [Math.ceil(lo), Math.round((lo + hi) / 2), Math.floor(hi)],
      });
      if (lossP.length > 1) {
        const stride = Math.max(1, Math.floor(lossP.length / 400));
        const ks = [], lp = [], lr = [];
        for (let i = 0; i < lossP.length; i += stride) {
          ks.push(i);
          lp.push(logv(lossP[i]));
          lr.push(logv(lossR[i]));
        }
        clipPlot(ctx, fr);
        P.line(ctx, fr, ks, lp, th.cyan, { width: 1.8 });
        P.line(ctx, fr, ks, lr, th.orange, { width: 1.8 });
        ctx.restore();
      }
    }

    function updateReadout() {
      const num = (v) => (v == null ? '…' : v.toExponential(1));
      $('s1-readout').textContent =
        'passo ' + stepCount +
        ' · MSE plain = ' + num(lossP[lossP.length - 1]) +
        ' · MSE residual = ' + num(lossR[lossR.length - 1]);
    }

    function redraw() {
      drawFit();
      drawLoss();
      updateReadout();
    }

    function stopRun() {
      running = false;
      btnRun.textContent = '▶ Treinar';
      DL.stopTicker(tick);
    }

    function tick(dt) {
      acc += dt * 90;
      let did = 0;
      while (acc >= 1 && did < 4 && stepCount < MAX_STEPS) {
        acc -= 1;
        stepBoth();
        did++;
      }
      redraw();
      if (stepCount >= MAX_STEPS) stopRun();
    }

    btnRun.addEventListener('click', () => {
      if (running) { stopRun(); return; }
      if (stepCount >= MAX_STEPS) rebuild();
      running = true;
      acc = 0;
      btnRun.textContent = '⏸ Pausar';
      DL.startTicker(tick);
    });
    btnReset.addEventListener('click', () => { stopRun(); rebuild(); redraw(); });
    btnSeed.addEventListener('click', () => { stopRun(); seed += 1; rebuild(); redraw(); });
    slDepth.addEventListener('input', () => {
      depth = +slDepth.value;
      $('s1-depth-val').textContent = depth;
      stopRun();
      rebuild();
      redraw();
    });

    rebuild();
    redraw();
    P.onRedraw(redraw);
    P.observeResize(cvFit, redraw);
    P.observeResize(cvLoss, redraw);
  }

  DL.sections.push({ name: 's1-plain-vs-residual', init });
})();
