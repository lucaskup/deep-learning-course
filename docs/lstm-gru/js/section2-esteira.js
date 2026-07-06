/* Seção 2: a esteira do estado de célula. Compara o produto de fatores do
   gradiente no caminho direto: RNN vanilla ∏ θ_hh·(1−tanh²), LSTM ∏ f_k e
   GRU ∏ (1−z_k), com células escalares rodando de verdade. */
(function () {
  'use strict';
  window.DL = window.DL || {};
  DL.sections = DL.sections || [];

  const LOG_MIN = -12, LOG_MAX = 6;

  function init() {
    const P = DL.plot;
    const $ = (id) => document.getElementById(id);
    const sig = (x) => 1 / (1 + Math.exp(-x));

    const cvGrad = $('s2-grad'), cvFac = $('s2-factors');
    const slT = $('s2-T'), slW = $('s2-w'), slBf = $('s2-bf');
    const btnSample = $('s2-resample');

    let T = +slT.value, thetaHH = +slW.value, bf = +slBf.value;
    let seed = 11;
    let facRNN = [], facLSTM = [], facGRU = [];
    let prodRNN = [], prodLSTM = [], prodGRU = [];

    function compute() {
      const rng = DL.utils.mulberry32(seed);
      const rw = () => (2 * rng() - 1) * 0.8;
      /* pesos aleatórios (escalares) das células */
      const wxRNN = rw();
      const wf = [rw(), rw()], wi = [rw(), rw()], wc = [rw(), rw()], wo = [rw(), rw()];
      const wr = [rw(), rw()], wz = [rw(), rw()], wh = [rw(), rw()];
      /* mesma sequência de entrada para as três células */
      const xs = [];
      for (let k = 0; k < T; k++) xs.push(DL.utils.randn(rng) * 0.8);

      facRNN = []; facLSTM = []; facGRU = [];
      let hR = 0, hL = 0, cL = 0, hG = 0;
      for (let k = 0; k < T; k++) {
        const x = xs[k];
        /* RNN vanilla: fator θ_hh·(1−tanh²(z_k)) */
        const z = thetaHH * hR + wxRNN * x;
        hR = Math.tanh(z);
        facRNN.push(Math.abs(thetaHH * (1 - hR * hR)));
        /* LSTM: fator do caminho direto é f_k */
        const f = sig(wf[0] * hL + wf[1] * x + bf);
        const i = sig(wi[0] * hL + wi[1] * x);
        const g = Math.tanh(wc[0] * hL + wc[1] * x);
        cL = f * cL + i * g;
        const o = sig(wo[0] * hL + wo[1] * x);
        hL = o * Math.tanh(cL);
        facLSTM.push(f);
        /* GRU: fator do caminho direto é (1−z_k); −bf empurra z para 0 */
        const r = sig(wr[0] * hG + wr[1] * x);
        const zg = sig(wz[0] * hG + wz[1] * x - bf);
        const hh = Math.tanh(wh[0] * (r * hG) + wh[1] * x);
        hG = (1 - zg) * hG + zg * hh;
        facGRU.push(1 - zg);
      }

      const accum = (fac) => {
        const out = [];
        let lg = 0;
        for (const v of fac) {
          lg += Math.log10(Math.max(v, 1e-300));
          out.push(Math.max(LOG_MIN, Math.min(LOG_MAX, lg)));
        }
        return out;
      };
      prodRNN = accum(facRNN);
      prodLSTM = accum(facLSTM);
      prodGRU = accum(facGRU);
    }

    function legend(ctx, fr, th) {
      P.label(ctx, fr.X(fr.xmin) + 8, fr.Y(fr.ymax) + 14, 'RNN vanilla', th.cyan);
      P.label(ctx, fr.X(fr.xmin) + 8, fr.Y(fr.ymax) + 28, 'LSTM (f_k)', th.orange);
      P.label(ctx, fr.X(fr.xmin) + 8, fr.Y(fr.ymax) + 42, 'GRU (1−z_k)', th.green);
    }

    function drawGrad() {
      const th = P.theme();
      const { ctx, w, h } = P.setup(cvGrad);
      P.clear(ctx, w, h);
      const fr = P.frame(ctx, w, h, 1, T, LOG_MIN, LOG_MAX);
      P.axes(ctx, fr, {
        xlabel: 'k', ylabel: 'log₁₀ do produto',
        xticks: [1, Math.round(T / 2), T],
        yticks: [-12, -6, 0, 6],
      });
      /* nível 1 (log = 0): gradiente perfeitamente preservado */
      P.line(ctx, fr, [1, T], [0, 0], th.comment, { width: 1, dash: [4, 4], alpha: 0.7 });
      const ks = prodRNN.map((_, k) => k + 1);
      P.line(ctx, fr, ks, prodRNN, th.cyan, { width: 2 });
      P.line(ctx, fr, ks, prodLSTM, th.orange, { width: 2 });
      P.line(ctx, fr, ks, prodGRU, th.green, { width: 2 });
      legend(ctx, fr, th);
    }

    function drawFactors() {
      const th = P.theme();
      const { ctx, w, h } = P.setup(cvFac);
      P.clear(ctx, w, h);
      const fr = P.frame(ctx, w, h, 1, T, 0, 1.6);
      P.axes(ctx, fr, {
        xlabel: 'k', ylabel: 'fator do passo',
        xticks: [1, Math.round(T / 2), T],
        yticks: [0, 0.5, 1, 1.5],
      });
      P.line(ctx, fr, [1, T], [1, 1], th.comment, { width: 1, dash: [4, 4], alpha: 0.7 });
      const ks = facRNN.map((_, k) => k + 1);
      const clip = (arr) => arr.map((v) => Math.min(1.6, v));
      P.line(ctx, fr, ks, clip(facRNN), th.cyan, { width: 1.6 });
      P.line(ctx, fr, ks, clip(facLSTM), th.orange, { width: 1.6 });
      P.line(ctx, fr, ks, clip(facGRU), th.green, { width: 1.6 });
    }

    function updateReadout() {
      const num = (lg) => (lg <= LOG_MIN ? '<1e−12' : Math.pow(10, lg).toExponential(1));
      $('s2-readout').textContent =
        'produto em k = ' + T + ': RNN = ' + num(prodRNN[T - 1]) +
        ' · LSTM = ' + num(prodLSTM[T - 1]) +
        ' · GRU = ' + num(prodGRU[T - 1]);
    }

    function redraw() {
      drawGrad();
      drawFactors();
      updateReadout();
    }

    slT.addEventListener('input', () => {
      T = +slT.value;
      $('s2-T-val').textContent = T;
      compute();
      redraw();
    });
    slW.addEventListener('input', () => {
      thetaHH = +slW.value;
      $('s2-w-val').textContent = thetaHH.toFixed(2);
      compute();
      redraw();
    });
    slBf.addEventListener('input', () => {
      bf = +slBf.value;
      $('s2-bf-val').textContent = bf.toFixed(1);
      compute();
      redraw();
    });
    btnSample.addEventListener('click', () => {
      seed = (seed * 1103515245 + 12345) >>> 0;
      compute();
      redraw();
    });

    compute();
    redraw();
    P.onRedraw(redraw);
    P.observeResize(cvGrad, redraw);
    P.observeResize(cvFac, redraw);
  }

  DL.sections.push({ name: 's2-esteira', init });
})();
