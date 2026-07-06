/* Seção 3: BPTT no exemplo numérico da aula.
   x = [1, 2], y = 3, h_0 = 0, θ_ih = θ_hh = 0.5, θ_ho = 1, b_h = b_o = 0.
   Cada iteração: forward, backward (somando as contribuições ao longo do
   tempo) e passo do optimizer θ ← θ − η·∂J/∂θ. */
(function () {
  'use strict';
  window.DL = window.DL || {};
  DL.sections = DL.sections || [];

  const X = [1, 2], MAX_ITERS = 400;
  const PARAMS = [
    { key: 'thho', tex: 'θ_{ho}' },
    { key: 'bo', tex: 'b_o' },
    { key: 'thih', tex: 'θ_{ih}' },
    { key: 'bh', tex: 'b_h' },
    { key: 'thhh', tex: 'θ_{hh}' },
  ];

  function init() {
    const P = DL.plot;
    const $ = (id) => document.getElementById(id);

    const cvGrads = $('s3-grads'), cvLoss = $('s3-loss');
    const slEta = $('s3-eta'), slY = $('s3-y');
    const btnStep = $('s3-step'), btnRun = $('s3-run'), btnReset = $('s3-reset');

    let eta = +slEta.value, target = +slY.value;
    let p, k, lossHist;
    let running = false, acc = 0;

    function initParams() {
      p = { thih: 0.5, thhh: 0.5, thho: 1, bh: 0, bo: 0 };
      k = 0;
      lossHist = [forwardBackward().J];
    }

    /* Forward de 2 passos + gradientes por BPTT (loss só no instante final). */
    function forwardBackward() {
      const z1 = X[0] * p.thih + 0 * p.thhh + p.bh;
      const h1 = Math.tanh(z1);
      const z2 = X[1] * p.thih + h1 * p.thhh + p.bh;
      const h2 = Math.tanh(z2);
      const yhat = h2 * p.thho + p.bo;
      const e = yhat - target;
      const J = 0.5 * e * e;
      /* δ_t = ∂J/∂z_t: o gradiente atravessa h_1 para chegar no passo 1 */
      const d2 = e * p.thho * (1 - h2 * h2);
      const d1 = d2 * p.thhh * (1 - h1 * h1);
      const g = {
        thho: e * h2,
        bo: e,
        thih: d2 * X[1] + d1 * X[0],
        bh: d2 + d1,
        thhh: d2 * h1 + d1 * 0,
      };
      return { h1, h2, yhat, J, g };
    }

    function stepOnce() {
      if (done()) return;
      const { g } = forwardBackward();
      for (const { key } of PARAMS) p[key] -= eta * g[key];
      k++;
      lossHist.push(forwardBackward().J);
    }

    function done() {
      const J = lossHist[lossHist.length - 1];
      return k >= MAX_ITERS || !isFinite(J) || J < 1e-10;
    }

    function drawGrads() {
      const th = P.theme();
      const { ctx, w, h } = P.setup(cvGrads);
      P.clear(ctx, w, h);
      const { g } = forwardBackward();
      const vals = PARAMS.map(({ key }) => g[key]);
      const lim = Math.max(0.1, ...vals.map(Math.abs)) * 1.25;
      const fr = P.frame(ctx, w, h, 0, 5, -lim, lim, { l: 44, r: 10, t: 10, b: 26 });
      P.axes(ctx, fr, {
        ylabel: '∂J/∂θ',
        yticks: [-Math.round(lim * 100) / 100, 0, Math.round(lim * 100) / 100],
      });
      P.line(ctx, fr, [0, 5], [0, 0], th.line, { width: 1 });
      const bw = fr.iw / 5 * 0.4;
      for (let i = 0; i < 5; i++) {
        const cx = fr.X(i + 0.5);
        ctx.fillStyle = th.orange;
        ctx.globalAlpha = 0.8;
        const y0 = fr.Y(0), y1 = fr.Y(vals[i]);
        ctx.fillRect(cx - bw / 2, Math.min(y0, y1), bw, Math.abs(y1 - y0));
        ctx.globalAlpha = 1;
        const above = vals[i] < 0;
        P.mathText(ctx, vals[i].toFixed(3), cx, above ? y0 - 6 : y0 + 14, th.fg, 'center', 10);
        P.mathText(ctx, PARAMS[i].tex, cx, fr.Y(-lim) + 16, th.comment, 'center', 11);
      }
    }

    function drawLoss() {
      const th = P.theme();
      const { ctx, w, h } = P.setup(cvLoss);
      P.clear(ctx, w, h);
      const logv = (v) => (isFinite(v) && v > 0 ? Math.max(-11, Math.log10(v)) : -11);
      const lg = lossHist.map(logv);
      const n = Math.max(20, lossHist.length - 1);
      const ymax = Math.ceil(Math.max(1, ...lg)) + 0.5;
      const fr = P.frame(ctx, w, h, 0, n, -11, ymax);
      P.axes(ctx, fr, {
        xlabel: 'k', ylabel: 'log₁₀ J',
        xticks: [0, Math.round(n / 2), n],
        yticks: [-10, -5, 0],
      });
      P.line(ctx, fr, lg.map((_, i) => i), lg, th.pink, { width: 1.8 });
    }

    function updateReadout() {
      const fb = forwardBackward();
      $('s3-readout').textContent =
        'k = ' + k + ' · h_1 = ' + fb.h1.toFixed(3) + ' · h_2 = ' + fb.h2.toFixed(3) +
        ' · ŷ_2 = ' + fb.yhat.toFixed(3) + ' · J = ' +
        (fb.J < 1e-3 ? fb.J.toExponential(2) : fb.J.toFixed(4));
    }

    function redraw() {
      drawGrads();
      drawLoss();
      updateReadout();
    }

    function stopRun() {
      running = false;
      btnRun.textContent = '▶ Rodar';
      DL.stopTicker(tick);
    }

    function tick(dt) {
      acc += dt * 10;
      while (acc >= 1) { acc -= 1; stepOnce(); }
      redraw();
      if (done()) stopRun();
    }

    btnStep.addEventListener('click', () => { stepOnce(); redraw(); });
    btnRun.addEventListener('click', () => {
      if (running) { stopRun(); return; }
      if (done()) initParams();
      running = true;
      acc = 0;
      btnRun.textContent = '⏸ Pausar';
      DL.startTicker(tick);
    });
    btnReset.addEventListener('click', () => { stopRun(); initParams(); redraw(); });

    slEta.addEventListener('input', () => {
      eta = +slEta.value;
      $('s3-eta-val').textContent = eta.toFixed(2);
      stopRun();
      initParams();
      redraw();
    });
    slY.addEventListener('input', () => {
      target = +slY.value;
      $('s3-y-val').textContent = target.toFixed(1);
      stopRun();
      initParams();
      redraw();
    });

    initParams();
    redraw();
    P.onRedraw(redraw);
    P.observeResize(cvGrads, redraw);
    P.observeResize(cvLoss, redraw);
  }

  DL.sections.push({ name: 's3-bptt', init });
})();
