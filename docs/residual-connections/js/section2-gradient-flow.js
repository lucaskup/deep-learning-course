/* Seção 2: gradiente quebrado (∂y/∂x errático em redes plain profundas) e
   norma do gradiente por camada, com e sem skip connection. Pesos He fixos
   por semente; o ganho g reescala os pesos dos blocos. */
(function () {
  'use strict';
  window.DL = window.DL || {};
  DL.sections = DL.sections || [];

  const W = 16, NX = 220, DOMX = 2, MAXD = 40;

  function init() {
    const P = DL.plot, U = DL.utils;
    const $ = (id) => document.getElementById(id);

    const cvDeriv = $('s2-deriv'), cvNorms = $('s2-norms');
    const slDepth = $('s2-depth'), slGain = $('s2-gain');
    const btnSeed = $('s2-seed');

    let seed = 11;
    let depth = +slDepth.value, gain = +slGain.value;
    let w0, b0, Ws, bs, wout;
    let plain, resid;

    function makeWeights() {
      const rng = U.mulberry32(seed);
      w0 = new Float64Array(W);
      b0 = new Float64Array(W);
      for (let i = 0; i < W; i++) {
        w0[i] = U.randn(rng) * Math.sqrt(2);
        b0[i] = U.randn(rng);
      }
      Ws = []; bs = [];
      const std = Math.sqrt(2 / W);
      for (let l = 0; l < MAXD; l++) {
        const Wl = new Float64Array(W * W), bl = new Float64Array(W);
        for (let i = 0; i < W * W; i++) Wl[i] = U.randn(rng) * std;
        for (let i = 0; i < W; i++) bl[i] = U.randn(rng) * 0.1;
        Ws.push(Wl);
        bs.push(bl);
      }
      wout = new Float64Array(W);
      for (let i = 0; i < W; i++) wout[i] = U.randn(rng) * Math.sqrt(1 / W);
    }

    /* Forward e backward em todo o grid de entradas. No ramo residual a
       entrada de cada bloco é normalizada pela escala RMS das ativações
       (o papel que o batch norm cumpre); a escala é tratada como constante
       no backward, como no argumento de variância dos slides. */
    function pass(residual) {
      const dydx = new Float64Array(NX);
      const lognorm = new Float64Array(depth + 1);
      const h = new Float64Array(W), z = new Float64Array(W);
      const a = new Float64Array(W), dh = new Float64Array(W), tmp = new Float64Array(W);
      const masks = new Uint8Array(MAXD * W), scale = new Float64Array(MAXD);
      const norm = (v) => {
        let s = 0;
        for (let i = 0; i < W; i++) s += v[i] * v[i];
        return Math.sqrt(s);
      };
      for (let n = 0; n < NX; n++) {
        const x = -DOMX + 2 * DOMX * (n + 0.5) / NX;
        for (let i = 0; i < W; i++) h[i] = w0[i] * x + b0[i];
        for (let l = 0; l < depth; l++) {
          let s = 1;
          if (residual) {
            let ms = 0;
            for (let i = 0; i < W; i++) ms += h[i] * h[i];
            s = Math.sqrt(ms / W) + 1e-8;
          }
          scale[l] = s;
          for (let j = 0; j < W; j++) {
            const u = h[j] / s;
            masks[l * W + j] = u > 0 ? 1 : 0;
            a[j] = u > 0 ? u : 0;
          }
          const Wl = Ws[l], bl = bs[l];
          for (let i = 0; i < W; i++) {
            let acc = bl[i];
            for (let j = 0; j < W; j++) acc += Wl[i * W + j] * a[j];
            z[i] = gain * acc;
          }
          if (residual) for (let i = 0; i < W; i++) h[i] += z[i];
          else for (let i = 0; i < W; i++) h[i] = z[i];
        }
        for (let i = 0; i < W; i++) dh[i] = wout[i];
        lognorm[depth] += Math.log10(norm(dh) + 1e-300);
        for (let l = depth - 1; l >= 0; l--) {
          const Wl = Ws[l], s = scale[l];
          for (let j = 0; j < W; j++) {
            let acc = 0;
            if (masks[l * W + j]) {
              for (let i = 0; i < W; i++) acc += Wl[i * W + j] * dh[i];
              acc = gain * acc / s;
            }
            tmp[j] = acc + (residual ? dh[j] : 0);
          }
          for (let j = 0; j < W; j++) dh[j] = tmp[j];
          lognorm[l] += Math.log10(norm(dh) + 1e-300);
        }
        let d = 0;
        for (let i = 0; i < W; i++) d += dh[i] * w0[i];
        dydx[n] = d;
      }
      for (let l = 0; l <= depth; l++) lognorm[l] /= NX;
      return { dydx, lognorm };
    }

    function compute() {
      plain = pass(false);
      resid = pass(true);
    }

    /* Centra e normaliza um vetor para comparar só a forma. */
    function normalized(v) {
      let mu = 0;
      for (const u of v) mu += u;
      mu /= v.length;
      let s2 = 0;
      for (const u of v) s2 += (u - mu) * (u - mu);
      const sd = Math.sqrt(s2 / v.length) || 1;
      return Array.from(v, (u) => (u - mu) / sd);
    }

    function drawDeriv() {
      const th = P.theme();
      const { ctx, w, h } = P.setup(cvDeriv);
      P.clear(ctx, w, h);
      const fr = P.frame(ctx, w, h, -DOMX, DOMX, -3.4, 3.4);
      P.axes(ctx, fr, { xlabel: 'x', ylabel: '∂y/∂x (norm.)', xticks: [-2, -1, 0, 1, 2], yticks: [-2, 0, 2] });
      const xs = [];
      for (let n = 0; n < NX; n++) xs.push(-DOMX + 2 * DOMX * (n + 0.5) / NX);
      ctx.save();
      ctx.beginPath();
      ctx.rect(fr.X(fr.xmin), fr.Y(fr.ymax), fr.iw, fr.ih);
      ctx.clip();
      P.line(ctx, fr, xs, normalized(plain.dydx), th.cyan, { width: 1.5, alpha: 0.9 });
      P.line(ctx, fr, xs, normalized(resid.dydx), th.orange, { width: 1.8 });
      ctx.restore();
      P.label(ctx, fr.X(fr.xmin) + 8, fr.Y(fr.ymax) + 14, 'plain', th.cyan);
      P.label(ctx, fr.X(fr.xmin) + 8, fr.Y(fr.ymax) + 28, 'residual', th.orange);
    }

    function drawNorms() {
      const th = P.theme();
      const { ctx, w, h } = P.setup(cvNorms);
      P.clear(ctx, w, h);
      let lo = Infinity, hi = -Infinity;
      for (let l = 0; l <= depth; l++) {
        lo = Math.min(lo, plain.lognorm[l], resid.lognorm[l]);
        hi = Math.max(hi, plain.lognorm[l], resid.lognorm[l]);
      }
      lo = Math.floor(lo) - 0.5;
      hi = Math.ceil(hi) + 0.5;
      const fr = P.frame(ctx, w, h, 0, Math.max(depth, 1), lo, hi);
      P.axes(ctx, fr, {
        xlabel: 'camada l', ylabel: 'log₁₀ ‖∂y/∂h_l‖',
        xticks: [0, Math.round(depth / 2), depth],
        yticks: [Math.ceil(lo), Math.round((lo + hi) / 2), Math.floor(hi)],
      });
      /* Linha de referência em 10^0. */
      if (lo < 0 && hi > 0) {
        P.line(ctx, fr, [0, depth], [0, 0], th.comment, { width: 1, dash: [4, 4], alpha: 0.6 });
      }
      const ls = [];
      for (let l = 0; l <= depth; l++) ls.push(l);
      P.line(ctx, fr, ls, Array.from(plain.lognorm), th.cyan, { width: 1.8 });
      P.line(ctx, fr, ls, Array.from(resid.lognorm), th.orange, { width: 1.8 });
    }

    function updateReadout() {
      const num = (v) => Math.pow(10, v).toExponential(1);
      $('s2-readout').textContent =
        '‖∂y/∂h_0‖ plain = ' + num(plain.lognorm[0]) +
        ' · residual = ' + num(resid.lognorm[0]);
    }

    function redraw() {
      drawDeriv();
      drawNorms();
      updateReadout();
    }

    function recompute() {
      compute();
      redraw();
    }

    slDepth.addEventListener('input', () => {
      depth = +slDepth.value;
      $('s2-depth-val').textContent = depth;
      recompute();
    });
    slGain.addEventListener('input', () => {
      gain = +slGain.value;
      $('s2-gain-val').textContent = gain.toFixed(2);
      recompute();
    });
    btnSeed.addEventListener('click', () => {
      seed += 1;
      makeWeights();
      recompute();
    });

    makeWeights();
    compute();
    redraw();
    P.onRedraw(redraw);
    P.observeResize(cvDeriv, redraw);
    P.observeResize(cvNorms, redraw);
  }

  DL.sections.push({ name: 's2-gradient-flow', init });
})();
