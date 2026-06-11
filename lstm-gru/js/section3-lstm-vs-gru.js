/* Seção 3: LSTM e GRU lado a lado na mesma sequência, com os pesos do
   exemplo numérico dos slides (o primeiro passo reproduz as contas da aula). */
(function () {
  'use strict';
  window.DL = window.DL || {};
  DL.sections = DL.sections || [];

  const T = 16;
  /* pesos do exemplo numérico dos slides: θ = [θ_h, θ_x], bias b */
  const W = {
    f: [0.5, -0.5, 0.3],
    i: [0.3, -0.4, 0.5],
    c: [0.1, -0.3, 0.7],
    o: [-0.5, 0.4, 0.5],
    r: [0.5, -0.4, 0.2],
    z: [-0.3, 0.6, 0.1],
    h: [0.4, 0.5, 0.0],
  };
  const C0 = 0.5, H0 = -0.6, X1 = 0.8;

  function init() {
    const P = DL.plot;
    const $ = (id) => document.getElementById(id);
    const sig = (x) => 1 / (1 + Math.exp(-x));

    const cvL = $('s3-lstm'), cvG = $('s3-gru');
    const btnStep = $('s3-step'), btnRun = $('s3-run'), btnReset = $('s3-reset'), btnSample = $('s3-resample');

    let seed = 3;
    let xs = [];
    let L = null, G = null; /* trajetórias completas */
    let t = 0;
    let running = false, acc = 0;

    function genInputs() {
      const rng = DL.utils.mulberry32(seed);
      xs = [0, X1]; /* x_1 = 0.8 fixo, para casar com os slides */
      for (let k = 2; k <= T; k++) {
        let v = 0;
        if (rng() < 0.25) v = (rng() < 0.5 ? -1 : 1) * (0.5 + 0.7 * rng());
        xs.push(v);
      }
    }

    function gate(wb, hv, xv) { return wb[0] * hv + wb[1] * xv + wb[2]; }

    function compute() {
      L = { c: [C0], h: [H0], f: [0], i: [0], o: [0] };
      G = { h: [H0], r: [0], z: [0] };
      for (let k = 1; k <= T; k++) {
        const x = xs[k];
        /* LSTM */
        const hl = L.h[k - 1];
        const f = sig(gate(W.f, hl, x));
        const i = sig(gate(W.i, hl, x));
        const cand = Math.tanh(gate(W.c, hl, x));
        const c = f * L.c[k - 1] + i * cand;
        const o = sig(gate(W.o, hl, x));
        L.f.push(f); L.i.push(i); L.o.push(o);
        L.c.push(c);
        L.h.push(o * Math.tanh(c));
        /* GRU */
        const hg = G.h[k - 1];
        const r = sig(gate(W.r, hg, x));
        const z = sig(gate(W.z, hg, x));
        const hh = Math.tanh(W.h[0] * (r * hg) + W.h[1] * x + W.h[2]);
        G.r.push(r); G.z.push(z);
        G.h.push((1 - z) * hg + z * hh);
      }
    }

    function bars(ctx, fr, th) {
      ctx.fillStyle = th.comment;
      ctx.globalAlpha = 0.3;
      const y0 = fr.Y(0);
      const bw = Math.max(3, fr.iw / T * 0.3);
      for (let k = 1; k <= T; k++) {
        if (xs[k] === 0) continue;
        const py = fr.Y(Math.max(-1.05, Math.min(1.05, xs[k])));
        ctx.fillRect(fr.X(k) - bw / 2, Math.min(py, y0), bw, Math.abs(py - y0));
      }
      ctx.globalAlpha = 1;
    }

    function marker(ctx, fr, th) {
      ctx.strokeStyle = th.comment;
      ctx.lineWidth = 1;
      ctx.setLineDash([4, 4]);
      ctx.beginPath();
      ctx.moveTo(fr.X(t), fr.Y(fr.ymin));
      ctx.lineTo(fr.X(t), fr.Y(fr.ymax));
      ctx.stroke();
      ctx.setLineDash([]);
    }

    function drawCell(cv, isLSTM) {
      const th = P.theme();
      const { ctx, w, h } = P.setup(cv);
      P.clear(ctx, w, h);
      const fr = P.frame(ctx, w, h, 0, T, -1.1, 1.1);
      P.axes(ctx, fr, { xlabel: 't', xticks: [0, 4, 8, 12, 16], yticks: [-1, 0, 1] });
      bars(ctx, fr, th);
      marker(ctx, fr, th);
      const ks = [];
      for (let k = 0; k <= t; k++) ks.push(k);
      const ksg = ks.slice(1); /* portas só existem a partir de t = 1 */
      if (isLSTM) {
        if (t >= 1) {
          P.line(ctx, fr, ksg, L.f.slice(1, t + 1), th.cyan, { width: 1.1, dash: [3, 3] });
          P.line(ctx, fr, ksg, L.i.slice(1, t + 1), th.purple, { width: 1.1, dash: [3, 3] });
          P.line(ctx, fr, ksg, L.o.slice(1, t + 1), th.pink, { width: 1.1, dash: [3, 3] });
        }
        P.line(ctx, fr, ks, L.c.slice(0, t + 1), th.orange, { width: 2.4 });
        P.line(ctx, fr, ks, L.h.slice(0, t + 1), th.green, { width: 2.4 });
        P.scatter(ctx, fr, [[t, L.c[t]], [t, L.h[t]]], th.fg, { r: 3, alpha: 1 });
        P.label(ctx, fr.X(0) + 8, fr.Y(1.1) + 14, 'c_t', th.orange);
        P.label(ctx, fr.X(0) + 8, fr.Y(1.1) + 28, 'h_t', th.green);
        P.label(ctx, fr.X(T) - 60, fr.Y(1.1) + 14, 'f', th.cyan);
        P.label(ctx, fr.X(T) - 44, fr.Y(1.1) + 14, 'i', th.purple);
        P.label(ctx, fr.X(T) - 28, fr.Y(1.1) + 14, 'o', th.pink);
      } else {
        if (t >= 1) {
          P.line(ctx, fr, ksg, G.r.slice(1, t + 1), th.cyan, { width: 1.1, dash: [3, 3] });
          P.line(ctx, fr, ksg, G.z.slice(1, t + 1), th.purple, { width: 1.1, dash: [3, 3] });
        }
        P.line(ctx, fr, ks, G.h.slice(0, t + 1), th.green, { width: 2.4 });
        P.scatter(ctx, fr, [[t, G.h[t]]], th.fg, { r: 3, alpha: 1 });
        P.label(ctx, fr.X(0) + 8, fr.Y(1.1) + 14, 'h_t', th.green);
        P.label(ctx, fr.X(T) - 44, fr.Y(1.1) + 14, 'r', th.cyan);
        P.label(ctx, fr.X(T) - 28, fr.Y(1.1) + 14, 'z', th.purple);
      }
    }

    function updateReadout() {
      const el = $('s3-readout');
      if (t === 0) {
        el.textContent = 't = 0 · estados iniciais: c_0 = ' + C0.toFixed(2) + ', h_0 = ' + H0.toFixed(2);
        return;
      }
      el.textContent =
        't = ' + t + ' · x = ' + xs[t].toFixed(2) +
        ' · LSTM: f = ' + L.f[t].toFixed(3) + ', i = ' + L.i[t].toFixed(3) +
        ', o = ' + L.o[t].toFixed(3) + ', c = ' + L.c[t].toFixed(3) + ', h = ' + L.h[t].toFixed(3) +
        ' · GRU: r = ' + G.r[t].toFixed(3) + ', z = ' + G.z[t].toFixed(3) + ', h = ' + G.h[t].toFixed(3);
    }

    function redraw() {
      drawCell(cvL, true);
      drawCell(cvG, false);
      updateReadout();
    }

    function stopRun() {
      running = false;
      btnRun.textContent = '▶ Rodar';
      DL.stopTicker(tick);
    }

    function tick(dt) {
      acc += dt * 4;
      while (acc >= 1 && t < T) { acc -= 1; t++; }
      redraw();
      if (t >= T) stopRun();
    }

    btnStep.addEventListener('click', () => {
      if (t < T) t++;
      redraw();
    });
    btnRun.addEventListener('click', () => {
      if (running) { stopRun(); return; }
      if (t >= T) t = 0;
      running = true;
      acc = 0;
      btnRun.textContent = '⏸ Pausar';
      DL.startTicker(tick);
    });
    btnReset.addEventListener('click', () => { stopRun(); t = 0; redraw(); });
    btnSample.addEventListener('click', () => {
      stopRun();
      seed = (seed * 1103515245 + 12345) >>> 0;
      genInputs();
      compute();
      t = 0;
      redraw();
    });

    genInputs();
    compute();
    redraw();
    P.onRedraw(redraw);
    P.observeResize(cvL, redraw);
    P.observeResize(cvG, redraw);
  }

  DL.sections.push({ name: 's3-lstm-vs-gru', init });
})();
