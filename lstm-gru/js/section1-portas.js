/* Seção 1: as portas da LSTM em ação, com f, i e o fixadas por sliders e o
   candidato simplificado c̃_t = tanh(x_t). */
(function () {
  'use strict';
  window.DL = window.DL || {};
  DL.sections = DL.sections || [];

  const T = 24;

  function init() {
    const P = DL.plot;
    const $ = (id) => document.getElementById(id);

    const cvTime = $('s1-timeline'), cvBars = $('s1-bars');
    const slF = $('s1-f'), slI = $('s1-i'), slO = $('s1-o');
    const btnStep = $('s1-step'), btnRun = $('s1-run'), btnReset = $('s1-reset'), btnSample = $('s1-resample');

    let f = +slF.value, i = +slI.value, o = +slO.value;
    let seed = 7;
    let xs = [], cands = [], cs = [], hs = [];
    let t = 0;
    let running = false, acc = 0;

    function genInputs() {
      const rng = DL.utils.mulberry32(seed);
      xs = [0];
      for (let k = 1; k <= T; k++) {
        let v = 0;
        if (rng() < 0.18) v = (rng() < 0.5 ? -1 : 1) * (0.6 + 0.6 * rng());
        xs.push(v);
      }
      /* garante um pulso cedo, para a memória ter o que reter */
      xs[2] = 1.0;
    }

    /* Trajetória completa (barata); o passo só avança o cursor t. */
    function compute() {
      cands = [0]; cs = [0]; hs = [0];
      for (let k = 1; k <= T; k++) {
        const cand = Math.tanh(xs[k]);
        const c = f * cs[k - 1] + i * cand;
        cands.push(cand);
        cs.push(c);
        hs.push(o * Math.tanh(c));
      }
    }

    function yMax() {
      let m = 1.4;
      for (const v of cs) m = Math.max(m, Math.abs(v));
      return m + 0.2;
    }

    function bars(ctx, fr, ks, vs, color, bw, alpha) {
      ctx.fillStyle = color;
      ctx.globalAlpha = alpha;
      const y0 = fr.Y(0);
      for (let k = 0; k < ks.length; k++) {
        if (vs[k] === 0) continue;
        const py = fr.Y(vs[k]);
        ctx.fillRect(fr.X(ks[k]) - bw / 2, Math.min(py, y0), bw, Math.abs(py - y0));
      }
      ctx.globalAlpha = 1;
    }

    function vline(ctx, fr, x, color) {
      ctx.strokeStyle = color;
      ctx.lineWidth = 1;
      ctx.setLineDash([4, 4]);
      ctx.beginPath();
      ctx.moveTo(fr.X(x), fr.Y(fr.ymin));
      ctx.lineTo(fr.X(x), fr.Y(fr.ymax));
      ctx.stroke();
      ctx.setLineDash([]);
    }

    function drawTimeline() {
      const th = P.theme();
      const { ctx, w, h } = P.setup(cvTime);
      P.clear(ctx, w, h);
      const ym = yMax();
      const fr = P.frame(ctx, w, h, 0, T, -ym, ym);
      P.axes(ctx, fr, { xlabel: 't', xticks: [0, 6, 12, 18, 24], yticks: [-1, 0, 1] });
      const ks = xs.map((_, k) => k);
      bars(ctx, fr, ks, xs, th.cyan, Math.max(3, fr.iw / T * 0.35), 0.45);
      vline(ctx, fr, t, th.comment);
      const upto = (arr) => arr.slice(0, t + 1);
      P.line(ctx, fr, ks.slice(0, t + 1), upto(cs), th.orange, { width: 2.2 });
      P.line(ctx, fr, ks.slice(0, t + 1), upto(hs), th.green, { width: 2.2 });
      P.scatter(ctx, fr, [[t, cs[t]]], th.orange, { r: 3.5, alpha: 1 });
      P.scatter(ctx, fr, [[t, hs[t]]], th.green, { r: 3.5, alpha: 1 });
      P.label(ctx, fr.X(0) + 8, fr.Y(ym) + 14, 'x_t', th.cyan);
      P.label(ctx, fr.X(0) + 8, fr.Y(ym) + 28, 'c_t', th.orange);
      P.label(ctx, fr.X(0) + 8, fr.Y(ym) + 42, 'h_t', th.green);
    }

    function drawBars() {
      const th = P.theme();
      const { ctx, w, h } = P.setup(cvBars);
      P.clear(ctx, w, h);
      const ym = yMax();
      const fr = P.frame(ctx, w, h, -0.5, 4.5, -ym, ym);
      P.axes(ctx, fr, { yticks: [-1, 0, 1] });
      const cPrev = t >= 1 ? cs[t - 1] : 0;
      const cand = t >= 1 ? cands[t] : 0;
      const items = [
        { lab: 'c_{t−1}', val: cPrev, col: th.comment },
        { lab: 'f·c_{t−1}', val: f * cPrev, col: th.purple },
        { lab: 'i·c̃_t', val: t >= 1 ? i * cand : 0, col: th.cyan },
        { lab: 'c_t', val: cs[t], col: th.orange },
        { lab: 'h_t', val: hs[t], col: th.green },
      ];
      /* linha do zero */
      ctx.strokeStyle = th.line;
      ctx.lineWidth = 1;
      ctx.beginPath();
      ctx.moveTo(fr.X(fr.xmin), fr.Y(0));
      ctx.lineTo(fr.X(fr.xmax), fr.Y(0));
      ctx.stroke();
      const bw = fr.iw / 5 * 0.5;
      const y0 = fr.Y(0);
      for (let k = 0; k < items.length; k++) {
        const py = fr.Y(items[k].val);
        ctx.fillStyle = items[k].col;
        ctx.globalAlpha = 0.85;
        ctx.fillRect(fr.X(k) - bw / 2, Math.min(py, y0), bw, Math.max(1, Math.abs(py - y0)));
        ctx.globalAlpha = 1;
        P.mathText(ctx, items[k].lab, fr.X(k), fr.Y(fr.ymin) + 14, th.comment, 'center', 10);
      }
    }

    function updateReadout() {
      $('s1-readout').textContent =
        't = ' + t + ' · x_t = ' + (t >= 1 ? xs[t].toFixed(2) : '—') +
        ' · c_t = ' + cs[t].toFixed(3) + ' · h_t = ' + hs[t].toFixed(3);
    }

    function redraw() {
      drawTimeline();
      drawBars();
      updateReadout();
    }

    function stopRun() {
      running = false;
      btnRun.textContent = '▶ Rodar';
      DL.stopTicker(tick);
    }

    function tick(dt) {
      acc += dt * 5;
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

    function bindSlider(sl, valId, fix, set) {
      sl.addEventListener('input', () => {
        set(+sl.value);
        $(valId).textContent = (+sl.value).toFixed(fix);
        compute();
        redraw();
      });
    }
    bindSlider(slF, 's1-f-val', 2, (v) => { f = v; });
    bindSlider(slI, 's1-i-val', 2, (v) => { i = v; });
    bindSlider(slO, 's1-o-val', 2, (v) => { o = v; });

    genInputs();
    compute();
    redraw();
    P.onRedraw(redraw);
    P.observeResize(cvTime, redraw);
    P.observeResize(cvBars, redraw);
  }

  DL.sections.push({ name: 's1-portas', init });
})();
