/* Seção 7: batch completo vs mini-batch vs SGD numa regressão linear ŷ = wx + b.
   L(w, b) = média dos erros quadráticos; instâncias amostradas sem reposição (épocas). */
(function () {
  'use strict';
  window.DL = window.DL || {};
  DL.sections = DL.sections || [];

  const N = 60, MAX_TRAIL = 800;

  function init() {
    const U = DL.utils, P = DL.plot;
    const $ = (id) => document.getElementById(id);

    const cvSurf = $('s7-surface'), cvData = $('s7-data');
    const tabBatch = $('s7-tab-batch'), tabMini = $('s7-tab-mini'), tabSgd = $('s7-tab-sgd');
    const slBs = $('s7-bs'), slEta = $('s7-eta');
    const bsWrap = $('s7-bs-wrap');
    const btnRun = $('s7-run'), btnReset = $('s7-reset'), btnResample = $('s7-resample');

    let seed = 7;
    let xs, ys;
    let A, B, C, D, E;                 // E[x²], E[x], E[xy], E[y], E[y²]
    let wopt, bopt, Lmin, Lmax, dom;
    let mode = 'batch', m = +slBs.value, eta = +slEta.value;
    let start, traj, kCount, diverged;
    let order, pos, rngBatch;
    let running = false, acc = 0;
    let frSurf = null, dragging = false;

    /* Forma quadrática da loss via estatísticas suficientes: O(1) por avaliação */
    function L(wv, bv) {
      return wv * wv * A + 2 * wv * bv * B + bv * bv - 2 * wv * C - 2 * bv * D + E;
    }

    function shuffle(arr) {
      for (let i = arr.length - 1; i > 0; i--) {
        const j = Math.floor(rngBatch() * (i + 1));
        const t = arr[i];
        arr[i] = arr[j];
        arr[j] = t;
      }
    }

    function resetTraj() {
      traj = [start.slice()];
      kCount = 0;
      diverged = false;
      pos = N;                         // força reembaralhar no próximo passo
    }

    function genData() {
      const rng = U.mulberry32(seed);
      xs = []; ys = [];
      for (let i = 0; i < N; i++) {
        const x = -2 + 4 * rng();
        xs.push(x);
        ys.push(1.5 * x + 0.5 + 0.5 * U.randn(rng));
      }
      A = 0; B = 0; C = 0; D = 0; E = 0;
      for (let i = 0; i < N; i++) {
        A += xs[i] * xs[i]; B += xs[i]; C += xs[i] * ys[i]; D += ys[i]; E += ys[i] * ys[i];
      }
      A /= N; B /= N; C /= N; D /= N; E /= N;
      /* Equações normais: mínimo de batch completo */
      wopt = (C - B * D) / (A - B * B);
      bopt = D - wopt * B;
      Lmin = L(wopt, bopt);
      dom = { w0: wopt - 2.5, w1: wopt + 2.5, b0: bopt - 2.5, b1: bopt + 2.5 };
      Lmax = Math.max(L(dom.w0, dom.b0), L(dom.w0, dom.b1), L(dom.w1, dom.b0), L(dom.w1, dom.b1));
      start = [wopt - 2, bopt + 2];
      rngBatch = U.mulberry32(seed ^ 0x5bd1e995);
      order = Array.from({ length: N }, (_, i) => i);
      resetTraj();
    }

    /* Amostragem sem reposição: reembaralha a cada época */
    function nextBatch(mm) {
      if (mm >= N) return order;
      const idx = [];
      while (idx.length < mm) {
        if (pos >= N) { shuffle(order); pos = 0; }
        idx.push(order[pos++]);
      }
      return idx;
    }

    function stepOnce() {
      if (diverged) return;
      const p = traj[traj.length - 1];
      const mm = mode === 'batch' ? N : (mode === 'mini' ? m : 1);
      const idx = nextBatch(mm);
      let gw = 0, gb = 0;
      for (const i of idx) {
        const e = p[0] * xs[i] + p[1] - ys[i];
        gw += 2 * e * xs[i];
        gb += 2 * e;
      }
      gw /= idx.length;
      gb /= idx.length;
      const pn = [p[0] - eta * gw, p[1] - eta * gb];
      traj.push(pn);
      kCount++;
      if (traj.length > MAX_TRAIL) traj.shift();
      if (!isFinite(pn[0]) || !isFinite(pn[1]) || Math.abs(pn[0]) > 1e4 || Math.abs(pn[1]) > 1e4) {
        diverged = true;
        stopRun();
      }
    }

    function clipPlot(ctx, fr) {
      ctx.save();
      ctx.beginPath();
      ctx.rect(fr.X(fr.xmin), fr.Y(fr.ymax), fr.iw, fr.ih);
      ctx.clip();
    }

    const clampW = (v) => (isFinite(v) ? Math.max(dom.w0 - 50, Math.min(dom.w1 + 50, v)) : dom.w1 + 50);
    const clampB = (v) => (isFinite(v) ? Math.max(dom.b0 - 50, Math.min(dom.b1 + 50, v)) : dom.b1 + 50);

    function drawSurface() {
      const th = P.theme();
      const { ctx, w, h } = P.setup(cvSurf);
      P.clear(ctx, w, h);
      const fr = P.frame(ctx, w, h, dom.w0, dom.w1, dom.b0, dom.b1);
      frSurf = fr;
      P.heatmap(ctx, fr, L, 150, 150, (v) => {
        const u = Math.pow(Math.min(1, (v - Lmin) / (Lmax - Lmin)), 0.35);
        const col = P.viridis(u);
        const lift = (u * 9) % 1 < 0.13 ? 42 : 0;
        return [Math.min(255, col[0] + lift), Math.min(255, col[1] + lift), Math.min(255, col[2] + lift), 235];
      });
      const r1 = (v) => Math.round(v * 10) / 10;
      P.axes(ctx, fr, {
        xlabel: 'w', ylabel: 'b',
        xticks: [r1(wopt - 2), r1(wopt), r1(wopt + 2)],
        yticks: [r1(bopt - 2), r1(bopt), r1(bopt + 2)],
      });

      clipPlot(ctx, fr);
      const txs = traj.map((p) => clampW(p[0]));
      const tys = traj.map((p) => clampB(p[1]));
      P.line(ctx, fr, txs, tys, th.cyan, { width: 1.6, alpha: 0.85 });
      P.scatter(ctx, fr, txs.map((x, i) => [x, tys[i]]), th.cyan, { r: 1.8, alpha: 0.7 });
      /* Ponto atual */
      const cur = traj[traj.length - 1];
      ctx.beginPath();
      ctx.arc(fr.X(clampW(cur[0])), fr.Y(clampB(cur[1])), 5.5, 0, 2 * Math.PI);
      ctx.fillStyle = th.pink;
      ctx.fill();
      ctx.lineWidth = 1.5;
      ctx.strokeStyle = th.bg;
      ctx.stroke();
      ctx.restore();

      /* Mínimo de batch completo: × verde */
      ctx.strokeStyle = th.green;
      ctx.lineWidth = 2;
      const mx = fr.X(wopt), my = fr.Y(bopt), s = 6;
      ctx.beginPath();
      ctx.moveTo(mx - s, my - s); ctx.lineTo(mx + s, my + s);
      ctx.moveTo(mx - s, my + s); ctx.lineTo(mx + s, my - s);
      ctx.stroke();
      P.label(ctx, mx + 9, my + 4, '(w*, b*)', th.green);
      /* Ponto inicial */
      ctx.beginPath();
      ctx.arc(fr.X(start[0]), fr.Y(start[1]), 6, 0, 2 * Math.PI);
      ctx.strokeStyle = th.fg;
      ctx.lineWidth = 2;
      ctx.stroke();

      if (diverged) P.label(ctx, fr.X((dom.w0 + dom.w1) / 2), fr.Y(dom.b1) + 18, 'divergiu!', th.red, 'center');
    }

    function drawData() {
      const th = P.theme();
      const { ctx, w, h } = P.setup(cvData);
      P.clear(ctx, w, h);
      let ymin = Infinity, ymax = -Infinity;
      for (const y of ys) { ymin = Math.min(ymin, y); ymax = Math.max(ymax, y); }
      ymin -= 0.6; ymax += 0.6;
      const fr = P.frame(ctx, w, h, -2.3, 2.3, ymin, ymax);
      P.axes(ctx, fr, {
        xlabel: 'x', ylabel: 'y',
        xticks: [-2, 0, 2],
        yticks: [Math.round(ymin + 0.5), 0, Math.round(ymax - 0.5)],
      });
      P.scatter(ctx, fr, xs.map((x, i) => [x, ys[i]]), th.cyan, { r: 2.5, alpha: 0.7 });
      clipPlot(ctx, fr);
      /* Ajuste de batch completo (referência) */
      P.line(ctx, fr, [-2.3, 2.3], [wopt * -2.3 + bopt, wopt * 2.3 + bopt],
        th.comment, { dash: [5, 4], width: 1.4 });
      /* Reta atual */
      const cur = traj[traj.length - 1];
      const wv = clampW(cur[0]), bv = clampB(cur[1]);
      P.line(ctx, fr, [-2.3, 2.3], [wv * -2.3 + bv, wv * 2.3 + bv], th.orange, { width: 2.2 });
      ctx.restore();
      P.label(ctx, fr.X(-2.3) + 8, fr.Y(ymax) + 14, 'ŷ = wx + b', th.orange);
      P.label(ctx, fr.X(-2.3) + 8, fr.Y(ymax) + 28, 'ajuste batch completo', th.comment);
    }

    function updateReadout() {
      const cur = traj[traj.length - 1];
      const num = (v) => (isFinite(v) ? v.toFixed(2) : '∞');
      const lcur = L(cur[0], cur[1]);
      $('s7-readout').textContent =
        'k = ' + kCount + ' · w = ' + num(cur[0]) + ' · b = ' + num(cur[1]) +
        ' · L = ' + (isFinite(lcur) ? lcur.toFixed(3) : '∞') +
        (diverged ? ' · divergiu' : '');
    }

    function redraw() {
      drawSurface();
      drawData();
      updateReadout();
    }

    function stopRun() {
      running = false;
      btnRun.textContent = '▶ Rodar';
      DL.stopTicker(tick);
    }

    function tick(dt) {
      const rate = mode === 'batch' ? 10 : (mode === 'mini' ? 30 : 60);
      acc += dt * rate;
      while (acc >= 1 && !diverged) { acc -= 1; stepOnce(); }
      redraw();
      if (diverged) stopRun();
    }

    function setMode(mo) {
      mode = mo;
      tabBatch.classList.toggle('active', mo === 'batch');
      tabMini.classList.toggle('active', mo === 'mini');
      tabSgd.classList.toggle('active', mo === 'sgd');
      bsWrap.classList.toggle('disabled', mo !== 'mini');
      slBs.disabled = mo !== 'mini';
      resetTraj();
      redraw();
    }

    function setStart(wv, bv) {
      start = [
        Math.max(dom.w0 + 0.05, Math.min(dom.w1 - 0.05, wv)),
        Math.max(dom.b0 + 0.05, Math.min(dom.b1 - 0.05, bv)),
      ];
      resetTraj();
      redraw();
    }

    cvSurf.addEventListener('pointerdown', (e) => {
      if (!frSurf) return;
      dragging = true;
      cvSurf.setPointerCapture(e.pointerId);
      stopRun();
      const r = cvSurf.getBoundingClientRect();
      setStart(frSurf.invX(e.clientX - r.left), frSurf.invY(e.clientY - r.top));
    });
    cvSurf.addEventListener('pointermove', (e) => {
      if (!dragging || !frSurf) return;
      const r = cvSurf.getBoundingClientRect();
      setStart(frSurf.invX(e.clientX - r.left), frSurf.invY(e.clientY - r.top));
    });
    cvSurf.addEventListener('pointerup', () => { dragging = false; });

    tabBatch.addEventListener('click', () => setMode('batch'));
    tabMini.addEventListener('click', () => setMode('mini'));
    tabSgd.addEventListener('click', () => setMode('sgd'));

    btnRun.addEventListener('click', () => {
      if (running) { stopRun(); return; }
      if (diverged) resetTraj();
      running = true;
      acc = 0;
      btnRun.textContent = '⏸ Pausar';
      DL.startTicker(tick);
    });
    btnReset.addEventListener('click', () => { stopRun(); resetTraj(); redraw(); });
    btnResample.addEventListener('click', () => {
      stopRun();
      seed = (seed * 1664525 + 1013904223) >>> 0;
      genData();
      redraw();
    });

    slBs.addEventListener('input', () => {
      m = +slBs.value;
      $('s7-bs-val').textContent = m;
    });
    slEta.addEventListener('input', () => {
      eta = +slEta.value;
      $('s7-eta-val').textContent = eta.toFixed(3);
    });

    genData();
    redraw();
    P.onRedraw(redraw);
    P.observeResize(cvSurf, redraw);
    P.observeResize(cvData, redraw);
  }

  DL.sections.push({ name: 's7-batches', init });
})();
