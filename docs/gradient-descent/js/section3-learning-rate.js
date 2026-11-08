/* Seção 3: taxa de aprendizado η em 1D. Duas paisagens: a convexa ℒ = θ²
   dos slides (θ₀ = 2, presets η = 0.02 / 0.3 / 1.1) e a não convexa
   f(θ) = (θ⁴ − 3θ² + θ)/4, com mínimo global em θ ≈ −1.30 e local em θ ≈ 1.13. */
(function () {
  'use strict';
  window.DL = window.DL || {};
  DL.sections = DL.sections || [];

  const MAX_STEPS = 300, TRAIL = 40;
  const MODES = {
    convex: {
      f: (x) => x * x, df: (x) => 2 * x,
      xmin: -4.6, xmax: 4.6, ymin: -1, ymax: 19, x0: 2,
      xticks: [-4, -2, 0, 2, 4], yticks: [0, 5, 10, 15], lossCap: 20, tan: 1.2, label: 'ℒ(θ) = θ²',
    },
    nonconvex: {
      f: (x) => (x * x * x * x - 3 * x * x + x) / 4, df: (x) => x * x * x - 1.5 * x + 0.25,
      xmin: -2.5, xmax: 2.5, ymin: -1.3, ymax: 6, x0: 2.3,
      xticks: [-2, -1, 0, 1, 2], yticks: [0, 2, 4, 6], lossCap: 8, tan: 0.65, label: 'f(θ) = (θ⁴ − 3θ² + θ)/4',
    },
  };
  let mode = MODES.nonconvex;
  let XMIN = mode.xmin, XMAX = mode.xmax, YMIN = mode.ymin, YMAX = mode.ymax;
  const f = (x) => mode.f(x);
  const df = (x) => mode.df(x);

  function init() {
    const P = DL.plot;
    const $ = (id) => document.getElementById(id);

    /* Rótulo sobre fundo opaco (legível sobre heatmap, curvas e pontos);
       com fr, fica dentro do retângulo do gráfico. */
    function pill(ctx, x, y, text, color, align, fr, size) {
      const th = P.theme();
      size = size || 11;
      ctx.font = size + 'px sans-serif';
      const tw = ctx.measureText(text.replace(/[_{}]/g, '')).width;
      const lead = align === 'center' ? tw / 2 : align === 'right' ? tw : 0;
      if (fr) {
        x = Math.max(fr.X(fr.xmin) + lead + 4, Math.min(fr.X(fr.xmax) - (tw - lead) - 4, x));
        y = Math.max(fr.Y(fr.ymax) + size + 3, Math.min(fr.Y(fr.ymin) - 6, y));
      }
      ctx.fillStyle = th.card;
      ctx.globalAlpha = 0.85;
      ctx.fillRect(x - lead - 3, y - size + 1, tw + 6, size + 4);
      ctx.globalAlpha = 1;
      P.mathText(ctx, text, x, y, color, align, size);
    }
    /* Fundo opaco atrás do rótulo do eixo x que P.axes desenha dentro do
       gráfico (sobre o amarelo do viridis o texto some). */
    function xlabelPill(ctx, fr, text) {
      const th = P.theme();
      ctx.font = '11px sans-serif';
      const tw = ctx.measureText(text.replace(/[_{}]/g, '')).width;
      ctx.fillStyle = th.card;
      ctx.globalAlpha = 0.85;
      ctx.fillRect(fr.X(fr.xmax) - 7 - tw, fr.Y(fr.ymin) - 18, tw + 6, 14);
      ctx.globalAlpha = 1;
    }

    const cvFn = $('s3-fn'), cvLoss = $('s3-loss');
    const slEta = $('s3-eta');
    const btnStep = $('s3-step'), btnRun = $('s3-run'), btnReset = $('s3-reset');

    /* Slider em escala log: η ∈ [0.001, 1.2] */
    const sliderToEta = (v) => 0.001 * Math.pow(1200, v / 100);
    const etaToSlider = (e) => 100 * Math.log(e / 0.001) / Math.log(1200);

    let eta = sliderToEta(+slEta.value);
    let x0 = mode.x0;
    let hist = [x0];
    let diverged = false;
    let running = false, acc = 0;
    let frFn = null, dragging = false;

    const last = () => hist[hist.length - 1];

    function stepOnce() {
      if (diverged || hist.length > MAX_STEPS) return;
      const x = last();
      const xn = x - eta * df(x);
      hist.push(xn);
      if (!isFinite(xn) || Math.abs(xn) > 1e3) diverged = true;
    }

    function converged() {
      return !diverged && isFinite(last()) && Math.abs(df(last())) < 1e-7;
    }

    function clipPlot(ctx, fr) {
      ctx.save();
      ctx.beginPath();
      ctx.rect(fr.X(fr.xmin), fr.Y(fr.ymax), fr.iw, fr.ih);
      ctx.clip();
    }

    function arrowSeg(ctx, ax, ay, bx, by, color, alpha) {
      ctx.strokeStyle = color;
      ctx.fillStyle = color;
      ctx.globalAlpha = alpha;
      ctx.lineWidth = 1.5;
      ctx.beginPath();
      ctx.moveTo(ax, ay);
      ctx.lineTo(bx, by);
      ctx.stroke();
      if (Math.hypot(bx - ax, by - ay) > 7) {
        const ang = Math.atan2(by - ay, bx - ax), hl = 5;
        ctx.beginPath();
        ctx.moveTo(bx, by);
        ctx.lineTo(bx - hl * Math.cos(ang - 0.45), by - hl * Math.sin(ang - 0.45));
        ctx.lineTo(bx - hl * Math.cos(ang + 0.45), by - hl * Math.sin(ang + 0.45));
        ctx.closePath();
        ctx.fill();
      }
      ctx.globalAlpha = 1;
    }

    function drawFn() {
      const th = P.theme();
      const { ctx, w, h } = P.setup(cvFn);
      P.clear(ctx, w, h);
      const fr = P.frame(ctx, w, h, XMIN, XMAX, YMIN, YMAX);
      frFn = fr;
      P.axes(ctx, fr, { xlabel: 'θ', ylabel: 'f(θ)', xticks: mode.xticks, yticks: mode.yticks });

      const xs = [], ys = [];
      for (let x = XMIN; x <= XMAX + 1e-9; x += 0.02) { xs.push(x); ys.push(f(x)); }
      P.line(ctx, fr, xs, ys, th.purple, { width: 2.2 });

      clipPlot(ctx, fr);
      /* Rastro com setas e alpha crescente (passos antigos esmaecem) */
      const lo = Math.max(0, hist.length - 1 - TRAIL);
      for (let i = lo; i < hist.length - 1; i++) {
        const a = 0.15 + 0.75 * (i - lo + 1) / (hist.length - lo);
        arrowSeg(ctx, fr.X(hist[i]), fr.Y(f(hist[i])), fr.X(hist[i + 1]), fr.Y(f(hist[i + 1])), th.cyan, a);
      }
      const xc = last();
      if (isFinite(xc)) {
        /* Tangente no ponto atual */
        const g = df(xc), t0 = xc - mode.tan, t1 = xc + mode.tan;
        P.line(ctx, fr, [t0, t1], [f(xc) + g * (t0 - xc), f(xc) + g * (t1 - xc)],
          th.orange, { dash: [5, 4], width: 1.6 });
        /* Bola no iterado atual (arrastável) */
        ctx.beginPath();
        ctx.arc(fr.X(xc), fr.Y(f(xc)), 7, 0, 2 * Math.PI);
        ctx.fillStyle = th.pink;
        ctx.fill();
        ctx.lineWidth = 1.5;
        ctx.strokeStyle = th.bg;
        ctx.stroke();
      }
      ctx.restore();

      pill(ctx, fr.X(XMIN) + 8, fr.Y(YMAX) + 14, mode.label, th.purple);
      pill(ctx, fr.X(XMIN) + 8, fr.Y(YMAX) + 28, 'tangente f′(θ_k)', th.orange);
      if (diverged) pill(ctx, fr.X(0), fr.Y(YMAX) + 18, 'divergiu!', th.red, 'center');
    }

    function drawLoss() {
      const th = P.theme();
      const { ctx, w, h } = P.setup(cvLoss);
      P.clear(ctx, w, h);
      const n = hist.length - 1;
      const kmax = Math.max(20, n);
      const vals = hist.map((x) => f(x));
      let vmax = 0;
      for (const v of vals) if (isFinite(v)) vmax = Math.max(vmax, Math.min(v, 50));
      const ymax = Math.max(1, Math.min(mode.lossCap, vmax * 1.08));
      const fr = P.frame(ctx, w, h, 0, kmax, -1.2, ymax);
      P.axes(ctx, fr, {
        xlabel: 'k', ylabel: 'f(θ_k)',
        xticks: [0, Math.round(kmax / 2), kmax],
        yticks: [0, Math.round(ymax * 10) / 10],
      });
      clipPlot(ctx, fr);
      const ks = vals.map((_, i) => i);
      const safe = vals.map((v) => (isFinite(v) ? v : ymax * 2));
      P.line(ctx, fr, ks, safe, th.cyan, { width: 1.8 });
      P.scatter(ctx, fr, safe.map((v, i) => [i, v]), th.cyan, { r: 2, alpha: 0.9 });
      ctx.restore();
    }

    function updateReadout() {
      const xc = last(), k = hist.length - 1;
      const num = (v) => (isFinite(v) ? v.toFixed(3) : '∞');
      $('s3-readout').textContent =
        'k = ' + k + ' · θ = ' + num(xc) + ' · f(θ) = ' + num(f(xc)) +
        (diverged ? ' · divergiu' : '');
    }

    function redraw() {
      drawFn();
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
      if (diverged || hist.length > MAX_STEPS || converged()) stopRun();
    }

    function setStart(x) {
      x0 = Math.min(XMAX - 0.05, Math.max(XMIN + 0.05, x));
      hist = [x0];
      diverged = false;
      redraw();
    }

    cvFn.addEventListener('pointerdown', (e) => {
      if (!frFn) return;
      dragging = true;
      cvFn.setPointerCapture(e.pointerId);
      stopRun();
      const r = cvFn.getBoundingClientRect();
      setStart(frFn.invX(e.clientX - r.left));
    });
    cvFn.addEventListener('pointermove', (e) => {
      if (!dragging || !frFn) return;
      const r = cvFn.getBoundingClientRect();
      setStart(frFn.invX(e.clientX - r.left));
    });
    cvFn.addEventListener('pointerup', () => { dragging = false; });

    btnStep.addEventListener('click', () => { stepOnce(); redraw(); });
    btnRun.addEventListener('click', () => {
      if (running) { stopRun(); return; }
      if (diverged || hist.length > MAX_STEPS || converged()) { hist = [x0]; diverged = false; }
      running = true;
      acc = 0;
      btnRun.textContent = '⏸ Pausar';
      DL.startTicker(tick);
    });
    btnReset.addEventListener('click', () => {
      stopRun();
      hist = [x0];
      diverged = false;
      redraw();
    });
    slEta.addEventListener('input', () => {
      eta = sliderToEta(+slEta.value);
      $('s3-eta-val').textContent = eta.toFixed(3);
    });
    /* Presets de η dos três painéis dos slides. */
    for (const b of document.querySelectorAll('[data-s3-eta]')) {
      b.addEventListener('click', () => {
        eta = +b.dataset.s3Eta;
        slEta.value = Math.round(etaToSlider(eta));
        $('s3-eta-val').textContent = eta.toFixed(3);
        stopRun();
        hist = [x0];
        diverged = false;
        redraw();
      });
    }
    function setMode(name) {
      stopRun();
      mode = MODES[name];
      XMIN = mode.xmin; XMAX = mode.xmax; YMIN = mode.ymin; YMAX = mode.ymax;
      x0 = mode.x0;
      hist = [x0];
      diverged = false;
      $('s3-tab-convex').classList.toggle('active', name === 'convex');
      $('s3-tab-nonconvex').classList.toggle('active', name === 'nonconvex');
      redraw();
    }
    $('s3-tab-convex').addEventListener('click', () => setMode('convex'));
    $('s3-tab-nonconvex').addEventListener('click', () => setMode('nonconvex'));

    $('s3-eta-val').textContent = eta.toFixed(3);
    redraw();
    P.onRedraw(redraw);
    P.observeResize(cvFn, redraw);
    P.observeResize(cvLoss, redraw);
  }

  DL.sections.push({ name: 's3-learning-rate', init });
})();
