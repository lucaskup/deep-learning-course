/* Seção 2: dinâmica do estado oculto iterado, h_t = tanh(θ_hh·h_{t−1}) ou
   h_t = θ_hh·h_{t−1}. Diagrama de teia de aranha com h_0 arrastável e
   trajetória h_t por passo: decaimento, explosão, saturação e oscilação. */
(function () {
  'use strict';
  window.DL = window.DL || {};
  DL.sections = DL.sections || [];

  const DOM = 1.6, NMAX = 30;

  function init() {
    const P = DL.plot;
    const $ = (id) => document.getElementById(id);

    const cvCob = $('s2-cobweb'), cvTraj = $('s2-traj');
    const slHh = $('s2-thhh');
    const tabTanh = $('s2-tab-tanh'), tabLin = $('s2-tab-linear');
    const btnStep = $('s2-step'), btnRun = $('s2-run'), btnReset = $('s2-reset');

    let thhh = +slHh.value;
    let mode = 'tanh';
    let h0 = 0.8;
    let cur = 0;               /* passos já dados (0..NMAX) */
    let running = false, acc = 0;
    let frCob = null, dragging = false;

    const f = (h) => (mode === 'tanh' ? Math.tanh(thhh * h) : thhh * h);
    const clampPlot = (v) => Math.max(-50, Math.min(50, v));

    function trajectory() {
      const hs = [h0];
      for (let t = 1; t <= NMAX; t++) hs.push(f(hs[t - 1]));
      return hs;
    }

    /* Ponto fixo positivo de tanh(θ·h) quando θ > 1 (iteração converge a h*). */
    function fixedPoint() {
      if (mode !== 'tanh' || thhh <= 1) return 0;
      let h = 1.5;
      for (let i = 0; i < 80; i++) h = Math.tanh(thhh * h);
      return h;
    }

    function clipPlot(ctx, fr) {
      ctx.save();
      ctx.beginPath();
      ctx.rect(fr.X(fr.xmin), fr.Y(fr.ymax), fr.iw, fr.ih);
      ctx.clip();
    }

    function drawCobweb() {
      const th = P.theme();
      const { ctx, w, h } = P.setup(cvCob);
      P.clear(ctx, w, h);
      const fr = P.frame(ctx, w, h, -DOM, DOM, -DOM, DOM);
      frCob = fr;
      P.axes(ctx, fr, { xlabel: 'h_{t−1}', ylabel: 'h_t', xticks: [-1, 0, 1], yticks: [-1, 0, 1] });

      clipPlot(ctx, fr);

      /* Diagonal h_t = h_{t−1} */
      P.line(ctx, fr, [-DOM, DOM], [-DOM, DOM], th.comment, { width: 1, dash: [4, 4] });

      /* Curva do mapa */
      const xs = [], ys = [];
      for (let i = 0; i <= 160; i++) {
        const x = -DOM + (2 * DOM * i) / 160;
        xs.push(x);
        ys.push(clampPlot(f(x)));
      }
      P.line(ctx, fr, xs, ys, th.purple, { width: 2 });

      /* Teia de aranha até o passo atual */
      const hs = trajectory();
      const px = [h0], py = [0];
      for (let t = 1; t <= cur; t++) {
        px.push(hs[t - 1]); py.push(hs[t]);    /* sobe até a curva */
        px.push(hs[t]); py.push(hs[t]);        /* anda até a diagonal */
      }
      P.line(ctx, fr, px.map(clampPlot), py.map(clampPlot), th.orange, { width: 1.6, alpha: 0.9 });
      ctx.restore();

      /* Pontos fixos */
      const hstar = fixedPoint();
      P.scatter(ctx, fr, [[0, 0]], hstar > 0 ? th.red : th.green, { r: 3.5, alpha: 1 });
      if (hstar > 0) {
        P.scatter(ctx, fr, [[hstar, hstar], [-hstar, -hstar]], th.green, { r: 3.5, alpha: 1 });
        P.label(ctx, fr.X(hstar) + 7, fr.Y(hstar) - 6, 'h*', th.green);
      }

      /* Marcador de h_0 (arrastável) */
      ctx.beginPath();
      ctx.arc(fr.X(h0), fr.Y(0), 6, 0, 2 * Math.PI);
      ctx.strokeStyle = th.fg;
      ctx.lineWidth = 2;
      ctx.stroke();
      P.label(ctx, fr.X(h0) + 9, fr.Y(0) - 8, 'h_0', th.fg);
    }

    function drawTraj() {
      const th = P.theme();
      const { ctx, w, h } = P.setup(cvTraj);
      P.clear(ctx, w, h);
      const hs = trajectory();
      const shown = hs.slice(0, cur + 1).map(clampPlot);
      const ymax = Math.max(1.4, Math.min(8, ...[1.4].concat(shown.map(Math.abs))));
      const fr = P.frame(ctx, w, h, 0, NMAX, -ymax, ymax);
      P.axes(ctx, fr, {
        xlabel: 't',
        xticks: [0, 10, 20, 30],
        yticks: [-Math.round(ymax * 10) / 10, 0, Math.round(ymax * 10) / 10],
      });
      P.line(ctx, fr, [0, NMAX], [0, 0], th.line, { width: 1, dash: [3, 3] });
      const hstar = fixedPoint();
      if (hstar > 0) {
        P.line(ctx, fr, [0, NMAX], [hstar, hstar], th.green, { width: 1, dash: [4, 4], alpha: 0.8 });
        P.line(ctx, fr, [0, NMAX], [-hstar, -hstar], th.green, { width: 1, dash: [4, 4], alpha: 0.8 });
      }
      clipPlot(ctx, fr);
      const ts = shown.map((_, i) => i);
      P.line(ctx, fr, ts, shown, th.orange, { width: 1.8 });
      P.scatter(ctx, fr, ts.map((t, i) => [t, shown[i]]), th.orange, { r: 2.4, alpha: 1 });
      ctx.restore();
    }

    function regime() {
      const a = Math.abs(thhh);
      if (mode === 'linear') {
        if (a < 1) return 'estado decai a zero, a memória evapora';
        if (a === 1) return 'estado persiste sem decair';
        return 'estado explode exponencialmente';
      }
      if (a <= 1) return 'estado decai a zero, a memória evapora';
      return 'estado satura no ponto fixo ±h*';
    }

    function updateReadout() {
      const hs = trajectory();
      const v = hs[cur];
      $('s2-readout').textContent =
        't = ' + cur + ' · h_t = ' + (Math.abs(v) > 1e4 ? v.toExponential(1) : v.toFixed(4)) +
        ' · ' + regime();
    }

    function redraw() {
      drawCobweb();
      drawTraj();
      updateReadout();
    }

    function stopRun() {
      running = false;
      btnRun.textContent = '▶ Rodar';
      DL.stopTicker(tick);
    }

    function tick(dt) {
      acc += dt * 4;
      while (acc >= 1 && cur < NMAX) { acc -= 1; cur++; }
      redraw();
      if (cur >= NMAX) stopRun();
    }

    function setH0(x) {
      h0 = Math.max(-DOM + 0.05, Math.min(DOM - 0.05, x));
      cur = 0;
      redraw();
    }

    cvCob.addEventListener('pointerdown', (e) => {
      if (!frCob) return;
      dragging = true;
      cvCob.setPointerCapture(e.pointerId);
      stopRun();
      const r = cvCob.getBoundingClientRect();
      setH0(frCob.invX(e.clientX - r.left));
    });
    cvCob.addEventListener('pointermove', (e) => {
      if (!dragging || !frCob) return;
      const r = cvCob.getBoundingClientRect();
      setH0(frCob.invX(e.clientX - r.left));
    });
    cvCob.addEventListener('pointerup', () => { dragging = false; });

    btnStep.addEventListener('click', () => {
      if (cur < NMAX) cur++;
      redraw();
    });
    btnRun.addEventListener('click', () => {
      if (running) { stopRun(); return; }
      if (cur >= NMAX) cur = 0;
      running = true;
      acc = 0;
      btnRun.textContent = '⏸ Pausar';
      DL.startTicker(tick);
    });
    btnReset.addEventListener('click', () => { stopRun(); cur = 0; redraw(); });

    slHh.addEventListener('input', () => {
      thhh = +slHh.value;
      $('s2-thhh-val').textContent = thhh.toFixed(2);
      cur = 0;
      redraw();
    });

    function setMode(m) {
      mode = m;
      tabTanh.classList.toggle('active', m === 'tanh');
      tabLin.classList.toggle('active', m === 'linear');
      cur = 0;
      stopRun();
      redraw();
    }
    tabTanh.addEventListener('click', () => setMode('tanh'));
    tabLin.addEventListener('click', () => setMode('linear'));

    redraw();
    P.onRedraw(redraw);
    P.observeResize(cvCob, redraw);
    P.observeResize(cvTraj, redraw);
  }

  DL.sections.push({ name: 's2-dinamica', init });
})();
