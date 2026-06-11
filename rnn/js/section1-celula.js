/* Seção 1: célula recorrente desenrolada no tempo.
   h_t = tanh(x_t·θ_ih + h_{t−1}·θ_hh), ŷ_t = h_t·θ_ho (b_h = b_o = 0).
   O aluno edita a sequência x arrastando barras e avança o tempo passo a passo. */
(function () {
  'use strict';
  window.DL = window.DL || {};
  DL.sections = DL.sections || [];

  const T = 5, XMAX = 2;

  function init() {
    const P = DL.plot;
    const $ = (id) => document.getElementById(id);

    const cvUnroll = $('s1-unroll'), cvState = $('s1-state');
    const slIh = $('s1-thih'), slHh = $('s1-thhh'), slHo = $('s1-thho');
    const btnStep = $('s1-step'), btnRun = $('s1-run');
    const btnReset = $('s1-reset'), btnRand = $('s1-rand');

    let thih = +slIh.value, thhh = +slHh.value, thho = +slHo.value;
    /* Exemplo da aula nos dois primeiros passos: x_1 = 1, x_2 = 2. */
    let xs = [1, 2, -1, 0.5, -1.5];
    let hs = [], ys = [];
    let cur = 0;               /* quantos passos já foram computados (0..T) */
    let seed = 7;
    let running = false, acc = 0;
    let layout = null, dragIdx = -1;

    function forward() {
      hs = [0]; ys = [0];
      for (let t = 1; t <= T; t++) {
        const h = Math.tanh(xs[t - 1] * thih + hs[t - 1] * thhh);
        hs.push(h);
        ys.push(h * thho);
      }
    }

    function arrow(ctx, x0, y0, x1, y1, color, width) {
      ctx.strokeStyle = color;
      ctx.fillStyle = color;
      ctx.lineWidth = width || 1.4;
      ctx.beginPath();
      ctx.moveTo(x0, y0);
      ctx.lineTo(x1, y1);
      ctx.stroke();
      const a = Math.atan2(y1 - y0, x1 - x0), s = 5;
      ctx.beginPath();
      ctx.moveTo(x1, y1);
      ctx.lineTo(x1 - s * Math.cos(a - 0.45), y1 - s * Math.sin(a - 0.45));
      ctx.lineTo(x1 - s * Math.cos(a + 0.45), y1 - s * Math.sin(a + 0.45));
      ctx.closePath();
      ctx.fill();
    }

    function roundRect(ctx, x, y, w, h, r) {
      ctx.beginPath();
      ctx.moveTo(x + r, y);
      ctx.arcTo(x + w, y, x + w, y + h, r);
      ctx.arcTo(x + w, y + h, x, y + h, r);
      ctx.arcTo(x, y + h, x, y, r);
      ctx.arcTo(x, y, x + w, y, r);
      ctx.closePath();
    }

    function drawUnroll() {
      const th = P.theme();
      const { ctx, w, h } = P.setup(cvUnroll);
      P.clear(ctx, w, h);

      const slotW = (w - 16) / T;
      const yOut = 26;                 /* centro dos círculos de saída */
      const cellTop = 58, cellH = 46;
      const barZero = h - 52;          /* linha do zero das barras de entrada */
      const barScale = 34 / XMAX;      /* px por unidade de x */
      const cellW = Math.min(slotW - 26, 86);
      layout = { slotW, barZero, barScale };

      /* Linha do zero das entradas */
      ctx.strokeStyle = th.line;
      ctx.lineWidth = 1;
      ctx.beginPath();
      ctx.moveTo(8, barZero);
      ctx.lineTo(w - 8, barZero);
      ctx.stroke();

      for (let t = 1; t <= T; t++) {
        const cx = 8 + slotW * (t - 0.5);
        const active = t <= cur;
        ctx.globalAlpha = active ? 1 : 0.35;

        /* Barra de entrada x_t (arrastável) */
        const bh = xs[t - 1] * barScale;
        ctx.globalAlpha = 1;
        ctx.fillStyle = th.cyan;
        ctx.globalAlpha = active ? 0.85 : 0.45;
        ctx.fillRect(cx - 9, Math.min(barZero, barZero - bh), 18, Math.abs(bh));
        ctx.globalAlpha = 1;
        P.mathText(ctx, 'x_' + t + '=' + xs[t - 1].toFixed(1), cx, h - 6, th.cyan, 'center', 10);

        ctx.globalAlpha = active ? 1 : 0.35;

        /* Seta da entrada para a célula */
        arrow(ctx, cx, barZero - Math.max(0, bh) - 2, cx, cellTop + cellH + 5, th.cyan, 1.3);

        /* Célula tanh */
        const cl = cx - cellW / 2;
        roundRect(ctx, cl, cellTop, cellW, cellH, 7);
        ctx.fillStyle = th.card;
        ctx.fill();
        ctx.strokeStyle = t === cur ? th.fg : th.green;
        ctx.lineWidth = t === cur ? 2.2 : 1.4;
        roundRect(ctx, cl, cellTop, cellW, cellH, 7);
        ctx.stroke();
        ctx.font = '11px sans-serif';
        ctx.fillStyle = th.green;
        ctx.textAlign = 'center';
        ctx.fillText('tanh', cx, cellTop + 17);
        if (active) {
          P.mathText(ctx, 'h_' + t + '=' + hs[t].toFixed(2), cx, cellTop + 35, th.orange, 'center', 11);
        }

        /* Seta da célula para a saída e círculo ŷ_t */
        arrow(ctx, cx, cellTop - 2, cx, yOut + 14, th.pink, 1.3);
        ctx.beginPath();
        ctx.arc(cx, yOut, 13, 0, 2 * Math.PI);
        ctx.fillStyle = th.card;
        ctx.fill();
        ctx.strokeStyle = th.pink;
        ctx.lineWidth = 1.4;
        ctx.stroke();
        if (active) {
          ctx.font = '10px sans-serif';
          ctx.fillStyle = th.pink;
          ctx.fillText(ys[t].toFixed(2), cx, yOut + 3.5);
        }
        P.mathText(ctx, 'ŷ_' + t, cx + 22, yOut + 4, th.pink, 'left', 10);

        /* Seta recorrente h_t entre células */
        if (t < T) {
          const nx = 8 + slotW * (t + 0.5) - cellW / 2;
          ctx.globalAlpha = t + 1 <= cur ? 1 : 0.35;
          arrow(ctx, cl + cellW + 2, cellTop + cellH / 2, nx - 3, cellTop + cellH / 2, th.orange, 1.6);
          if (t <= cur) {
            P.mathText(ctx, hs[t].toFixed(2), (cl + cellW + nx) / 2, cellTop + cellH / 2 - 6, th.orange, 'center', 10);
          }
        }
        ctx.globalAlpha = 1;
      }

      /* h_0 = 0 entrando na primeira célula */
      const c1 = 8 + slotW * 0.5 - cellW / 2;
      ctx.globalAlpha = 0.8;
      arrow(ctx, c1 - 24, cellTop + cellH / 2, c1 - 3, cellTop + cellH / 2, th.orange, 1.4);
      P.mathText(ctx, 'h_0=0', c1 - 14, cellTop + cellH / 2 - 7, th.orange, 'center', 10);
      ctx.globalAlpha = 1;
    }

    function drawState() {
      const th = P.theme();
      const { ctx, w, h } = P.setup(cvState);
      P.clear(ctx, w, h);
      const fr = P.frame(ctx, w, h, 0, T, -2.2, 2.2);
      P.axes(ctx, fr, { xlabel: 't', xticks: [0, 1, 2, 3, 4, 5], yticks: [-2, -1, 0, 1, 2] });
      P.line(ctx, fr, [0, T], [0, 0], th.line, { width: 1, dash: [3, 3] });
      if (cur > 0) {
        const ts = [], hv = [], yv = [];
        for (let t = 0; t <= cur; t++) { ts.push(t); hv.push(hs[t]); yv.push(ys[t]); }
        P.line(ctx, fr, ts, hv, th.orange, { width: 1.8 });
        P.scatter(ctx, fr, ts.map((t, i) => [t, hv[i]]), th.orange, { r: 2.6, alpha: 1 });
        P.line(ctx, fr, ts.slice(1), yv.slice(1), th.pink, { width: 1.8, dash: [5, 3] });
        P.scatter(ctx, fr, ts.slice(1).map((t, i) => [t, yv[i + 1]]), th.pink, { r: 2.6, alpha: 1 });
      }
      P.label(ctx, fr.X(0) + 8, fr.Y(2.2) + 12, 'h_t', th.orange);
      P.label(ctx, fr.X(0) + 8, fr.Y(2.2) + 26, 'ŷ_t', th.pink);
    }

    function updateReadout() {
      if (cur === 0) {
        $('s1-readout').textContent = 't = 0 · h_0 = 0 (estado inicial)';
        return;
      }
      const t = cur;
      $('s1-readout').textContent =
        't = ' + t + '/' + T +
        ' · h_' + t + ' = tanh(' + xs[t - 1].toFixed(1) + '·' + thih.toFixed(2) +
        ' + ' + hs[t - 1].toFixed(3) + '·' + thhh.toFixed(2) + ') = ' + hs[t].toFixed(3) +
        ' · ŷ_' + t + ' = ' + ys[t].toFixed(3);
    }

    function redraw() {
      forward();
      drawUnroll();
      drawState();
      updateReadout();
    }

    function stopRun() {
      running = false;
      btnRun.textContent = '▶ Rodar';
      DL.stopTicker(tick);
    }

    function tick(dt) {
      acc += dt * 1.6;
      while (acc >= 1 && cur < T) { acc -= 1; cur++; }
      redraw();
      if (cur >= T) stopRun();
    }

    /* Arrastar barras de entrada */
    function barIndexAt(px, py) {
      if (!layout) return -1;
      if (py < layout.barZero - XMAX * layout.barScale - 8) return -1;
      const i = Math.floor((px - 8) / layout.slotW);
      return i >= 0 && i < T ? i : -1;
    }
    cvUnroll.addEventListener('pointerdown', (e) => {
      const r = cvUnroll.getBoundingClientRect();
      const i = barIndexAt(e.clientX - r.left, e.clientY - r.top);
      if (i < 0) return;
      dragIdx = i;
      cvUnroll.setPointerCapture(e.pointerId);
      stopRun();
      const v = (layout.barZero - (e.clientY - r.top)) / layout.barScale;
      xs[i] = Math.max(-XMAX, Math.min(XMAX, Math.round(v * 10) / 10));
      redraw();
    });
    cvUnroll.addEventListener('pointermove', (e) => {
      if (dragIdx < 0 || !layout) return;
      const r = cvUnroll.getBoundingClientRect();
      const v = (layout.barZero - (e.clientY - r.top)) / layout.barScale;
      xs[dragIdx] = Math.max(-XMAX, Math.min(XMAX, Math.round(v * 10) / 10));
      redraw();
    });
    cvUnroll.addEventListener('pointerup', () => { dragIdx = -1; });

    btnStep.addEventListener('click', () => {
      if (cur < T) cur++;
      redraw();
    });
    btnRun.addEventListener('click', () => {
      if (running) { stopRun(); return; }
      if (cur >= T) cur = 0;
      running = true;
      acc = 0;
      btnRun.textContent = '⏸ Pausar';
      DL.startTicker(tick);
    });
    btnReset.addEventListener('click', () => { stopRun(); cur = 0; redraw(); });
    btnRand.addEventListener('click', () => {
      stopRun();
      seed = (seed * 1103515245 + 12345) >>> 0;
      const rng = DL.utils.mulberry32(seed);
      xs = xs.map(() => Math.round((4 * rng() - 2) * 10) / 10);
      cur = 0;
      redraw();
    });

    function bindSlider(sl, valId, fmt, set) {
      sl.addEventListener('input', () => {
        set(+sl.value);
        $(valId).textContent = fmt(+sl.value);
        redraw();
      });
    }
    bindSlider(slIh, 's1-thih-val', (v) => v.toFixed(2), (v) => { thih = v; });
    bindSlider(slHh, 's1-thhh-val', (v) => v.toFixed(2), (v) => { thhh = v; });
    bindSlider(slHo, 's1-thho-val', (v) => v.toFixed(2), (v) => { thho = v; });

    cur = 2;   /* abre já com os dois passos do exemplo da aula computados */
    redraw();
    P.onRedraw(redraw);
    P.observeResize(cvUnroll, redraw);
    P.observeResize(cvState, redraw);
  }

  DL.sections.push({ name: 's1-celula', init });
})();
