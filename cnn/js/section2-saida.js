/* Seção 2: padding P, stride S e dilation D, varredura do kernel posição a
   posição e a fórmula do tamanho da saída O = ⌊(N + 2P − D·(K−1) − 1)/S⌋ + 1. */
(function () {
  'use strict';
  window.DL = window.DL || {};
  DL.sections = DL.sections || [];

  function init() {
    const P = DL.plot;
    const G = DL.grid;
    const $ = (id) => document.getElementById(id);

    const cvIn = $('s2-input'), cvOut = $('s2-output');
    const sliders = { n: $('s2-n'), k: $('s2-k'), p: $('s2-p'), s: $('s2-s'), d: $('s2-d') };
    const btnStep = $('s2-step'), btnRun = $('s2-run'), btnReset = $('s2-reset');

    let n = +sliders.n.value, k = +sliders.k.value, p = +sliders.p.value;
    let s = +sliders.s.value, d = +sliders.d.value;
    let cur = 0;                  /* índice da janela atual na varredura */
    let running = false, acc = 0;

    function outSize() {
      return Math.floor((n + 2 * p - d * (k - 1) - 1) / s) + 1;
    }

    function drawInput() {
      const th = P.theme();
      const { ctx, w, h } = P.setup(cvIn);
      P.clear(ctx, w, h);
      const T = n + 2 * p;
      const L = G.layout(w, h, T, T, 14);
      const o = outSize();

      /* Células internas (entrada) preenchidas; anel de padding vazio */
      for (let i = 0; i < n; i++) {
        for (let j = 0; j < n; j++) {
          ctx.fillStyle = G.mix(th.card, th.cyan, 0.18);
          ctx.fillRect(L.x(j + p), L.y(i + p), L.cs, L.cs);
        }
      }

      /* Posições lidas pelo kernel na janela atual */
      if (o >= 1) {
        const r = Math.floor(cur / o), c = cur % o;
        ctx.fillStyle = th.orange;
        ctx.globalAlpha = 0.55;
        for (let a = 0; a < k; a++) {
          for (let b = 0; b < k; b++) {
            ctx.fillRect(L.x(c * s + b * d), L.y(r * s + a * d), L.cs, L.cs);
          }
        }
        ctx.globalAlpha = 1;
      }

      G.lines(ctx, L, th.line);

      /* Borda tracejada separando padding da entrada */
      if (p > 0) {
        ctx.setLineDash([4, 3]);
        G.outline(ctx, L, p, p, n, n, th.comment, 1.5);
        ctx.setLineDash([]);
      }

      /* Contorno da extensão efetiva do kernel (com dilation) */
      if (o >= 1) {
        const r = Math.floor(cur / o), c = cur % o;
        const ext = d * (k - 1) + 1;
        G.outline(ctx, L, r * s, c * s, ext, ext, th.orange, 2.5);
      }
      P.label(ctx, L.x(0), L.y(0) - 5, 'entrada ' + n + '×' + n + (p > 0 ? ' + padding' : ''), th.comment);
    }

    function drawOutput() {
      const th = P.theme();
      const { ctx, w, h } = P.setup(cvOut);
      P.clear(ctx, w, h);
      const o = outSize();
      if (o < 1) {
        ctx.fillStyle = th.pink;
        ctx.font = '13px sans-serif';
        ctx.textAlign = 'center';
        ctx.fillText('kernel não cabe na entrada: O ≤ 0', w / 2, h / 2);
        return;
      }
      const L = G.layout(w, h, o, o, 14);
      for (let i = 0; i < o; i++) {
        for (let j = 0; j < o; j++) {
          const idx = i * o + j;
          if (idx < cur) {
            ctx.fillStyle = th.green;
            ctx.globalAlpha = 0.4;
          } else if (idx === cur) {
            ctx.fillStyle = th.orange;
            ctx.globalAlpha = 0.7;
          } else {
            ctx.fillStyle = th.card;
            ctx.globalAlpha = 1;
          }
          ctx.fillRect(L.x(j), L.y(i), L.cs, L.cs);
          ctx.globalAlpha = 1;
        }
      }
      G.lines(ctx, L, th.line);
      P.label(ctx, L.x(0), L.y(0) - 5, 'saída ' + o + '×' + o, th.comment);
    }

    function updateReadout() {
      const o = outSize();
      const f = 'O = ⌊(' + n + ' + ' + (2 * p) + ' − ' + d + '·(' + k + '−1) − 1)/' + s + '⌋ + 1 = ' + o;
      if (o < 1) {
        $('s2-readout').textContent = f + ' · kernel não cabe';
      } else {
        $('s2-readout').textContent = f + ' · janela ' + (cur + 1) + '/' + (o * o);
      }
    }

    function redraw() {
      drawInput();
      drawOutput();
      updateReadout();
    }

    function stopRun() {
      running = false;
      btnRun.textContent = '▶ Animar';
      DL.stopTicker(tick);
    }

    function tick(dt) {
      const o = outSize();
      if (o < 1) { stopRun(); redraw(); return; }
      acc += dt * 7;
      while (acc >= 1) {
        acc -= 1;
        if (cur >= o * o - 1) { stopRun(); break; }
        cur++;
      }
      redraw();
    }

    btnStep.addEventListener('click', () => {
      const o = outSize();
      if (o < 1) return;
      cur = (cur + 1) % (o * o);
      redraw();
    });
    btnRun.addEventListener('click', () => {
      if (running) { stopRun(); redraw(); return; }
      const o = outSize();
      if (o < 1) return;
      if (cur >= o * o - 1) cur = 0;
      running = true;
      acc = 0;
      btnRun.textContent = '⏸ Pausar';
      DL.startTicker(tick);
    });
    btnReset.addEventListener('click', () => {
      stopRun();
      cur = 0;
      redraw();
    });

    function onSlider() {
      n = +sliders.n.value; k = +sliders.k.value; p = +sliders.p.value;
      s = +sliders.s.value; d = +sliders.d.value;
      $('s2-n-val').textContent = n;
      $('s2-k-val').textContent = k;
      $('s2-p-val').textContent = p;
      $('s2-s-val').textContent = s;
      $('s2-d-val').textContent = d;
      stopRun();
      cur = 0;
      redraw();
    }
    for (const key of Object.keys(sliders)) sliders[key].addEventListener('input', onSlider);

    redraw();
    P.onRedraw(redraw);
    P.observeResize(cvIn, redraw);
    P.observeResize(cvOut, redraw);
  }

  DL.sections.push({ name: 's2-saida', init });
})();
