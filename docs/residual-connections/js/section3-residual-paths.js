/* Seção 3: os 2^n caminhos da entrada à saída numa rede com n blocos
   residuais. ⏭ percorre os caminhos; o histograma mostra o ensemble
   implícito de profundidades (coeficiente binomial). */
(function () {
  'use strict';
  window.DL = window.DL || {};
  DL.sections = DL.sections || [];

  function init() {
    const P = DL.plot;
    const $ = (id) => document.getElementById(id);

    const cvGraph = $('s3-graph'), cvHist = $('s3-hist');
    const slN = $('s3-n');
    const btnStep = $('s3-step'), btnRun = $('s3-run'), btnReset = $('s3-reset');

    let n = +slN.value;
    let path = 0;
    let running = false, accT = 0;

    const popcount = (p) => {
      let c = 0;
      while (p) { c += p & 1; p >>= 1; }
      return c;
    };

    function binom(m, k) {
      let r = 1;
      for (let i = 0; i < k; i++) r = r * (m - i) / (i + 1);
      return Math.round(r);
    }

    function drawGraph() {
      const th = P.theme();
      const { ctx, w, h } = P.setup(cvGraph);
      P.clear(ctx, w, h);
      const mid = h * 0.64, top = h * 0.16;
      const left = 40, right = w - 44;
      const span = (right - left) / n;
      const jx = (i) => left + i * span;
      const r = 10;
      const boxW = Math.min(46, span * 0.36), boxH = Math.min(34, h * 0.22);

      function segment(x1, x2, color, width) {
        ctx.strokeStyle = color;
        ctx.lineWidth = width;
        ctx.beginPath();
        ctx.moveTo(x1, mid);
        ctx.lineTo(x2, mid);
        ctx.stroke();
      }
      function arc(i, color, width) {
        ctx.strokeStyle = color;
        ctx.lineWidth = width;
        ctx.beginPath();
        ctx.moveTo(jx(i - 1), mid - (i === 1 ? 6 : r));
        ctx.quadraticCurveTo((jx(i - 1) + jx(i)) / 2, top, jx(i), mid - r);
        ctx.stroke();
      }

      /* Arestas de base (apagadas) e destaque do caminho atual. */
      for (let i = 1; i <= n; i++) {
        segment(jx(i - 1) + (i === 1 ? 8 : r), jx(i) - r, th.line, 1.5);
        arc(i, th.line, 1.5);
      }
      for (let i = 1; i <= n; i++) {
        const applied = (path >> (i - 1)) & 1;
        if (applied) segment(jx(i - 1) + (i === 1 ? 8 : r), jx(i) - r, th.orange, 3);
        else arc(i, th.orange, 3);
      }
      /* Seta de saída. */
      segment(jx(n) + r, right + 24, th.orange, 3);

      /* Caixas dos blocos f_i por cima das arestas. */
      for (let i = 1; i <= n; i++) {
        const applied = (path >> (i - 1)) & 1;
        const cx = (jx(i - 1) + jx(i)) / 2;
        ctx.fillStyle = th.card;
        ctx.strokeStyle = applied ? th.orange : th.comment;
        ctx.lineWidth = applied ? 2.5 : 1.5;
        ctx.beginPath();
        ctx.rect(cx - boxW / 2, mid - boxH / 2, boxW, boxH);
        ctx.fill();
        ctx.stroke();
        P.mathText(ctx, 'f_' + i, cx, mid + 4, applied ? th.orange : th.fg, 'center', 13);
      }
      /* Junções de soma. */
      for (let i = 1; i <= n; i++) {
        ctx.fillStyle = th.card;
        ctx.strokeStyle = th.fg;
        ctx.lineWidth = 1.5;
        ctx.beginPath();
        ctx.arc(jx(i), mid, r, 0, 2 * Math.PI);
        ctx.fill();
        ctx.stroke();
        ctx.fillStyle = th.fg;
        ctx.font = '13px sans-serif';
        ctx.textAlign = 'center';
        ctx.fillText('+', jx(i), mid + 4.5);
      }
      P.mathText(ctx, 'x', jx(0) - 4, mid + 4, th.fg, 'right', 14);
      P.mathText(ctx, 'y', right + 30, mid + 4, th.fg, 'left', 14);
    }

    function drawHist() {
      const th = P.theme();
      const { ctx, w, h } = P.setup(cvHist);
      P.clear(ctx, w, h);
      const counts = [];
      let cmax = 0;
      for (let k = 0; k <= n; k++) {
        counts.push(binom(n, k));
        cmax = Math.max(cmax, counts[k]);
      }
      const fr = P.frame(ctx, w, h, -0.6, n + 0.6, 0, cmax * 1.25);
      const ticks = [];
      for (let k = 0; k <= n; k++) ticks.push(k);
      P.axes(ctx, fr, { xlabel: 'profundidade k', ylabel: 'nº de caminhos', xticks: ticks, yticks: [0, cmax] });
      const cur = popcount(path);
      for (let k = 0; k <= n; k++) {
        const x0 = fr.X(k - 0.34), x1 = fr.X(k + 0.34);
        const y0 = fr.Y(0), y1 = fr.Y(counts[k]);
        ctx.fillStyle = k === cur ? th.orange : th.purple;
        ctx.globalAlpha = k === cur ? 0.95 : 0.45;
        ctx.fillRect(x0, y1, x1 - x0, y0 - y1);
        ctx.globalAlpha = 1;
        P.label(ctx, fr.X(k), y1 - 5, String(counts[k]), th.comment, 'center');
      }
    }

    function termString() {
      let t = 'x';
      for (let i = 1; i <= n; i++) {
        if ((path >> (i - 1)) & 1) t = 'f' + i + '[' + t + ']';
      }
      return t;
    }

    function updateReadout() {
      const total = 1 << n;
      const k = popcount(path);
      $('s3-readout').textContent =
        'caminho ' + (path + 1) + ' de ' + total +
        ' · profundidade ' + k +
        ' · termo: ' + termString() +
        ' · 2^(n−1) = ' + (1 << (n - 1)) + ' caminhos passam por f1';
    }

    function redraw() {
      drawGraph();
      drawHist();
      updateReadout();
    }

    function nextPath() {
      path = (path + 1) % (1 << n);
      redraw();
    }

    function stopRun() {
      running = false;
      btnRun.textContent = '▶ Percorrer';
      DL.stopTicker(tick);
    }

    function tick(dt) {
      accT += dt;
      if (accT >= 0.8) {
        accT -= 0.8;
        nextPath();
      }
    }

    btnStep.addEventListener('click', () => { stopRun(); nextPath(); });
    btnRun.addEventListener('click', () => {
      if (running) { stopRun(); return; }
      running = true;
      accT = 0;
      btnRun.textContent = '⏸ Pausar';
      DL.startTicker(tick);
    });
    btnReset.addEventListener('click', () => { stopRun(); path = 0; redraw(); });
    slN.addEventListener('input', () => {
      n = +slN.value;
      $('s3-n-val').textContent = n;
      path = 0;
      redraw();
    });

    redraw();
    P.onRedraw(redraw);
    P.observeResize(cvGraph, redraw);
    P.observeResize(cvHist, redraw);
  }

  DL.sections.push({ name: 's3-residual-paths', init });
})();
