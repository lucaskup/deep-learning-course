/* Seção 3: o papel da escala 1/√d. Para componentes i.i.d. N(0,1),
   E[q·k] = 0 e var[q·k] = d; dividir por √d devolve variância 1 e
   evita que a softmax sature em quase one-hot. */
(function () {
  'use strict';
  window.DL = window.DL || {};
  DL.sections = DL.sections || [];

  const DS = [2, 4, 8, 16, 32, 64, 128, 256, 512];
  const NPAIRS = 1500;     /* pares (q, k) amostrados por configuração */
  const M = 10;            /* scores que entram na softmax */
  const NBINS = 80;

  function init() {
    const P = DL.plot, U = DL.utils;
    const $ = (id) => document.getElementById(id);

    const cvHist = $('s3-hist'), cvSoft = $('s3-softmax');
    const slD = $('s3-d');
    const fmt = (v, d) => v.toFixed(d == null ? 2 : d);

    let seed = 3;
    let d = DS[+slD.value];
    let dots = [];     /* q·k sem escala */
    let z = [];        /* M valores base N(0,1): scores escalados */
    let empVar = 0;

    function softmax(s) {
      const mx = Math.max(...s);
      const e = s.map((v) => Math.exp(v - mx));
      const t = e.reduce((a, b) => a + b, 0);
      return e.map((v) => v / t);
    }

    function recompute() {
      const rng = U.mulberry32(seed);
      z = [];
      for (let i = 0; i < M; i++) z.push(U.randn(rng));
      dots = new Float64Array(NPAIRS);
      let s2 = 0;
      for (let n = 0; n < NPAIRS; n++) {
        let s = 0;
        for (let i = 0; i < d; i++) s += U.randn(rng) * U.randn(rng);
        dots[n] = s;
        s2 += s * s;
      }
      empVar = s2 / NPAIRS;
    }

    /* Janela do histograma: ±3.4·√d (par), para que d = 2 não vire três barras. */
    function range() { return Math.max(6, 2 * Math.ceil(1.7 * Math.sqrt(d))); }

    function bins(values, scale) {
      const RANGE = range();
      const b = new Float64Array(NBINS);
      const bw = (2 * RANGE) / NBINS;
      for (const v of values) {
        const k = Math.floor((v * scale + RANGE) / bw);
        if (k >= 0 && k < NBINS) b[k]++;
      }
      const mx = Math.max(1, ...b);
      for (let k = 0; k < NBINS; k++) b[k] /= mx;
      return b;
    }

    /* Bloco de legenda com fundo semitransparente, para que o texto continue
       legível quando barras ou curvas passam por baixo. items: [[texto, cor]]. */
    function legendBox(ctx, th, x, y, items, align) {
      ctx.font = '11px sans-serif';
      let wmax = 0;
      for (const it of items) wmax = Math.max(wmax, ctx.measureText(it[0].replace(/[_{}]/g, '')).width);
      const bx = align === 'right' ? x - wmax - 6 : x - 6;
      ctx.save();
      ctx.globalAlpha = 0.85;
      ctx.fillStyle = th.card;
      ctx.fillRect(bx, y, wmax + 12, 14 * items.length + 6);
      ctx.restore();
      items.forEach((it, k) => P.label(ctx, x, y + 14 * (k + 1), it[0], it[1], align || 'left'));
    }

    function drawHist() {
      const th = P.theme();
      const { ctx, w, h } = P.setup(cvHist);
      P.clear(ctx, w, h);
      const RANGE = range();
      const fr = P.frame(ctx, w, h, -RANGE, RANGE, 0, 1.12);
      P.axes(ctx, fr, { xlabel: 'q·k', xticks: [-RANGE, -RANGE / 2, 0, RANGE / 2, RANGE] });

      const bw = (2 * RANGE) / NBINS;
      const drawBins = (b, color, alpha) => {
        ctx.fillStyle = color;
        ctx.globalAlpha = alpha;
        for (let k = 0; k < NBINS; k++) {
          if (b[k] <= 0) continue;
          const x0 = fr.X(-RANGE + k * bw), x1 = fr.X(-RANGE + (k + 1) * bw);
          ctx.fillRect(x0, fr.Y(b[k]), x1 - x0, fr.Y(0) - fr.Y(b[k]));
        }
        ctx.globalAlpha = 1;
      };
      drawBins(bins(dots, 1), th.cyan, 0.55);
      drawBins(bins(dots, 1 / Math.sqrt(d)), th.orange, 0.55);

      /* desvio padrão teórico ±√d dos scores sem escala */
      const sd = Math.sqrt(d);
      ctx.strokeStyle = th.cyan;
      ctx.setLineDash([4, 4]);
      ctx.lineWidth = 1.2;
      for (const s of [-sd, sd]) {
        ctx.beginPath();
        ctx.moveTo(fr.X(s), fr.Y(0));
        ctx.lineTo(fr.X(s), fr.Y(1.05));
        ctx.stroke();
      }
      ctx.setLineDash([]);
      const sdText = '±√d = ±' + sd.toFixed(1);
      const legend = [['sem escala: q·k, var = d', th.cyan], ['com escala: q·k/√d, var = 1', th.orange]];
      /* Em canvas estreito o rótulo da linha tracejada cairia sobre a legenda:
         vira uma terceira linha dela. */
      ctx.font = '11px sans-serif';
      const legendRight = fr.X(-RANGE) + 10 + ctx.measureText(legend[1][0]).width;
      if (fr.X(sd) + 6 > legendRight + 12) {
        P.mathText(ctx, sdText, fr.X(sd) + 6, fr.Y(1.0), th.cyan, 'left', 11);
      } else {
        legend.push([sdText + ' (tracejado)', th.cyan]);
      }
      legendBox(ctx, th, fr.X(-RANGE) + 10, fr.Y(1.05) - 4, legend);
    }

    function drawSoftmax() {
      const th = P.theme();
      const { ctx, w, h } = P.setup(cvSoft);
      P.clear(ctx, w, h);
      const sd = Math.sqrt(d);
      const aRaw = softmax(z.map((v) => v * sd));    /* scores ~ N(0, d) */
      const aScaled = softmax(z);                    /* scores ~ N(0, 1) */

      const fr = P.frame(ctx, w, h, 0, M, 0, 1.12, { l: 34, r: 8, t: 8, b: 26 });
      P.axes(ctx, fr, { ylabel: 'a_i', yticks: [0, 0.5, 1] });
      const cw = fr.iw / M;
      aRaw.forEach((v, i) => {
        const x = fr.X(i) + cw * 0.16;
        ctx.fillStyle = th.cyan;
        ctx.fillRect(x, fr.Y(v), cw * 0.3, fr.Y(0) - fr.Y(v));
        ctx.fillStyle = th.orange;
        ctx.fillRect(x + cw * 0.36, fr.Y(aScaled[i]), cw * 0.3, fr.Y(0) - fr.Y(aScaled[i]));
        ctx.fillStyle = th.comment;
        ctx.font = '10px sans-serif';
        ctx.textAlign = 'center';
        ctx.fillText(String(i + 1), fr.X(i) + cw / 2, fr.Y(0) + 13);
      });
      legendBox(ctx, th, fr.X(0) + 6, fr.Y(1.05) - 10, [['sem escala', th.cyan]]);
      legendBox(ctx, th, fr.X(0) + 86, fr.Y(1.05) - 10, [['com escala', th.orange]]);
    }

    function updateReadout() {
      const sd = Math.sqrt(d);
      const mxRaw = Math.max(...softmax(z.map((v) => v * sd)));
      const mxScaled = Math.max(...softmax(z));
      $('s3-readout').textContent =
        'var(q·k) ≈ ' + fmt(empVar, 1) + ' (teoria: d = ' + d + ')' +
        ' · max aᵢ sem escala = ' + fmt(mxRaw) +
        ' · com escala = ' + fmt(mxScaled);
    }

    function redraw() {
      drawHist();
      drawSoftmax();
      updateReadout();
    }

    slD.addEventListener('input', () => {
      d = DS[+slD.value];
      $('s3-d-val').textContent = d;
      recompute();
      redraw();
    });
    $('s3-rand').addEventListener('click', () => {
      seed = (seed * 7919 + 11) % 100000;
      recompute();
      redraw();
    });

    recompute();
    redraw();
    P.onRedraw(redraw);
    P.observeResize(cvHist, redraw);
    P.observeResize(cvSoft, redraw);
  }

  DL.sections.push({ name: 's3-scaling', init });
})();
