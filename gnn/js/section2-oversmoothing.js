/* Seção 2: oversmoothing com a propagação real da GCN.
   Duas comunidades com atributos 2D; aplicamos k vezes a matriz
   normalizada simétrica com auto-laços e pintamos cada nó pelo
   embedding atual. O gráfico ao lado mostra a distância média
   entre pares de embeddings em função da profundidade. */
(function () {
  'use strict';
  window.DL = window.DL || {};
  DL.sections = DL.sections || [];

  const KMAX = 10;

  /* Duas comunidades de 8 nós; posições em [0,10] x [0,6]. */
  const POS = [
    [1.2, 4.6], [2.6, 5.2], [3.4, 4.0], [2.0, 3.2], [0.8, 2.6],
    [2.2, 1.6], [3.6, 2.4], [1.2, 1.0],
    [6.6, 4.8], [8.0, 5.4], [9.2, 4.4], [7.4, 3.6], [8.8, 3.0],
    [6.8, 2.2], [8.2, 1.4], [9.4, 2.0],
  ];
  const EDGES = [
    /* comunidade A (densa) */
    [0, 1], [0, 3], [0, 4], [1, 2], [1, 3], [2, 3], [2, 6], [3, 5],
    [4, 5], [4, 7], [5, 6], [5, 7], [3, 6],
    /* comunidade B (densa) */
    [8, 9], [8, 11], [9, 10], [9, 11], [10, 12], [11, 12], [11, 13],
    [12, 15], [13, 14], [14, 15], [12, 14], [8, 13],
    /* pontes */
    [2, 8], [6, 13],
  ];
  const N = POS.length;
  const ADJ = Array.from({ length: N }, () => []);
  for (const [a, b] of EDGES) { ADJ[a].push(b); ADJ[b].push(a); }

  const state = { k: 0, seed: 7, layers: null, dists: null };

  /* Pré-computa H_0 ... H_KMAX e a distância média por profundidade. */
  function recompute() {
    const rng = DL.utils.mulberry32(state.seed);
    let H = [];
    for (let i = 0; i < N; i++) {
      const comm = i < 8 ? [1, 0] : [0, 1];
      H.push([
        Math.abs(comm[0] + 0.35 * DL.utils.randn(rng)),
        Math.abs(comm[1] + 0.35 * DL.utils.randn(rng)),
      ]);
    }
    const deg = ADJ.map((nb) => nb.length + 1); /* auto-laço incluso */
    const layers = [H];
    for (let k = 1; k <= KMAX; k++) {
      const prev = layers[k - 1];
      const next = [];
      for (let i = 0; i < N; i++) {
        const acc = [prev[i][0] / deg[i], prev[i][1] / deg[i]];
        for (const j of ADJ[i]) {
          const w = 1 / Math.sqrt(deg[i] * deg[j]);
          acc[0] += w * prev[j][0];
          acc[1] += w * prev[j][1];
        }
        next.push(acc);
      }
      layers.push(next);
    }
    const dists = layers.map((L) => {
      let s = 0, c = 0;
      for (let i = 0; i < N; i++) {
        for (let j = i + 1; j < N; j++) {
          s += Math.hypot(L[i][0] - L[j][0], L[i][1] - L[j][1]);
          c++;
        }
      }
      return s / c;
    });
    state.layers = layers;
    state.dists = dists;
  }

  /* Cor do nó: interpolação ciano-laranja pela fração da 2a coordenada. */
  function nodeColor(t, f) {
    const a = DL.plot.rgb(t.cyan), b = DL.plot.rgb(t.orange);
    const s = f[1] / (f[0] + f[1] + 1e-9);
    const mix = a.map((v, i) => Math.round(v + s * (b[i] - v)));
    return 'rgb(' + mix.join(',') + ')';
  }

  function init() {
    const cvG = document.getElementById('s2-graph');
    const cvD = document.getElementById('s2-dist');
    const slider = document.getElementById('s2-k');
    const kVal = document.getElementById('s2-k-val');
    const readout = document.getElementById('s2-readout');

    function drawGraph() {
      const { ctx, w, h } = DL.plot.setup(cvG);
      const t = DL.plot.theme();
      DL.plot.clear(ctx, w, h);
      const fr = DL.plot.frame(ctx, w, h, 0, 10.2, 0.2, 6.2,
        { l: 8, r: 8, t: 8, b: 8 });
      const H = state.layers[state.k];

      ctx.strokeStyle = t.line;
      ctx.lineWidth = 1.2;
      for (const [a, b] of EDGES) {
        ctx.beginPath();
        ctx.moveTo(fr.X(POS[a][0]), fr.Y(POS[a][1]));
        ctx.lineTo(fr.X(POS[b][0]), fr.Y(POS[b][1]));
        ctx.stroke();
      }
      for (let i = 0; i < N; i++) {
        ctx.fillStyle = nodeColor(t, H[i]);
        ctx.beginPath();
        ctx.arc(fr.X(POS[i][0]), fr.Y(POS[i][1]), 8, 0, 2 * Math.PI);
        ctx.fill();
      }
    }

    function drawDist() {
      const { ctx, w, h } = DL.plot.setup(cvD);
      const t = DL.plot.theme();
      DL.plot.clear(ctx, w, h);
      const ymax = state.dists[0] * 1.15;
      const fr = DL.plot.frame(ctx, w, h, 0, KMAX, 0, ymax);
      DL.plot.axes(ctx, fr, {
        xlabel: 'camadas k', ylabel: 'dist. média',
        xticks: [0, 2, 4, 6, 8, 10],
        yticks: [0, Math.round(ymax * 50) / 100, Math.round(ymax * 100) / 100],
      });
      const xs = state.dists.map((_, i) => i);
      DL.plot.line(ctx, fr, xs, state.dists, t.purple, { width: 2 });
      /* Marcador da profundidade atual. */
      ctx.fillStyle = t.pink;
      ctx.beginPath();
      ctx.arc(fr.X(state.k), fr.Y(state.dists[state.k]), 5, 0, 2 * Math.PI);
      ctx.fill();
    }

    function redraw() {
      drawGraph();
      drawDist();
      const pct = 100 * state.dists[state.k] / state.dists[0];
      readout.textContent = 'distância média entre embeddings: ' +
        state.dists[state.k].toFixed(3) + ' (' + pct.toFixed(0) +
        '% do valor inicial)';
    }

    slider.addEventListener('input', () => {
      state.k = parseInt(slider.value, 10);
      kVal.textContent = slider.value;
      redraw();
    });
    document.getElementById('s2-reseed').addEventListener('click', () => {
      state.seed = (state.seed * 1103515245 + 12345) % 2147483647;
      recompute();
      redraw();
    });

    recompute();
    DL.plot.observeResize(cvG, redraw);
    DL.plot.observeResize(cvD, redraw);
    DL.plot.onRedraw(redraw);
    redraw();
  }

  DL.sections.push({ name: 'oversmoothing', init });
})();
