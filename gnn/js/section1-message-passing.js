/* Seção 1: message passing e receptive field.
   Clique escolhe o nó central; o slider K colore os nós por distância
   em saltos (BFS), mostrando o alcance de K camadas de message passing. */
(function () {
  'use strict';
  window.DL = window.DL || {};
  DL.sections = DL.sections || [];

  /* Grafo fixo, posições em coordenadas de dados [0,10] x [0,6]. */
  const POS = [
    [0.8, 4.8], [2.2, 5.4], [1.6, 3.4], [3.2, 4.4], [3.0, 2.4],
    [4.6, 5.0], [4.8, 3.0], [6.0, 4.4], [6.4, 2.4], [7.6, 5.2],
    [8.2, 3.4], [9.2, 4.6], [7.4, 1.2], [9.0, 1.8],
  ];
  const EDGES = [
    [0, 1], [0, 2], [1, 3], [2, 3], [2, 4], [3, 4], [3, 5], [4, 6],
    [5, 6], [5, 7], [5, 9], [6, 8], [6, 10], [7, 8], [7, 9], [8, 10],
    [8, 12], [9, 10], [9, 11], [10, 12], [10, 13], [11, 13],
  ];
  const N = POS.length;
  const ADJ = Array.from({ length: N }, () => []);
  for (const [a, b] of EDGES) { ADJ[a].push(b); ADJ[b].push(a); }

  function bfs(src) {
    const dist = new Array(N).fill(Infinity);
    dist[src] = 0;
    const queue = [src];
    while (queue.length) {
      const u = queue.shift();
      for (const w of ADJ[u]) {
        if (dist[w] === Infinity) { dist[w] = dist[u] + 1; queue.push(w); }
      }
    }
    return dist;
  }

  const state = { center: 0, k: 2 };

  function init() {
    const cv = document.getElementById('s1-graph');
    const slider = document.getElementById('s1-k');
    const kVal = document.getElementById('s1-k-val');
    const readout = document.getElementById('s1-readout');

    function draw() {
      const { ctx, w, h } = DL.plot.setup(cv);
      const t = DL.plot.theme();
      DL.plot.clear(ctx, w, h);
      const fr = DL.plot.frame(ctx, w, h, -0.4, 10.4, -0.4, 6.4,
        { l: 8, r: 8, t: 8, b: 8 });
      /* Paleta por distância: 0 (centro) até 4 saltos. */
      const hopColor = [t.pink, t.cyan, t.green, t.orange, t.purple];
      const dist = bfs(state.center);

      ctx.strokeStyle = t.line;
      ctx.lineWidth = 1.4;
      for (const [a, b] of EDGES) {
        /* Aresta destacada se liga dois nós já alcançados. */
        const inside = dist[a] <= state.k && dist[b] <= state.k;
        ctx.globalAlpha = inside ? 0.95 : 0.3;
        ctx.beginPath();
        ctx.moveTo(fr.X(POS[a][0]), fr.Y(POS[a][1]));
        ctx.lineTo(fr.X(POS[b][0]), fr.Y(POS[b][1]));
        ctx.stroke();
      }
      ctx.globalAlpha = 1;

      let reached = 0;
      for (let i = 0; i < N; i++) {
        const d = dist[i];
        const inside = d <= state.k;
        if (inside) reached++;
        const x = fr.X(POS[i][0]), y = fr.Y(POS[i][1]);
        ctx.beginPath();
        ctx.arc(x, y, i === state.center ? 11 : 9, 0, 2 * Math.PI);
        if (inside) {
          ctx.fillStyle = hopColor[Math.min(d, hopColor.length - 1)];
          ctx.globalAlpha = 1;
        } else {
          ctx.fillStyle = t.comment;
          ctx.globalAlpha = 0.25;
        }
        ctx.fill();
        ctx.globalAlpha = 1;
        if (inside) {
          ctx.fillStyle = t.bg;
          ctx.font = 'bold 10px sans-serif';
          ctx.textAlign = 'center';
          ctx.textBaseline = 'middle';
          ctx.fillText(i === state.center ? 'v' : String(d), x, y + 0.5);
          ctx.textBaseline = 'alphabetic';
        }
      }

      /* Legenda. */
      ctx.font = '11px sans-serif';
      ctx.textAlign = 'left';
      let lx = fr.X(-0.2), ly = fr.Y(6.3);
      for (let d = 0; d <= 4; d++) {
        ctx.fillStyle = hopColor[d];
        ctx.beginPath();
        ctx.arc(lx + 5, ly - 4, 5, 0, 2 * Math.PI);
        ctx.fill();
        ctx.fillStyle = t.comment;
        const txt = d === 0 ? 'centro' : d + (d === 1 ? ' salto' : ' saltos');
        ctx.fillText(txt, lx + 14, ly);
        lx += 24 + ctx.measureText(txt).width;
      }

      readout.textContent = 'receptive field de v: ' + reached + ' de ' + N +
        ' nós com K = ' + state.k;
    }

    cv.addEventListener('pointerdown', (ev) => {
      const rect = cv.getBoundingClientRect();
      const px = ev.clientX - rect.left, py = ev.clientY - rect.top;
      const { w, h } = DL.plot.setup(cv);
      const fr = DL.plot.frame(null, w, h, -0.4, 10.4, -0.4, 6.4,
        { l: 8, r: 8, t: 8, b: 8 });
      let best = -1, bestD = 18;
      for (let i = 0; i < N; i++) {
        const d = Math.hypot(fr.X(POS[i][0]) - px, fr.Y(POS[i][1]) - py);
        if (d < bestD) { bestD = d; best = i; }
      }
      if (best >= 0) { state.center = best; draw(); }
    });

    slider.addEventListener('input', () => {
      state.k = parseInt(slider.value, 10);
      kVal.textContent = slider.value;
      draw();
    });

    DL.plot.observeResize(cv, draw);
    DL.plot.onRedraw(draw);
    draw();
  }

  DL.sections.push({ name: 'message-passing', init });
})();
