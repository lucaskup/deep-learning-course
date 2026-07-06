/* Seção 2: mapas de ativação dos neurônios da primeira camada escondida.
   Lê o modelo da seção 1 via DL.mlpPlayground e se atualiza quando o
   treinamento avança (a cada ~30 quadros) ou pausa. */
(function () {
  'use strict';
  window.DL = window.DL || {};
  DL.sections = DL.sections || [];

  const MAXN = 8;   // máximo de neurônios na primeira camada escondida
  const GRID = 60;  // resolução do heatmap de cada neurônio

  function init() {
    const P = DL.plot;
    const $ = (id) => document.getElementById(id);
    if (!DL.mlpPlayground) return;   // a seção 1 não inicializou

    const boxes = [], canvases = [];
    for (let i = 0; i < MAXN; i++) {
      boxes.push($('s5-box' + i));
      canvases.push($('s5-n' + i));
    }
    const buf = new Float32Array(MAXN);

    function drawUnit(j, st, th, cNeg, cPos) {
      const { ctx, w, h } = P.setup(canvases[j]);
      P.clear(ctx, w, h);
      const R = st.range;
      const fr = P.frame(ctx, w, h, -R, R, -R, R, { l: 4, r: 4, t: 4, b: 4 });
      P.heatmap(ctx, fr, (x, y) => { st.model.hidden1(x, y, buf); return buf[j]; },
        GRID, GRID, (v) => {
          const u = st.squash(v);
          const c = u < 0 ? cNeg : cPos;
          return [c[0], c[1], c[2], Math.min(1, Math.abs(u)) * 160];
        });
      P.scatter(ctx, fr, st.data.filter((_, i) => st.labels[i] === 0), th.cyan, { r: 1.3, alpha: 0.25 });
      P.scatter(ctx, fr, st.data.filter((_, i) => st.labels[i] === 1), th.orange, { r: 1.3, alpha: 0.25 });
    }

    function redraw() {
      const st = DL.mlpPlayground.getState();
      if (!st.model) return;
      const th = P.theme();
      const cNeg = P.rgb(th.purple), cPos = P.rgb(th.green);
      for (let j = 0; j < MAXN; j++) {
        const visible = j < st.h1;
        boxes[j].classList.toggle('hidden', !visible);
        if (visible) drawUnit(j, st, th, cNeg, cPos);
      }
    }

    DL.mlpPlayground.subscribe(redraw);
    redraw();
    P.onRedraw(redraw);
    P.observeResize(canvases[0], redraw);
  }

  DL.sections.push({ name: 's5-hidden', init });
})();
