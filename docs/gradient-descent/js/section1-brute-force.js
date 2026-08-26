/* Seção 1: varredura por força bruta na loss L(w, b) de uma unidade sigmoide.
   Como no exemplo dos slides: f(x) = σ(wx + b), loss = erro quadrático médio.
   Uma grade k×k é avaliada sobre (w, b) e comparada com o mínimo real. */
(function () {
  'use strict';
  window.DL = window.DL || {};
  DL.sections = DL.sections || [];

  const WMIN = -6, WMAX = 6, BMIN = -6, BMAX = 6;
  const XMIN = -3, XMAX = 3;
  const N = 40, TRUE_W = 1.8, TRUE_B = -0.8;
  const NF = 110; /* resolução da varredura fina (mínimo de referência) */

  function init() {
    const P = DL.plot;
    const $ = (id) => document.getElementById(id);

    const cvSurf = $('s1-surface'), cvData = $('s1-data');
    const slK = $('s1-k');

    let seed = 1;
    let xs = [], ys = [];
    let lossLo = 0, lossHi = 1;
    let fineMin = { w: 0, b: 0, L: Infinity };
    let k = +slK.value;
    let grid = [];        /* pontos avaliados da grade: {w, b, L} */
    let best = null;      /* melhor ponto da grade */
    let selected = null;  /* ponto da grade clicado pelo usuário */
    let frSurf = null;

    const sigm = (z) => 1 / (1 + Math.exp(-z));

    function loss(w, b) {
      let s = 0;
      for (let i = 0; i < N; i++) {
        const d = ys[i] - sigm(w * xs[i] + b);
        s += d * d;
      }
      return s / N;
    }

    function genData() {
      const rng = DL.utils.mulberry32(seed);
      xs = []; ys = [];
      for (let i = 0; i < N; i++) {
        const x = XMIN + (XMAX - XMIN) * rng();
        xs.push(x);
        ys.push(rng() < sigm(TRUE_W * x + TRUE_B) ? 1 : 0);
      }
      /* Varredura fina: normalização do mapa de cores + mínimo de referência. */
      lossLo = Infinity; lossHi = -Infinity;
      fineMin = { w: 0, b: 0, L: Infinity };
      for (let j = 0; j < NF; j++) {
        const b = BMIN + (j + 0.5) / NF * (BMAX - BMIN);
        for (let i = 0; i < NF; i++) {
          const w = WMIN + (i + 0.5) / NF * (WMAX - WMIN);
          const L = loss(w, b);
          if (L < lossLo) lossLo = L;
          if (L > lossHi) lossHi = L;
          if (L < fineMin.L) fineMin = { w, b, L };
        }
      }
    }

    function sweep() {
      grid = [];
      best = null;
      for (let j = 0; j < k; j++) {
        const b = k === 1 ? 0 : BMIN + j / (k - 1) * (BMAX - BMIN);
        for (let i = 0; i < k; i++) {
          const w = k === 1 ? 0 : WMIN + i / (k - 1) * (WMAX - WMIN);
          const p = { w, b, L: loss(w, b) };
          grid.push(p);
          if (!best || p.L < best.L) best = p;
        }
      }
      selected = null;
    }

    function drawSurf() {
      const th = P.theme();
      const { ctx, w, h } = P.setup(cvSurf);
      P.clear(ctx, w, h);
      const fr = P.frame(ctx, w, h, WMIN, WMAX, BMIN, BMAX);
      frSurf = fr;
      /* Raiz quadrada no t para dar contraste perto do mínimo; o clamp evita
         NaN quando a grade do heatmap amostra fora da faixa [lossLo, lossHi]. */
      P.heatmap(ctx, fr, loss, 90, 90, (v) => {
        const u = Math.max(0, Math.min(1, (v - lossLo) / (lossHi - lossLo)));
        const c = P.viridis(1 - Math.sqrt(u));
        return [c[0], c[1], c[2], 235];
      });
      P.axes(ctx, fr, { xlabel: 'w', ylabel: 'b', xticks: [-6, -3, 0, 3, 6], yticks: [-6, -3, 0, 3, 6] });

      /* Grade avaliada */
      ctx.fillStyle = '#ffffff';
      for (const p of grid) {
        ctx.globalAlpha = 0.8;
        ctx.beginPath();
        ctx.arc(fr.X(p.w), fr.Y(p.b), k > 25 ? 1.4 : 2.2, 0, 2 * Math.PI);
        ctx.fill();
      }
      ctx.globalAlpha = 1;

      /* Mínimo real (varredura fina) como × */
      const mx = fr.X(fineMin.w), my = fr.Y(fineMin.b);
      ctx.strokeStyle = th.green;
      ctx.lineWidth = 2;
      ctx.beginPath();
      ctx.moveTo(mx - 5, my - 5); ctx.lineTo(mx + 5, my + 5);
      ctx.moveTo(mx - 5, my + 5); ctx.lineTo(mx + 5, my - 5);
      ctx.stroke();

      /* Ponto selecionado pelo clique */
      if (selected) {
        ctx.beginPath();
        ctx.arc(fr.X(selected.w), fr.Y(selected.b), 6, 0, 2 * Math.PI);
        ctx.strokeStyle = th.orange;
        ctx.lineWidth = 2.5;
        ctx.stroke();
      }

      /* Melhor ponto da grade */
      if (best) {
        ctx.beginPath();
        ctx.arc(fr.X(best.w), fr.Y(best.b), 6.5, 0, 2 * Math.PI);
        ctx.fillStyle = th.pink;
        ctx.fill();
        ctx.lineWidth = 1.5;
        ctx.strokeStyle = th.bg;
        ctx.stroke();
      }

      P.label(ctx, fr.X(WMIN) + 8, fr.Y(BMAX) + 14, 'melhor da grade', th.pink);
      P.label(ctx, fr.X(WMIN) + 8, fr.Y(BMAX) + 28, 'mínimo real ×', th.green);
    }

    function drawData() {
      const th = P.theme();
      const { ctx, w, h } = P.setup(cvData);
      P.clear(ctx, w, h);
      const fr = P.frame(ctx, w, h, XMIN, XMAX, -0.12, 1.12);
      P.axes(ctx, fr, { xlabel: 'x', ylabel: 'y', xticks: [-3, -1.5, 0, 1.5, 3], yticks: [0, 0.5, 1] });

      /* Dados binários */
      P.scatter(ctx, fr, xs.map((x, i) => [x, ys[i]]), th.comment, { r: 3, alpha: 0.65 });

      const curve = (wPar, bPar) => {
        const cx = [], cy = [];
        for (let x = XMIN; x <= XMAX + 1e-9; x += 0.05) { cx.push(x); cy.push(sigm(wPar * x + bPar)); }
        return [cx, cy];
      };

      /* Mínimo real tracejado, melhor da grade sólido, selecionado em laranja */
      let c = curve(fineMin.w, fineMin.b);
      P.line(ctx, fr, c[0], c[1], th.green, { dash: [5, 4], width: 1.6 });
      if (best) {
        c = curve(best.w, best.b);
        P.line(ctx, fr, c[0], c[1], th.cyan, { width: 2.2 });
      }
      if (selected) {
        c = curve(selected.w, selected.b);
        P.line(ctx, fr, c[0], c[1], th.orange, { width: 1.8 });
      }

      P.label(ctx, fr.X(XMIN) + 8, fr.Y(1.12) + 14, 'melhor da grade', th.cyan);
      P.label(ctx, fr.X(XMIN) + 8, fr.Y(1.12) + 28, 'mínimo real', th.green);
      if (selected) P.label(ctx, fr.X(XMIN) + 8, fr.Y(1.12) + 42, 'ponto clicado', th.orange);
    }

    function updateReadout() {
      let txt = 'k² = ' + (k * k) + ' avaliações · melhor L = ' + best.L.toFixed(3) +
        ' em (w, b) = (' + best.w.toFixed(1) + ', ' + best.b.toFixed(1) + ')' +
        ' · mínimo real L = ' + fineMin.L.toFixed(3);
      if (selected) txt += ' · clicado: L = ' + selected.L.toFixed(3);
      $('s1-readout').textContent = txt;
      /* Extrapolação do custo para P = 1000 parâmetros: k^1000 = 10^(1000·log10 k) */
      const expo = Math.round(1000 * Math.log10(k));
      $('s1-extrap').innerHTML = '10<sup>' + expo + '</sup>';
    }

    function redraw() {
      drawSurf();
      drawData();
      updateReadout();
    }

    cvSurf.addEventListener('pointerdown', (e) => {
      if (!frSurf || grid.length === 0) return;
      const r = cvSurf.getBoundingClientRect();
      const wc = frSurf.invX(e.clientX - r.left);
      const bc = frSurf.invY(e.clientY - r.top);
      let bestD = Infinity, sel = null;
      for (const p of grid) {
        const d = (p.w - wc) * (p.w - wc) + (p.b - bc) * (p.b - bc);
        if (d < bestD) { bestD = d; sel = p; }
      }
      selected = sel;
      redraw();
    });

    slK.addEventListener('input', () => {
      k = +slK.value;
      $('s1-k-val').textContent = String(k);
      sweep();
      redraw();
    });
    $('s1-resample').addEventListener('click', () => {
      seed = (seed * 1103515245 + 12345) & 0x7fffffff;
      genData();
      sweep();
      redraw();
    });

    genData();
    sweep();
    $('s1-k-val').textContent = String(k);
    redraw();
    P.onRedraw(redraw);
    P.observeResize(cvSurf, redraw);
    P.observeResize(cvData, redraw);
  }

  DL.sections.push({ name: 's1-brute-force', init });
})();
