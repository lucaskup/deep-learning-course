/* Seção 2: matriz de similaridade contrastiva S = UVᵀ.
   N vetores unitários de imagem e N de texto, parametrizados por ângulo,
   arrastáveis em duas circunferências. S_ij = û_i·v̂_j = cos(a_i − b_j). */
(function () {
  'use strict';
  window.DL = window.DL || {};
  DL.sections = DL.sections || [];

  const SEED = 7;

  function init() {
    if (!document.getElementById('s2-circles')) return;
    const P = DL.plot;
    const $ = (id) => document.getElementById(id);
    const cvCirc = $('s2-circles'), cvHeat = $('s2-heat');
    const slN = $('s2-n');

    let N = +slN.value;
    let imgA = [], txtA = [];
    let geom = null;          // centros e raios das duas circunferências
    let drag = null;          // { set:'img'|'txt', idx }

    function reset() {
      const rng = DL.utils.mulberry32(SEED);
      imgA = []; txtA = [];
      for (let i = 0; i < N; i++) {
        /* Espalha as imagens e dá aos textos um deslocamento pequeno do par. */
        const base = -Math.PI / 2 + i * (2 * Math.PI / N);
        imgA.push(base + 0.25 * (rng() - 0.5));
        txtA.push(base + 0.9 * (rng() - 0.5));
      }
    }

    const sim = (i, j) => Math.cos(imgA[i] - txtA[j]);

    function makeGeom(w, h) {
      const r = Math.min(h * 0.34, w * 0.2);
      return {
        r,
        img: { cx: w * 0.27, cy: h * 0.54 },
        txt: { cx: w * 0.73, cy: h * 0.54 },
      };
    }

    function ptOf(set, k) {
      const g = set === 'img' ? geom.img : geom.txt;
      const a = (set === 'img' ? imgA : txtA)[k];
      return [g.cx + geom.r * Math.cos(a), g.cy + geom.r * Math.sin(a)];
    }

    function drawCircles() {
      const th = P.theme();
      const { ctx, w, h } = P.setup(cvCirc);
      P.clear(ctx, w, h);
      geom = makeGeom(w, h);

      [['img', geom.img, th.cyan, 'û'], ['txt', geom.txt, th.orange, 'v̂']].forEach(([set, g, col, sym]) => {
        ctx.strokeStyle = th.line; ctx.lineWidth = 1.2;
        ctx.beginPath(); ctx.arc(g.cx, g.cy, geom.r, 0, 2 * Math.PI); ctx.stroke();
        P.mathText(ctx, set === 'img' ? 'imagens' : 'textos', g.cx, g.cy - geom.r - 12, col, 'center', 12);
        const arr = set === 'img' ? imgA : txtA;
        for (let k = 0; k < N; k++) {
          const p = ptOf(set, k);
          /* Pequeno traço do centro à ponta para sugerir o vetor. */
          ctx.strokeStyle = col; ctx.globalAlpha = 0.35; ctx.lineWidth = 1.4;
          ctx.beginPath(); ctx.moveTo(g.cx, g.cy); ctx.lineTo(p[0], p[1]); ctx.stroke();
          ctx.globalAlpha = 1;
          ctx.fillStyle = col;
          ctx.beginPath(); ctx.arc(p[0], p[1], 6, 0, 2 * Math.PI); ctx.fill();
          ctx.strokeStyle = th.bg; ctx.lineWidth = 1.5; ctx.stroke();
          const lx = g.cx + (geom.r + 14) * Math.cos(arr[k]);
          const ly = g.cy + (geom.r + 14) * Math.sin(arr[k]);
          P.mathText(ctx, sym + '_' + (k + 1), lx, ly + 4, col, 'center', 11);
        }
      });
    }

    function drawHeat() {
      const th = P.theme();
      const { ctx, w, h } = P.setup(cvHeat);
      P.clear(ctx, w, h);
      const padL = 34, padT = 30, padR = 12, padB = 14;
      const gw = w - padL - padR, gh = h - padT - padB;
      const cell = Math.min(gw, gh) / N;
      const ox = padL + (gw - cell * N) / 2;
      const oy = padT + (gh - cell * N) / 2;

      P.mathText(ctx, 'textos  v̂_j →', ox + cell * N / 2, padT - 14, th.orange, 'center', 11);
      ctx.save();
      ctx.translate(ox - 18, oy + cell * N / 2);
      ctx.rotate(-Math.PI / 2);
      P.mathText(ctx, 'imagens  û_i', 0, 0, th.cyan, 'center', 11);
      ctx.restore();

      for (let i = 0; i < N; i++) {
        for (let j = 0; j < N; j++) {
          const s = sim(i, j);
          const t = (s + 1) / 2;                 // [-1,1] → [0,1]
          const col = P.viridis(t);
          const x = ox + j * cell, y = oy + i * cell;
          ctx.fillStyle = 'rgb(' + col[0] + ',' + col[1] + ',' + col[2] + ')';
          ctx.fillRect(x, y, cell - 1, cell - 1);
          /* valor */
          ctx.fillStyle = t > 0.55 ? '#101010' : '#f0f0f0';
          ctx.font = (cell > 54 ? 12 : 10) + 'px sans-serif';
          ctx.textAlign = 'center';
          ctx.fillText(s.toFixed(2), x + cell / 2, y + cell / 2 + 4);
          if (i === j) {
            ctx.strokeStyle = th.green; ctx.lineWidth = 2.5;
            ctx.strokeRect(x + 1.5, y + 1.5, cell - 4, cell - 4);
          }
        }
      }
    }

    function updateReadout() {
      $('s2-readout').textContent =
        'positivos = ' + N + ' (diagonal) · negativos = ' + N * (N - 1);
    }

    function redraw() { drawCircles(); drawHeat(); updateReadout(); }

    function pick(px, py) {
      if (!geom) return null;
      let best = null, bd = 22;
      for (const set of ['img', 'txt']) {
        for (let k = 0; k < N; k++) {
          const p = ptOf(set, k);
          const d = Math.hypot(px - p[0], py - p[1]);
          if (d < bd) { bd = d; best = { set, idx: k }; }
        }
      }
      return best;
    }

    cvCirc.addEventListener('pointerdown', (e) => {
      const r = cvCirc.getBoundingClientRect();
      const hit = pick(e.clientX - r.left, e.clientY - r.top);
      if (!hit) return;
      drag = hit;
      cvCirc.setPointerCapture(e.pointerId);
    });
    cvCirc.addEventListener('pointermove', (e) => {
      if (!drag) return;
      const r = cvCirc.getBoundingClientRect();
      const g = drag.set === 'img' ? geom.img : geom.txt;
      const a = Math.atan2(e.clientY - r.top - g.cy, e.clientX - r.left - g.cx);
      (drag.set === 'img' ? imgA : txtA)[drag.idx] = a;
      redraw();
    });
    cvCirc.addEventListener('pointerup', () => { drag = null; });
    cvCirc.addEventListener('pointercancel', () => { drag = null; });

    slN.addEventListener('input', () => {
      N = +slN.value;
      $('s2-n-val').textContent = N;
      reset();
      redraw();
    });
    $('s2-reset').addEventListener('click', () => { reset(); redraw(); });

    reset();
    $('s2-n-val').textContent = N;
    redraw();
    P.onRedraw(redraw);
    P.observeResize(cvCirc, redraw);
    P.observeResize(cvHeat, redraw);
  }

  DL.sections.push({ name: 's2-similarity', init });
})();
