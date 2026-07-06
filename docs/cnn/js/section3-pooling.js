/* Seção 3: max e average pooling 2×2 (stride 2) e invariância a translação,
   comparando a saída do padrão deslocado com a saída original deslocada. */
(function () {
  'use strict';
  window.DL = window.DL || {};
  DL.sections = DL.sections || [];

  const N = 8, POOL = 2, M = N / POOL, DMAX = 5;

  /* Padrão assimétrico 3×3 (um "L" com um detalhe fraco) */
  const PAT = [
    [1.0, 0.6, 0.0],
    [1.0, 0.0, 0.0],
    [1.0, 1.0, 1.0],
  ];

  function init() {
    const P = DL.plot;
    const G = DL.grid;
    const $ = (id) => document.getElementById(id);

    const cvIn = $('s3-input'), cvOut = $('s3-output');
    const slDx = $('s3-dx'), slDy = $('s3-dy');

    let mode = 'max';
    let dx = +slDx.value, dy = +slDy.value;
    let dragging = false, dragCell = null, dragD0 = null;

    function makeInput(ddx, ddy) {
      const g = [];
      for (let i = 0; i < N; i++) g.push(new Array(N).fill(0));
      for (let a = 0; a < 3; a++) {
        for (let b = 0; b < 3; b++) {
          const i = a + ddy, j = b + ddx;
          if (i < N && j < N) g[i][j] = PAT[a][b];
        }
      }
      return g;
    }

    function pool(g) {
      const out = [];
      for (let i = 0; i < M; i++) {
        const row = [];
        for (let j = 0; j < M; j++) {
          const vals = [
            g[2 * i][2 * j], g[2 * i][2 * j + 1],
            g[2 * i + 1][2 * j], g[2 * i + 1][2 * j + 1],
          ];
          row.push(mode === 'max'
            ? Math.max(vals[0], vals[1], vals[2], vals[3])
            : (vals[0] + vals[1] + vals[2] + vals[3]) / 4);
        }
        out.push(row);
      }
      return out;
    }

    /* Referência: saída do padrão sem deslocamento, deslocada ⌊Δ/2⌋ no espaço da saída. */
    function shiftedRef() {
      const base = pool(makeInput(0, 0));
      const sj = Math.floor(dx / POOL), si = Math.floor(dy / POOL);
      const out = [];
      for (let i = 0; i < M; i++) {
        const row = [];
        for (let j = 0; j < M; j++) {
          const ii = i - si, jj = j - sj;
          row.push(ii >= 0 && ii < M && jj >= 0 && jj < M ? base[ii][jj] : 0);
        }
        out.push(row);
      }
      return out;
    }

    function fmt(v) {
      return String(Math.round(v * 100) / 100);
    }

    function drawInput() {
      const th = P.theme();
      const { ctx, w, h } = P.setup(cvIn);
      P.clear(ctx, w, h);
      const L = G.layout(w, h, N, N, 12);
      const g = makeInput(dx, dy);

      for (let i = 0; i < N; i++) {
        for (let j = 0; j < N; j++) {
          ctx.fillStyle = G.mix(th.card, th.cyan, g[i][j] * 0.85);
          ctx.fillRect(L.x(j), L.y(i), L.cs, L.cs);
        }
      }
      G.lines(ctx, L, th.line);
      /* Janelas 2×2 do pooling demarcadas com linha mais forte */
      ctx.strokeStyle = th.comment;
      ctx.lineWidth = 1.6;
      ctx.beginPath();
      for (let i = 0; i <= N; i += POOL) {
        ctx.moveTo(L.x(0), L.y(i)); ctx.lineTo(L.x(N), L.y(i));
        ctx.moveTo(L.x(i), L.y(0)); ctx.lineTo(L.x(i), L.y(N));
      }
      ctx.stroke();

      if (L.cs >= 24) {
        for (let i = 0; i < N; i++) {
          for (let j = 0; j < N; j++) {
            if (g[i][j] > 0) {
              G.cellText(ctx, L, i, j, fmt(g[i][j]), g[i][j] > 0.55 ? th.bg : th.comment);
            }
          }
        }
      }
      P.label(ctx, L.x(0), L.y(0) - 5, 'entrada (arraste o padrão)', th.comment);
    }

    function drawOutput() {
      const th = P.theme();
      const { ctx, w, h } = P.setup(cvOut);
      P.clear(ctx, w, h);
      const L = G.layout(w, h, M, M, 12);
      const out = pool(makeInput(dx, dy));
      const ref = shiftedRef();

      let ndiff = 0;
      for (let i = 0; i < M; i++) {
        for (let j = 0; j < M; j++) {
          ctx.fillStyle = G.mix(th.card, th.cyan, Math.min(1, out[i][j]) * 0.85);
          ctx.fillRect(L.x(j), L.y(i), L.cs, L.cs);
        }
      }
      G.lines(ctx, L, th.line);
      for (let i = 0; i < M; i++) {
        for (let j = 0; j < M; j++) {
          G.cellText(ctx, L, i, j, fmt(out[i][j]), out[i][j] > 0.55 ? th.bg : th.fg);
          if (Math.abs(out[i][j] - ref[i][j]) > 1e-9) {
            ndiff++;
            ctx.save();
            G.outline(ctx, L, i, j, 1, 1, th.pink, 3);
            ctx.restore();
          }
        }
      }
      P.label(ctx, L.x(0), L.y(0) - 5, (mode === 'max' ? 'max' : 'average') + ' pooling 2×2', th.comment);

      $('s3-readout').textContent =
        'Δ = (' + dx + ', ' + dy + ') · ' + ndiff + ' de ' + (M * M) + ' células diferem da referência deslocada';
    }

    function redraw() {
      drawInput();
      drawOutput();
    }

    function syncSliders() {
      slDx.value = dx;
      slDy.value = dy;
      $('s3-dx-val').textContent = dx;
      $('s3-dy-val').textContent = dy;
    }

    $('s3-tab-max').addEventListener('click', () => {
      mode = 'max';
      $('s3-tab-max').classList.add('active');
      $('s3-tab-avg').classList.remove('active');
      redraw();
    });
    $('s3-tab-avg').addEventListener('click', () => {
      mode = 'avg';
      $('s3-tab-avg').classList.add('active');
      $('s3-tab-max').classList.remove('active');
      redraw();
    });

    slDx.addEventListener('input', () => {
      dx = +slDx.value;
      $('s3-dx-val').textContent = dx;
      redraw();
    });
    slDy.addEventListener('input', () => {
      dy = +slDy.value;
      $('s3-dy-val').textContent = dy;
      redraw();
    });

    $('s3-reset').addEventListener('click', () => {
      dx = 0; dy = 0;
      syncSliders();
      redraw();
    });

    function cellFromEvent(e) {
      const r = cvIn.getBoundingClientRect();
      const { w, h } = P.setup(cvIn);
      const L = G.layout(w, h, N, N, 12);
      return L.cellAt(e.clientX - r.left, e.clientY - r.top);
    }

    cvIn.addEventListener('pointerdown', (e) => {
      const c = cellFromEvent(e);
      if (!c) return;
      cvIn.setPointerCapture(e.pointerId);
      dragging = true;
      dragCell = c;
      dragD0 = { dx, dy };
    });
    cvIn.addEventListener('pointermove', (e) => {
      if (!dragging) return;
      const c = cellFromEvent(e);
      if (!c) return;
      dx = Math.max(0, Math.min(DMAX, dragD0.dx + (c.j - dragCell.j)));
      dy = Math.max(0, Math.min(DMAX, dragD0.dy + (c.i - dragCell.i)));
      syncSliders();
      redraw();
    });
    cvIn.addEventListener('pointerup', () => { dragging = false; });

    redraw();
    P.onRedraw(redraw);
    P.observeResize(cvIn, redraw);
    P.observeResize(cvOut, redraw);
  }

  DL.sections.push({ name: 's3-pooling', init });
})();
