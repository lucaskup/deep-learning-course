/* Seção 1: convolução 2D interativa, entrada desenhável 8×8, kernels clássicos
   editáveis e mapa de características z = w ∗ x recalculado ao vivo. */
(function () {
  'use strict';
  window.DL = window.DL || {};
  DL.sections = DL.sections || [];

  const N = 8;

  const KERNELS = {
    sobelv: [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],
    sobelh: [[-1, -2, -1], [0, 0, 0], [1, 2, 1]],
    blur: [[0.11, 0.11, 0.11], [0.11, 0.11, 0.11], [0.11, 0.11, 0.11]],
    sharpen: [[0, -1, 0], [-1, 5, -1], [0, -1, 0]],
  };

  const PATTERNS = {
    'borda-v': (i, j) => (j >= N / 2 ? 1 : 0),
    quadrado: (i, j) => (i >= 2 && i <= 5 && j >= 2 && j <= 5 ? 1 : 0),
    diagonal: (i, j) => (i === j ? 1 : 0),
    xis: (i, j) => (i === j || i + j === N - 1 ? 1 : 0),
    gradiente: (i, j) => j / (N - 1),
  };

  function init() {
    const P = DL.plot;
    const G = DL.grid;
    const $ = (id) => document.getElementById(id);

    const cvIn = $('s1-input'), cvOut = $('s1-output');
    const selPattern = $('s1-pattern');
    const tabIds = ['sobelv', 'sobelh', 'blur', 'sharpen', 'custom'];
    const kInputs = [];
    for (let i = 0; i < 9; i++) kInputs.push($('s1-k' + i));

    let input = [];
    let kernel = KERNELS.sobelv.map((r) => r.slice());
    let output = [];
    let hover = null;            /* {i, j} ou null */
    let painting = false, paintValue = 1;
    let seed = 7;

    function fillPattern() {
      const f = PATTERNS[selPattern.value];
      input = [];
      for (let i = 0; i < N; i++) {
        const row = [];
        for (let j = 0; j < N; j++) row.push(f(i, j));
        input.push(row);
      }
    }

    function readKernel() {
      for (let m = 0; m < 3; m++) {
        for (let n = 0; n < 3; n++) {
          const v = parseFloat(kInputs[3 * m + n].value);
          kernel[m][n] = isFinite(v) ? v : 0;
        }
      }
    }

    function writeKernel() {
      for (let m = 0; m < 3; m++) {
        for (let n = 0; n < 3; n++) {
          kInputs[3 * m + n].value = String(kernel[m][n]);
        }
      }
    }

    function setActiveTab(name) {
      for (const t of tabIds) $('s1-tab-' + t).classList.toggle('active', t === name);
    }

    function recompute() {
      output = G.conv2dSame3(input, kernel);
    }

    /* Layout com margem de 1 célula: o anel externo representa o zero padding. */
    function makeLayout(w, h) {
      return G.layout(w, h, N + 2, N + 2, 6);
    }

    function fmtIn(v) {
      if (v === 0) return '0';
      if (v === 1) return '1';
      return v.toFixed(1);
    }

    function drawInput() {
      const th = P.theme();
      const { ctx, w, h } = P.setup(cvIn);
      P.clear(ctx, w, h);
      const L = makeLayout(w, h);

      /* Anel de padding (zeros), tracejado e apagado */
      ctx.setLineDash([3, 3]);
      G.outline(ctx, L, 0, 0, N + 2, N + 2, th.line, 1);
      ctx.setLineDash([]);

      for (let i = 0; i < N; i++) {
        for (let j = 0; j < N; j++) {
          ctx.fillStyle = G.mix(th.card, th.cyan, input[i][j] * 0.85);
          ctx.fillRect(L.x(j + 1), L.y(i + 1), L.cs, L.cs);
        }
      }
      /* Grade interna apenas na região N×N */
      ctx.strokeStyle = th.line;
      ctx.lineWidth = 1;
      ctx.beginPath();
      for (let i = 1; i <= N + 1; i++) {
        ctx.moveTo(L.x(1), L.y(i)); ctx.lineTo(L.x(N + 1), L.y(i));
        ctx.moveTo(L.x(i), L.y(1)); ctx.lineTo(L.x(i), L.y(N + 1));
      }
      ctx.stroke();

      if (L.cs >= 24) {
        for (let i = 0; i < N; i++) {
          for (let j = 0; j < N; j++) {
            const v = input[i][j];
            ctx.globalAlpha = 0.85;
            G.cellText(ctx, L, i + 1, j + 1, fmtIn(v), v > 0.55 ? th.bg : th.comment);
            ctx.globalAlpha = 1;
          }
        }
      }

      if (hover) {
        G.outline(ctx, L, hover.i, hover.j, 3, 3, th.orange, 2.5);
      }
      P.label(ctx, L.x(1), L.y(1) - 5, 'zero padding', th.comment);
    }

    function drawOutput() {
      const th = P.theme();
      const { ctx, w, h } = P.setup(cvOut);
      P.clear(ctx, w, h);
      const L = makeLayout(w, h);

      let vmax = 1e-9;
      for (let i = 0; i < N; i++) {
        for (let j = 0; j < N; j++) vmax = Math.max(vmax, Math.abs(output[i][j]));
      }
      for (let i = 0; i < N; i++) {
        for (let j = 0; j < N; j++) {
          ctx.fillStyle = G.diverge(output[i][j], vmax, th);
          ctx.fillRect(L.x(j + 1), L.y(i + 1), L.cs, L.cs);
        }
      }
      ctx.strokeStyle = th.line;
      ctx.lineWidth = 1;
      ctx.beginPath();
      for (let i = 1; i <= N + 1; i++) {
        ctx.moveTo(L.x(1), L.y(i)); ctx.lineTo(L.x(N + 1), L.y(i));
        ctx.moveTo(L.x(i), L.y(1)); ctx.lineTo(L.x(i), L.y(N + 1));
      }
      ctx.stroke();

      if (L.cs >= 24) {
        for (let i = 0; i < N; i++) {
          for (let j = 0; j < N; j++) {
            const v = output[i][j];
            ctx.globalAlpha = 0.9;
            G.cellText(ctx, L, i + 1, j + 1, (Math.round(v * 10) / 10).toString(), th.fg);
            ctx.globalAlpha = 1;
          }
        }
      }

      if (hover) {
        G.outline(ctx, L, hover.i + 1, hover.j + 1, 1, 1, th.orange, 2.5);
      }
      P.label(ctx, L.x(1), L.y(1) - 5, 'z = w ∗ x', th.comment);
    }

    function updateReadout() {
      let vmax = 0;
      for (let i = 0; i < N; i++) {
        for (let j = 0; j < N; j++) vmax = Math.max(vmax, Math.abs(output[i][j]));
      }
      let txt = 'máx |z| = ' + (Math.round(vmax * 100) / 100);
      if (hover) {
        const z = output[hover.i][hover.j];
        txt = 'z(' + (hover.i + 1) + ', ' + (hover.j + 1) + ') = ' + (Math.round(z * 100) / 100) + ' · ' + txt;
      }
      $('s1-readout').textContent = txt;
    }

    function redraw() {
      drawInput();
      drawOutput();
      updateReadout();
    }

    /* Centro da janela 3×3 sob o ponteiro (em coordenadas da entrada N×N). */
    function cellFromEvent(cv, e) {
      const r = cv.getBoundingClientRect();
      const { w, h } = P.setup(cv);
      const L = makeLayout(w, h);
      const c = L.cellAt(e.clientX - r.left, e.clientY - r.top);
      if (!c) return null;
      const i = c.i - 1, j = c.j - 1;
      return i >= 0 && i < N && j >= 0 && j < N ? { i, j } : null;
    }

    cvIn.addEventListener('pointerdown', (e) => {
      const c = cellFromEvent(cvIn, e);
      if (!c) return;
      cvIn.setPointerCapture(e.pointerId);
      painting = true;
      paintValue = input[c.i][c.j] > 0.5 ? 0 : 1;
      input[c.i][c.j] = paintValue;
      recompute();
      redraw();
    });
    cvIn.addEventListener('pointermove', (e) => {
      const c = cellFromEvent(cvIn, e);
      hover = c;
      if (painting && c) {
        input[c.i][c.j] = paintValue;
        recompute();
      }
      redraw();
    });
    cvIn.addEventListener('pointerup', () => { painting = false; });
    cvIn.addEventListener('pointerleave', () => {
      if (!painting) { hover = null; redraw(); }
    });

    cvOut.addEventListener('pointermove', (e) => {
      hover = cellFromEvent(cvOut, e);
      redraw();
    });
    cvOut.addEventListener('pointerleave', () => { hover = null; redraw(); });

    for (const t of ['sobelv', 'sobelh', 'blur', 'sharpen']) {
      $('s1-tab-' + t).addEventListener('click', () => {
        kernel = KERNELS[t].map((r) => r.slice());
        writeKernel();
        setActiveTab(t);
        recompute();
        redraw();
      });
    }
    $('s1-tab-custom').addEventListener('click', () => setActiveTab('custom'));

    for (const inp of kInputs) {
      inp.addEventListener('input', () => {
        readKernel();
        setActiveTab('custom');
        recompute();
        redraw();
      });
    }

    selPattern.addEventListener('change', () => {
      fillPattern();
      recompute();
      redraw();
    });

    $('s1-noise').addEventListener('click', () => {
      const rng = DL.utils.mulberry32(seed);
      seed = (seed * 16807 + 11) % 2147483647;
      for (let i = 0; i < N; i++) {
        for (let j = 0; j < N; j++) {
          input[i][j] = Math.max(0, Math.min(1, input[i][j] + (rng() - 0.5) * 0.6));
        }
      }
      recompute();
      redraw();
    });

    $('s1-clear').addEventListener('click', () => {
      fillPattern();
      recompute();
      redraw();
    });

    fillPattern();
    readKernel();
    recompute();
    redraw();
    P.onRedraw(redraw);
    P.observeResize(cvIn, redraw);
    P.observeResize(cvOut, redraw);
  }

  DL.sections.push({ name: 's1-convolucao', init });
})();
