/* Seção 4: leis de escala. Curva isoFLOP L(N, C/6N) na forma paramétrica
   de Hoffmann et al., L(N,D) = E + A/N^α + B/D^β, com constantes calibradas
   para que o ótimo siga a regra didática dos slides: N* ∝ C^0.5 e D* ≈ 20 N*. */
(function () {
  'use strict';
  window.DL = window.DL || {};
  DL.sections = DL.sections || [];

  const E = 1.69, A = 148.4, B = 410.7, AL = 0.34, BE = 0.34;
  const HALF = 1.8; /* meia-largura da janela em décadas de N */

  const MODELS = [
    { name: 'GPT-3', N: 175e9, D: 300e9 },
    { name: 'Gopher', N: 280e9, D: 300e9 },
    { name: 'Chinchilla', N: 70e9, D: 1.4e12 },
    { name: 'Llama 3 8B', N: 8e9, D: 15e12 },
  ];

  const loss = (N, D) => E + A / Math.pow(N, AL) + B / Math.pow(D, BE);
  const lossIso = (C, N) => loss(N, C / (6 * N));
  const nStar = (C) => Math.sqrt(C / 120);          /* N* ∝ C^0.5 */
  const dStar = (C) => C / (6 * nStar(C));          /* D* = 20·N* */

  function fmtCount(v) {
    if (v >= 1e12) return +(v / 1e12).toPrecision(2) + 'T';
    if (v >= 1e9) return +(v / 1e9).toPrecision(2) + 'B';
    return +(v / 1e6).toPrecision(2) + 'M';
  }

  function init() {
    const P = DL.plot;
    const $ = (id) => document.getElementById(id);

    const cvIso = $('s4-iso'), cvOpt = $('s4-opt');
    const slC = $('s4-c');

    let logC = +slC.value;
    let userLogN = null;     /* null = ponto acompanha o ótimo */
    let frIso = null, dragging = false;

    const C = () => Math.pow(10, logC);
    const curLogN = () => {
      const c = Math.log10(nStar(C()));
      if (userLogN === null) return c;
      return Math.max(c - HALF, Math.min(c + HALF, userLogN));
    };

    function drawIso() {
      const th = P.theme();
      const { ctx, w, h } = P.setup(cvIso);
      P.clear(ctx, w, h);
      const c = C();
      const lc = Math.log10(nStar(c));
      const x0 = lc - HALF, x1 = lc + HALF;

      /* Janela vertical: do mínimo até o valor nas bordas */
      const lmin = lossIso(c, nStar(c));
      const ledge = lossIso(c, Math.pow(10, x0));
      const y0 = lmin - 0.06 * (ledge - lmin);
      const y1 = ledge + 0.10 * (ledge - lmin);
      const fr = P.frame(ctx, w, h, x0, x1, y0, y1, { l: 44, r: 12, t: 12, b: 26 });
      frIso = fr;
      P.axes(ctx, fr, {
        xlabel: 'log₁₀ N', ylabel: 'L',
        xticks: [Math.ceil(x0), Math.round(lc), Math.floor(x1)],
        yticks: [+lmin.toFixed(2), +((lmin + ledge) / 2).toFixed(2), +ledge.toFixed(2)],
      });

      const M = 90;
      const xs = [], ysMain = [], ysLo = [], ysHi = [];
      for (let k = 0; k <= M; k++) {
        const lx = x0 + (k / M) * (x1 - x0);
        const N = Math.pow(10, lx);
        xs.push(lx);
        ysMain.push(lossIso(c, N));
        ysLo.push(lossIso(c / 30, N));
        ysHi.push(lossIso(c * 30, N));
      }
      /* Orçamentos vizinhos (contexto): C/30 e C·30 */
      P.line(ctx, fr, xs, ysLo, th.comment, { width: 1.2, alpha: 0.45, dash: [4, 4] });
      P.line(ctx, fr, xs, ysHi, th.comment, { width: 1.2, alpha: 0.45, dash: [4, 4] });
      P.line(ctx, fr, xs, ysMain, th.purple, { width: 2.2 });
      P.label(ctx, fr.X(x1) - 64, fr.Y(ysLo[M]) - 6, 'C / 30', th.comment);
      P.label(ctx, fr.X(x1) - 64, fr.Y(ysHi[M]) + 14, 'C × 30', th.comment);

      /* Ótimo (N*) e a regra 20N (coincidem por construção; o rótulo reforça) */
      P.scatter(ctx, fr, [[lc, lmin]], th.green, { r: 5, alpha: 1 });
      P.label(ctx, fr.X(lc) + 8, fr.Y(lmin) + 16, 'ótimo: N* (D* = 20·N*)', th.green);

      /* Ponto do aluno (arrastável) */
      const ux = curLogN();
      const uy = lossIso(c, Math.pow(10, ux));
      ctx.beginPath();
      ctx.arc(fr.X(ux), fr.Y(uy), 7, 0, 2 * Math.PI);
      ctx.fillStyle = th.bg;
      ctx.fill();
      ctx.strokeStyle = th.fg;
      ctx.lineWidth = 2;
      ctx.stroke();
      P.label(ctx, fr.X(ux) + 10, fr.Y(uy) - 8, 'N = ' + fmtCount(Math.pow(10, ux)), th.fg);
    }

    function drawOpt() {
      const th = P.theme();
      const { ctx, w, h } = P.setup(cvOpt);
      P.clear(ctx, w, h);
      const fr = P.frame(ctx, w, h, 20, 26, 8, 14, { l: 40, r: 10, t: 12, b: 26 });
      P.axes(ctx, fr, {
        xlabel: 'log₁₀ C', ylabel: 'log₁₀ (N, D)',
        xticks: [20, 22, 24, 26], yticks: [8, 10, 12, 14],
      });
      const xs = [], yn = [], yd = [];
      for (let k = 0; k <= 60; k++) {
        const lx = 20 + (k / 60) * 6;
        const c = Math.pow(10, lx);
        xs.push(lx);
        yn.push(Math.log10(nStar(c)));
        yd.push(Math.log10(dStar(c)));
      }
      P.line(ctx, fr, xs, yn, th.cyan, { width: 2 });
      P.line(ctx, fr, xs, yd, th.orange, { width: 2 });
      P.label(ctx, fr.X(20.2), fr.Y(yn[0]) - 8, 'N* ∝ C^0.5', th.cyan);
      P.label(ctx, fr.X(20.2), fr.Y(yd[0]) - 8, 'D* ∝ C^0.5', th.orange);

      /* C atual */
      P.line(ctx, fr, [logC, logC], [8, 14], th.comment, { width: 1, dash: [4, 4], alpha: 0.7 });
      P.scatter(ctx, fr, [[logC, Math.log10(nStar(C()))]], th.cyan, { r: 4.5, alpha: 1 });
      P.scatter(ctx, fr, [[logC, Math.log10(dStar(C()))]], th.orange, { r: 4.5, alpha: 1 });

      /* Modelos reais: (C = 6ND, N) e (C, D) */
      for (const m of MODELS) {
        const lx = Math.log10(6 * m.N * m.D);
        P.scatter(ctx, fr, [[lx, Math.log10(m.N)]], th.cyan, { r: 3.5, alpha: 0.95 });
        P.scatter(ctx, fr, [[lx, Math.log10(m.D)]], th.orange, { r: 3.5, alpha: 0.95 });
        P.label(ctx, fr.X(lx) + 5, fr.Y(Math.log10(m.N)) + 11, m.name, th.comment);
      }
    }

    function updateReadout() {
      const c = C();
      const ns = nStar(c), ds = dStar(c);
      const uN = Math.pow(10, curLogN());
      const uD = c / (6 * uN);
      const uL = loss(uN, uD), oL = lossIso(c, ns);
      $('s4-readout').textContent =
        'N* = ' + fmtCount(ns) + ' · D* = ' + fmtCount(ds) + ' tokens (D*/N* ≈ 20)' +
        ' · seu ponto: N = ' + fmtCount(uN) + ', D = ' + fmtCount(uD) +
        ', L = ' + uL.toFixed(3) + ' (ótimo: ' + oL.toFixed(3) + ')';
    }

    function redraw() {
      drawIso();
      drawOpt();
      updateReadout();
    }

    cvIso.addEventListener('pointerdown', (e) => {
      if (!frIso) return;
      dragging = true;
      cvIso.setPointerCapture(e.pointerId);
      const r = cvIso.getBoundingClientRect();
      userLogN = frIso.invX(e.clientX - r.left);
      redraw();
    });
    cvIso.addEventListener('pointermove', (e) => {
      if (!dragging || !frIso) return;
      const r = cvIso.getBoundingClientRect();
      userLogN = frIso.invX(e.clientX - r.left);
      redraw();
    });
    cvIso.addEventListener('pointerup', () => { dragging = false; });

    slC.addEventListener('input', () => {
      logC = +slC.value;
      $('s4-c-val').textContent = logC.toFixed(1);
      redraw();
    });
    $('s4-reset').addEventListener('click', () => { userLogN = null; redraw(); });

    redraw();
    P.onRedraw(redraw);
    P.observeResize(cvIso, redraw);
    P.observeResize(cvOpt, redraw);
  }

  DL.sections.push({ name: 's4-scaling-laws', init });
})();
