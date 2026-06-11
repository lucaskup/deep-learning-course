/* Seção 1: mudança de variável em 1D, x = f(z) monotônica ajustável sobre
   z ~ N(0,1), com o termo do jacobiano visível (p_X = p_Z / |f'|). */
(function () {
  'use strict';
  window.DL = window.DL || {};
  DL.sections = DL.sections || [];

  const ZMIN = -3.5, ZMAX = 3.5, W = 0.7, NSAMP = 3000, NGRID = 400;

  function init() {
    const P = DL.plot;
    const U = DL.utils;
    const $ = (id) => document.getElementById(id);

    const cvBase = $('s1-base'), cvMap = $('s1-map'), cvOut = $('s1-out');
    const slA = $('s1-a'), slB = $('s1-b'), slC = $('s1-c');

    let a = +slA.value, b = +slB.value, c = +slC.value;
    let z0 = 1.0;
    let seed = 1;
    let zs = [];
    let frMap = null, dragging = false;

    const f = (z) => b + a * z + c * Math.tanh(z / W);
    function fp(z) {
      const th = Math.tanh(z / W);
      return a + (c / W) * (1 - th * th);
    }

    function resample() {
      const rng = U.mulberry32(seed);
      zs = [];
      for (let i = 0; i < NSAMP; i++) zs.push(U.randn(rng));
    }

    /* Grade de z e curvas paramétricas (x = f(z), densidades). */
    function gridData() {
      const z = [], x = [], pz = [], px = [], dz = (ZMAX - ZMIN) / (NGRID - 1);
      for (let i = 0; i < NGRID; i++) {
        const zi = ZMIN + i * dz;
        z.push(zi);
        x.push(f(zi));
        const p = U.gaussPdf(zi, 0, 1);
        pz.push(p);
        px.push(p / fp(zi));
      }
      /* Área da curva sem jacobiano: ∫ p_Z(f⁻¹(x)) dx = ∫ p_Z(z)·f'(z) dz. */
      let areaNoJac = 0;
      for (let i = 1; i < NGRID; i++) {
        areaNoJac += 0.5 * (pz[i] * fp(z[i]) + pz[i - 1] * fp(z[i - 1])) * dz;
      }
      return { z, x, pz, px, areaNoJac };
    }

    function xRange() {
      const lo = f(ZMIN), hi = f(ZMAX);
      const m = 0.06 * (hi - lo);
      return [lo - m, hi + m];
    }

    function drawBase() {
      const th = P.theme();
      const { ctx, w, h } = P.setup(cvBase);
      P.clear(ctx, w, h);
      const fr = P.frame(ctx, w, h, ZMIN, ZMAX, 0, 0.5);
      P.axes(ctx, fr, { xlabel: 'z', ylabel: 'p(z)', xticks: [-3, 0, 3], yticks: [0, 0.2, 0.4] });
      /* Amostras como pontinhos junto ao eixo */
      const dots = [];
      for (let i = 0; i < 350; i++) dots.push([zs[i], 0.012 + 0.012 * (i % 3)]);
      P.scatter(ctx, fr, dots, th.cyan, { r: 1.2, alpha: 0.3 });
      const g = gridData();
      P.line(ctx, fr, g.z, g.pz, th.cyan, { width: 2 });
      /* Marcador z0 */
      P.line(ctx, fr, [z0, z0], [0, 0.48], th.fg, { width: 1, dash: [4, 4], alpha: 0.6 });
      P.label(ctx, fr.X(z0) + 5, fr.Y(0.45), 'z_0', th.fg);
      P.label(ctx, fr.X(ZMIN) + 8, fr.Y(0.5) + 12, 'p_Z(z) = N(0,1)', th.cyan);
    }

    function drawMap() {
      const th = P.theme();
      const { ctx, w, h } = P.setup(cvMap);
      P.clear(ctx, w, h);
      const [xlo, xhi] = xRange();
      const fr = P.frame(ctx, w, h, ZMIN, ZMAX, xlo, xhi);
      frMap = fr;
      P.axes(ctx, fr, {
        xlabel: 'z', ylabel: 'x = f(z)',
        xticks: [-3, 0, 3],
        yticks: [Math.ceil(xlo), 0, Math.floor(xhi)],
      });
      /* Identidade de referência */
      P.line(ctx, fr, [ZMIN, ZMAX], [ZMIN, ZMAX], th.comment, { width: 1, dash: [4, 4], alpha: 0.5 });
      const g = gridData();
      P.line(ctx, fr, g.z, g.x, th.orange, { width: 2.2 });
      /* Guias e ponto arrastável em z0 */
      const x0 = f(z0);
      P.line(ctx, fr, [z0, z0], [xlo, x0], th.cyan, { width: 1, dash: [3, 3], alpha: 0.7 });
      P.line(ctx, fr, [ZMIN, z0], [x0, x0], th.green, { width: 1, dash: [3, 3], alpha: 0.7 });
      ctx.beginPath();
      ctx.arc(fr.X(z0), fr.Y(x0), 6, 0, 2 * Math.PI);
      ctx.fillStyle = th.fg;
      ctx.fill();
      P.label(ctx, fr.X(z0) + 9, fr.Y(x0) - 8, "f'(z_0) = " + fp(z0).toFixed(2), th.fg);
      P.label(ctx, fr.X(ZMIN) + 8, fr.Y(xhi) + 12, 'x = f(z)', th.orange);
    }

    function drawOut() {
      const th = P.theme();
      const { ctx, w, h } = P.setup(cvOut);
      P.clear(ctx, w, h);
      const [xlo, xhi] = xRange();
      const g = gridData();
      let ymax = 0.45;
      for (const v of g.px) if (v > ymax) ymax = v;
      ymax = Math.min(3.5, ymax * 1.12);
      const fr = P.frame(ctx, w, h, xlo, xhi, 0, ymax);
      P.axes(ctx, fr, {
        xlabel: 'x', ylabel: 'p(x)',
        xticks: [Math.ceil(xlo), 0, Math.floor(xhi)],
        yticks: [0, +(ymax / 2).toFixed(1)],
      });
      /* Histograma das amostras transformadas */
      const xsamp = zs.map(f);
      P.histogram(ctx, fr, xsamp, 60, th.orange, { alpha: 0.25, range: [xlo, xhi] });
      /* Sem jacobiano (tracejada) e com jacobiano (sólida) */
      P.line(ctx, fr, g.x, g.pz, th.pink, { width: 1.6, dash: [5, 4], alpha: 0.9 });
      P.line(ctx, fr, g.x, g.px, th.orange, { width: 2.2 });
      /* Marcador x0 = f(z0) */
      const x0 = f(z0);
      P.line(ctx, fr, [x0, x0], [0, ymax * 0.95], th.fg, { width: 1, dash: [4, 4], alpha: 0.6 });
      P.label(ctx, fr.X(x0) + 5, fr.Y(ymax * 0.9), 'x_0', th.fg);
      P.label(ctx, fr.X(xlo) + 8, fr.Y(ymax) + 12, 'p_X(x) = p_Z(z)/|f′(z)|', th.orange);
      P.label(ctx, fr.X(xlo) + 8, fr.Y(ymax) + 26, 'sem jacobiano', th.pink);
    }

    function updateReadout() {
      const g = gridData();
      const d = fp(z0);
      $('s1-readout').textContent =
        'z₀ = ' + z0.toFixed(2) +
        ' · f′(z₀) = ' + d.toFixed(2) +
        ' · p_X(x₀) = ' + (U.gaussPdf(z0, 0, 1) / d).toFixed(3) +
        ' · área: com jacobiano = 1.00, sem = ' + g.areaNoJac.toFixed(2);
    }

    function redraw() {
      drawBase();
      drawMap();
      drawOut();
      updateReadout();
    }

    function bindSlider(sl, valId, set, fmtFn) {
      sl.addEventListener('input', () => {
        set(+sl.value);
        $(valId).textContent = fmtFn(+sl.value);
        redraw();
      });
    }
    bindSlider(slA, 's1-a-val', (v) => { a = v; }, (v) => v.toFixed(2));
    bindSlider(slB, 's1-b-val', (v) => { b = v; }, (v) => v.toFixed(1));
    bindSlider(slC, 's1-c-val', (v) => { c = v; }, (v) => v.toFixed(2));

    $('s1-resample').addEventListener('click', () => {
      seed = (seed * 1103515245 + 12345) % 2147483647;
      resample();
      redraw();
    });

    function setZ0(px) {
      if (!frMap) return;
      z0 = Math.max(ZMIN + 0.05, Math.min(ZMAX - 0.05, frMap.invX(px)));
      redraw();
    }
    cvMap.addEventListener('pointerdown', (e) => {
      dragging = true;
      cvMap.setPointerCapture(e.pointerId);
      setZ0(e.clientX - cvMap.getBoundingClientRect().left);
    });
    cvMap.addEventListener('pointermove', (e) => {
      if (!dragging) return;
      setZ0(e.clientX - cvMap.getBoundingClientRect().left);
    });
    cvMap.addEventListener('pointerup', () => { dragging = false; });

    resample();
    redraw();
    P.onRedraw(redraw);
    P.observeResize(cvBase, redraw);
    P.observeResize(cvMap, redraw);
    P.observeResize(cvOut, redraw);
  }

  DL.sections.push({ name: 's1-change-of-variable', init });
})();
