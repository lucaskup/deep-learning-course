/* Seção 3: funções de custo na saída sigmoide. Compara a entropia cruzada
   binária (NLL da Bernoulli, slides) com o MSE aplicado após a sigmoide,
   em função do logit z, para um único exemplo com alvo y ∈ {0, 1}. */
(function () {
  'use strict';
  window.DL = window.DL || {};
  DL.sections = DL.sections || [];

  const ZMIN = -8, ZMAX = 8, NPTS = 321;

  function init() {
    const P = DL.plot;
    const $ = (id) => document.getElementById(id);

    const cvLoss = $('s3-loss'), cvGrad = $('s3-grad');
    const tabY1 = $('s3-y1'), tabY0 = $('s3-y0');

    let y = 1;
    let z0 = -5;
    let frLoss = null, frGrad = null;

    function sigm(z) { return 1 / (1 + Math.exp(-z)); }
    /* ℒ_BCE = −[y·log σ + (1−y)·log(1−σ)];  dℒ_BCE/dz = σ − y */
    function lossCE(z) { const s = sigm(z); return -(y * Math.log(s) + (1 - y) * Math.log(1 - s)); }
    function gradCE(z) { return sigm(z) - y; }
    /* ℒ_MSE = (σ − y)²;  dℒ_MSE/dz = 2(σ − y)·σ(1−σ) */
    function lossMSE(z) { const s = sigm(z); return (s - y) * (s - y); }
    function gradMSE(z) { const s = sigm(z); return 2 * (s - y) * s * (1 - s); }

    function redraw() {
      const th = P.theme();
      const zs = [], ce = [], mse = [], gce = [], gmse = [];
      for (let i = 0; i < NPTS; i++) {
        const z = ZMIN + (ZMAX - ZMIN) * i / (NPTS - 1);
        zs.push(z);
        ce.push(lossCE(z));
        mse.push(lossMSE(z));
        gce.push(Math.abs(gradCE(z)));
        gmse.push(Math.abs(gradMSE(z)));
      }

      /* ── ℒ(z) ── */
      let { ctx, w, h } = P.setup(cvLoss);
      P.clear(ctx, w, h);
      const lmax = 8.5;
      frLoss = P.frame(ctx, w, h, ZMIN, ZMAX, 0, lmax);
      P.axes(ctx, frLoss, { xlabel: 'z', xticks: [-8, -4, 0, 4, 8], yticks: [0, 4, 8] });
      P.line(ctx, frLoss, zs, ce, th.purple, { width: 2.2 });
      P.line(ctx, frLoss, zs, mse, th.orange, { width: 2.2 });
      P.label(ctx, frLoss.X(ZMIN) + 10, frLoss.Y(lmax) + 14, 'ℒ_{BCE}', th.purple);
      P.label(ctx, frLoss.X(ZMIN) + 10, frLoss.Y(lmax) + 30, 'ℒ_{MSE}', th.orange);
      P.line(ctx, frLoss, [z0, z0], [0, lmax], th.comment, { width: 1, dash: [3, 3] });
      P.scatter(ctx, frLoss, [[z0, Math.min(lmax, lossCE(z0))]], th.purple, { r: 5, alpha: 1 });
      P.scatter(ctx, frLoss, [[z0, lossMSE(z0)]], th.orange, { r: 5, alpha: 1 });
      P.label(ctx, frLoss.X(z0) + 8, frLoss.Y(lmax) + 14, 'z_0', th.pink);

      /* ── |dℒ/dz| ── */
      ({ ctx, w, h } = P.setup(cvGrad));
      P.clear(ctx, w, h);
      frGrad = P.frame(ctx, w, h, ZMIN, ZMAX, 0, 1.05);
      P.axes(ctx, frGrad, { xlabel: 'z', xticks: [-8, -4, 0, 4, 8], yticks: [0, 0.5, 1] });
      P.line(ctx, frGrad, zs, gce, th.purple, { width: 2 });
      P.line(ctx, frGrad, zs, gmse, th.orange, { width: 2 });
      P.line(ctx, frGrad, [z0, z0], [0, 1.05], th.comment, { width: 1, dash: [3, 3] });
      P.scatter(ctx, frGrad, [[z0, Math.abs(gradCE(z0))]], th.purple, { r: 4, alpha: 1 });
      P.scatter(ctx, frGrad, [[z0, Math.abs(gradMSE(z0))]], th.orange, { r: 4, alpha: 1 });

      $('s3-readout').innerHTML =
        'y = ' + y +
        ' · z<sub>0</sub> = ' + z0.toFixed(2) +
        ' · σ(z<sub>0</sub>) = ' + sigm(z0).toFixed(3) +
        ' · ℒ<sub>BCE</sub> = ' + lossCE(z0).toFixed(3) +
        ' · ℒ<sub>MSE</sub> = ' + lossMSE(z0).toFixed(3) +
        ' · |dℒ<sub>BCE</sub>/dz| = ' + Math.abs(gradCE(z0)).toFixed(3) +
        ' · |dℒ<sub>MSE</sub>/dz| = ' + Math.abs(gradMSE(z0)).toFixed(3);
    }

    function setTarget(val) {
      y = val;
      tabY1.classList.toggle('active', y === 1);
      tabY0.classList.toggle('active', y === 0);
      redraw();
    }
    tabY1.addEventListener('click', () => setTarget(1));
    tabY0.addEventListener('click', () => setTarget(0));

    /* Arrasto do marcador z0 nos dois canvases. */
    function attachDrag(canvas, getFrame) {
      let dragging = false;
      const move = (e) => {
        const fr = getFrame();
        if (!fr) return;
        const r = canvas.getBoundingClientRect();
        z0 = Math.max(ZMIN, Math.min(ZMAX, fr.invX(e.clientX - r.left)));
        redraw();
      };
      canvas.addEventListener('pointerdown', (e) => {
        dragging = true;
        canvas.setPointerCapture(e.pointerId);
        move(e);
      });
      canvas.addEventListener('pointermove', (e) => { if (dragging) move(e); });
      canvas.addEventListener('pointerup', () => { dragging = false; });
      canvas.addEventListener('pointercancel', () => { dragging = false; });
    }
    attachDrag(cvLoss, () => frLoss);
    attachDrag(cvGrad, () => frGrad);

    redraw();
    P.onRedraw(redraw);
    P.observeResize(cvLoss, redraw);
    P.observeResize(cvGrad, redraw);
  }

  DL.sections.push({ name: 's3-losses', init });
})();
