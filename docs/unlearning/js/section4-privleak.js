/* Seção 4: PrivLeak estimado por um ataque de inferência de pertencimento.
   O atacante prevê "membro" quando a loss é baixa. A força do ataque é a AUC da ROC,
   AUC = #{pares com loss(membro) < loss(não-membro)} / (n_in·n_out). Ideal ≈ 0,5. */
(function () {
  'use strict';
  window.DL = window.DL || {};
  DL.sections = DL.sections || [];

  const fm = (v, d) => v.toFixed(d == null ? 2 : d).replace('.', ',');

  const PRESETS = {
    leak: { mem: [0.10, 0.20, 0.30, 0.65], non: [0.50, 0.60, 0.70, 0.80] },
    priv: { mem: [0.30, 0.45, 0.60, 0.75], non: [0.35, 0.50, 0.55, 0.70] },
  };

  function init() {
    if (!document.getElementById('s4-loss')) return;
    const P = DL.plot;
    const $ = (id) => document.getElementById(id);

    const cvLoss = $('s4-loss'), cvRoc = $('s4-roc');

    let MEM = PRESETS.leak.mem.slice();
    let NON = PRESETS.leak.non.slice();
    let frLoss = null, drag = null;

    const LANE_M = 0.66, LANE_N = 0.33;

    /* AUC por contagem de pares (membro com loss menor conta como acerto). */
    function auc() {
      let c = 0;
      for (const m of MEM) for (const n of NON) {
        if (m < n) c += 1; else if (m === n) c += 0.5;
      }
      return c / (MEM.length * NON.length);
    }

    /* Pontos da ROC: prevê membro se loss < limiar, varrendo o limiar. */
    function rocPoints() {
      const thrs = Array.from(new Set(MEM.concat(NON))).sort((a, b) => a - b);
      thrs.push(1.01);
      const pts = [[0, 0]];
      for (const t of thrs) {
        let tp = 0, fp = 0;
        for (const m of MEM) if (m < t) tp++;
        for (const n of NON) if (n < t) fp++;
        pts.push([fp / NON.length, tp / MEM.length]);
      }
      pts.push([1, 1]);
      pts.sort((a, b) => (a[0] - b[0]) || (a[1] - b[1]));
      return pts;
    }

    /* ── Losses arrastáveis ── */
    function drawLoss() {
      const th = P.theme();
      const { ctx, w, h } = P.setup(cvLoss);
      P.clear(ctx, w, h);
      const fr = P.frame(ctx, w, h, 0, 1, 0, 1, { l: 18, r: 14, t: 28, b: 24 });
      frLoss = fr;

      ctx.strokeStyle = th.line;
      ctx.lineWidth = 1;
      for (const lane of [LANE_M, LANE_N]) {
        ctx.beginPath();
        ctx.moveTo(fr.X(0), fr.Y(lane));
        ctx.lineTo(fr.X(1), fr.Y(lane));
        ctx.stroke();
      }
      ctx.fillStyle = th.comment;
      ctx.font = '11px sans-serif';
      ctx.textAlign = 'center';
      for (const v of [0, 0.25, 0.5, 0.75, 1]) ctx.fillText(fm(v), fr.X(v), fr.Y(0) + 6);
      P.mathText(ctx, 'loss →', fr.X(1), fr.Y(0) + 18, th.comment, 'right', 10);

      P.mathText(ctx, 'membros (in)', fr.X(0) + 2, fr.Y(LANE_M) - 12, th.orange, 'left', 11);
      P.mathText(ctx, 'não-membros (out)', fr.X(0) + 2, fr.Y(LANE_N) - 12, th.cyan, 'left', 11);

      for (const m of MEM) P.scatter(ctx, fr, [[m, LANE_M]], th.orange, { r: 6, alpha: 0.95 });
      for (const n of NON) P.scatter(ctx, fr, [[n, LANE_N]], th.cyan, { r: 6, alpha: 0.95 });
    }

    /* ── Curva ROC ── */
    function drawRoc() {
      const th = P.theme();
      const { ctx, w, h } = P.setup(cvRoc);
      P.clear(ctx, w, h);
      const fr = P.frame(ctx, w, h, 0, 1, 0, 1);
      P.axes(ctx, fr, { xlabel: 'FPR', ylabel: 'TPR', xticks: [0, 0.5, 1], yticks: [0, 0.5, 1] });

      /* Diagonal de referência (AUC = 0,5). */
      P.line(ctx, fr, [0, 1], [0, 1], th.comment, { width: 1.2, dash: [4, 4] });

      const pts = rocPoints();
      const xs = pts.map((p) => p[0]), ys = pts.map((p) => p[1]);
      /* Área sob a ROC levemente sombreada. */
      ctx.save();
      ctx.beginPath();
      ctx.moveTo(fr.X(0), fr.Y(0));
      for (let i = 0; i < pts.length; i++) ctx.lineTo(fr.X(xs[i]), fr.Y(ys[i]));
      ctx.lineTo(fr.X(1), fr.Y(0));
      ctx.closePath();
      ctx.fillStyle = th.purple;
      ctx.globalAlpha = 0.12;
      ctx.fill();
      ctx.restore();

      P.line(ctx, fr, xs, ys, th.purple, { width: 2.2 });
      P.scatter(ctx, fr, pts, th.purple, { r: 3, alpha: 0.9 });
      P.mathText(ctx, 'AUC = ' + fm(auc(), 3), fr.X(0.40), fr.Y(0.10), th.purple, 'left', 12);
    }

    function updateReadout() {
      const a = auc();
      const good = Math.abs(a - 0.5) <= 0.1;
      $('s4-readout').textContent =
        'AUC = ' + fm(a, 3) + ' · alvo: AUC ≈ 0,5 (faixa 0,40 a 0,60)';
      const badge = $('s4-verdict');
      badge.textContent = good ? 'sem vazamento (AUC ≈ 0,5)' : 'PrivLeak: pertencimento vaza';
      badge.className = 'un-badge ' + (good ? 'ok' : 'warn');
    }

    function redraw() { drawLoss(); drawRoc(); updateReadout(); }

    function hitTest(px, py) {
      let best = null, bestD = 14;
      const test = (arr, lane) => {
        for (let i = 0; i < arr.length; i++) {
          const d = Math.hypot(px - frLoss.X(arr[i]), py - frLoss.Y(lane));
          if (d < bestD) { bestD = d; best = { arr, i }; }
        }
      };
      test(MEM, LANE_M);
      test(NON, LANE_N);
      return best;
    }

    cvLoss.addEventListener('pointerdown', (e) => {
      if (!frLoss) return;
      const r = cvLoss.getBoundingClientRect();
      drag = hitTest(e.clientX - r.left, e.clientY - r.top);
      if (drag) cvLoss.setPointerCapture(e.pointerId);
    });
    cvLoss.addEventListener('pointermove', (e) => {
      if (!drag || !frLoss) return;
      const r = cvLoss.getBoundingClientRect();
      drag.arr[drag.i] = Math.max(0.02, Math.min(0.98, frLoss.invX(e.clientX - r.left)));
      redraw();
    });
    cvLoss.addEventListener('pointerup', () => { drag = null; });

    const apply = (key) => { MEM = PRESETS[key].mem.slice(); NON = PRESETS[key].non.slice(); redraw(); };
    $('s4-leak').addEventListener('click', () => apply('leak'));
    $('s4-private').addEventListener('click', () => apply('priv'));
    $('s4-reset').addEventListener('click', () => apply('leak'));

    redraw();
    P.onRedraw(redraw);
    P.observeResize(cvLoss, redraw);
    P.observeResize(cvRoc, redraw);
  }

  DL.sections.push({ name: 's4-privleak', init });
})();
