/* Seção 2: bounding boxes arrastáveis com IoU, Dice e MSE ao vivo.
   Convenção dos slides: y = (x, y, h, w), origem (0,0) no canto superior
   esquerdo e (1,1) no inferior direito. */
(function () {
  'use strict';
  window.DL = window.DL || {};
  DL.sections = DL.sections || [];

  const MIN_SIZE = 0.05;
  const fm = (v, d) => v.toFixed(d == null ? 2 : d).replace('.', ',');

  function init() {
    const P = DL.plot;
    const $ = (id) => document.getElementById(id);

    const cvBoxes = $('s2-boxes'), cvCurve = $('s2-curve');
    const btnReset = $('s2-reset'), btnRand = $('s2-rand');

    /* Exemplo dos slides: A com (x, y) = (0.2, 0.1), h = 0.5, w = 0.7. */
    let A, B;
    let seed = 7;
    let fr = null, drag = null;

    function resetBoxes() {
      A = { x: 0.2, y: 0.1, w: 0.7, h: 0.5 };
      B = { x: 0.32, y: 0.32, w: 0.55, h: 0.45 };
    }

    function randBoxes() {
      seed = (seed * 16807 + 13) % 2147483647;
      const rng = DL.utils.mulberry32(seed);
      const make = function () {
        const w = 0.2 + rng() * 0.45, h = 0.2 + rng() * 0.45;
        return { x: rng() * (1 - w), y: rng() * (1 - h), w: w, h: h };
      };
      A = make();
      B = make();
    }

    function interArea(a, b) {
      const ix = Math.max(0, Math.min(a.x + a.w, b.x + b.w) - Math.max(a.x, b.x));
      const iy = Math.max(0, Math.min(a.y + a.h, b.y + b.h) - Math.max(a.y, b.y));
      return ix * iy;
    }

    function metrics() {
      const aA = A.w * A.h, aB = B.w * B.h;
      const inter = interArea(A, B);
      const union = aA + aB - inter;
      const iou = union > 0 ? inter / union : 0;
      const dice = (aA + aB) > 0 ? 2 * inter / (aA + aB) : 0;
      /* MSE dos slides (N = 1): ½[(Δx)² + (Δy)² + (Δh)² + (Δw)²] */
      const mse = 0.5 * (
        Math.pow(A.x - B.x, 2) + Math.pow(A.y - B.y, 2) +
        Math.pow(A.h - B.h, 2) + Math.pow(A.w - B.w, 2)
      );
      return { inter: inter, union: union, iou: iou, dice: dice, mse: mse };
    }

    /* Mapeamento imagem (y para baixo) → frame do plot (y para cima). */
    const Yi = (v) => fr.Y(1 - v);
    const invYi = (py) => 1 - fr.invY(py);

    function drawBox(ctx, b, color, name) {
      const x0 = fr.X(b.x), y0 = Yi(b.y);
      const bw = fr.X(b.x + b.w) - x0, bh = Yi(b.y + b.h) - y0;
      ctx.fillStyle = color;
      ctx.globalAlpha = 0.16;
      ctx.fillRect(x0, y0, bw, bh);
      ctx.globalAlpha = 1;
      ctx.strokeStyle = color;
      ctx.lineWidth = 2;
      ctx.strokeRect(x0, y0, bw, bh);
      /* Alça de redimensionamento no canto inferior direito */
      ctx.fillStyle = color;
      ctx.fillRect(x0 + bw - 5, y0 + bh - 5, 10, 10);
      /* Nome e vetor y = (x, y, h, w) */
      P.mathText(ctx, name, x0 + 4, y0 + 13, color, 'left', 12);
      const vec = '(' + fm(b.x) + ', ' + fm(b.y) + ', ' + fm(b.h) + ', ' + fm(b.w) + ')';
      P.mathText(ctx, vec, x0 + 4, y0 + bh - 6, color, 'left', 10);
    }

    function drawBoxes() {
      const th = P.theme();
      const sz = P.setup(cvBoxes);
      const ctx = sz.ctx, w = sz.w, h = sz.h;
      P.clear(ctx, w, h);
      fr = P.frame(ctx, w, h, 0, 1, 0, 1, { l: 26, r: 26, t: 20, b: 22 });

      /* Moldura da imagem em coordenadas normalizadas */
      ctx.strokeStyle = th.line;
      ctx.lineWidth = 1.5;
      ctx.strokeRect(fr.X(0), Yi(0), fr.X(1) - fr.X(0), Yi(1) - Yi(0));
      P.mathText(ctx, '(0, 0)', fr.X(0), Yi(0) - 5, th.comment, 'left', 10);
      P.mathText(ctx, '(1, 1)', fr.X(1), Yi(1) + 13, th.comment, 'right', 10);

      drawBox(ctx, A, th.red, 'ground truth A');
      drawBox(ctx, B, th.cyan, 'predição B');

      /* Interseção A ∩ B */
      const ix0 = Math.max(A.x, B.x), iy0 = Math.max(A.y, B.y);
      const ix1 = Math.min(A.x + A.w, B.x + B.w), iy1 = Math.min(A.y + A.h, B.y + B.h);
      if (ix1 > ix0 && iy1 > iy0) {
        ctx.strokeStyle = th.fg;
        ctx.lineWidth = 1.6;
        ctx.setLineDash([5, 4]);
        ctx.strokeRect(fr.X(ix0), Yi(iy0), fr.X(ix1) - fr.X(ix0), Yi(iy1) - Yi(iy0));
        ctx.setLineDash([]);
        P.mathText(ctx, 'A ∩ B', (fr.X(ix0) + fr.X(ix1)) / 2, (Yi(iy0) + Yi(iy1)) / 2 + 4, th.fg, 'center', 11);
      }
    }

    function drawCurve() {
      const th = P.theme();
      const sz = P.setup(cvCurve);
      const ctx = sz.ctx, w = sz.w, h = sz.h;
      P.clear(ctx, w, h);
      const fc = P.frame(ctx, w, h, 0, 1, 0, 1);
      P.axes(ctx, fc, { xlabel: 'IoU', ylabel: 'valor da métrica', xticks: [0, 0.5, 1], yticks: [0, 0.5, 1] });

      const n = 60;
      const xs = [], dice = [], ident = [];
      for (let i = 0; i <= n; i++) {
        const u = i / n;
        xs.push(u);
        dice.push(2 * u / (1 + u));
        ident.push(u);
      }
      P.line(ctx, fc, xs, ident, th.comment, { width: 1.2, dash: [4, 4] });
      P.line(ctx, fc, xs, dice, th.orange, { width: 2 });

      const m = metrics();
      P.line(ctx, fc, [m.iou, m.iou], [0, m.dice], th.comment, { width: 1, dash: [2, 3] });
      P.scatter(ctx, fc, [[m.iou, m.dice]], th.orange, { r: 5, alpha: 1 });
      P.scatter(ctx, fc, [[m.iou, m.iou]], th.cyan, { r: 5, alpha: 1 });
      P.mathText(ctx, 'Dice = 2·IoU/(1 + IoU)', fc.X(0.45), fc.Y(0.88), th.orange, 'left', 11);
      P.mathText(ctx, 'IoU', fc.X(0.75), fc.Y(0.6), th.cyan, 'left', 11);
    }

    function updateReadout() {
      const m = metrics();
      $('s2-readout').textContent =
        'A ∩ B = ' + fm(m.inter, 3) + ' · A ∪ B = ' + fm(m.union, 3) +
        ' · IoU = ' + fm(m.iou, 3) + ' · Dice = ' + fm(m.dice, 3) +
        ' · MSE = ' + fm(m.mse, 4);
    }

    function redraw() {
      drawBoxes();
      drawCurve();
      updateReadout();
    }

    function hitTest(px, py) {
      /* Alças primeiro (B por cima de A), depois o corpo das caixas. */
      const order = [{ b: B }, { b: A }];
      for (const e of order) {
        const hx = fr.X(e.b.x + e.b.w), hy = Yi(e.b.y + e.b.h);
        if (Math.hypot(px - hx, py - hy) < 12) return { b: e.b, kind: 'resize' };
      }
      for (const e of order) {
        const x0 = fr.X(e.b.x), x1 = fr.X(e.b.x + e.b.w);
        const y0 = Yi(e.b.y), y1 = Yi(e.b.y + e.b.h);
        if (px >= x0 && px <= x1 && py >= y0 && py <= y1) {
          return { b: e.b, kind: 'move', dx: fr.invX(px) - e.b.x, dy: invYi(py) - e.b.y };
        }
      }
      return null;
    }

    cvBoxes.addEventListener('pointerdown', function (e) {
      if (!fr) return;
      const r = cvBoxes.getBoundingClientRect();
      drag = hitTest(e.clientX - r.left, e.clientY - r.top);
      if (drag) cvBoxes.setPointerCapture(e.pointerId);
    });
    cvBoxes.addEventListener('pointermove', function (e) {
      if (!drag || !fr) return;
      const r = cvBoxes.getBoundingClientRect();
      const u = fr.invX(e.clientX - r.left), v = invYi(e.clientY - r.top);
      const b = drag.b;
      if (drag.kind === 'move') {
        b.x = Math.max(0, Math.min(1 - b.w, u - drag.dx));
        b.y = Math.max(0, Math.min(1 - b.h, v - drag.dy));
      } else {
        b.w = Math.max(MIN_SIZE, Math.min(1 - b.x, u - b.x));
        b.h = Math.max(MIN_SIZE, Math.min(1 - b.y, v - b.y));
      }
      redraw();
    });
    cvBoxes.addEventListener('pointerup', function () { drag = null; });

    btnReset.addEventListener('click', function () { resetBoxes(); redraw(); });
    btnRand.addEventListener('click', function () { randBoxes(); redraw(); });

    resetBoxes();
    redraw();
    P.onRedraw(redraw);
    P.observeResize(cvBoxes, redraw);
    P.observeResize(cvCurve, redraw);
  }

  DL.sections.push({ name: 's2-iou', init });
})();
