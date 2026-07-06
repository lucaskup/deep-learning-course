/* Seção 4: RoPE em um bloco 2D. Rotação de q e k por i·θ e j·θ no plano,
   score q̃ᵢ·k̃ⱼ dependendo apenas do offset j − i (slider δ desloca a dupla). */
(function () {
  'use strict';
  window.DL = window.DL || {};
  DL.sections = DL.sections || [];

  const LIM = 1.6, OFFMAX = 16;

  function rot(v, a) {
    const c = Math.cos(a), s = Math.sin(a);
    return [c * v[0] - s * v[1], s * v[0] + c * v[1]];
  }

  function dot(a, b) { return a[0] * b[0] + a[1] * b[1]; }

  function init() {
    const P = DL.plot;
    const $ = (id) => document.getElementById(id);

    const cvPlane = $('s4-plane'), cvScore = $('s4-score');
    const slI = $('s4-i'), slJ = $('s4-j'), slTheta = $('s4-theta'), slDelta = $('s4-delta');
    const btnReset = $('s4-reset');

    let q = [1, 0], k = [0, 1];
    let i = +slI.value, j = +slJ.value, theta = +slTheta.value, delta = +slDelta.value;
    let frPlane = null, dragging = null;

    /* Forma fechada do score: (q·k)cos(Δθ) + (q₁k₀ − q₀k₁)sin(Δθ). */
    function scoreAt(off) {
      return dot(q, k) * Math.cos(off * theta) + (q[1] * k[0] - q[0] * k[1]) * Math.sin(off * theta);
    }

    function arrow(ctx, fr, v, color, dash, name) {
      const x0 = fr.X(0), y0 = fr.Y(0), x1 = fr.X(v[0]), y1 = fr.Y(v[1]);
      ctx.strokeStyle = color;
      ctx.fillStyle = color;
      ctx.lineWidth = 2;
      if (dash) ctx.setLineDash([5, 4]);
      ctx.beginPath();
      ctx.moveTo(x0, y0);
      ctx.lineTo(x1, y1);
      ctx.stroke();
      ctx.setLineDash([]);
      const ang = Math.atan2(y1 - y0, x1 - x0);
      ctx.beginPath();
      ctx.moveTo(x1, y1);
      ctx.lineTo(x1 - 9 * Math.cos(ang - 0.4), y1 - 9 * Math.sin(ang - 0.4));
      ctx.lineTo(x1 - 9 * Math.cos(ang + 0.4), y1 - 9 * Math.sin(ang + 0.4));
      ctx.closePath();
      ctx.fill();
      P.mathText(ctx, name, x1 + (x1 >= x0 ? 8 : -8), y1 - 6, color, x1 >= x0 ? 'left' : 'right', 12);
    }

    function drawPlane() {
      const th = P.theme();
      const { ctx, w, h } = P.setup(cvPlane);
      P.clear(ctx, w, h);
      const fr = P.frame(ctx, w, h, -LIM, LIM, -LIM, LIM);
      frPlane = fr;
      P.axes(ctx, fr, { xlabel: 'dim 0', ylabel: 'dim 1', xticks: [-1, 0, 1], yticks: [-1, 0, 1] });
      /* Eixos centrais e círculo unitário. */
      P.line(ctx, fr, [-LIM, LIM], [0, 0], th.line, { width: 1 });
      P.line(ctx, fr, [0, 0], [-LIM, LIM], th.line, { width: 1 });
      ctx.strokeStyle = th.line;
      ctx.lineWidth = 1;
      ctx.setLineDash([3, 4]);
      ctx.beginPath();
      ctx.arc(fr.X(0), fr.Y(0), Math.abs(fr.X(1) - fr.X(0)), 0, 2 * Math.PI);
      ctx.stroke();
      ctx.setLineDash([]);

      const ie = i + delta, je = j + delta;
      const qr = rot(q, ie * theta), kr = rot(k, je * theta);
      arrow(ctx, fr, q, th.cyan, true, 'q');
      arrow(ctx, fr, k, th.orange, true, 'k');
      arrow(ctx, fr, qr, th.cyan, false, 'q̃_i');
      arrow(ctx, fr, kr, th.orange, false, 'k̃_j');
      P.label(ctx, fr.X(-LIM) + 8, fr.Y(LIM) + 14, 'ângulo q: ' + (ie * theta).toFixed(2) + ' rad', th.cyan);
      P.label(ctx, fr.X(-LIM) + 8, fr.Y(LIM) + 28, 'ângulo k: ' + (je * theta).toFixed(2) + ' rad', th.orange);
    }

    function drawScore() {
      const th = P.theme();
      const { ctx, w, h } = P.setup(cvScore);
      P.clear(ctx, w, h);
      const amp = Math.hypot(q[0], q[1]) * Math.hypot(k[0], k[1]) + 0.1;
      const fr = P.frame(ctx, w, h, -OFFMAX, OFFMAX, -amp, amp);
      P.axes(ctx, fr, {
        xlabel: 'offset j − i', ylabel: 'score',
        xticks: [-OFFMAX, 0, OFFMAX],
        yticks: [-1, 0, 1],
      });
      P.line(ctx, fr, [-OFFMAX, OFFMAX], [0, 0], th.line, { width: 1, dash: [4, 4] });
      P.line(ctx, fr, [0, 0], [-amp, amp], th.line, { width: 1, dash: [4, 4] });
      const xs = [], ys = [];
      const n = 400;
      for (let t = 0; t <= n; t++) {
        const off = -OFFMAX + (2 * OFFMAX * t) / n;
        xs.push(off);
        ys.push(scoreAt(off));
      }
      P.line(ctx, fr, xs, ys, th.purple, { width: 1.8 });
      const cur = j - i;
      P.scatter(ctx, fr, [[cur, scoreAt(cur)]], th.green, { r: 5, alpha: 1 });
      P.label(ctx, fr.X(cur), fr.Y(scoreAt(cur)) - 10, 'j − i = ' + cur, th.green, 'center');
    }

    function updateReadout() {
      const ie = i + delta, je = j + delta;
      const sShift = dot(rot(q, ie * theta), rot(k, je * theta));
      const sBase = dot(rot(q, i * theta), rot(k, j * theta));
      $('s4-readout').textContent =
        'score(i=' + ie + ', j=' + je + ') = ' + sShift.toFixed(3) +
        ' · sem δ: ' + sBase.toFixed(3) + ' · offset = ' + (j - i);
    }

    function redraw() {
      drawPlane();
      drawScore();
      updateReadout();
    }

    function dragTo(e) {
      if (!frPlane || !dragging) return;
      const r = cvPlane.getBoundingClientRect();
      let x = frPlane.invX(e.clientX - r.left);
      let y = frPlane.invY(e.clientY - r.top);
      const n = Math.hypot(x, y);
      if (n < 0.15) return;
      if (n > 1.45) { x *= 1.45 / n; y *= 1.45 / n; }
      if (dragging === 'q') q = [x, y]; else k = [x, y];
      redraw();
    }

    cvPlane.addEventListener('pointerdown', (e) => {
      if (!frPlane) return;
      const r = cvPlane.getBoundingClientRect();
      const px = e.clientX - r.left, py = e.clientY - r.top;
      const dq = Math.hypot(px - frPlane.X(q[0]), py - frPlane.Y(q[1]));
      const dk = Math.hypot(px - frPlane.X(k[0]), py - frPlane.Y(k[1]));
      if (Math.min(dq, dk) > 28) return;
      dragging = dq <= dk ? 'q' : 'k';
      cvPlane.setPointerCapture(e.pointerId);
      dragTo(e);
    });
    cvPlane.addEventListener('pointermove', dragTo);
    cvPlane.addEventListener('pointerup', () => { dragging = null; });

    slI.addEventListener('input', () => {
      i = +slI.value;
      $('s4-i-val').textContent = i;
      redraw();
    });
    slJ.addEventListener('input', () => {
      j = +slJ.value;
      $('s4-j-val').textContent = j;
      redraw();
    });
    slTheta.addEventListener('input', () => {
      theta = +slTheta.value;
      $('s4-theta-val').textContent = theta.toFixed(2);
      redraw();
    });
    slDelta.addEventListener('input', () => {
      delta = +slDelta.value;
      $('s4-delta-val').textContent = delta;
      redraw();
    });
    btnReset.addEventListener('click', () => {
      q = [1, 0];
      k = [0, 1];
      i = 1; j = 3; theta = 1; delta = 0;
      slI.value = i; slJ.value = j; slTheta.value = theta; slDelta.value = delta;
      $('s4-i-val').textContent = i;
      $('s4-j-val').textContent = j;
      $('s4-theta-val').textContent = theta.toFixed(2);
      $('s4-delta-val').textContent = delta;
      redraw();
    });

    redraw();
    P.onRedraw(redraw);
    P.observeResize(cvPlane, redraw);
    P.observeResize(cvScore, redraw);
  }

  DL.sections.push({ name: 's4-rope', init });
})();
