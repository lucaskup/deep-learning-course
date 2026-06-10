/* Seção 2: portas lógicas com um Perceptron (AND, OR e o problema XOR). */
(function () {
  'use strict';
  window.DL = window.DL || {};
  DL.sections = DL.sections || [];

  const R = 1.8;
  const INPUTS = [[-1, -1], [-1, 1], [1, -1], [1, 1]];
  const TARGETS = {
    and: [-1, -1, -1, 1],
    or: [-1, 1, 1, 1],
    xor: [-1, 1, 1, -1],
  };
  /* Soluções dos slides: θ = [θ₀, θ₁, θ₂] */
  const PRESETS = { and: [-1.5, 1, 1], or: [0.5, 1, 1] };

  function init() {
    const U = DL.utils, P = DL.plot;
    const $ = (id) => document.getElementById(id);

    const cv = $('s2-plane');
    const sl = { t1: $('s2-t1'), t2: $('s2-t2'), t0: $('s2-t0') };
    const tabs = { and: $('s2-tab-and'), or: $('s2-tab-or'), xor: $('s2-tab-xor') };

    let gate = 'and';
    let anim = null;

    function theta() { return [+sl.t0.value, +sl.t1.value, +sl.t2.value]; }
    function preact(th, x1, x2) { return th[0] + th[1] * x1 + th[2] * x2; }
    function predict(th, x1, x2) { return preact(th, x1, x2) >= 0 ? 1 : -1; }
    function fmtNum(v) { return (v < 0 ? '−' : '') + Math.abs(v).toFixed(1); }

    function arrow(ctx, fr, x0, y0, x1, y1, color, width, alpha) {
      const px0 = fr.X(x0), py0 = fr.Y(y0), px1 = fr.X(x1), py1 = fr.Y(y1);
      const ang = Math.atan2(py1 - py0, px1 - px0);
      const hs = 5 + 2 * width;
      ctx.save();
      ctx.globalAlpha = alpha != null ? alpha : 1;
      ctx.strokeStyle = color;
      ctx.fillStyle = color;
      ctx.lineWidth = width;
      ctx.beginPath();
      ctx.moveTo(px0, py0);
      ctx.lineTo(px1, py1);
      ctx.stroke();
      ctx.beginPath();
      ctx.moveTo(px1, py1);
      ctx.lineTo(px1 - hs * Math.cos(ang - 0.45), py1 - hs * Math.sin(ang - 0.45));
      ctx.lineTo(px1 - hs * Math.cos(ang + 0.45), py1 - hs * Math.sin(ang + 0.45));
      ctx.closePath();
      ctx.fill();
      ctx.restore();
    }

    function drawPlane() {
      const th = P.theme();
      const t = theta();
      const targets = TARGETS[gate];
      const { ctx, w, h } = P.setup(cv);
      P.clear(ctx, w, h);
      const fr = P.frame(ctx, w, h, -R, R, -R, R);

      const cp = P.rgb(th.cyan), cn = P.rgb(th.orange);
      P.heatmap(ctx, fr, (x, y) => preact(t, x, y), 60, 60,
        (v) => (v >= 0 ? [cp[0], cp[1], cp[2], 30] : [cn[0], cn[1], cn[2], 30]));
      P.axes(ctx, fr, { xlabel: 'x₁', ylabel: 'x₂', xticks: [-1, 0, 1], yticks: [-1, 0, 1] });

      const n2 = t[1] * t[1] + t[2] * t[2];
      if (n2 > 1e-6) {
        ctx.save();
        ctx.beginPath();
        ctx.rect(fr.X(fr.xmin), fr.Y(fr.ymax), fr.iw, fr.ih);
        ctx.clip();
        const nrm = Math.sqrt(n2);
        const fx = -t[0] * t[1] / n2, fy = -t[0] * t[2] / n2;
        const ux = t[1] / nrm, uy = t[2] / nrm;
        const vx = -uy, vy = ux;
        const L = 3 * R;
        P.line(ctx, fr, [fx - L * vx, fx + L * vx], [fy - L * vy, fy + L * vy], th.fg, { width: 1.8 });
        arrow(ctx, fr, fx, fy, fx + 0.7 * ux, fy + 0.7 * uy, th.green, 2.2);
        P.label(ctx, fr.X(fx + 0.9 * ux), fr.Y(fy + 0.9 * uy) + 4, '(θ₁, θ₂)', th.green, 'center');
        ctx.restore();
      }

      /* Quatro entradas: preenchimento pela saída desejada, contorno vermelho se errada */
      for (let i = 0; i < 4; i++) {
        const [x1, x2] = INPUTS[i];
        const px = fr.X(x1), py = fr.Y(x2);
        ctx.fillStyle = targets[i] === 1 ? th.cyan : th.orange;
        ctx.beginPath();
        ctx.arc(px, py, 10, 0, 2 * Math.PI);
        ctx.fill();
        if (predict(t, x1, x2) !== targets[i]) {
          ctx.strokeStyle = th.red;
          ctx.lineWidth = 2.5;
          ctx.setLineDash([4, 3]);
          ctx.beginPath();
          ctx.arc(px, py, 14, 0, 2 * Math.PI);
          ctx.stroke();
          ctx.setLineDash([]);
        }
      }
      P.label(ctx, fr.X(-R) + 8, 16, 'y = +1', th.cyan);
      P.label(ctx, fr.X(-R) + 8, 30, 'y = −1', th.orange);
    }

    function updateTable() {
      const t = theta();
      const targets = TARGETS[gate];
      let correct = 0;
      let html = '<table class="truth-table"><thead><tr>' +
        '<th>x₁</th><th>x₂</th><th>y</th><th>xθ<sup>⊤</sup></th><th>ŷ</th><th></th>' +
        '</tr></thead><tbody>';
      for (let i = 0; i < 4; i++) {
        const [x1, x2] = INPUTS[i];
        const z = preact(t, x1, x2);
        const yhat = z >= 0 ? 1 : -1;
        const ok = yhat === targets[i];
        if (ok) correct++;
        html += '<tr>' +
          '<td>' + x1 + '</td><td>' + x2 + '</td>' +
          '<td class="' + (targets[i] === 1 ? 'pos' : 'neg') + '">' + (targets[i] === 1 ? '+1' : '−1') + '</td>' +
          '<td class="formula-cell">' + fmtNum(z) + '</td>' +
          '<td class="' + (yhat === 1 ? 'pos' : 'neg') + '">' + (yhat === 1 ? '+1' : '−1') + '</td>' +
          '<td class="' + (ok ? 'ok' : 'bad') + '">' + (ok ? '✓' : '✗') + '</td>' +
          '</tr>';
      }
      html += '</tbody></table>';
      $('s2-table').innerHTML = html;
      $('s2-score').textContent = 'corretas: ' + correct + ' / 4' +
        (gate === 'xor' ? ' (máximo de uma reta: 3)' : '');
    }

    function redraw() {
      const t = theta();
      $('s2-t0-val').textContent = fmtNum(t[0]);
      $('s2-t1-val').textContent = fmtNum(t[1]);
      $('s2-t2-val').textContent = fmtNum(t[2]);
      drawPlane();
      updateTable();
    }

    /* Anima os sliders até uma configuração alvo (θ₀, θ₁, θ₂). */
    function animateTo(target) {
      if (anim) { DL.stopTicker(anim); anim = null; }
      const from = theta();
      let prog = 0;
      anim = function (dt) {
        prog = Math.min(1, prog + dt / 0.6);
        const e = 1 - (1 - prog) * (1 - prog);
        sl.t0.value = from[0] + (target[0] - from[0]) * e;
        sl.t1.value = from[1] + (target[1] - from[1]) * e;
        sl.t2.value = from[2] + (target[2] - from[2]) * e;
        if (prog >= 1) {
          sl.t0.value = target[0]; sl.t1.value = target[1]; sl.t2.value = target[2];
          DL.stopTicker(anim);
          anim = null;
        }
        redraw();
      };
      DL.startTicker(anim);
    }

    function setGate(g) {
      gate = g;
      for (const k in tabs) tabs[k].classList.toggle('active', k === g);
      redraw();
    }

    tabs.and.addEventListener('click', () => setGate('and'));
    tabs.or.addEventListener('click', () => setGate('or'));
    tabs.xor.addEventListener('click', () => setGate('xor'));
    $('s2-preset-and').addEventListener('click', () => animateTo(PRESETS.and));
    $('s2-preset-or').addEventListener('click', () => animateTo(PRESETS.or));
    for (const k of ['t0', 't1', 't2']) sl[k].addEventListener('input', redraw);

    redraw();
    P.onRedraw(redraw);
    P.observeResize(cv, drawPlane);
  }

  DL.sections.push({ name: 's2-gates', init });
})();
