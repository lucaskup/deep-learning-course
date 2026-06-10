/* Seção 3: algoritmo de aprendizado do Perceptron (PLA) ao vivo. */
(function () {
  'use strict';
  window.DL = window.DL || {};
  DL.sections = DL.sections || [];

  const N = 80;
  const HIST_WINDOW = 400;
  const MODES = {
    sep: { dist: 3.2, noise: 0.7, range: 4.2 },
    nonsep: { dist: 1.1, noise: 0.95, range: 3.2 },
  };

  function init() {
    const U = DL.utils, P = DL.plot;
    const $ = (id) => document.getElementById(id);

    const cvPlane = $('s3-plane'), cvErr = $('s3-errors');
    const slEta = $('s3-eta');
    const btnStep = $('s3-step'), btnTrain = $('s3-train');
    const btnReset = $('s3-reset'), btnResample = $('s3-resample');
    const tabSep = $('s3-tab-sep'), tabNonsep = $('s3-tab-nonsep');

    let seed = 7, mode = 'sep';
    let data = null, ys = null, thetaInit = null, theta = null;
    let cursor = 0, updates = 0, errHist = [], lastUpd = null, converged = false;
    let training = false, acc = 0;

    function eta() { return +slEta.value; }
    function predict(th, p) { return th[0] + th[1] * p[0] + th[2] * p[1] >= 0 ? 1 : -1; }

    function countErrors() {
      let e = 0;
      for (let i = 0; i < N; i++) if (predict(theta, data[i]) !== ys[i]) e++;
      return e;
    }

    function regenerate() {
      const rng = U.mulberry32(seed);
      const cfg = MODES[mode];
      const d = U.blobs(N, cfg.dist, cfg.noise, rng);
      data = d.pts;
      ys = d.lab.map((c) => (c === 0 ? -1 : 1));
      thetaInit = [U.randn(rng) * 0.5, U.randn(rng) * 0.5, U.randn(rng) * 0.5];
      resetTraining();
    }

    function resetTraining() {
      stopTraining();
      theta = thetaInit.slice();
      cursor = 0; updates = 0; lastUpd = null; converged = false;
      errHist = [countErrors()];
      updateStatus();
      drawPlane();
      drawErrors();
    }

    /* Um passo do PLA: acha a próxima observação mal classificada (varredura
       cíclica a partir do cursor) e aplica θ ← θ + η·y⁽ⁱ⁾·x⁽ⁱ⁾, com x₀ = 1. */
    function step() {
      if (converged) return false;
      let idx = -1;
      for (let k = 0; k < N; k++) {
        const i = (cursor + k) % N;
        if (predict(theta, data[i]) !== ys[i]) { idx = i; break; }
      }
      if (idx === -1) { converged = true; lastUpd = null; return false; }
      cursor = (idx + 1) % N;
      const old = theta.slice();
      const x = [1, data[idx][0], data[idx][1]];
      const e = eta();
      for (let j = 0; j < 3; j++) theta[j] = old[j] + e * ys[idx] * x[j];
      lastUpd = { i: idx, old };
      updates++;
      errHist.push(countErrors());
      if (errHist.length > HIST_WINDOW + 1) errHist.shift();
      return true;
    }

    function updateStatus() {
      $('s3-status').textContent = 'atualizações: ' + updates +
        ' · mal classificados: ' + errHist[errHist.length - 1] + ' / ' + N +
        (converged ? ' · convergiu ✓' : '');
    }

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
      const R = MODES[mode].range;
      const { ctx, w, h } = P.setup(cvPlane);
      P.clear(ctx, w, h);
      const fr = P.frame(ctx, w, h, -R, R, -R, R);

      const cp = P.rgb(th.cyan), cn = P.rgb(th.orange);
      const t = theta;
      P.heatmap(ctx, fr, (x, y) => t[0] + t[1] * x + t[2] * y, 60, 60,
        (v) => (v >= 0 ? [cp[0], cp[1], cp[2], 18] : [cn[0], cn[1], cn[2], 18]));
      P.axes(ctx, fr, { xlabel: 'x₁', ylabel: 'x₂', xticks: [-3, 0, 3], yticks: [-3, 0, 3] });

      ctx.save();
      ctx.beginPath();
      ctx.rect(fr.X(fr.xmin), fr.Y(fr.ymax), fr.iw, fr.ih);
      ctx.clip();

      /* Fronteira atual */
      const n2 = t[1] * t[1] + t[2] * t[2];
      if (n2 > 1e-6) {
        const nrm = Math.sqrt(n2);
        const fx = -t[0] * t[1] / n2, fy = -t[0] * t[2] / n2;
        const vx = -t[2] / nrm, vy = t[1] / nrm;
        const L = 3 * R;
        P.line(ctx, fr, [fx - L * vx, fx + L * vx], [fy - L * vy, fy + L * vy], th.fg, { width: 1.8 });
      }

      /* Pontos pela classe verdadeira */
      P.scatter(ctx, fr, data.filter((_, i) => ys[i] === 1), th.cyan, { r: 3.5, alpha: 0.85 });
      P.scatter(ctx, fr, data.filter((_, i) => ys[i] === -1), th.orange, { r: 3.5, alpha: 0.85 });

      /* Soma vetorial da última atualização: θ antigo (esmaecido),
         contribuição η·y·x (rosa) e θ novo (verde), em escala comum. */
      const normW = Math.hypot(t[1], t[2]);
      if (lastUpd) {
        const o = lastUpd.old;
        const normO = Math.hypot(o[1], o[2]);
        const s = (0.45 * R) / Math.max(0.6, normW, normO);
        arrow(ctx, fr, 0, 0, s * o[1], s * o[2], th.comment, 2, 0.55);
        arrow(ctx, fr, s * o[1], s * o[2], s * t[1], s * t[2], th.pink, 2);
        arrow(ctx, fr, 0, 0, s * t[1], s * t[2], th.green, 2.4);
        const mx = s * (o[1] + t[1]) / 2, my = s * (o[2] + t[2]) / 2;
        P.label(ctx, fr.X(mx) + 8, fr.Y(my), 'η·y·x', th.pink);
        P.label(ctx, fr.X(s * t[1]) + 8, fr.Y(s * t[2]) - 6, 'θ novo', th.green);
        /* Destaque do ponto mal classificado usado na atualização */
        const p = data[lastUpd.i];
        ctx.strokeStyle = th.red;
        ctx.lineWidth = 2.5;
        ctx.beginPath();
        ctx.arc(fr.X(p[0]), fr.Y(p[1]), 8, 0, 2 * Math.PI);
        ctx.stroke();
      } else if (normW > 1e-6) {
        const s = (0.45 * R) / Math.max(0.6, normW);
        arrow(ctx, fr, 0, 0, s * t[1], s * t[2], th.green, 2.4);
        P.label(ctx, fr.X(s * t[1]) + 8, fr.Y(s * t[2]) - 6, '(θ₁, θ₂)', th.green);
      }
      ctx.restore();
    }

    function drawErrors() {
      const th = P.theme();
      const { ctx, w, h } = P.setup(cvErr);
      P.clear(ctx, w, h);
      const start = Math.max(0, updates - HIST_WINDOW);
      const end = Math.max(start + 20, updates);
      let ymax = 5;
      for (const e of errHist) if (e > ymax) ymax = e;
      const fr = P.frame(ctx, w, h, start, end, 0, ymax * 1.08);
      P.axes(ctx, fr, {
        xlabel: 'atualizações', ylabel: 'erros',
        xticks: [start, Math.round((start + end) / 2), end],
        yticks: [0, Math.round(ymax / 2), ymax],
      });
      const xs = [], vals = [];
      for (let k = 0; k < errHist.length; k++) {
        xs.push(start + k);
        vals.push(errHist[k]);
      }
      P.line(ctx, fr, [start, end], [0, 0], th.line, { dash: [3, 4], width: 1 });
      P.line(ctx, fr, xs, vals, th.purple, { width: 2 });
      if (converged) P.label(ctx, fr.X(end) - 6, fr.Y(0) - 8, 'convergiu: 0 erros', th.green, 'right');
    }

    function redrawAllLocal() {
      drawPlane();
      drawErrors();
    }

    /* ~6 atualizações por segundo durante o treino animado */
    function tick(dt) {
      acc += dt * 6;
      let changed = false;
      while (acc >= 1) {
        acc -= 1;
        const ok = step();
        changed = true;
        if (!ok) { stopTraining(); break; }
      }
      if (changed) { updateStatus(); redrawAllLocal(); }
    }

    function stopTraining() {
      training = false;
      btnTrain.textContent = '▶ Treinar';
      DL.stopTicker(tick);
    }

    btnTrain.addEventListener('click', () => {
      if (training) { stopTraining(); return; }
      if (converged) return;
      training = true;
      acc = 0;
      btnTrain.textContent = '⏸ Pausar';
      DL.startTicker(tick);
    });

    btnStep.addEventListener('click', () => {
      stopTraining();
      step();
      updateStatus();
      redrawAllLocal();
    });

    btnReset.addEventListener('click', resetTraining);
    btnResample.addEventListener('click', () => {
      seed = (seed * 1664525 + 1013904223) >>> 0;
      regenerate();
    });

    slEta.addEventListener('input', () => {
      $('s3-eta-val').textContent = eta().toFixed(2);
    });

    function setMode(m) {
      mode = m;
      tabSep.classList.toggle('active', m === 'sep');
      tabNonsep.classList.toggle('active', m === 'nonsep');
      regenerate();
    }
    tabSep.addEventListener('click', () => setMode('sep'));
    tabNonsep.addEventListener('click', () => setMode('nonsep'));

    regenerate();
    P.onRedraw(redrawAllLocal);
    P.observeResize(cvPlane, drawPlane);
    P.observeResize(cvErr, drawErrors);
  }

  DL.sections.push({ name: 's3-pla', init });
})();
