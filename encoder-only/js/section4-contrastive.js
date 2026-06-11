/* Seção 4: aprendizado contrastivo com InfoNCE e in-batch negatives.
   Embeddings unitários em 2D (parametrizados pelo ângulo): cada consulta qᵢ é
   puxada para seu documento dᵢ e empurrada dos demais documentos do batch. */
(function () {
  'use strict';
  window.DL = window.DL || {};
  DL.sections = DL.sections || [];

  const MAX_STEPS = 5000;

  function init() {
    const P = DL.plot;
    const U = DL.utils;
    const $ = (id) => document.getElementById(id);

    const cvCircle = $('s4-circle'), cvMatrix = $('s4-matrix');
    const slTau = $('s4-tau'), slN = $('s4-n');
    const btnRun = $('s4-run');

    let seed = 21;
    let N = +slN.value;
    let tau = +slTau.value;
    let thQ = [], thD = [];
    let steps = 0;
    let running = false, accum = 0;
    let frCircle = null, drag = null;

    function initAngles() {
      const rng = U.mulberry32(seed);
      thQ = [];
      thD = [];
      for (let i = 0; i < N; i++) {
        thQ.push(2 * Math.PI * rng());
        thD.push(2 * Math.PI * rng());
      }
      steps = 0;
    }

    /* sᵢⱼ = cos(θqᵢ − θdⱼ); pᵢⱼ = softmax na linha de sᵢⱼ/τ. */
    function forward() {
      const S = [], Prob = [];
      for (let i = 0; i < N; i++) {
        const srow = new Array(N);
        let mx = -Infinity;
        for (let j = 0; j < N; j++) {
          srow[j] = Math.cos(thQ[i] - thD[j]);
          if (srow[j] / tau > mx) mx = srow[j] / tau;
        }
        let z = 0;
        const prow = srow.map((s) => {
          const e = Math.exp(s / tau - mx);
          z += e;
          return e;
        });
        S.push(srow);
        Prob.push(prow.map((e) => e / z));
      }
      return { S, Prob };
    }

    function loss(Prob) {
      let l = 0;
      for (let i = 0; i < N; i++) l += -Math.log(Math.max(Prob[i][i], 1e-12));
      return l / N;
    }

    function stepOnce() {
      const { Prob } = forward();
      const gQ = new Array(N).fill(0);
      const gD = new Array(N).fill(0);
      for (let i = 0; i < N; i++) {
        for (let j = 0; j < N; j++) {
          const coef = (Prob[i][j] - (i === j ? 1 : 0)) / (N * tau);
          const sn = Math.sin(thQ[i] - thD[j]);
          gQ[i] += coef * (-sn);
          gD[j] += coef * sn;
        }
      }
      const lr = 4.0 * tau; /* passo proporcional a τ para estabilidade */
      for (let i = 0; i < N; i++) {
        thQ[i] -= lr * gQ[i];
        thD[i] -= lr * gD[i];
      }
      steps++;
    }

    function done() {
      const { Prob } = forward();
      return steps > MAX_STEPS || loss(Prob) < 0.005;
    }

    function pairColor(i) {
      const th = P.theme();
      const palette = [th.cyan, th.orange, th.green, th.pink, th.purple, th.red];
      return palette[i % palette.length];
    }

    function drawCircle() {
      const th = P.theme();
      const { ctx, w, h } = P.setup(cvCircle);
      P.clear(ctx, w, h);
      const fr = P.frame(ctx, w, h, -1.45, 1.45, -1.45, 1.45, { l: 10, r: 10, t: 10, b: 10 });
      frCircle = fr;

      /* círculo unitário */
      ctx.strokeStyle = th.line;
      ctx.lineWidth = 1.2;
      ctx.beginPath();
      ctx.arc(fr.X(0), fr.Y(0), fr.X(1) - fr.X(0), 0, 2 * Math.PI);
      ctx.stroke();

      for (let i = 0; i < N; i++) {
        const color = pairColor(i);
        const qx = Math.cos(thQ[i]), qy = Math.sin(thQ[i]);
        const dx = Math.cos(thD[i]), dy = Math.sin(thD[i]);
        /* corda ligando o par */
        P.line(ctx, fr, [qx, dx], [qy, dy], color, { width: 1.2, alpha: 0.45, dash: i >= 6 ? [4, 3] : null });
        /* consulta: círculo cheio */
        ctx.beginPath();
        ctx.arc(fr.X(qx), fr.Y(qy), 5.5, 0, 2 * Math.PI);
        ctx.fillStyle = color;
        ctx.fill();
        /* documento: quadrado vazado */
        ctx.strokeStyle = color;
        ctx.lineWidth = 2;
        ctx.strokeRect(fr.X(dx) - 4.5, fr.Y(dy) - 4.5, 9, 9);
        /* rótulos para batches pequenos */
        if (N <= 6) {
          P.mathText(ctx, 'q_' + (i + 1), fr.X(1.18 * qx), fr.Y(1.18 * qy) + 4, color, 'center');
          P.mathText(ctx, 'd_' + (i + 1), fr.X(1.18 * dx), fr.Y(1.18 * dy) + 4, color, 'center');
        }
      }
    }

    function drawMatrix() {
      const th = P.theme();
      const { ctx, w, h } = P.setup(cvMatrix);
      P.clear(ctx, w, h);
      const { Prob } = forward();

      const ml = 34, mt = 24, mb = 8, mr = 8;
      const cell = Math.min((w - ml - mr) / N, (h - mt - mb) / N);
      const x0 = ml + (w - ml - mr - cell * N) / 2;
      const y0 = mt;
      mat = { cell, x0, y0 };

      const pc = P.rgb(th.purple);
      for (let i = 0; i < N; i++) {
        for (let j = 0; j < N; j++) {
          const x = x0 + j * cell, y = y0 + i * cell;
          const a = 0.05 + 0.92 * Prob[i][j];
          ctx.fillStyle = 'rgba(' + pc[0] + ',' + pc[1] + ',' + pc[2] + ',' + a.toFixed(3) + ')';
          ctx.fillRect(x, y, cell - 1, cell - 1);
          if (i === j) {
            ctx.strokeStyle = th.green;
            ctx.lineWidth = 1.6;
            ctx.strokeRect(x + 0.5, y + 0.5, cell - 2, cell - 2);
          }
          if (cell >= 30) {
            ctx.fillStyle = Prob[i][j] > 0.45 ? th.bg : th.comment;
            ctx.font = '9px sans-serif';
            ctx.textAlign = 'center';
            ctx.fillText(Prob[i][j].toFixed(2), x + cell / 2, y + cell / 2 + 3);
          }
        }
      }
      for (let i = 0; i < N; i++) {
        P.mathText(ctx, 'q_' + (i + 1), x0 - 18, y0 + i * cell + cell / 2 + 3, th.comment);
        P.mathText(ctx, 'd_' + (i + 1), x0 + i * cell + cell / 2, y0 - 8, th.comment, 'center');
      }
    }

    function updateReadout() {
      const { S, Prob } = forward();
      let cp = 0, cn = 0, hits = 0;
      for (let i = 0; i < N; i++) {
        cp += S[i][i];
        let best = 0;
        for (let j = 0; j < N; j++) {
          if (j !== i) cn += S[i][j];
          if (Prob[i][j] > Prob[i][best]) best = j;
        }
        if (best === i) hits++;
      }
      cp /= N;
      cn /= N * Math.max(1, N - 1);
      $('s4-readout').textContent =
        'passo ' + steps + ' · ℒ = ' + loss(Prob).toFixed(3) +
        ' · cos(qᵢ,dᵢ) médio = ' + cp.toFixed(2) +
        ' · cos negativos = ' + cn.toFixed(2) +
        ' · diagonal: ' + hits + '/' + N;
    }

    function redraw() {
      drawCircle();
      drawMatrix();
      updateReadout();
    }

    function stopRun() {
      running = false;
      btnRun.textContent = '▶ Treinar';
      DL.stopTicker(tick);
    }

    function tick(dt) {
      accum += dt * 60;
      while (accum >= 1) { accum -= 1; stepOnce(); }
      redraw();
      if (done()) stopRun();
    }

    /* Arrastar pontos: pega o mais próximo do clique e fixa pelo ângulo. */
    cvCircle.addEventListener('pointerdown', (e) => {
      if (!frCircle) return;
      const r = cvCircle.getBoundingClientRect();
      const px = e.clientX - r.left, py = e.clientY - r.top;
      let best = null, bd = 14;
      for (let i = 0; i < N; i++) {
        const dq = Math.hypot(frCircle.X(Math.cos(thQ[i])) - px, frCircle.Y(Math.sin(thQ[i])) - py);
        const dd = Math.hypot(frCircle.X(Math.cos(thD[i])) - px, frCircle.Y(Math.sin(thD[i])) - py);
        if (dq < bd) { bd = dq; best = { kind: 'q', i }; }
        if (dd < bd) { bd = dd; best = { kind: 'd', i }; }
      }
      if (best) {
        drag = best;
        cvCircle.setPointerCapture(e.pointerId);
        stopRun();
      }
    });
    cvCircle.addEventListener('pointermove', (e) => {
      if (!drag || !frCircle) return;
      const r = cvCircle.getBoundingClientRect();
      const x = frCircle.invX(e.clientX - r.left);
      const y = frCircle.invY(e.clientY - r.top);
      const ang = Math.atan2(y, x);
      if (drag.kind === 'q') thQ[drag.i] = ang;
      else thD[drag.i] = ang;
      redraw();
    });
    cvCircle.addEventListener('pointerup', () => { drag = null; });

    $('s4-step').addEventListener('click', () => { stepOnce(); redraw(); });
    btnRun.addEventListener('click', () => {
      if (running) { stopRun(); return; }
      running = true;
      accum = 0;
      btnRun.textContent = '⏸ Pausar';
      DL.startTicker(tick);
    });
    $('s4-reset').addEventListener('click', () => {
      stopRun();
      initAngles();
      redraw();
    });
    $('s4-seed').addEventListener('click', () => {
      stopRun();
      seed = (seed + 1) >>> 0;
      initAngles();
      redraw();
    });
    slTau.addEventListener('input', () => {
      tau = +slTau.value;
      $('s4-tau-val').textContent = tau.toFixed(2);
      redraw();
    });
    slN.addEventListener('input', () => {
      N = +slN.value;
      $('s4-n-val').textContent = N;
      stopRun();
      initAngles();
      redraw();
    });

    initAngles();
    redraw();
    P.onRedraw(redraw);
    P.observeResize(cvCircle, redraw);
    P.observeResize(cvMatrix, redraw);
  }

  DL.sections.push({ name: 's4-contrastive', init });
})();
