/* Seção 3: objetivo do RLHF J(θ) = E_π[r_φ] − β·KL(π_θ ‖ π_ref) numa política
   categórica de brinquedo; com β pequeno a política colapsa no pico falso do RM
   (reward hacking), com β grande permanece presa ao SFT. */
(function () {
  'use strict';
  window.DL = window.DL || {};
  DL.sections = DL.sections || [];

  const K = 41, XMIN = 0, XMAX = 10;
  const LR = 0.3, MAX_ITERS = 300;

  function init() {
    const P = DL.plot;
    const $ = (id) => document.getElementById(id);

    const cvDist = $('s3-dist'), cvCurves = $('s3-curves');
    const slBeta = $('s3-beta');
    const btnStep = $('s3-step'), btnRun = $('s3-run'), btnReset = $('s3-reset');

    let beta = +slBeta.value;
    let running = false, acc = 0;

    /* eixo de "tipos de resposta" e funções fixas do cenário */
    const xs = [];
    for (let i = 0; i < K; i++) xs.push(XMIN + (XMAX - XMIN) * i / (K - 1));

    const uOf = (x) => 2.2 * Math.exp(-(x - 6) * (x - 6) / 2.2);
    const hackOf = (x) => 3.2 * Math.exp(-(x - 9.2) * (x - 9.2) / 0.18);
    const rOf = (x) => uOf(x) + hackOf(x);

    const u = xs.map(uOf);
    const r = xs.map(rOf);
    const rho = (() => {
      const raw = xs.map((x) => Math.exp(-(x - 4.5) * (x - 4.5) / (2 * 1.3 * 1.3)));
      const s = raw.reduce((a, b) => a + b, 0);
      return raw.map((v) => v / s);
    })();
    const logRho = rho.map((v) => Math.log(v));

    let z, k, histR, histU, histKL;

    function softmax(zz) {
      const m = Math.max(...zz);
      const e = zz.map((v) => Math.exp(v - m));
      const s = e.reduce((a, b) => a + b, 0);
      return e.map((v) => v / s);
    }

    function metrics(pi) {
      let er = 0, eu = 0, kl = 0;
      for (let i = 0; i < K; i++) {
        er += pi[i] * r[i];
        eu += pi[i] * u[i];
        if (pi[i] > 1e-12) kl += pi[i] * (Math.log(pi[i]) - logRho[i]);
      }
      return { er, eu, kl };
    }

    function pushHist() {
      const m = metrics(softmax(z));
      histR.push(m.er);
      histU.push(m.eu);
      histKL.push(m.kl);
    }

    function resetOpt() {
      z = logRho.slice();
      k = 0;
      histR = [];
      histU = [];
      histKL = [];
      pushHist();
    }

    /* Subida de gradiente espelhada em J(π) = Σ π_i (r_i − β log(π_i/ρ_i)):
       z_i ← z_i + η·∂J/∂π_i. O ponto fixo é a solução fechada do RLHF,
       π* ∝ π_ref·exp(r/β), a mesma usada na derivação do DPO. */
    function stepOnce() {
      if (k >= MAX_ITERS) return;
      const pi = softmax(z);
      for (let i = 0; i < K; i++) {
        const logRatio = Math.max(-30, Math.log(Math.max(pi[i], 1e-13)) - logRho[i]);
        z[i] += LR * (r[i] - beta * (logRatio + 1));
      }
      k++;
      pushHist();
    }

    function drawDist() {
      const th = P.theme();
      const { ctx, w, h } = P.setup(cvDist);
      P.clear(ctx, w, h);
      const YMAX = 6;
      const fr = P.frame(ctx, w, h, XMIN, XMAX, 0, YMAX);
      P.axes(ctx, fr, {
        xlabel: 'tipo de resposta y', ylabel: 'reward',
        xticks: [0, 2, 4, 6, 8, 10],
        yticks: [0, 2, 4, 6],
      });

      const pi = softmax(z);
      const maxBar = Math.max(...pi, ...rho);
      const scale = 4.6 / maxBar;
      /* barras da política (escala própria, normalizada pelo máximo visível) */
      const pc = P.rgb(th.purple);
      const bw = (fr.X(xs[1]) - fr.X(xs[0])) * 0.8;
      ctx.fillStyle = 'rgba(' + pc[0] + ',' + pc[1] + ',' + pc[2] + ',0.55)';
      for (let i = 0; i < K; i++) {
        const hh = fr.Y(0) - fr.Y(Math.min(YMAX, pi[i] * scale));
        ctx.fillRect(fr.X(xs[i]) - bw / 2, fr.Y(0) - hh, bw, hh);
      }
      /* referência SFT (mesma escala das barras) */
      P.line(ctx, fr, xs, rho.map((v) => Math.min(YMAX, v * scale)), th.comment, { width: 1.6, dash: [5, 4] });
      /* RM e utilidade verdadeira (escala absoluta do eixo y) */
      P.line(ctx, fr, xs, r.map((v) => Math.min(YMAX, v)), th.orange, { width: 2 });
      P.line(ctx, fr, xs, u, th.green, { width: 2 });

      P.label(ctx, fr.X(0.2), fr.Y(YMAX) + 14, 'π_θ (barras)', th.purple);
      P.label(ctx, fr.X(0.2), fr.Y(YMAX) + 28, 'π_{ref} (SFT)', th.comment);
      P.label(ctx, fr.X(6), fr.Y(uOf(6)) - 8, 'u: utilidade real', th.green, 'center');
      P.label(ctx, fr.X(9.2), fr.Y(Math.min(YMAX, rOf(9.2))) - 8, 'r_φ: pico falso', th.orange, 'center');
    }

    function drawCurves() {
      const th = P.theme();
      const { ctx, w, h } = P.setup(cvCurves);
      P.clear(ctx, w, h);
      const n = Math.max(30, histR.length - 1);
      const ymax = Math.min(12, Math.ceil(Math.max(3, ...histR, ...histKL)) + 1);
      const fr = P.frame(ctx, w, h, 0, n, 0, ymax);
      P.axes(ctx, fr, {
        xlabel: 'iteração', ylabel: '',
        xticks: [0, Math.round(n / 2), n],
        yticks: [0, Math.round(ymax / 2), ymax],
      });
      const ks = histR.map((_, i) => i);
      const cap = (v) => Math.min(ymax, v);
      P.line(ctx, fr, ks, histR.map(cap), th.orange, { width: 1.8 });
      P.line(ctx, fr, ks, histU.map(cap), th.green, { width: 1.8 });
      P.line(ctx, fr, ks, histKL.map(cap), th.pink, { width: 1.8 });
      P.label(ctx, fr.X(0) + 8, fr.Y(ymax) + 14, 'E[r_φ] (RM)', th.orange);
      P.label(ctx, fr.X(0) + 8, fr.Y(ymax) + 28, 'E[u] (real)', th.green);
      P.label(ctx, fr.X(0) + 8, fr.Y(ymax) + 42, 'KL(π_θ ‖ π_{ref})', th.pink);
    }

    function updateReadout() {
      const m = metrics(softmax(z));
      $('s3-readout').textContent =
        'k = ' + k + ' · E[r_φ] = ' + m.er.toFixed(2) +
        ' · E[u] = ' + m.eu.toFixed(2) + ' · KL = ' + m.kl.toFixed(2);
    }

    function redraw() {
      drawDist();
      drawCurves();
      updateReadout();
    }

    function stopRun() {
      running = false;
      btnRun.textContent = '▶ Otimizar';
      DL.stopTicker(tick);
    }

    function tick(dt) {
      acc += dt * 20;
      while (acc >= 1 && k < MAX_ITERS) { acc -= 1; stepOnce(); }
      redraw();
      if (k >= MAX_ITERS) stopRun();
    }

    btnStep.addEventListener('click', () => { stepOnce(); redraw(); });
    btnRun.addEventListener('click', () => {
      if (running) { stopRun(); return; }
      if (k >= MAX_ITERS) resetOpt();
      running = true;
      acc = 0;
      btnRun.textContent = '⏸ Pausar';
      DL.startTicker(tick);
    });
    btnReset.addEventListener('click', () => { stopRun(); resetOpt(); redraw(); });

    slBeta.addEventListener('input', () => {
      beta = +slBeta.value;
      $('s3-beta-val').textContent = beta.toFixed(2);
      stopRun();
      resetOpt();
      redraw();
    });

    resetOpt();
    redraw();
    P.onRedraw(redraw);
    P.observeResize(cvDist, redraw);
    P.observeResize(cvCurves, redraw);
  }

  DL.sections.push({ name: 's3-rlhf-kl', init });
})();
