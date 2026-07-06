/* Seção 4: vantagem por grupo do GRPO, Â_i = (r_i − r̄)/σ_r, com rewards
   binários verificáveis e uma política Bernoulli treinada via REINFORCE. */
(function () {
  'use strict';
  window.DL = window.DL || {};
  DL.sections = DL.sections || [];

  const P0 = 0.3, MAX_ITERS = 400;
  const sigmoid = (z) => 1 / (1 + Math.exp(-z));
  const logit = (p) => Math.log(p / (1 - p));

  function init() {
    const P = DL.plot;
    const $ = (id) => document.getElementById(id);

    const cvGroup = $('s4-group'), cvPolicy = $('s4-policy');
    const slG = $('s4-G'), slEta = $('s4-eta');
    const btnSample = $('s4-sample'), btnStep = $('s4-step');
    const btnRun = $('s4-run'), btnReset = $('s4-reset');

    let G = +slG.value, eta = +slEta.value;
    let theta, hist, group, rng;
    let frGroup = null;
    let running = false, acc = 0;

    function resetAll() {
      theta = logit(P0);
      hist = [sigmoid(theta)];
      rng = DL.utils.mulberry32(42);
      sampleGroup();
    }

    function sampleGroup() {
      const p = sigmoid(theta);
      group = [];
      for (let i = 0; i < G; i++) group.push(rng() < p ? 1 : 0);
    }

    function stats() {
      const mean = group.reduce((a, b) => a + b, 0) / G;
      let v = 0;
      for (const r of group) v += (r - mean) * (r - mean);
      const std = Math.sqrt(v / G); /* desvio populacional, como nos slides */
      const adv = group.map((r) => (std < 1e-9 ? 0 : (r - mean) / std));
      return { mean, std, adv };
    }

    /* REINFORCE com a vantagem de grupo: θ ← θ + η·(1/G)·Σ Â_i·∇log π(a_i). */
    function applyUpdate() {
      if (hist.length > MAX_ITERS) return;
      const p = sigmoid(theta);
      const { adv } = stats();
      let g = 0;
      for (let i = 0; i < G; i++) {
        g += adv[i] * (group[i] === 1 ? 1 - p : -p);
      }
      theta += eta * g / G;
      hist.push(sigmoid(theta));
      sampleGroup();
    }

    function drawGroup() {
      const th = P.theme();
      const { ctx, w, h } = P.setup(cvGroup);
      P.clear(ctx, w, h);
      const fr = P.frame(ctx, w, h, 0.5, G + 0.5, -2.6, 2.6);
      frGroup = fr;
      P.axes(ctx, fr, {
        xlabel: 'amostra i', ylabel: '',
        xticks: G <= 8 ? group.map((_, i) => i + 1) : [1, Math.round(G / 2), G],
        yticks: [-2, -1, 0, 1, 2],
      });
      const { mean, std, adv } = stats();
      P.line(ctx, fr, [0.5, G + 0.5], [0, 0], th.comment, { width: 1, alpha: 0.5 });
      /* linha da média dos rewards (na escala 0..1 dos rewards) */
      P.line(ctx, fr, [0.5, G + 0.5], [mean, mean], th.cyan, { width: 1, alpha: 0.6, dash: [3, 4] });

      const slot = fr.iw / G;
      const wAdv = Math.min(34, slot * 0.42);
      const wR = Math.max(4, wAdv * 0.3);
      for (let i = 0; i < G; i++) {
        const cx = fr.X(i + 1);
        /* barra fina do reward r_i ∈ {0, 1} */
        ctx.fillStyle = th.cyan;
        ctx.globalAlpha = 0.85;
        const hR = fr.Y(0) - fr.Y(group[i]);
        ctx.fillRect(cx - wAdv / 2 - wR - 2, fr.Y(0) - hR, wR, Math.max(1.5, hR));
        /* barra da vantagem Â_i */
        ctx.fillStyle = adv[i] >= 0 ? th.green : th.red;
        const y0 = fr.Y(Math.max(0, adv[i]));
        const hh = Math.abs(fr.Y(adv[i]) - fr.Y(0));
        ctx.fillRect(cx - wAdv / 2, y0, wAdv, Math.max(1.5, hh));
        ctx.globalAlpha = 1;
        /* rótulo de acerto/erro */
        P.label(ctx, cx, fr.Y(-2.6) + 16, group[i] === 1 ? '✓' : '✗',
          group[i] === 1 ? th.green : th.red, 'center');
      }
      P.label(ctx, fr.X(0.6), fr.Y(2.6) + 14, 'r_i (ciano fino) · Â_i (barra larga)', th.comment);
      P.label(ctx, fr.X(G + 0.45), fr.Y(mean) - 5, 'r̄ = ' + mean.toFixed(2), th.cyan, 'right');
      if (std < 1e-9) {
        P.label(ctx, fr.X((G + 1) / 2), fr.Y(1.6), 'grupo unânime: σ_r = 0, sem gradiente', th.orange, 'center');
      }
    }

    function drawPolicy() {
      const th = P.theme();
      const { ctx, w, h } = P.setup(cvPolicy);
      P.clear(ctx, w, h);
      const n = Math.max(30, hist.length - 1);
      const fr = P.frame(ctx, w, h, 0, n, 0, 1.05);
      P.axes(ctx, fr, {
        xlabel: 'iteração', ylabel: 'p',
        xticks: [0, Math.round(n / 2), n],
        yticks: [0, 0.5, 1],
      });
      P.line(ctx, fr, [0, n], [1, 1], th.comment, { width: 1, alpha: 0.5, dash: [3, 4] });
      P.line(ctx, fr, hist.map((_, i) => i), hist, th.purple, { width: 2 });
      const last = hist[hist.length - 1];
      P.scatter(ctx, fr, [[hist.length - 1, last]], th.purple, { r: 4, alpha: 1 });
    }

    function updateReadout() {
      const { mean, std } = stats();
      const p = sigmoid(theta);
      $('s4-readout').textContent =
        'k = ' + (hist.length - 1) + ' · p = ' + p.toFixed(3) +
        ' · r̄ = ' + mean.toFixed(2) + ' · σ_r = ' + std.toFixed(2) +
        (std < 1e-9 ? ' · Â_i = 0 para todos' : '');
    }

    function redraw() {
      drawGroup();
      drawPolicy();
      updateReadout();
    }

    function done() {
      return hist.length > MAX_ITERS || sigmoid(theta) > 0.995;
    }

    function stopRun() {
      running = false;
      btnRun.textContent = '▶ Treinar';
      DL.stopTicker(tick);
    }

    function tick(dt) {
      acc += dt * 15;
      while (acc >= 1 && !done()) { acc -= 1; applyUpdate(); }
      redraw();
      if (done()) stopRun();
    }

    /* clique numa amostra alterna acerto/erro do grupo exibido */
    cvGroup.addEventListener('pointerdown', (e) => {
      if (!frGroup) return;
      const rect = cvGroup.getBoundingClientRect();
      const xi = Math.round(frGroup.invX(e.clientX - rect.left));
      if (xi >= 1 && xi <= G) {
        stopRun();
        group[xi - 1] = 1 - group[xi - 1];
        redraw();
      }
    });

    btnSample.addEventListener('click', () => { stopRun(); sampleGroup(); redraw(); });
    btnStep.addEventListener('click', () => { applyUpdate(); redraw(); });
    btnRun.addEventListener('click', () => {
      if (running) { stopRun(); return; }
      if (done()) resetAll();
      running = true;
      acc = 0;
      btnRun.textContent = '⏸ Pausar';
      DL.startTicker(tick);
    });
    btnReset.addEventListener('click', () => { stopRun(); resetAll(); redraw(); });

    slG.addEventListener('input', () => {
      G = +slG.value;
      $('s4-G-val').textContent = G;
      stopRun();
      sampleGroup();
      redraw();
    });
    slEta.addEventListener('input', () => {
      eta = +slEta.value;
      $('s4-eta-val').textContent = eta.toFixed(2);
    });

    resetAll();
    redraw();
    P.onRedraw(redraw);
    P.observeResize(cvGroup, redraw);
    P.observeResize(cvPolicy, redraw);
  }

  DL.sections.push({ name: 's4-grpo', init });
})();
