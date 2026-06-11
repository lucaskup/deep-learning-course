/* Seção 3: reparametrization trick. A amostragem z* = μ + σ·ε vira uma
   transformação determinística do ruído ε ~ N(0,1): os sliders movem a
   reta, as amostras seguem, e o gradiente flui por μ e σ. Inclui o KL
   em forma fechada contra o prior N(0,1). */
(function () {
  'use strict';
  window.DL = window.DL || {};
  DL.sections = DL.sections || [];

  const ER = 3.5, ZRMAX = 4.5;
  const MU_R = 2.5, SIG_LO = 0.05, SIG_HI = 2.5;
  const BUF_MAX = 400;

  function init() {
    const P = DL.plot, U = DL.utils;
    const $ = (id) => document.getElementById(id);

    const cvMap = $('s3-map'), cvKL = $('s3-kl');
    const slMu = $('s3-mu'), slSig = $('s3-sigma');
    const btnSample = $('s3-sample'), btnAnim = $('s3-anim'), btnReset = $('s3-reset');

    let mu = +slMu.value, sig = +slSig.value;
    let seed = 21;
    let rngAnim = U.mulberry32(seed);
    let buf = [];
    let animating = false, acc = 0;
    let frKL = null, dragKL = false;

    const klOf = (m, s) => 0.5 * (m * m + s * s - 1 - 2 * Math.log(s));

    function fillBuf(n) {
      const rng = U.mulberry32(seed);
      buf = [];
      for (let i = 0; i < n; i++) buf.push(U.randn(rng));
    }

    function drawMap() {
      const th = P.theme();
      const { ctx, w, h } = P.setup(cvMap);
      P.clear(ctx, w, h);
      const fr = P.frame(ctx, w, h, -ER, ER, -ZRMAX, ZRMAX);
      P.axes(ctx, fr, { xlabel: 'ε', ylabel: 'z', xticks: [-3, 0, 3], yticks: [-4, 0, 4] });

      ctx.save();
      ctx.beginPath();
      ctx.rect(fr.X(fr.xmin), fr.Y(fr.ymax), fr.iw, fr.ih);
      ctx.clip();

      /* densidade de ε ~ N(0,1) ao longo do eixo inferior */
      const exs = [], eys = [];
      for (let i = 0; i <= 120; i++) {
        const e = -ER + (i / 120) * 2 * ER;
        exs.push(e);
        eys.push(-ZRMAX + U.gaussPdf(e, 0, 1) * 2.4);
      }
      P.line(ctx, fr, exs, eys, th.comment, { width: 1.4, alpha: 0.8 });

      /* reta z = μ + σ·ε */
      P.line(ctx, fr, [-ER, ER], [mu - sig * ER, mu + sig * ER], th.orange, { width: 2.2 });

      /* histograma de z na vertical (barras horizontais a partir do eixo) */
      if (buf.length > 4) {
        const nb = 40;
        const bins = new Float64Array(nb);
        const bw = 2 * ZRMAX / nb;
        for (const e of buf) {
          const z = mu + sig * e;
          const b = Math.floor((z + ZRMAX) / bw);
          if (b >= 0 && b < nb) bins[b]++;
        }
        let dmax = 0;
        for (let b = 0; b < nb; b++) dmax = Math.max(dmax, bins[b] / (buf.length * bw));
        dmax = Math.max(dmax, U.gaussPdf(mu, mu, sig * sig));
        const scale = 1.5 / dmax;
        ctx.fillStyle = th.cyan;
        ctx.globalAlpha = 0.4;
        for (let b = 0; b < nb; b++) {
          const d = bins[b] / (buf.length * bw);
          const z0 = -ZRMAX + b * bw, z1 = z0 + bw;
          ctx.fillRect(fr.X(-ER), fr.Y(z1), fr.X(-ER + d * scale) - fr.X(-ER), fr.Y(z0) - fr.Y(z1));
        }
        ctx.globalAlpha = 1;
        /* densidade teórica N(μ, σ²) sobre o histograma */
        const txs = [], tys = [];
        for (let i = 0; i <= 140; i++) {
          const z = -ZRMAX + (i / 140) * 2 * ZRMAX;
          txs.push(-ER + U.gaussPdf(z, mu, sig * sig) * scale);
          tys.push(z);
        }
        P.line(ctx, fr, txs, tys, th.green, { width: 1.6 });
      }

      /* amostras sobre a reta */
      const pts = buf.map((e) => [e, mu + sig * e]);
      P.scatter(ctx, fr, pts, th.cyan, { r: 2, alpha: 0.55 });

      /* guia da amostra mais recente: ε sobe até a reta e vai ao eixo z */
      if (buf.length) {
        const e = buf[buf.length - 1];
        const z = mu + sig * e;
        P.line(ctx, fr, [e, e], [-ZRMAX, z], th.pink, { width: 1.2, dash: [4, 3] });
        P.line(ctx, fr, [e, -ER], [z, z], th.pink, { width: 1.2, dash: [4, 3] });
        P.scatter(ctx, fr, [[e, z]], th.pink, { r: 3.5, alpha: 1 });
      }
      ctx.restore();

      P.label(ctx, fr.X(ER) - 4, fr.Y(mu + sig * ER * 0.92) + (sig > 0 ? 14 : -8), 'z = μ + σ·ε', th.orange, 'right');
      P.label(ctx, fr.X(-ER) + 8, fr.Y(ZRMAX) + 14, 'q(z|x,φ) = N(μ, σ²)', th.green);
      P.label(ctx, fr.X(-ER) + 8, fr.Y(-ZRMAX) - 8, 'ε ~ N(0,1)', th.comment);
    }

    function drawKL() {
      const th = P.theme();
      const { ctx, w, h } = P.setup(cvKL);
      P.clear(ctx, w, h);
      const fr = P.frame(ctx, w, h, -MU_R, MU_R, SIG_LO, SIG_HI);
      frKL = fr;
      P.heatmap(ctx, fr, klOf, 90, 90, (v) => {
        const u = Math.pow(Math.min(1, v / 6), 0.45);
        const col = P.viridis(u);
        const lift = (u * 9) % 1 < 0.12 ? 42 : 0;
        return [Math.min(255, col[0] + lift), Math.min(255, col[1] + lift), Math.min(255, col[2] + lift), 235];
      });
      P.axes(ctx, fr, { xlabel: 'μ', ylabel: 'σ', xticks: [-2, 0, 2], yticks: [0.5, 1, 1.5, 2] });

      /* mínimo: q = prior, KL = 0 */
      P.scatter(ctx, fr, [[0, 1]], th.green, { r: 4.5, alpha: 1 });
      P.label(ctx, fr.X(0) + 8, fr.Y(1) - 6, 'KL = 0 em (μ=0, σ=1)', th.green);

      /* ponto atual (arrastável) */
      ctx.beginPath();
      ctx.arc(fr.X(mu), fr.Y(sig), 6.5, 0, 2 * Math.PI);
      ctx.strokeStyle = th.fg;
      ctx.lineWidth = 2.2;
      ctx.stroke();
      P.label(ctx, fr.X(mu) + 10, fr.Y(sig) + 4, '(μ, σ)', th.fg);
    }

    function updateReadout() {
      $('s3-mu-val').textContent = mu.toFixed(2);
      $('s3-sigma-val').textContent = sig.toFixed(2);
      const kl = klOf(mu, sig);
      $('s3-readout').textContent =
        'KL = ½(μ² + σ² − 1 − log σ²) = ½(' + (mu * mu).toFixed(2) + ' + ' + (sig * sig).toFixed(2) +
        ' − 1 + ' + (-2 * Math.log(sig)).toFixed(2) + ') = ' + kl.toFixed(3) +
        ' · ∂z/∂μ = 1 · ∂z/∂σ = ε';
    }

    function redraw() {
      drawMap();
      drawKL();
      updateReadout();
    }

    function stopAnim() {
      animating = false;
      btnAnim.textContent = '▶ Chuva de amostras';
      DL.stopTicker(tick);
    }

    function tick(dt) {
      acc += dt * 25;
      let changed = false;
      while (acc >= 1) {
        acc -= 1;
        buf.push(U.randn(rngAnim));
        if (buf.length > BUF_MAX) buf.shift();
        changed = true;
      }
      if (changed) redraw();
    }

    function setParams(m, s) {
      mu = Math.max(-MU_R, Math.min(MU_R, m));
      sig = Math.max(SIG_LO, Math.min(SIG_HI, s));
      slMu.value = mu;
      slSig.value = sig;
      redraw();
    }

    cvKL.addEventListener('pointerdown', (e) => {
      if (!frKL) return;
      dragKL = true;
      cvKL.setPointerCapture(e.pointerId);
      const r = cvKL.getBoundingClientRect();
      setParams(frKL.invX(e.clientX - r.left), frKL.invY(e.clientY - r.top));
    });
    cvKL.addEventListener('pointermove', (e) => {
      if (!dragKL || !frKL) return;
      const r = cvKL.getBoundingClientRect();
      setParams(frKL.invX(e.clientX - r.left), frKL.invY(e.clientY - r.top));
    });
    cvKL.addEventListener('pointerup', () => { dragKL = false; });

    slMu.addEventListener('input', () => { mu = +slMu.value; redraw(); });
    slSig.addEventListener('input', () => { sig = +slSig.value; redraw(); });
    btnSample.addEventListener('click', () => {
      seed = seed * 3 + 7;
      rngAnim = U.mulberry32(seed + 1);
      fillBuf(250);
      redraw();
    });
    btnAnim.addEventListener('click', () => {
      if (animating) { stopAnim(); return; }
      animating = true;
      acc = 0;
      btnAnim.textContent = '⏸ Pausar';
      DL.startTicker(tick);
    });
    btnReset.addEventListener('click', () => {
      stopAnim();
      seed = 21;
      rngAnim = U.mulberry32(seed);
      setParams(0.5, 0.8);
      fillBuf(250);
      redraw();
    });

    fillBuf(250);
    redraw();
    P.onRedraw(redraw);
    P.observeResize(cvMap, redraw);
    P.observeResize(cvKL, redraw);
  }

  DL.sections.push({ name: 's3-reparam', init });
})();
