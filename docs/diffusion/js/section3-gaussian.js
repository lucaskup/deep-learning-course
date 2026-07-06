/* Seção 3: kernel q(z_t|x) e posterior q(z_{t-1}|z_t,x) como produto de gaussianas. */
(function () {
  'use strict';
  window.DM = window.DM || {};
  DM.sections = DM.sections || [];

  const T = 50, ZRANGE = 3.2;
  const K_DRAW = 30, K_SIM = 300, M_KERNEL = 200;

  function init() {
    const U = DM.utils, P = DM.plot;
    const $ = (id) => document.getElementById(id);
    const sched = U.linearSchedule(T, 1e-4, 0.2);

    /* ── 3a: kernel ───────────────────────────────────────── */
    const cvTraj = $('s3a-traj'), cvDist = $('s3a-dist');
    let ta = 25, xa = 1.5, seedA = 42;
    let noiseSeq = null, kernelEps = null, traj = null;

    function regenA() {
      const rng = U.mulberry32(seedA);
      noiseSeq = [];
      for (let k = 0; k < K_SIM; k++) {
        const row = new Float32Array(T + 1);
        for (let s = 1; s <= T; s++) row[s] = U.randn(rng);
        noiseSeq.push(row);
      }
      kernelEps = new Float32Array(M_KERNEL);
      for (let m = 0; m < M_KERNEL; m++) kernelEps[m] = U.randn(rng);
      rebuildTraj();
    }

    /* Trajetórias sequenciais determinísticas dado o ruído pré-amostrado */
    function rebuildTraj() {
      traj = [];
      for (let k = 0; k < K_SIM; k++) {
        const row = new Float32Array(T + 1);
        row[0] = xa;
        for (let s = 1; s <= T; s++)
          row[s] = Math.sqrt(1 - sched.beta[s]) * row[s - 1] + Math.sqrt(sched.beta[s]) * noiseSeq[k][s];
        traj.push(row);
      }
    }

    function drawA() {
      const th = P.theme();
      let { ctx, w, h } = P.setup(cvTraj);
      P.clear(ctx, w, h);
      let fr = P.frame(ctx, w, h, 0, T, -ZRANGE, ZRANGE);
      P.axes(ctx, fr, { xlabel: 't', ylabel: 'z', xticks: [0, 25, 50], yticks: [-3, 0, 3] });
      const ts = Array.from({ length: T + 1 }, (_, s) => s);
      for (let k = 0; k < K_DRAW; k++)
        P.line(ctx, fr, ts, ts.map((s) => traj[k][s]), th.cyan, { width: 0.8, alpha: 0.25 });
      P.scatter(ctx, fr, [[0, xa]], th.pink, { r: 5, alpha: 1 });
      P.label(ctx, fr.X(0) + 8, fr.Y(xa) - 8, 'x', th.pink);
      P.line(ctx, fr, [ta, ta], [-ZRANGE, ZRANGE], th.comment, { dash: [3, 3], width: 1 });

      ({ ctx, w, h } = P.setup(cvDist));
      P.clear(ctx, w, h);
      const a = sched.alpha[ta];
      const mu = Math.sqrt(a) * xa, v = 1 - a;
      const seq = [];
      for (let k = 0; k < K_SIM; k++) seq.push(traj[k][ta]);
      const oneShot = [];
      for (let m = 0; m < M_KERNEL; m++) oneShot.push(mu + Math.sqrt(v) * kernelEps[m]);
      const xs = [], pdf = [];
      let ymax = 0;
      for (let x = -ZRANGE; x <= ZRANGE; x += 0.05) {
        xs.push(x);
        const p = U.gaussPdf(x, mu, v);
        pdf.push(p);
        if (p > ymax) ymax = p;
      }
      fr = P.frame(ctx, w, h, -ZRANGE, ZRANGE, 0, Math.max(0.6, ymax * 1.15));
      P.axes(ctx, fr, { xlabel: 'z_t', xticks: [-3, 0, 3] });
      P.histogram(ctx, fr, seq, 36, th.cyan, { alpha: 0.4 });
      P.histogram(ctx, fr, oneShot, 36, th.green, { alpha: 0.4 });
      P.line(ctx, fr, xs, pdf, th.purple, { width: 2.2 });
      P.label(ctx, fr.X(-3) + 8, 14, 'sequencial (' + K_SIM + ' cadeias)', th.cyan);
      P.label(ctx, fr.X(-3) + 8, 28, 'kernel, 1 passo (' + M_KERNEL + ')', th.green);
      P.label(ctx, fr.X(-3) + 8, 42, '𝒩(√α_t·x, 1−α_t)', th.purple);
    }

    $('s3a-t').addEventListener('input', (e) => { ta = +e.target.value; $('s3a-t-val').textContent = ta; drawA(); });
    $('s3a-x').addEventListener('input', (e) => {
      xa = +e.target.value;
      $('s3a-x-val').textContent = xa.toFixed(1);
      rebuildTraj();
      drawA();
    });
    $('s3a-resample').addEventListener('click', () => { seedA = (seedA * 1664525 + 1013904223) >>> 0; regenA(); drawA(); });

    /* ── 3b: posterior, produto de gaussianas ─────────────── */
    const cvPost = $('s3b-plot');
    let tb = 25, xb = 1.0, ztb = 0.5;
    let dragging = null;

    function posterior() {
      const b = sched.beta[tb];
      const at = sched.alpha[tb], at1 = sched.alpha[tb - 1];
      const mean = ((1 - at1) / (1 - at)) * Math.sqrt(1 - b) * ztb
        + (Math.sqrt(at1) * b / (1 - at)) * xb;
      const variance = b * (1 - at1) / (1 - at);
      return { b, at, at1, mean, variance };
    }

    function drawB() {
      const th = P.theme();
      const { ctx, w, h } = P.setup(cvPost);
      P.clear(ctx, w, h);
      const { b, at, at1, mean, variance } = posterior();

      const muPrior = Math.sqrt(at1) * xb, vPrior = 1 - at1;
      const muLik = ztb / Math.sqrt(1 - b), vLik = b / (1 - b);

      const xs = [], prior = [], lik = [], post = [];
      let ymax = 0;
      for (let x = -ZRANGE; x <= ZRANGE; x += 0.02) {
        xs.push(x);
        const p1 = U.gaussPdf(x, muPrior, vPrior);
        const p2 = U.gaussPdf(x, muLik, vLik);
        const p3 = U.gaussPdf(x, mean, variance);
        prior.push(p1); lik.push(p2); post.push(p3);
        ymax = Math.max(ymax, p1, p2, p3);
      }
      const fr = P.frame(ctx, w, h, -ZRANGE, ZRANGE, 0, ymax * 1.18, { l: 38, r: 10, t: 10, b: 34 });
      P.axes(ctx, fr, { xlabel: 'z_{t−1}', xticks: [-3, -2, -1, 0, 1, 2, 3] });
      P.line(ctx, fr, xs, prior, th.orange, { width: 2 });
      P.line(ctx, fr, xs, lik, th.cyan, { width: 2 });
      P.line(ctx, fr, xs, post, th.purple, { width: 2.6 });
      P.label(ctx, fr.X(-3) + 8, 14, 'prior q(z_{t−1}|x)', th.orange);
      P.label(ctx, fr.X(-3) + 8, 28, 'verossimilhança q(z_t|z_{t−1}) renorm.', th.cyan);
      P.label(ctx, fr.X(-3) + 8, 42, 'posterior q(z_{t−1}|z_t, x)', th.purple);

      /* Marcadores arrastáveis de x e z_t na base do gráfico */
      const y0 = fr.Y(0);
      ctx.fillStyle = th.orange;
      ctx.beginPath();
      ctx.moveTo(fr.X(xb), y0 + 4); ctx.lineTo(fr.X(xb) - 6, y0 + 14); ctx.lineTo(fr.X(xb) + 6, y0 + 14);
      ctx.closePath(); ctx.fill();
      P.label(ctx, fr.X(xb), y0 + 26, 'x', th.orange, 'center');
      ctx.fillStyle = th.cyan;
      ctx.beginPath();
      ctx.arc(fr.X(ztb), y0 + 9, 6, 0, 2 * Math.PI);
      ctx.fill();
      P.label(ctx, fr.X(ztb), y0 + 26, 'z_t', th.cyan, 'center');

      $('s3b-readout').innerHTML =
        'x = ' + xb.toFixed(2) + ' · z<sub>t</sub> = ' + ztb.toFixed(2) +
        ' · média = ' + mean.toFixed(3) + ' · variância = ' + variance.toFixed(4);
      $('s3b-formula').innerHTML =
        'q(z<sub>t−1</sub> | z<sub>t</sub>, x) = 𝒩( ' +
        '<span style="color:var(--purple)">' + mean.toFixed(3) + '</span>, ' +
        '<span style="color:var(--purple)">' + variance.toFixed(4) + '</span> )<br />' +
        'média = ((1−α<sub>t−1</sub>)/(1−α<sub>t</sub>))·√(1−β<sub>t</sub>)·z<sub>t</sub>' +
        ' + (√α<sub>t−1</sub>·β<sub>t</sub>/(1−α<sub>t</sub>))·x = ' + mean.toFixed(3) + '<br />' +
        'variância = β<sub>t</sub>(1−α<sub>t−1</sub>)/(1−α<sub>t</sub>) = ' + variance.toFixed(4) + '<br />' +
        'β<sub>t</sub> = ' + b.toFixed(4) + ' · α<sub>t</sub> = ' + at.toFixed(3) +
        ' · α<sub>t−1</sub> = ' + at1.toFixed(3);

      return fr;
    }

    let frB = null;
    function redrawB() { frB = drawB(); }

    cvPost.addEventListener('pointerdown', (e) => {
      if (!frB) return;
      const rect = cvPost.getBoundingClientRect();
      const px = e.clientX - rect.left;
      const dx = Math.abs(px - frB.X(xb)), dz = Math.abs(px - frB.X(ztb));
      if (Math.min(dx, dz) > 25) return;
      dragging = dx < dz ? 'x' : 'z';
      cvPost.setPointerCapture(e.pointerId);
    });
    cvPost.addEventListener('pointermove', (e) => {
      if (!dragging || !frB) return;
      const rect = cvPost.getBoundingClientRect();
      const px = e.clientX - rect.left;
      const val = Math.max(-ZRANGE, Math.min(ZRANGE,
        frB.xmin + (px - frB.pad.l) / frB.iw * (frB.xmax - frB.xmin)));
      if (dragging === 'x') xb = val; else ztb = val;
      redrawB();
    });
    const release = () => { dragging = null; };
    cvPost.addEventListener('pointerup', release);
    cvPost.addEventListener('pointercancel', release);

    $('s3b-t').addEventListener('input', (e) => { tb = +e.target.value; $('s3b-t-val').textContent = tb; redrawB(); });

    regenA();
    drawA();
    redrawB();
    P.onRedraw(() => { drawA(); redrawB(); });
    P.observeResize(cvTraj, drawA);
    P.observeResize(cvPost, redrawB);
  }

  DM.sections.push({ name: 's3-gaussian', init });
})();
