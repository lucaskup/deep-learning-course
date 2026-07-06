/* Seção 1: playground do processo de difusão (forward), 1D e 2D. */
(function () {
  'use strict';
  window.DM = window.DM || {};
  DM.sections = DM.sections || [];

  const T = 50, N1D = 300, N2D = 500, ZRANGE = 3.2;

  function init() {
    const U = DM.utils, P = DM.plot;
    const $ = (id) => document.getElementById(id);

    const cvTraj = $('s1-traj'), cvDens = $('s1-density'), cvMoons = $('s1-moons');
    const slider = $('s1-t');
    const btnPlay = $('s1-play'), btnResample = $('s1-resample');
    const tab1d = $('s1-tab-1d'), tab2d = $('s1-tab-2d');

    const sched = U.linearSchedule(T, 1e-4, 0.2);
    let seed = 42;
    let t = 0, view = '1d', playing = false, tAcc = 0;
    let traj1d = null, hi = [0, 1, 2];
    let moonsTraj = null, moonsLab = null;

    /* Pré-computa toda a aleatoriedade: o slider de t é determinístico nos dois sentidos. */
    function regenerate() {
      const rng = U.mulberry32(seed);
      traj1d = [];
      const z0 = new Float32Array(N1D);
      for (let i = 0; i < N1D; i++) z0[i] = U.sampleBimodal(rng);
      traj1d.push(z0);
      for (let s = 1; s <= T; s++) {
        const prev = traj1d[s - 1], cur = new Float32Array(N1D);
        const ka = Math.sqrt(1 - sched.beta[s]), kb = Math.sqrt(sched.beta[s]);
        for (let i = 0; i < N1D; i++) cur[i] = ka * prev[i] + kb * U.randn(rng);
        traj1d.push(cur);
      }
      /* 3 instâncias destacadas: a menor, a maior e a mediana em t = 0 */
      const order = Array.from({ length: N1D }, (_, i) => i).sort((a, b) => z0[a] - z0[b]);
      hi = [order[0], order[Math.floor(N1D / 2)], order[N1D - 1]];

      const { pts, lab } = U.twoMoons(N2D, 0.08, rng);
      moonsLab = lab;
      moonsTraj = [pts.map((p) => [p[0], p[1]])];
      for (let s = 1; s <= T; s++) {
        const prev = moonsTraj[s - 1];
        const ka = Math.sqrt(1 - sched.beta[s]), kb = Math.sqrt(sched.beta[s]);
        moonsTraj.push(prev.map((p) => [ka * p[0] + kb * U.randn(rng), ka * p[1] + kb * U.randn(rng)]));
      }
    }

    function drawTraj() {
      const th = P.theme();
      const { ctx, w, h } = P.setup(cvTraj);
      P.clear(ctx, w, h);
      const fr = P.frame(ctx, w, h, 0, T, -ZRANGE, ZRANGE);
      P.axes(ctx, fr, { xlabel: 't', ylabel: 'z', xticks: [0, 25, 50], yticks: [-3, 0, 3] });
      const ts = Array.from({ length: T + 1 }, (_, s) => s);
      for (let i = 0; i < N1D; i += 5) {
        if (hi.includes(i)) continue;
        P.line(ctx, fr, ts, ts.map((s) => traj1d[s][i]), th.cyan, { width: 0.7, alpha: 0.12 });
      }
      const hiColors = [th.pink, th.orange, th.green];
      hi.forEach((idx, k) => {
        P.line(ctx, fr, ts, ts.map((s) => traj1d[s][idx]), hiColors[k], { width: 2 });
      });
      P.line(ctx, fr, [t, t], [-ZRANGE, ZRANGE], th.comment, { dash: [3, 3], width: 1 });
      hi.forEach((idx, k) => {
        P.scatter(ctx, fr, [[t, traj1d[t][idx]]], hiColors[k], { r: 4, alpha: 1 });
      });
    }

    function drawDensity() {
      const th = P.theme();
      const { ctx, w, h } = P.setup(cvDens);
      P.clear(ctx, w, h);
      const fr = P.frame(ctx, w, h, -ZRANGE, ZRANGE, 0, 0.85);
      P.axes(ctx, fr, { xlabel: 'z', xticks: [-3, 0, 3], yticks: [0, 0.4, 0.8] });
      P.histogram(ctx, fr, traj1d[t], 40, th.cyan);
      const xs = [], exact = [], ref = [];
      for (let x = -ZRANGE; x <= ZRANGE; x += 0.05) {
        xs.push(x);
        exact.push(U.bimodalMarginalPdf(x, sched.alpha[t]));
        ref.push(U.gaussPdf(x, 0, 1));
      }
      P.line(ctx, fr, xs, ref, th.comment, { dash: [5, 4], width: 1.4 });
      P.line(ctx, fr, xs, exact, th.purple, { width: 2.2 });
      P.label(ctx, fr.X(-3) + 8, fr.Y(0.85) + 12, 'q(z_t) exata', th.purple);
      P.label(ctx, fr.X(-3) + 8, fr.Y(0.85) + 26, '𝒩(0, 1)', th.comment);
    }

    function drawMoons() {
      const th = P.theme();
      const { ctx, w, h } = P.setup(cvMoons);
      P.clear(ctx, w, h);
      const fr = P.frame(ctx, w, h, -3, 3, -3, 3);
      P.axes(ctx, fr, { xlabel: 'z₁', ylabel: 'z₂', xticks: [-2, 0, 2], yticks: [-2, 0, 2] });
      ctx.strokeStyle = th.comment;
      ctx.setLineDash([4, 4]);
      for (const r of [1, 2]) {
        ctx.beginPath();
        ctx.ellipse(fr.X(0), fr.Y(0), r * fr.iw / 6, r * fr.ih / 6, 0, 0, 2 * Math.PI);
        ctx.stroke();
      }
      ctx.setLineDash([]);
      const pts = moonsTraj[t];
      const a = pts.filter((_, i) => moonsLab[i] === 0);
      const b = pts.filter((_, i) => moonsLab[i] === 1);
      P.scatter(ctx, fr, a, th.cyan, { r: 2.5, alpha: 0.7 });
      P.scatter(ctx, fr, b, th.orange, { r: 2.5, alpha: 0.7 });
    }

    function redraw() {
      $('s1-t-val').textContent = t;
      if (view === '1d') { drawTraj(); drawDensity(); }
      else drawMoons();
    }

    function tick(dt) {
      tAcc += dt * 10;
      while (tAcc >= 1 && t < T) { t++; tAcc -= 1; }
      slider.value = t;
      redraw();
      if (t >= T) stop();
    }

    function stop() {
      playing = false;
      btnPlay.textContent = '▶ Difundir';
      DM.stopTicker(tick);
    }

    btnPlay.addEventListener('click', () => {
      if (playing) { stop(); return; }
      if (t >= T) { t = 0; slider.value = 0; }
      playing = true;
      tAcc = 0;
      btnPlay.textContent = '⏸ Pausar';
      DM.startTicker(tick);
    });

    slider.addEventListener('input', () => { stop(); t = +slider.value; redraw(); });
    btnResample.addEventListener('click', () => {
      seed = (seed * 1664525 + 1013904223) >>> 0;
      regenerate();
      redraw();
    });

    tab1d.addEventListener('click', () => {
      view = '1d';
      tab1d.classList.add('active'); tab2d.classList.remove('active');
      $('s1-row-1d').classList.remove('hidden'); $('s1-row-2d').classList.add('hidden');
      redraw();
    });
    tab2d.addEventListener('click', () => {
      view = '2d';
      tab2d.classList.add('active'); tab1d.classList.remove('active');
      $('s1-row-2d').classList.remove('hidden'); $('s1-row-1d').classList.add('hidden');
      redraw();
    });

    regenerate();
    redraw();
    P.onRedraw(redraw);
    P.observeResize(cvTraj, redraw);
    P.observeResize(cvMoons, redraw);
  }

  DM.sections.push({ name: 's1-forward', init });
})();
