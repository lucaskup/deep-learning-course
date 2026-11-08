/* Seção 2: Forget Quality via teste de duas amostras de Kolmogorov-Smirnov.
   Compara as CDFs empíricas dos Truth Ratios do modelo desaprendido (θ_u) e do
   oráculo (θ*). D = sup|F_u − F*| e o p-valor exato do teste KS. */
(function () {
  'use strict';
  window.DL = window.DL || {};
  DL.sections = DL.sections || [];

  const XMIN = 0, XMAX = 3;
  const fm = (v, d) => v.toFixed(d == null ? 2 : d).replace('.', ',');

  const PRESETS = {
    reset: {
      u: [0.6, 0.8, 0.95, 1.05, 1.2, 1.5],
      s: [0.7, 0.9, 1.0, 1.1, 1.3, 1.6],
    },
    good: {
      u: [0.75, 0.9, 1.0, 1.05, 1.15, 1.35],
      s: [0.7, 0.92, 1.02, 1.08, 1.2, 1.4],
    },
    bad: {
      u: [0.2, 0.35, 0.5, 0.6, 0.75, 0.9],
      s: [1.4, 1.6, 1.8, 2.0, 2.2, 2.5],
    },
  };

  function init() {
    if (!document.getElementById('s2-cdf')) return;
    const P = DL.plot;
    const $ = (id) => document.getElementById(id);

    const cvCdf = $('s2-cdf'), cvDots = $('s2-dots');

    let U = PRESETS.reset.u.slice();
    let S = PRESETS.reset.s.slice();
    let frDots = null, drag = null;

    const LANE_U = 0.66, LANE_S = 0.33;

    /* F(t) = (#valores ≤ t) / n, avaliada num ponto. */
    function ecdfAt(vals, t) {
      let c = 0;
      for (const v of vals) if (v <= t) c++;
      return c / vals.length;
    }

    /* Polilinha em degraus da CDF empírica sobre [XMIN, XMAX]. */
    function stepPoly(vals) {
      const sorted = vals.slice().sort((a, b) => a - b);
      const xs = [XMIN], ys = [0];
      let cum = 0, i = 0;
      const n = sorted.length;
      while (i < n) {
        let j = i;
        while (j < n && sorted[j] === sorted[i]) j++;
        const v = sorted[i], mult = j - i;
        xs.push(v); ys.push(cum / n);
        cum += mult;
        xs.push(v); ys.push(cum / n);
        i = j;
      }
      xs.push(XMAX); ys.push(1);
      return { xs, ys };
    }

    /* D = sup|F_u − F*|, avaliado na união dos valores (onde os saltos ocorrem). */
    function ksStat() {
      const cands = U.concat(S);
      let D = 0, xStar = cands[0], fuStar = 0, fsStar = 0;
      for (const t of cands) {
        const fu = ecdfAt(U, t), fs = ecdfAt(S, t);
        const d = Math.abs(fu - fs);
        if (d > D) { D = d; xStar = t; fuStar = fu; fsStar = fs; }
      }
      return { D, xStar, fuStar, fsStar };
    }

    /* p-valor assintótico (Kolmogorov), usado só para amostras grandes. */
    function ksPValueAsymptotic(D, n, m) {
      const ne = (n * m) / (n + m);
      const sne = Math.sqrt(ne);
      const lambda = (sne + 0.12 + 0.11 / sne) * D;
      let sum = 0;
      for (let k = 1; k <= 100; k++) {
        sum += (k % 2 === 1 ? 1 : -1) * Math.exp(-2 * k * k * lambda * lambda);
      }
      return Math.max(0, Math.min(1, 2 * sum));
    }

    /* p-valor exato do teste KS de duas amostras: P(D_{n,m} ≥ D_obs).
       Sob H0 todas as C(n+m, n) ordenações das duas amostras são
       equiprováveis; cada uma é um caminho de (0,0) a (n,m) na grade, e
       D < D_obs equivale ao caminho ficar estritamente dentro da faixa
       |i/n − j/m| < D_obs. Contamos esses caminhos por programação dinâmica. */
    function ksPValue(D) {
      const n = U.length, m = S.length;
      if (D <= 1e-12) return 1;
      if (n * m > 10000) return ksPValueAsymptotic(D, n, m);
      const EPS = 1e-12;
      const W = m + 1;
      const paths = new Float64Array((n + 1) * W);
      for (let i = 0; i <= n; i++) {
        for (let j = 0; j <= m; j++) {
          const inside = Math.abs(i / n - j / m) < D - EPS;
          if (!inside) { paths[i * W + j] = 0; continue; }
          if (i === 0 && j === 0) { paths[0] = 1; continue; }
          paths[i * W + j] = (i > 0 ? paths[(i - 1) * W + j] : 0) + (j > 0 ? paths[i * W + j - 1] : 0);
        }
      }
      let total = 1;                       /* C(n+m, n) em double */
      for (let k = 1; k <= n; k++) total = total * (m + k) / k;
      const p = 1 - paths[n * W + m] / total;
      return Math.max(0, Math.min(1, p));
    }

    /* ── CDFs empíricas ── */
    function drawCdf() {
      const th = P.theme();
      const { ctx, w, h } = P.setup(cvCdf);
      P.clear(ctx, w, h);
      const fr = P.frame(ctx, w, h, XMIN, XMAX, 0, 1.02);
      P.axes(ctx, fr, {
        xlabel: 'Truth Ratio', ylabel: 'CDF',
        xticks: [0, 1, 2, 3], yticks: [0, 0.5, 1],
      });

      const pu = stepPoly(U), ps = stepPoly(S);
      P.line(ctx, fr, ps.xs, ps.ys, th.orange, { width: 2 });
      P.line(ctx, fr, pu.xs, pu.ys, th.cyan, { width: 2 });

      const k = ksStat();
      /* Segmento vertical em D. */
      ctx.strokeStyle = th.red;
      ctx.lineWidth = 2.4;
      ctx.beginPath();
      ctx.moveTo(fr.X(k.xStar), fr.Y(k.fuStar));
      ctx.lineTo(fr.X(k.xStar), fr.Y(k.fsStar));
      ctx.stroke();
      /* Rótulo do lado em que cabe no quadro. */
      const dRight = fr.X(k.xStar) + 6 + 52 < fr.X(XMAX);
      P.mathText(ctx, 'D = ' + fm(k.D), fr.X(k.xStar) + (dRight ? 6 : -6),
        fr.Y((k.fuStar + k.fsStar) / 2) + 4, th.red, dRight ? 'left' : 'right', 11);

      P.mathText(ctx, 'F_u', fr.X(0.08), fr.Y(0.95), th.cyan, 'left', 12);
      P.mathText(ctx, 'F_*', fr.X(0.08), fr.Y(0.82), th.orange, 'left', 12);
    }

    /* ── Pontos arrastáveis ── */
    function drawDots() {
      const th = P.theme();
      const { ctx, w, h } = P.setup(cvDots);
      P.clear(ctx, w, h);
      const fr = P.frame(ctx, w, h, XMIN, XMAX, 0, 1, { l: 16, r: 12, t: 28, b: 24 });
      frDots = fr;

      /* Eixo de valores. */
      ctx.strokeStyle = th.line;
      ctx.lineWidth = 1;
      for (const lane of [LANE_U, LANE_S]) {
        ctx.beginPath();
        ctx.moveTo(fr.X(XMIN), fr.Y(lane));
        ctx.lineTo(fr.X(XMAX), fr.Y(lane));
        ctx.stroke();
      }
      ctx.fillStyle = th.comment;
      ctx.font = '11px sans-serif';
      ctx.textAlign = 'center';
      for (const v of [0, 1, 2, 3]) ctx.fillText(String(v), fr.X(v), fr.Y(0) + 6);

      P.mathText(ctx, 'θ_u', fr.X(XMIN) + 2, fr.Y(LANE_U) - 12, th.cyan, 'left', 12);
      P.mathText(ctx, 'θ_*', fr.X(XMIN) + 2, fr.Y(LANE_S) - 12, th.orange, 'left', 12);

      for (let i = 0; i < U.length; i++) {
        P.scatter(ctx, fr, [[U[i], LANE_U]], th.cyan, { r: 6, alpha: 0.95 });
      }
      for (let i = 0; i < S.length; i++) {
        P.scatter(ctx, fr, [[S[i], LANE_S]], th.orange, { r: 6, alpha: 0.95 });
      }
    }

    function updateReadout() {
      const k = ksStat();
      const p = ksPValue(k.D);
      const ok = p >= 0.05;
      $('s2-readout').textContent =
        'D = ' + fm(k.D, 3) + ' · p = ' + fm(p, 3) + ' · ' +
        (ok ? 'p alto ⇒ o teste não distingue as distribuições ⇒ esquecimento compatível com o oráculo'
            : 'p baixo ⇒ distribuições distintas ⇒ esquecimento insuficiente');
    }

    function redraw() { drawCdf(); drawDots(); updateReadout(); }

    /* Acerta o ponto mais próximo (na lane) e o arrasta horizontalmente. */
    function hitTest(px, py) {
      let best = null, bestD = 14;
      const test = (arr, lane, key) => {
        for (let i = 0; i < arr.length; i++) {
          const d = Math.hypot(px - frDots.X(arr[i]), py - frDots.Y(lane));
          if (d < bestD) { bestD = d; best = { arr, i, key }; }
        }
      };
      test(U, LANE_U, 'u');
      test(S, LANE_S, 's');
      return best;
    }

    cvDots.addEventListener('pointerdown', (e) => {
      if (!frDots) return;
      const r = cvDots.getBoundingClientRect();
      drag = hitTest(e.clientX - r.left, e.clientY - r.top);
      if (drag) cvDots.setPointerCapture(e.pointerId);
    });
    cvDots.addEventListener('pointermove', (e) => {
      if (!drag || !frDots) return;
      const r = cvDots.getBoundingClientRect();
      const v = Math.max(XMIN + 0.05, Math.min(XMAX - 0.05, frDots.invX(e.clientX - r.left)));
      drag.arr[drag.i] = v;
      redraw();
    });
    cvDots.addEventListener('pointerup', () => { drag = null; });
    cvDots.addEventListener('pointercancel', () => { drag = null; });

    const apply = (key) => { U = PRESETS[key].u.slice(); S = PRESETS[key].s.slice(); redraw(); };
    $('s2-good').addEventListener('click', () => apply('good'));
    $('s2-bad').addEventListener('click', () => apply('bad'));
    $('s2-reset').addEventListener('click', () => apply('reset'));

    redraw();
    P.onRedraw(redraw);
    P.observeResize(cvCdf, redraw);
    P.observeResize(cvDots, redraw);
  }

  DL.sections.push({ name: 's2-ks', init });
})();
