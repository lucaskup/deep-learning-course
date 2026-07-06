/* Seção 1: capacidade, sobreajuste e regularização L2 (ridge em base polinomial). */
(function () {
  'use strict';
  window.DL = window.DL || {};
  DL.sections = DL.sections || [];

  const DMAX = 15, NTRAIN = 25, NVAL = 25, NOISE = 0.3;
  const YMIN = -2, YMAX = 2;

  function trueF(x) { return Math.sin(2 * Math.PI * x); }

  /* Linha da matriz de design: polinômios de Chebyshev T_0..T_d em x ∈ [−1,1].
     Mesmo espaço de funções dos monômios x^k, mas XᵀX fica bem condicionada
     até grau alto (monômios puros explodem numericamente perto de d = 15). */
  function chebRow(x, d) {
    const r = new Float64Array(d + 1);
    r[0] = 1;
    if (d >= 1) r[1] = x;
    for (let k = 2; k <= d; k++) r[k] = 2 * x * r[k - 1] - r[k - 2];
    return r;
  }

  /* Resolve A·w = b por eliminação gaussiana com pivoteamento parcial. */
  function solve(A, b) {
    const n = b.length;
    for (let col = 0; col < n; col++) {
      let piv = col;
      for (let r = col + 1; r < n; r++) if (Math.abs(A[r][col]) > Math.abs(A[piv][col])) piv = r;
      if (piv !== col) {
        const tr = A[piv]; A[piv] = A[col]; A[col] = tr;
        const tb = b[piv]; b[piv] = b[col]; b[col] = tb;
      }
      const d = A[col][col] || 1e-12;
      for (let r = col + 1; r < n; r++) {
        const f = A[r][col] / d;
        if (f === 0) continue;
        for (let c = col; c < n; c++) A[r][c] -= f * A[col][c];
        b[r] -= f * b[col];
      }
    }
    const w = new Float64Array(n);
    for (let r = n - 1; r >= 0; r--) {
      let s = b[r];
      for (let c = r + 1; c < n; c++) s -= A[r][c] * w[c];
      w[r] = s / (A[r][r] || 1e-12);
    }
    return w;
  }

  /* Ridge em forma fechada: w = (XᵀX + λI)⁻¹·Xᵀy, sem penalizar o viés (T_0). */
  function fitRidge(xs, ys, d, lambda) {
    const n = d + 1;
    const A = [];
    for (let i = 0; i < n; i++) A.push(new Float64Array(n));
    const b = new Float64Array(n);
    for (let p = 0; p < xs.length; p++) {
      const row = chebRow(xs[p], d);
      for (let i = 0; i < n; i++) {
        b[i] += row[i] * ys[p];
        for (let j = 0; j < n; j++) A[i][j] += row[i] * row[j];
      }
    }
    for (let i = 1; i < n; i++) A[i][i] += lambda;
    return solve(A, b);
  }

  function predict(w, x) {
    const row = chebRow(x, w.length - 1);
    let s = 0;
    for (let i = 0; i < w.length; i++) s += w[i] * row[i];
    return s;
  }

  function mse(w, xs, ys) {
    let s = 0;
    for (let i = 0; i < xs.length; i++) {
      const e = predict(w, xs[i]) - ys[i];
      s += e * e;
    }
    return s / xs.length;
  }

  function weightNorm2(w) {
    let s = 0;
    for (let i = 1; i < w.length; i++) s += w[i] * w[i];   /* viés fora da norma */
    return s;
  }

  /* Exposto para teste no console/node: DL.ridgeTest.fitRidge(...) */
  DL.ridgeTest = { chebRow, solve, fitRidge, predict, mse, weightNorm2 };

  function init() {
    const U = DL.utils, P = DL.plot;
    const $ = (id) => document.getElementById(id);

    const cvFit = $('s1-fit'), cvMse = $('s1-mse');
    const sliderD = $('s1-d'), sliderLam = $('s1-lam');
    const btnResample = $('s1-resample');

    let seed = 42;
    let train = null, val = null;
    let w = null;
    let mseTrainByD = [], mseValByD = [];

    function lambdaFromSlider() {
      const pos = +sliderLam.value;
      if (pos === 0) return 0;
      return Math.pow(10, -9 + 0.5 * (pos - 1));   /* 10⁻⁹ … 10¹ */
    }

    const SUP = { '0': '⁰', '1': '¹', '2': '²', '3': '³', '4': '⁴', '5': '⁵', '6': '⁶', '7': '⁷', '8': '⁸', '9': '⁹', '-': '⁻' };
    function pow10str(e) {
      return '10' + String(e).split('').map((c) => SUP[c]).join('');
    }
    function fmtLam(pos) {
      if (pos === 0) return '0';
      const e = -9 + 0.5 * (pos - 1);
      if (e === 0) return '1';
      if (e === 0.5) return '3.2';
      if (e === 1) return '10';
      if (Number.isInteger(e)) return pow10str(e);
      return '3.2×' + pow10str(Math.floor(e));
    }

    function fmtMse(v) {
      if (v >= 100) return v.toFixed(0);
      if (v >= 0.01) return v.toFixed(3);
      return v.toExponential(1);
    }

    function regenerate() {
      const rng = U.mulberry32(seed);
      train = U.noisyCurve(NTRAIN, trueF, NOISE, -1, 1, rng);
      val = U.noisyCurve(NVAL, trueF, NOISE, -1, 1, rng);
    }

    function recompute() {
      const d = +sliderD.value;
      const lam = lambdaFromSlider();
      w = fitRidge(train.xs, train.ys, d, lam);
      mseTrainByD = []; mseValByD = [];
      for (let dd = 1; dd <= DMAX; dd++) {
        const wd = dd === d ? w : fitRidge(train.xs, train.ys, dd, lam);
        mseTrainByD.push(mse(wd, train.xs, train.ys));
        mseValByD.push(mse(wd, val.xs, val.ys));
      }
      $('s1-d-val').textContent = d;
      $('s1-lam-val').textContent = fmtLam(+sliderLam.value);
      $('s1-readout').textContent =
        'MSE treino: ' + fmtMse(mseTrainByD[d - 1]) +
        ' · MSE validação: ' + fmtMse(mseValByD[d - 1]) +
        ' · ‖w‖²: ' + fmtMse(weightNorm2(w));
    }

    function drawFit() {
      const th = P.theme();
      const { ctx, w: cw, h: ch } = P.setup(cvFit);
      P.clear(ctx, cw, ch);
      const fr = P.frame(ctx, cw, ch, -1.08, 1.08, YMIN, YMAX);
      P.axes(ctx, fr, { xlabel: 'x', ylabel: 'y', xticks: [-1, 0, 1], yticks: [-2, 0, 2] });

      /* Curvas recortadas à área do gráfico (graus altos disparam fora do range). */
      ctx.save();
      ctx.beginPath();
      ctx.rect(fr.X(fr.xmin), fr.Y(fr.ymax), fr.iw, fr.ih);
      ctx.clip();
      const xs = [], yTrue = [], yFit = [];
      for (let x = -1.08; x <= 1.081; x += 0.01) {
        xs.push(x);
        yTrue.push(trueF(x));
        yFit.push(predict(w, x));
      }
      P.line(ctx, fr, xs, yTrue, th.comment, { dash: [5, 4], width: 1.4 });
      P.line(ctx, fr, xs, yFit, th.purple, { width: 2.2 });
      ctx.restore();

      P.scatter(ctx, fr, train.xs.map((x, i) => [x, train.ys[i]]), th.cyan, { r: 3, alpha: 0.9 });
      ctx.strokeStyle = th.orange;
      ctx.lineWidth = 1.4;
      for (let i = 0; i < val.xs.length; i++) {
        const px = fr.X(val.xs[i]), py = fr.Y(val.ys[i]);
        if (py < fr.pad.t || py > fr.pad.t + fr.ih) continue;
        ctx.beginPath();
        ctx.arc(px, py, 3, 0, 2 * Math.PI);
        ctx.stroke();
      }

      P.label(ctx, fr.X(-1.08) + 8, 14, 'ajuste (grau ' + sliderD.value + ')', th.purple);
      P.label(ctx, fr.X(-1.08) + 8, 28, 'sin(2πx)', th.comment);
    }

    function drawMse() {
      const th = P.theme();
      const { ctx, w: cw, h: ch } = P.setup(cvMse);
      P.clear(ctx, cw, ch);
      const ymin = -3, ymax = 1.4;   /* log10 do MSE */
      const fr = P.frame(ctx, cw, ch, 0.5, DMAX + 0.5, ymin, ymax);
      P.axes(ctx, fr, { xlabel: 'd', xticks: [1, 5, 10, 15] });

      for (const v of [0.01, 0.1, 1, 10]) {
        P.line(ctx, fr, [0.5, DMAX + 0.5], [Math.log10(v), Math.log10(v)], th.line, { dash: [3, 4], width: 1 });
        P.label(ctx, fr.X(0.5) - 4, fr.Y(Math.log10(v)) + 4, String(v), th.comment, 'right');
      }

      const clamp = (v) => Math.min(ymax, Math.max(ymin, Math.log10(Math.max(v, 1e-12))));
      const ds = [];
      for (let dd = 1; dd <= DMAX; dd++) ds.push(dd);
      P.line(ctx, fr, ds, mseTrainByD.map(clamp), th.cyan, { width: 2 });
      P.line(ctx, fr, ds, mseValByD.map(clamp), th.orange, { width: 2 });

      const d = +sliderD.value;
      P.line(ctx, fr, [d, d], [ymin, ymax], th.comment, { dash: [3, 3], width: 1 });
      P.scatter(ctx, fr, [[d, clamp(mseTrainByD[d - 1])]], th.cyan, { r: 4, alpha: 1 });
      P.scatter(ctx, fr, [[d, clamp(mseValByD[d - 1])]], th.orange, { r: 4, alpha: 1 });

      P.label(ctx, fr.X(0.5) + 8, 14, 'validação', th.orange);
      P.label(ctx, fr.X(0.5) + 8, 28, 'treino', th.cyan);
    }

    function redraw() { drawFit(); drawMse(); }
    function update() { recompute(); redraw(); }

    sliderD.addEventListener('input', update);
    sliderLam.addEventListener('input', update);
    btnResample.addEventListener('click', () => {
      seed = (seed * 1664525 + 1013904223) >>> 0;
      regenerate();
      update();
    });

    regenerate();
    update();
    P.onRedraw(redraw);
    P.observeResize(cvFit, drawFit);
    P.observeResize(cvMse, drawMse);
  }

  DL.sections.push({ name: 's1-ridge', init });
})();
