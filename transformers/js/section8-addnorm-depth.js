/* Seção 4: Add & Norm. LayerNorm por token (batch antes/depois) e o efeito
   da conexão residual ao empilhar N blocos: norma do sinal por profundidade. */
(function () {
  'use strict';
  window.DL = window.DL || {};
  DL.sections = DL.sections || [];

  const T = 6, D = 16, MAXN = 32;

  function init() {
    const P = DL.plot;
    const U = DL.utils;
    const $ = (id) => document.getElementById(id);

    const cvLN = $('s8-ln'), cvDepth = $('s8-depth');
    const slMu = $('s8-mu'), slSigma = $('s8-sigma'), slGamma = $('s8-gamma'), slBeta = $('s8-beta');
    const slN = $('s8-N'), slGain = $('s8-gain');

    let mu = +slMu.value, sigma = +slSigma.value, gamma = +slGamma.value, beta = +slBeta.value;
    let N = +slN.value, gain = +slGain.value;
    let wseed = 3;

    /* ── base fixa do batch: ruído por célula + deslocamento/escala por token ── */
    const base = [], shift = [], scl = [];
    {
      const rng = U.mulberry32(5);
      for (let t = 0; t < T; t++) {
        const row = new Array(D);
        for (let k = 0; k < D; k++) row[k] = U.randn(rng);
        base.push(row);
        shift.push(U.randn(rng) * 1.2);
        scl.push(0.5 + rng());
      }
    }

    function lnBatch() {
      const before = [], after = [];
      for (let t = 0; t < T; t++) {
        const x = base[t].map((b) => mu * shift[t] + sigma * scl[t] * b);
        let m = 0;
        for (const v of x) m += v;
        m /= D;
        let s2 = 0;
        for (const v of x) s2 += (v - m) * (v - m);
        const sd = Math.sqrt(s2 / D) || 1;
        before.push(x);
        after.push(x.map((v) => gamma * ((v - m) / sd) + beta));
      }
      return { before, after };
    }

    function cellColor(th, v) {
      const c = P.rgb(v >= 0 ? th.cyan : th.orange);
      const a = 0.06 + 0.94 * Math.min(1, Math.abs(v) / 3);
      return 'rgba(' + c[0] + ',' + c[1] + ',' + c[2] + ',' + a + ')';
    }

    function drawMatrix(ctx, th, M_, x0, y0, cw, chh, title) {
      ctx.fillStyle = th.fg;
      ctx.font = '11px sans-serif';
      ctx.textAlign = 'left';
      ctx.fillText(title, x0, y0 - 6);
      for (let t = 0; t < T; t++) {
        for (let k = 0; k < D; k++) {
          ctx.fillStyle = cellColor(th, M_[t][k]);
          ctx.fillRect(x0 + k * cw, y0 + t * chh, cw + 0.5, chh + 0.5);
        }
      }
      ctx.strokeStyle = th.line;
      ctx.strokeRect(x0, y0, D * cw, T * chh);
      ctx.fillStyle = th.comment;
      ctx.font = '9px sans-serif';
      ctx.textAlign = 'right';
      for (let t = 0; t < T; t++) ctx.fillText('x_' + (t + 1), x0 - 3, y0 + t * chh + chh / 2 + 3);
    }

    function drawLN(stats) {
      const th = P.theme();
      const { ctx, w, h } = P.setup(cvLN);
      P.clear(ctx, w, h);
      const padL = 30, gap = 44, top = 26;
      const panelW = (w - 2 * padL - gap) / 2;
      const cw = panelW / D;
      const chh = Math.min(24, (h - top - 26) / T);
      drawMatrix(ctx, th, stats.before, padL, top, cw, chh, 'x (antes)');
      drawMatrix(ctx, th, stats.after, padL + panelW + gap, top, cw, chh, 'LN(x) (depois)');
      /* seta entre painéis */
      const ay = top + (T * chh) / 2;
      ctx.strokeStyle = th.fg;
      ctx.lineWidth = 1.5;
      ctx.beginPath();
      ctx.moveTo(padL + panelW + 8, ay);
      ctx.lineTo(padL + panelW + gap - 8, ay);
      ctx.stroke();
      ctx.fillStyle = th.fg;
      ctx.beginPath();
      ctx.moveTo(padL + panelW + gap - 8, ay);
      ctx.lineTo(padL + panelW + gap - 14, ay - 4);
      ctx.lineTo(padL + panelW + gap - 14, ay + 4);
      ctx.fill();
      ctx.font = '10px sans-serif';
      ctx.textAlign = 'center';
      ctx.fillStyle = th.comment;
      ctx.fillText('LN', padL + panelW + gap / 2, ay - 8);
      /* estatísticas por linha (token 1) */
      const row = stats.before[0], rowA = stats.after[0];
      const mean = (xs) => xs.reduce((a, b) => a + b, 0) / xs.length;
      const std = (xs) => {
        const m = mean(xs);
        return Math.sqrt(xs.reduce((a, b) => a + (b - m) * (b - m), 0) / xs.length);
      };
      ctx.textAlign = 'left';
      ctx.fillText(
        'token 1: μ = ' + mean(row).toFixed(2) + ', σ = ' + std(row).toFixed(2) +
        '  →  μ = ' + mean(rowA).toFixed(2) + ', σ = ' + std(rowA).toFixed(2),
        padL, top + T * chh + 16);
    }

    /* ── pilha de N blocos: F(x) = W2·relu(W1·x), pesos com ganho g ── */
    let blocks = [];
    function buildBlocks() {
      const rng = U.mulberry32(wseed);
      blocks = [];
      for (let l = 0; l < MAXN; l++) {
        const W1 = [], W2 = [];
        for (let r = 0; r < D; r++) {
          const r1 = new Array(D), r2 = new Array(D);
          for (let c = 0; c < D; c++) {
            r1[c] = U.randn(rng) / Math.sqrt(D);
            r2[c] = U.randn(rng) / Math.sqrt(D);
          }
          W1.push(r1);
          W2.push(r2);
        }
        blocks.push({ W1, W2 });
      }
    }

    function matVec(W, x) {
      const out = new Array(D).fill(0);
      for (let r = 0; r < D; r++) {
        const Wr = W[r];
        let s = 0;
        for (let c = 0; c < D; c++) s += Wr[c] * x[c];
        out[r] = s;
      }
      return out;
    }

    function F(l, x) {
      const hdn = matVec(blocks[l].W1, x).map((v) => Math.max(0, v) * Math.SQRT2);
      return matVec(blocks[l].W2, hdn).map((v) => v * gain);
    }

    function layerNorm(x) {
      let m = 0;
      for (const v of x) m += v;
      m /= D;
      let s2 = 0;
      for (const v of x) s2 += (v - m) * (v - m);
      const sd = Math.sqrt(s2 / D) || 1;
      return x.map((v) => (v - m) / sd);
    }

    const norm = (x) => Math.sqrt(x.reduce((a, b) => a + b * b, 0));
    const cosSim = (a, b) => {
      let s = 0;
      for (let i = 0; i < D; i++) s += a[i] * b[i];
      return s / ((norm(a) * norm(b)) || 1);
    };

    let x0 = [];
    {
      const rng = U.mulberry32(99);
      for (let i = 0; i < D; i++) x0.push(U.randn(rng));
    }

    function runStacks() {
      let plain = x0.slice(), res = x0.slice(), resLN = layerNorm(x0);
      const nP = [norm(plain)], nR = [norm(res)], nL = [norm(resLN)];
      for (let l = 0; l < N; l++) {
        plain = F(l, plain);
        const fr2 = F(l, res);
        res = res.map((v, i) => v + fr2[i]);
        const fr3 = F(l, resLN);
        resLN = layerNorm(resLN.map((v, i) => v + fr3[i]));
        nP.push(norm(plain));
        nR.push(norm(res));
        nL.push(norm(resLN));
      }
      return { nP, nR, nL, plain, res, resLN };
    }

    function drawDepth(run) {
      const th = P.theme();
      const { ctx, w, h } = P.setup(cvDepth);
      P.clear(ctx, w, h);
      const lg = (v) => Math.max(-6, Math.min(6, v > 0 ? Math.log10(v) : -6));
      const yP = run.nP.map(lg), yR = run.nR.map(lg), yL = run.nL.map(lg);
      const all = yP.concat(yR, yL);
      const ymin = Math.min(-1, Math.floor(Math.min(...all))) - 0.3;
      const ymax = Math.max(1.5, Math.ceil(Math.max(...all))) + 0.3;
      const fr = P.frame(ctx, w, h, 0, N, ymin, ymax);
      P.axes(ctx, fr, {
        xlabel: 'bloco ℓ', ylabel: 'log₁₀ ‖x‖',
        xticks: [0, Math.round(N / 2), N],
        yticks: [Math.ceil(ymin), 0, Math.floor(ymax)],
      });
      const ls = yP.map((_, i) => i);
      P.line(ctx, fr, ls, yP, th.orange, { width: 1.8 });
      P.line(ctx, fr, ls, yR, th.green, { width: 1.8 });
      P.line(ctx, fr, ls, yL, th.cyan, { width: 1.8 });
      P.label(ctx, fr.X(0) + 8, fr.Y(ymax) + 14, 'F(x) puro', th.orange);
      P.label(ctx, fr.X(0) + 8, fr.Y(ymax) + 28, 'x + F(x)', th.green);
      P.label(ctx, fr.X(0) + 8, fr.Y(ymax) + 42, 'LN(x + F(x))', th.cyan);
    }

    function updateReadout(run) {
      $('s8-readout').textContent =
        'cos(x^(0), x^(N)): com residual = ' + cosSim(x0, run.res).toFixed(2) +
        ' · sem residual = ' + cosSim(x0, run.plain).toFixed(2);
    }

    function redraw() {
      drawLN(lnBatch());
      const run = runStacks();
      drawDepth(run);
      updateReadout(run);
    }

    function bindSlider(sl, idVal, fmt, set) {
      sl.addEventListener('input', () => {
        set(+sl.value);
        $(idVal).textContent = fmt(+sl.value);
        redraw();
      });
    }
    bindSlider(slMu, 's8-mu-val', (v) => v.toFixed(1), (v) => { mu = v; });
    bindSlider(slSigma, 's8-sigma-val', (v) => v.toFixed(1), (v) => { sigma = v; });
    bindSlider(slGamma, 's8-gamma-val', (v) => v.toFixed(1), (v) => { gamma = v; });
    bindSlider(slBeta, 's8-beta-val', (v) => v.toFixed(1), (v) => { beta = v; });
    bindSlider(slN, 's8-N-val', (v) => String(v), (v) => { N = v; });
    bindSlider(slGain, 's8-gain-val', (v) => v.toFixed(2), (v) => { gain = v; });

    $('s8-resample').addEventListener('click', () => {
      wseed = (wseed * 1664525 + 1013904223) >>> 0;
      buildBlocks();
      redraw();
    });

    buildBlocks();
    redraw();
    P.onRedraw(redraw);
    P.observeResize(cvLN, redraw);
    P.observeResize(cvDepth, redraw);
  }

  DL.sections.push({ name: 's8-addnorm-depth', init });
})();
