/* Seção 1: limiar de decisão e tipos de erro.
   Duas densidades gaussianas de escore (classe y=0 e y=1), limiar arrastável
   e métricas (FPR, FNR, PPV, taxa de positivos, acurácia) em barras. */
(function () {
  'use strict';
  window.DL = window.DL || {};
  DL.sections = DL.sections || [];

  /* Parâmetros do classificador sintético (iguais aos da seção 2). */
  const M0 = 0.35, M1 = 0.65, SD = 0.13;

  /* CDF da normal padrão via aproximação de erf (Abramowitz-Stegun 7.1.26). */
  function phi(z) {
    /* Φ(z) = ½(1 + erf(z/√2)); a aproximação abaixo é de erf(x), x = |z|/√2. */
    const x = Math.abs(z) / Math.SQRT2;
    const t = 1 / (1 + 0.3275911 * x);
    const y = 1 - (((((1.061405429 * t - 1.453152027) * t) + 1.421413741) * t - 0.284496736) * t + 0.254829592) * t * Math.exp(-x * x);
    const e = 0.5 * (1 + y);
    return z >= 0 ? e : 1 - e;
  }

  /* Métricas analíticas dadas taxa-base p e limiar tau. */
  function metrics(p, tau) {
    const tpr = 1 - phi((tau - M1) / SD);
    const fpr = 1 - phi((tau - M0) / SD);
    const rate = p * tpr + (1 - p) * fpr;
    const ppv = rate > 1e-9 ? (p * tpr) / rate : 0;
    const acc = p * tpr + (1 - p) * (1 - fpr);
    return { tpr, fpr, fnr: 1 - tpr, rate, ppv, acc };
  }

  function gauss(x, mu) {
    return DL.utils.gaussPdf(x, mu, SD * SD);
  }

  const state = { tau: 0.5, p: 0.45 };

  function init() {
    const distCv = document.getElementById('s1-dist');
    const metCv = document.getElementById('s1-metrics');
    const tauSlider = document.getElementById('s1-tau');
    const tauVal = document.getElementById('s1-tau-val');
    const pSlider = document.getElementById('s1-p');
    const pVal = document.getElementById('s1-p-val');
    const readout = document.getElementById('s1-readout');

    const YMAX = 3.4;

    function drawDist() {
      const { ctx, w, h } = DL.plot.setup(distCv);
      const t = DL.plot.theme();
      DL.plot.clear(ctx, w, h);
      /* Margem inferior maior para o rótulo do eixo ficar abaixo dos ticks,
         longe da cauda das densidades. */
      const fr = DL.plot.frame(ctx, w, h, 0, 1, 0, YMAX, { l: 38, r: 10, t: 10, b: 36 });
      DL.plot.axes(ctx, fr, { xticks: [0, 0.5, 1] });
      DL.plot.label(ctx, fr.X(1), fr.Y(0) + 28, 'escore s', t.comment, 'right');

      const p = state.p, tau = state.tau;
      const f0 = (x) => (1 - p) * gauss(x, M0);
      const f1 = (x) => p * gauss(x, M1);

      /* Áreas de erro: FP (classe 0 à direita de tau), FN (classe 1 à esquerda). */
      function shade(f, lo, hi, color) {
        ctx.fillStyle = color;
        ctx.globalAlpha = 0.30;
        ctx.beginPath();
        ctx.moveTo(fr.X(lo), fr.Y(0));
        for (let x = lo; x <= hi + 1e-9; x += 0.004) ctx.lineTo(fr.X(x), fr.Y(f(x)));
        ctx.lineTo(fr.X(hi), fr.Y(0));
        ctx.closePath();
        ctx.fill();
        ctx.globalAlpha = 1;
      }
      shade(f0, tau, 1, t.red);
      shade(f1, 0, tau, t.cyan);

      /* Curvas de densidade. */
      const xs = [], y0 = [], y1 = [];
      for (let x = 0; x <= 1.0001; x += 0.004) { xs.push(x); y0.push(f0(x)); y1.push(f1(x)); }
      DL.plot.line(ctx, fr, xs, y0, t.red, { width: 2 });
      DL.plot.line(ctx, fr, xs, y1, t.cyan, { width: 2 });
      /* Rótulos centrados acima de cada pico, presos dentro do quadro e sobre
         uma pílula (em telas estreitas a curva passa por trás do texto). */
      function peakLabel(x, y, text, color) {
        ctx.font = '11px sans-serif';
        const tw = ctx.measureText(text).width;
        const cx = Math.max(fr.X(0) + tw / 2 + 2, Math.min(fr.X(1) - tw / 2 - 2, fr.X(x)));
        const cy = Math.max(fr.Y(YMAX) + 12, fr.Y(y) - 12);
        ctx.fillStyle = t.bg;
        ctx.globalAlpha = 0.85;
        ctx.fillRect(cx - tw / 2 - 3, cy - 11, tw + 6, 15);
        ctx.globalAlpha = 1;
        DL.plot.label(ctx, cx, cy, text, color, 'center');
      }
      /* Levemente deslocados para fora, longe do limiar inicial em 0,5. */
      peakLabel(M0 - 0.04, f0(M0), 'y = 0 (não reincide)', t.red);
      peakLabel(M1 + 0.04, f1(M1), 'y = 1 (reincide)', t.cyan);

      /* Limiar. */
      ctx.strokeStyle = t.fg;
      ctx.lineWidth = 2;
      ctx.setLineDash([5, 4]);
      ctx.beginPath();
      ctx.moveTo(fr.X(tau), fr.Y(0));
      ctx.lineTo(fr.X(tau), fr.Y(YMAX * 0.97));
      ctx.stroke();
      ctx.setLineDash([]);
      DL.plot.label(ctx, fr.X(tau) + 6, fr.Y(YMAX * 0.94), 'τ', t.fg, 'left');
      DL.plot.label(ctx, fr.X(tau) + 6, fr.Y(YMAX * 0.86), 'ŷ = 1 →', t.comment, 'left');

      /* Rótulos das áreas. */
      DL.plot.label(ctx, fr.X(Math.min(tau + 0.07, 0.97)), fr.Y(0.25), 'FP', t.red, 'left');
      DL.plot.label(ctx, fr.X(Math.max(tau - 0.07, 0.03)), fr.Y(0.25), 'FN', t.cyan, 'right');
      return fr;
    }

    function drawMetrics() {
      const { ctx, w, h } = DL.plot.setup(metCv);
      const t = DL.plot.theme();
      DL.plot.clear(ctx, w, h);
      const m = metrics(state.p, state.tau);
      const rows = [
        { name: 'FPR', v: m.fpr, c: t.red },
        { name: 'FNR', v: m.fnr, c: t.cyan },
        { name: 'PPV', v: m.ppv, c: t.green },
        { name: 'P(ŷ=1)', v: m.rate, c: t.purple },
        { name: 'acurácia', v: m.acc, c: t.orange },
      ];
      const padL = 70, padR = 46, padT = 14, rowH = (h - padT - 12) / rows.length;
      rows.forEach((r, i) => {
        const y = padT + i * rowH;
        ctx.fillStyle = t.comment;
        ctx.font = '11px sans-serif';
        ctx.textAlign = 'right';
        ctx.fillText(r.name, padL - 6, y + rowH * 0.62);
        ctx.fillStyle = t.line;
        ctx.fillRect(padL, y + rowH * 0.25, w - padL - padR, rowH * 0.5);
        ctx.fillStyle = r.c;
        ctx.fillRect(padL, y + rowH * 0.25, (w - padL - padR) * Math.max(0, Math.min(1, r.v)), rowH * 0.5);
        ctx.fillStyle = t.fg;
        ctx.textAlign = 'left';
        ctx.fillText((100 * r.v).toFixed(1) + '%', padL + (w - padL - padR) + 5, y + rowH * 0.62);
      });
    }

    let fr = null;
    function redraw() {
      fr = drawDist();
      drawMetrics();
      const m = metrics(state.p, state.tau);
      readout.textContent = 'τ = ' + state.tau.toFixed(2) +
        ' · FPR ' + (100 * m.fpr).toFixed(1) + '% · FNR ' + (100 * m.fnr).toFixed(1) +
        '% · PPV ' + (100 * m.ppv).toFixed(1) + '%';
    }

    /* Arrastar o limiar diretamente no gráfico. */
    let dragging = false;
    function pointerTau(ev) {
      const rect = distCv.getBoundingClientRect();
      const x = fr.invX(ev.clientX - rect.left);
      state.tau = Math.max(0.05, Math.min(0.95, x));
      tauSlider.value = String(state.tau);
      tauVal.textContent = state.tau.toFixed(2);
      redraw();
    }
    distCv.addEventListener('pointerdown', (ev) => { dragging = true; distCv.setPointerCapture(ev.pointerId); pointerTau(ev); });
    distCv.addEventListener('pointermove', (ev) => { if (dragging) pointerTau(ev); });
    distCv.addEventListener('pointerup', () => { dragging = false; });
    distCv.addEventListener('pointercancel', () => { dragging = false; });

    tauSlider.addEventListener('input', () => {
      state.tau = parseFloat(tauSlider.value);
      tauVal.textContent = state.tau.toFixed(2);
      redraw();
    });
    pSlider.addEventListener('input', () => {
      state.p = parseFloat(pSlider.value);
      pVal.textContent = state.p.toFixed(2);
      redraw();
    });

    DL.plot.observeResize(distCv, redraw);
    DL.plot.observeResize(metCv, redraw);
    DL.plot.onRedraw(redraw);
    redraw();
  }

  DL.sections.push({ name: 'limiar', init });
})();
