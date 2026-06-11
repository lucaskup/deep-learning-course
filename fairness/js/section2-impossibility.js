/* Seção 2: teorema da impossibilidade (caso COMPAS).
   Dois grupos com o mesmo classificador e taxas-base distintas; limiar por
   grupo; indicadores mostram quais critérios de fairness valem (tol. 3 p.p.). */
(function () {
  'use strict';
  window.DL = window.DL || {};
  DL.sections = DL.sections || [];

  const M0 = 0.35, M1 = 0.65, SD = 0.13;
  const TOL = 0.03;

  function phi(z) {
    const t = 1 / (1 + 0.3275911 * Math.abs(z));
    const y = 1 - (((((1.061405429 * t - 1.453152027) * t) + 1.421413741) * t - 0.284496736) * t + 0.254829592) * t * Math.exp(-z * z);
    const e = 0.5 * (1 + y);
    return z >= 0 ? e : 1 - e;
  }

  function metrics(p, tau) {
    const tpr = 1 - phi((tau - M1) / SD);
    const fpr = 1 - phi((tau - M0) / SD);
    const rate = p * tpr + (1 - p) * fpr;
    const ppv = rate > 1e-9 ? (p * tpr) / rate : 0;
    return { tpr, fpr, fnr: 1 - tpr, rate, ppv };
  }

  function gauss(x, mu) {
    return DL.utils.gaussPdf(x, mu, SD * SD);
  }

  const state = { pa: 0.51, pb: 0.39, ta: 0.5, tb: 0.5 };

  function init() {
    const cvA = document.getElementById('s2-distA');
    const cvB = document.getElementById('s2-distB');
    const cvM = document.getElementById('s2-metrics');
    const badgesBox = document.getElementById('s2-badges');

    const sliders = {
      pa: { el: document.getElementById('s2-pa'), val: document.getElementById('s2-pa-val') },
      pb: { el: document.getElementById('s2-pb'), val: document.getElementById('s2-pb-val') },
      ta: { el: document.getElementById('s2-ta'), val: document.getElementById('s2-ta-val') },
      tb: { el: document.getElementById('s2-tb'), val: document.getElementById('s2-tb-val') },
    };

    /* Indicadores de critérios, criados uma vez. */
    const CRITERIA = [
      { key: 'parity', label: 'Paridade demográfica', diff: (a, b) => Math.abs(a.rate - b.rate) },
      { key: 'opportunity', label: 'Igualdade de oportunidade (TPR)', diff: (a, b) => Math.abs(a.tpr - b.tpr) },
      { key: 'fprEq', label: 'Igualdade de FPR', diff: (a, b) => Math.abs(a.fpr - b.fpr) },
      { key: 'calibration', label: 'Calibração (PPV)', diff: (a, b) => Math.abs(a.ppv - b.ppv) },
    ];
    const badges = CRITERIA.map((c) => {
      const span = document.createElement('span');
      span.className = 'readout';
      badgesBox.appendChild(span);
      return span;
    });

    const YMAX = 3.4;

    function drawDist(cv, p, tau, color, name) {
      const { ctx, w, h } = DL.plot.setup(cv);
      const t = DL.plot.theme();
      DL.plot.clear(ctx, w, h);
      const fr = DL.plot.frame(ctx, w, h, 0, 1, 0, YMAX);
      DL.plot.axes(ctx, fr, { xlabel: 'escore s', xticks: [0, 0.5, 1] });

      const f0 = (x) => (1 - p) * gauss(x, M0);
      const f1 = (x) => p * gauss(x, M1);
      const xs = [], y0 = [], y1 = [];
      for (let x = 0; x <= 1.0001; x += 0.005) { xs.push(x); y0.push(f0(x)); y1.push(f1(x)); }
      DL.plot.line(ctx, fr, xs, y0, t.red, { width: 1.8 });
      DL.plot.line(ctx, fr, xs, y1, color, { width: 1.8 });

      ctx.strokeStyle = t.fg;
      ctx.lineWidth = 2;
      ctx.setLineDash([5, 4]);
      ctx.beginPath();
      ctx.moveTo(fr.X(tau), fr.Y(0));
      ctx.lineTo(fr.X(tau), fr.Y(YMAX * 0.97));
      ctx.stroke();
      ctx.setLineDash([]);
      DL.plot.label(ctx, fr.X(tau) + 5, fr.Y(YMAX * 0.92), name, t.fg, 'left');
      DL.plot.label(ctx, fr.X(0.03), fr.Y(YMAX * 0.92), 'p = ' + p.toFixed(2), t.comment, 'left');
      return fr;
    }

    function drawMetrics() {
      const { ctx, w, h } = DL.plot.setup(cvM);
      const t = DL.plot.theme();
      DL.plot.clear(ctx, w, h);
      const a = metrics(state.pa, state.ta);
      const b = metrics(state.pb, state.tb);
      const rows = [
        { name: 'P(ŷ=1)', va: a.rate, vb: b.rate },
        { name: 'TPR', va: a.tpr, vb: b.tpr },
        { name: 'FPR', va: a.fpr, vb: b.fpr },
        { name: 'PPV', va: a.ppv, vb: b.ppv },
      ];
      const padL = 56, padR = 44, padT = 10, rowH = (h - padT - 10) / rows.length;
      rows.forEach((r, i) => {
        const y = padT + i * rowH;
        ctx.fillStyle = t.comment;
        ctx.font = '11px sans-serif';
        ctx.textAlign = 'right';
        ctx.fillText(r.name, padL - 6, y + rowH * 0.58);
        const bw = w - padL - padR;
        ctx.fillStyle = t.line;
        ctx.fillRect(padL, y + rowH * 0.16, bw, rowH * 0.26);
        ctx.fillRect(padL, y + rowH * 0.52, bw, rowH * 0.26);
        ctx.fillStyle = t.cyan;
        ctx.fillRect(padL, y + rowH * 0.16, bw * r.va, rowH * 0.26);
        ctx.fillStyle = t.orange;
        ctx.fillRect(padL, y + rowH * 0.52, bw * r.vb, rowH * 0.26);
        ctx.fillStyle = t.fg;
        ctx.textAlign = 'left';
        ctx.font = '10px sans-serif';
        ctx.fillText((100 * r.va).toFixed(0) + '%', padL + bw + 4, y + rowH * 0.38);
        ctx.fillText((100 * r.vb).toFixed(0) + '%', padL + bw + 4, y + rowH * 0.74);
      });
    }

    function updateBadges() {
      const a = metrics(state.pa, state.ta);
      const b = metrics(state.pb, state.tb);
      CRITERIA.forEach((c, i) => {
        const d = c.diff(a, b);
        const ok = d < TOL;
        badges[i].textContent = (ok ? '✓ ' : '✗ ') + c.label + ' (Δ ' + (100 * d).toFixed(1) + ' p.p.)';
        badges[i].style.color = ok ? 'var(--green)' : 'var(--red)';
      });
    }

    let frA = null, frB = null;
    function redraw() {
      frA = drawDist(cvA, state.pa, state.ta, DL.plot.theme().cyan, 'τ_A');
      frB = drawDist(cvB, state.pb, state.tb, DL.plot.theme().orange, 'τ_B');
      drawMetrics();
      updateBadges();
    }

    function syncSliders() {
      sliders.pa.el.value = String(state.pa); sliders.pa.val.textContent = state.pa.toFixed(2);
      sliders.pb.el.value = String(state.pb); sliders.pb.val.textContent = state.pb.toFixed(2);
      sliders.ta.el.value = String(state.ta); sliders.ta.val.textContent = state.ta.toFixed(2);
      sliders.tb.el.value = String(state.tb); sliders.tb.val.textContent = state.tb.toFixed(2);
    }

    function bindSlider(key) {
      sliders[key].el.addEventListener('input', () => {
        state[key] = parseFloat(sliders[key].el.value);
        sliders[key].val.textContent = state[key].toFixed(2);
        redraw();
      });
    }
    Object.keys(sliders).forEach(bindSlider);

    /* Arrastar limiares nos gráficos dos grupos. */
    function attachDrag(cv, key, getFr) {
      let dragging = false;
      function apply(ev) {
        const fr = getFr();
        if (!fr) return;
        const rect = cv.getBoundingClientRect();
        state[key] = Math.max(0.05, Math.min(0.95, fr.invX(ev.clientX - rect.left)));
        syncSliders();
        redraw();
      }
      cv.addEventListener('pointerdown', (ev) => { dragging = true; cv.setPointerCapture(ev.pointerId); apply(ev); });
      cv.addEventListener('pointermove', (ev) => { if (dragging) apply(ev); });
      cv.addEventListener('pointerup', () => { dragging = false; });
    }
    attachDrag(cvA, 'ta', () => frA);
    attachDrag(cvB, 'tb', () => frB);

    document.getElementById('s2-compas').addEventListener('click', () => {
      state.pa = 0.51; state.pb = 0.39; state.ta = 0.5; state.tb = 0.5;
      syncSliders(); redraw();
    });
    document.getElementById('s2-equal').addEventListener('click', () => {
      state.pa = 0.45; state.pb = 0.45; state.ta = 0.5; state.tb = 0.5;
      syncSliders(); redraw();
    });

    DL.plot.observeResize(cvA, redraw);
    DL.plot.observeResize(cvB, redraw);
    DL.plot.observeResize(cvM, redraw);
    DL.plot.onRedraw(redraw);
    syncSliders();
    redraw();
  }

  DL.sections.push({ name: 'impossibilidade', init });
})();
