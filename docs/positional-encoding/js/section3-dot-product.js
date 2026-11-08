/* Seção 3: PE_i · PE_j = Σ_k cos((i−j)·ω_k), função apenas do offset.
   Heatmap Toeplitz L×L e curvas de similaridade transladadas. */
(function () {
  'use strict';
  window.DL = window.DL || {};
  DL.sections = DL.sections || [];

  /* Rótulo com fundo semitransparente (cor do canvas), legível sobre curvas e heatmaps. */
  function pillText(ctx, th, text, x, y, color, align, size) {
    size = size || 11;
    ctx.font = size + 'px sans-serif';
    const tw = ctx.measureText(text.replace(/_\{([^}]*)\}/g, '$1').replace(/_(.)/g, '$1')).width;
    const x0 = align === 'right' ? x - tw : (align === 'center' ? x - tw / 2 : x);
    ctx.fillStyle = th.bg;
    ctx.globalAlpha = 0.85;
    ctx.fillRect(x0 - 3, y - size, tw + 6, size + 4);
    ctx.globalAlpha = 1;
    DL.plot.mathText(ctx, text, x, y, color, align, size);
  }

  function omega(k, d) { return Math.pow(10000, -2 * k / d); }

  /* Produto interno normalizado em função do offset m: (2/d)·Σ_k cos(m·ω_k). */
  function buildTable(L, d) {
    const t = new Float64Array(L);
    const half = d / 2;
    for (let m = 0; m < L; m++) {
      let s = 0;
      for (let k = 0; k < half; k++) s += Math.cos(m * omega(k, d));
      t[m] = s / half;
    }
    return t;
  }

  /* Produto interno bruto entre PE(a) e PE(b), calculado dos vetores completos
     (verificação independente de que só o offset importa). */
  function rawDot(a, b, d) {
    let s = 0;
    for (let k = 0; k < d / 2; k++) {
      const w = omega(k, d);
      s += Math.sin(a * w) * Math.sin(b * w) + Math.cos(a * w) * Math.cos(b * w);
    }
    return s / (d / 2);
  }

  function init() {
    const P = DL.plot;
    const $ = (id) => document.getElementById(id);

    const cvHeat = $('s3-heat'), cvCurve = $('s3-curve');
    const slD = $('s3-d'), slL = $('s3-len'), slI = $('s3-i'), slI2 = $('s3-i2');

    let d = +slD.value, L = +slL.value, i = +slI.value, i2 = +slI2.value;
    let table = buildTable(L, d);

    function drawHeat() {
      const th = P.theme();
      const { ctx, w, h } = P.setup(cvHeat);
      P.clear(ctx, w, h);
      const fr = P.frame(ctx, w, h, 0, L, 0, L);
      const oc = P.rgb(th.orange), cc = P.rgb(th.cyan);
      P.heatmap(ctx, fr, (x, y) => {
        const jj = Math.min(L - 1, Math.max(0, Math.floor(x)));
        const ii = Math.min(L - 1, Math.max(0, Math.floor(y)));
        return table[Math.abs(ii - jj)];
      }, L, L, (v) => {
        const a = Math.round(10 + 220 * Math.min(1, Math.abs(v)));
        return v >= 0 ? [oc[0], oc[1], oc[2], a] : [cc[0], cc[1], cc[2], a];
      });
      P.axes(ctx, fr, {
        ylabel: 'posição i',
        xticks: [0, Math.round(L / 2), L],
        yticks: [0, Math.round(L / 2), L],
      });
      /* Linhas de referência i e i′ (cada linha é a curva ao lado). */
      ctx.strokeStyle = th.cyan;
      ctx.lineWidth = 2;
      ctx.strokeRect(fr.X(0), fr.Y(i + 1), fr.iw, fr.Y(i) - fr.Y(i + 1));
      ctx.strokeStyle = th.orange;
      ctx.strokeRect(fr.X(0), fr.Y(i2 + 1), fr.iw, fr.Y(i2) - fr.Y(i2 + 1));
      /* Rótulos das linhas: acima da faixa; se a faixa encosta no topo, abaixo dela.
         Com i = i′ o segundo rótulo desce para não cobrir o primeiro. */
      const rowLabelY = (row) => {
        const y = fr.Y(row + 1) - 3;
        return y < fr.Y(L) + 11 ? fr.Y(row) + 12 : y;
      };
      pillText(ctx, th, 'i = ' + i, fr.X(L) - 6, rowLabelY(i), th.cyan, 'right');
      pillText(ctx, th, "i' = " + i2, fr.X(L) - 6, i === i2 ? fr.Y(i2) + 12 : rowLabelY(i2), th.orange, 'right');
      pillText(ctx, th, 'posição j', fr.X(L) - 4, fr.Y(0) - 8, th.comment, 'right');
    }

    function drawCurve() {
      const th = P.theme();
      const { ctx, w, h } = P.setup(cvCurve);
      P.clear(ctx, w, h);
      let ymin = 0;
      for (let m = 0; m < L; m++) ymin = Math.min(ymin, table[m]);
      const fr = P.frame(ctx, w, h, 0, L - 1, ymin - 0.1, 1.1);
      P.axes(ctx, fr, {
        ylabel: 'PE_i · PE_j',
        xticks: [0, Math.round((L - 1) / 2), L - 1],
        yticks: [0, 0.5, 1],
      });
      P.line(ctx, fr, [0, L - 1], [0, 0], th.line, { width: 1, dash: [4, 4] });
      const js = [], yi = [], yi2 = [];
      for (let j = 0; j < L; j++) {
        js.push(j);
        yi.push(table[Math.abs(i - j)]);
        yi2.push(table[Math.abs(i2 - j)]);
      }
      P.line(ctx, fr, js, yi, th.cyan, { width: 1.8 });
      P.line(ctx, fr, js, yi2, th.orange, { width: 1.8 });
      P.scatter(ctx, fr, [[i, 1]], th.cyan, { r: 4, alpha: 1 });
      P.scatter(ctx, fr, [[i2, 1]], th.orange, { r: 4, alpha: 1 });
      /* Rótulos dos picos, puxados para dentro do quadro quando i está numa borda. */
      ctx.font = '11px sans-serif';
      const clampX = (x, txt) => {
        const half = ctx.measureText(txt).width / 2 + 2;
        return Math.max(fr.X(0) + half, Math.min(fr.X(L - 1) - half, x));
      };
      pillText(ctx, th, 'j = i', clampX(fr.X(i), 'j = i'), fr.Y(1.1) + 12, th.cyan, 'center');
      pillText(ctx, th, "j = i'", clampX(fr.X(i2), "j = i'"), fr.Y(1.1) + 12, th.orange, 'center');
      pillText(ctx, th, 'posição j', fr.X(L - 1) - 4, fr.Y(fr.ymin) - 8, th.comment, 'right');
    }

    function updateReadout() {
      const off = 8;
      const a = i + off < L ? rawDot(i, i + off, d) : null;
      const b = i2 + off < L ? rawDot(i2, i2 + off, d) : null;
      let txt = 'mesma forma, apenas transladada';
      if (a !== null && b !== null) {
        txt = 'PE_i·PE_(i+8) = ' + a.toFixed(4) + " · PE_i'·PE_(i'+8) = " + b.toFixed(4);
      }
      $('s3-readout').textContent = txt;
    }

    function redraw() {
      drawHeat();
      drawCurve();
      updateReadout();
    }

    function syncPositions() {
      slI.max = L - 1;
      slI2.max = L - 1;
      if (i > L - 1) i = L - 1;
      if (i2 > L - 1) i2 = L - 1;
      slI.value = i;
      slI2.value = i2;
      $('s3-i-val').textContent = i;
      $('s3-i2-val').textContent = i2;
    }

    slD.addEventListener('input', () => {
      d = +slD.value;
      $('s3-d-val').textContent = d;
      table = buildTable(L, d);
      redraw();
    });
    slL.addEventListener('input', () => {
      L = +slL.value;
      $('s3-len-val').textContent = L;
      table = buildTable(L, d);
      syncPositions();
      redraw();
    });
    slI.addEventListener('input', () => {
      i = +slI.value;
      $('s3-i-val').textContent = i;
      redraw();
    });
    slI2.addEventListener('input', () => {
      i2 = +slI2.value;
      $('s3-i2-val').textContent = i2;
      redraw();
    });

    syncPositions();
    redraw();
    P.onRedraw(redraw);
    P.observeResize(cvHeat, redraw);
    P.observeResize(cvCurve, redraw);
  }

  DL.sections.push({ name: 's3-dot-product', init });
})();
