/* Seção 6: positional encoding sinusoidal P_{i,j} (Vaswani et al., 2017),
   heatmap posição × dimensão e produto escalar P_i·P_j com pico em j = i. */
(function () {
  'use strict';
  window.DL = window.DL || {};
  DL.sections = DL.sections || [];

  const NPOS = 96;

  function init() {
    const P = DL.plot;
    const $ = (id) => document.getElementById(id);

    const cvHeat = $('s6-heatmap'), cvDot = $('s6-dot');
    const slD = $('s6-d'), slN = $('s6-n'), slPos = $('s6-pos');

    let d = +slD.value;
    let n = Math.round(Math.pow(10, +slN.value));
    let pos = +slPos.value;
    let PE = [], dots = [];
    let frHeat = null, dragging = false;

    function pe(i, j) {
      const k = Math.floor(j / 2);
      const ang = i / Math.pow(n, (2 * k) / d);
      return j % 2 === 0 ? Math.sin(ang) : Math.cos(ang);
    }

    function recompute() {
      PE = [];
      for (let i = 0; i < NPOS; i++) {
        const row = new Array(d);
        for (let j = 0; j < d; j++) row[j] = pe(i, j);
        PE.push(row);
      }
      const ref = PE[pos];
      let norm2 = 0;
      for (const v of ref) norm2 += v * v;
      norm2 = norm2 || 1;
      dots = PE.map((row) => {
        let s = 0;
        for (let j = 0; j < d; j++) s += ref[j] * row[j];
        return s / norm2;
      });
    }

    /* Bloco de legenda com fundo semitransparente, para que o texto continue
       legível quando barras ou curvas passam por baixo. items: [[texto, cor]]. */
    function legendBox(ctx, th, x, y, items, align) {
      ctx.font = '11px sans-serif';
      let wmax = 0;
      for (const it of items) wmax = Math.max(wmax, ctx.measureText(it[0].replace(/[_{}]/g, '')).width);
      const bx = align === 'right' ? x - wmax - 6 : x - 6;
      ctx.save();
      ctx.globalAlpha = 0.85;
      ctx.fillStyle = th.card;
      ctx.fillRect(bx, y, wmax + 12, 14 * items.length + 6);
      ctx.restore();
      items.forEach((it, k) => P.label(ctx, x, y + 14 * (k + 1), it[0], it[1], align || 'left'));
    }

    function drawHeat() {
      const th = P.theme();
      const { ctx, w, h } = P.setup(cvHeat);
      P.clear(ctx, w, h);
      const fr = P.frame(ctx, w, h, 0, d, 0, NPOS, { l: 38, r: 10, t: 10, b: 36 });
      frHeat = fr;
      const cy = P.rgb(th.cyan), or = P.rgb(th.orange);
      /* posição 0 no topo: linha y do frame cresce para baixo via NPOS − i */
      P.heatmap(ctx, fr, (x, y) => {
        const i = Math.min(NPOS - 1, Math.max(0, Math.floor(NPOS - y)));
        const j = Math.min(d - 1, Math.max(0, Math.floor(x)));
        return PE[i][j];
      }, d, NPOS, (v) => {
        const c = v >= 0 ? cy : or;
        return [c[0], c[1], c[2], 20 + 235 * Math.min(1, Math.abs(v))];
      });
      P.axes(ctx, fr, {
        ylabel: 'posição i',
        xticks: [0, Math.round(d / 2), d],
        yticks: [],
      });
      /* rótulo abaixo dos ticks, fora do heatmap */
      P.label(ctx, fr.X(d), fr.Y(0) + 30, 'dimensão j', th.comment, 'right');
      ctx.fillStyle = th.comment;
      ctx.font = '11px sans-serif';
      ctx.textAlign = 'right';
      for (const i of [0, 32, 64, NPOS - 1]) {
        ctx.fillText(String(i), fr.X(0) - 4, fr.Y(NPOS - i) + 4);
      }
      /* marcador da posição selecionada */
      const ym = fr.Y(NPOS - pos - 0.5);
      ctx.strokeStyle = th.green;
      ctx.lineWidth = 2;
      ctx.beginPath();
      ctx.moveTo(fr.X(0), ym);
      ctx.lineTo(fr.X(d), ym);
      ctx.stroke();
      legendBox(ctx, th, fr.X(d) - 8, ym - 20, [['i = ' + pos, th.green]], 'right');
    }

    function drawDot() {
      const th = P.theme();
      const { ctx, w, h } = P.setup(cvDot);
      P.clear(ctx, w, h);
      const ymin = Math.min(0, Math.floor(Math.min(...dots) * 10) / 10);
      const fr = P.frame(ctx, w, h, 0, NPOS, ymin, 1.05);
      P.axes(ctx, fr, {
        xlabel: 'posição j', ylabel: 'P_i·P_j (norm.)',
        xticks: [0, 32, 64, NPOS - 1],
        yticks: [0, 0.5, 1],
      });
      const xs = dots.map((_, j) => j);
      P.line(ctx, fr, xs, dots, th.cyan, { width: 1.8 });
      P.scatter(ctx, fr, [[pos, 1]], th.green, { r: 4, alpha: 1 });
      P.label(ctx, fr.X(pos) + 7, fr.Y(1) + 3, 'j = i', th.green);
    }

    function updateReadout() {
      /* maior pico secundário fora da vizinhança imediata de i */
      let side = -Infinity;
      for (let j = 0; j < NPOS; j++) {
        if (Math.abs(j - pos) > 3 && dots[j] > side) side = dots[j];
      }
      $('s6-readout').textContent =
        'd = ' + d + ' · n = ' + n + ' · maior pico secundário = ' + side.toFixed(2);
    }

    function redraw() {
      drawHeat();
      drawDot();
      updateReadout();
    }

    function setPos(p) {
      pos = Math.max(0, Math.min(NPOS - 1, Math.round(p)));
      slPos.value = pos;
      $('s6-pos-val').textContent = pos;
      recompute();
      redraw();
    }

    cvHeat.addEventListener('pointerdown', (e) => {
      if (!frHeat) return;
      dragging = true;
      cvHeat.setPointerCapture(e.pointerId);
      const r = cvHeat.getBoundingClientRect();
      setPos(NPOS - frHeat.invY(e.clientY - r.top) - 0.5);
    });
    cvHeat.addEventListener('pointermove', (e) => {
      if (!dragging || !frHeat) return;
      const r = cvHeat.getBoundingClientRect();
      setPos(NPOS - frHeat.invY(e.clientY - r.top) - 0.5);
    });
    cvHeat.addEventListener('pointerup', () => { dragging = false; });

    slD.addEventListener('input', () => {
      d = +slD.value;
      $('s6-d-val').textContent = d;
      recompute();
      redraw();
    });
    slN.addEventListener('input', () => {
      n = Math.round(Math.pow(10, +slN.value));
      $('s6-n-val').textContent = n;
      recompute();
      redraw();
    });
    slPos.addEventListener('input', () => setPos(+slPos.value));

    recompute();
    redraw();
    P.onRedraw(redraw);
    P.observeResize(cvHeat, redraw);
    P.observeResize(cvDot, redraw);
  }

  DL.sections.push({ name: 's6-positional-encoding', init });
})();
