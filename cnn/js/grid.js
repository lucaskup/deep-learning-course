/* Helpers de grades de pixels compartilhados pelas seções da demo de CNN. */
(function () {
  'use strict';
  window.DL = window.DL || {};

  /* Geometria: centraliza uma grade rows×cols de células quadradas no canvas. */
  function layout(w, h, rows, cols, pad) {
    pad = pad == null ? 12 : pad;
    const cs = Math.min((w - 2 * pad) / cols, (h - 2 * pad) / rows);
    const ox = (w - cs * cols) / 2;
    const oy = (h - cs * rows) / 2;
    return {
      cs, ox, oy, rows, cols,
      x: (j) => ox + j * cs,
      y: (i) => oy + i * cs,
      cellAt: (px, py) => {
        const j = Math.floor((px - ox) / cs);
        const i = Math.floor((py - oy) / cs);
        return i >= 0 && i < rows && j >= 0 && j < cols ? { i, j } : null;
      },
    };
  }

  /* Mistura linear entre duas cores CSS do tema; devolve string rgb(). */
  function mix(c1, c2, t) {
    const a = DL.plot.rgb(c1), b = DL.plot.rgb(c2);
    t = Math.max(0, Math.min(1, t));
    const ch = (k) => Math.round(a[k] + (b[k] - a[k]) * t);
    return 'rgb(' + ch(0) + ',' + ch(1) + ',' + ch(2) + ')';
  }

  /* Cor divergente: positivo → verde, negativo → rosa, zero → cor do card. */
  function diverge(v, vmax, th) {
    const t = vmax > 0 ? Math.min(1, Math.abs(v) / vmax) : 0;
    return mix(th.card, v >= 0 ? th.green : th.pink, t);
  }

  /* Linhas internas e borda da grade. */
  function lines(ctx, L, color, width) {
    ctx.strokeStyle = color;
    ctx.lineWidth = width || 1;
    ctx.beginPath();
    for (let i = 0; i <= L.rows; i++) {
      ctx.moveTo(L.x(0), L.y(i));
      ctx.lineTo(L.x(L.cols), L.y(i));
    }
    for (let j = 0; j <= L.cols; j++) {
      ctx.moveTo(L.x(j), L.y(0));
      ctx.lineTo(L.x(j), L.y(L.rows));
    }
    ctx.stroke();
  }

  /* Texto centrado em uma célula. */
  function cellText(ctx, L, i, j, text, color, size) {
    ctx.fillStyle = color;
    ctx.font = (size || Math.max(9, Math.min(13, L.cs * 0.34))) + 'px sans-serif';
    ctx.textAlign = 'center';
    ctx.textBaseline = 'middle';
    ctx.fillText(text, L.x(j) + L.cs / 2, L.y(i) + L.cs / 2);
    ctx.textBaseline = 'alphabetic';
  }

  /* Contorno de um retângulo de células: canto (i, j), tamanho di×dj. */
  function outline(ctx, L, i, j, di, dj, color, width) {
    ctx.strokeStyle = color;
    ctx.lineWidth = width || 2;
    ctx.strokeRect(L.x(j), L.y(i), L.cs * dj, L.cs * di);
  }

  /* Convolução 2D de kernel 3×3 com zero padding "same". */
  function conv2dSame3(input, k) {
    const n = input.length;
    const out = [];
    for (let i = 0; i < n; i++) {
      const row = [];
      for (let j = 0; j < n; j++) {
        let s = 0;
        for (let m = 0; m < 3; m++) {
          for (let q = 0; q < 3; q++) {
            const ii = i + m - 1, jj = j + q - 1;
            if (ii >= 0 && ii < n && jj >= 0 && jj < n) s += k[m][q] * input[ii][jj];
          }
        }
        row.push(s);
      }
      out.push(row);
    }
    return out;
  }

  DL.grid = { layout, mix, diverge, lines, cellText, outline, conv2dSame3 };
})();
