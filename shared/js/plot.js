/* Helpers de canvas compartilhados: DPR, cache de cores do tema e primitivas. */
(function () {
  'use strict';
  window.DL = window.DL || {};

  let themeCache = null;
  const redrawCallbacks = [];

  function theme() {
    if (!themeCache) {
      const cs = getComputedStyle(document.documentElement);
      const v = (name) => cs.getPropertyValue(name).trim();
      themeCache = {
        bg: v('--bg'), card: v('--card'), line: v('--line'), fg: v('--fg'),
        comment: v('--comment'), cyan: v('--cyan'), green: v('--green'),
        orange: v('--orange'), pink: v('--pink'), purple: v('--purple'),
        red: v('--red') || '#ff5555',
      };
    }
    return themeCache;
  }

  function onRedraw(cb) { redrawCallbacks.push(cb); }
  function redrawAll() { for (const cb of redrawCallbacks) cb(); }

  new MutationObserver(() => { themeCache = null; redrawAll(); })
    .observe(document.documentElement, { attributes: true, attributeFilter: ['data-theme'] });

  /* Converte cor CSS (#rgb, #rrggbb ou rgb(...)) em [r, g, b]. */
  function rgb(color) {
    color = color.trim();
    if (color[0] === '#') {
      if (color.length === 4) {
        return [parseInt(color[1] + color[1], 16), parseInt(color[2] + color[2], 16), parseInt(color[3] + color[3], 16)];
      }
      return [parseInt(color.slice(1, 3), 16), parseInt(color.slice(3, 5), 16), parseInt(color.slice(5, 7), 16)];
    }
    const m = color.match(/rgba?\(([^)]+)\)/);
    if (m) return m[1].split(',').slice(0, 3).map((s) => Math.round(parseFloat(s)));
    return [128, 128, 128];
  }

  /* Dimensiona o backing store pelo devicePixelRatio; desenho em px CSS. */
  function setup(canvas) {
    const rect = canvas.getBoundingClientRect();
    const dpr = window.devicePixelRatio || 1;
    const w = Math.max(1, Math.round(rect.width));
    const h = Math.max(1, Math.round(rect.height));
    if (canvas.width !== w * dpr || canvas.height !== h * dpr) {
      canvas.width = w * dpr;
      canvas.height = h * dpr;
    }
    const ctx = canvas.getContext('2d');
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    return { ctx, w, h };
  }

  function observeResize(canvas, redraw) {
    new ResizeObserver(() => { setup(canvas); redraw(); }).observe(canvas);
  }

  /* Sistema de coordenadas: mapeia [xmin,xmax]x[ymin,ymax] no retângulo do canvas. */
  function frame(ctx, w, h, xmin, xmax, ymin, ymax, pad) {
    pad = pad || { l: 38, r: 10, t: 10, b: 24 };
    const iw = w - pad.l - pad.r, ih = h - pad.t - pad.b;
    const X = (x) => pad.l + (x - xmin) / (xmax - xmin) * iw;
    const Y = (y) => pad.t + ih - (y - ymin) / (ymax - ymin) * ih;
    /* Inversas: de px CSS para coordenadas de dados (útil para arrastar). */
    const invX = (px) => xmin + (px - pad.l) / iw * (xmax - xmin);
    const invY = (py) => ymin + (pad.t + ih - py) / ih * (ymax - ymin);
    return { X, Y, invX, invY, pad, iw, ih, xmin, xmax, ymin, ymax };
  }

  function clear(ctx, w, h) {
    ctx.clearRect(0, 0, w, h);
  }

  /* Desenha texto com subscritos no estilo TeX: "z_t", "q(z_{t−1}|x)".
     Canvas não renderiza HTML/LaTeX, então o subscrito é desenhado
     manualmente com fonte menor e baseline deslocada. */
  function mathText(ctx, text, x, y, color, align, size) {
    size = size || 11;
    const subSize = Math.round(size * 0.75);
    const segs = [];
    let i = 0, cur = '';
    while (i < text.length) {
      if (text[i] === '_') {
        if (cur) { segs.push({ t: cur, sub: false }); cur = ''; }
        if (text[i + 1] === '{') {
          let end = text.indexOf('}', i + 2);
          if (end === -1) end = text.length;
          segs.push({ t: text.slice(i + 2, end), sub: true });
          i = end + 1;
        } else {
          segs.push({ t: text[i + 1] || '', sub: true });
          i += 2;
        }
      } else { cur += text[i]; i++; }
    }
    if (cur) segs.push({ t: cur, sub: false });

    let total = 0;
    for (const s of segs) {
      ctx.font = (s.sub ? subSize : size) + 'px sans-serif';
      s.w = ctx.measureText(s.t).width;
      total += s.w;
    }
    let cx = x;
    if (align === 'center') cx = x - total / 2;
    else if (align === 'right') cx = x - total;
    ctx.fillStyle = color || theme().fg;
    ctx.textAlign = 'left';
    for (const s of segs) {
      ctx.font = (s.sub ? subSize : size) + 'px sans-serif';
      ctx.fillText(s.t, cx, y + (s.sub ? size * 0.25 : 0));
      cx += s.w;
    }
  }

  function axes(ctx, fr, opts) {
    opts = opts || {};
    const t = theme();
    ctx.strokeStyle = t.line;
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(fr.X(fr.xmin), fr.Y(fr.ymin));
    ctx.lineTo(fr.X(fr.xmax), fr.Y(fr.ymin));
    ctx.moveTo(fr.X(fr.xmin), fr.Y(fr.ymin));
    ctx.lineTo(fr.X(fr.xmin), fr.Y(fr.ymax));
    ctx.stroke();
    /* Rótulo do eixo x no canto inferior direito, dentro da área do
       gráfico, para não colidir com os ticks nem com marcadores. */
    if (opts.xlabel) {
      mathText(ctx, opts.xlabel, fr.X(fr.xmax) - 4, fr.Y(fr.ymin) - 8, t.comment, 'right');
    }
    if (opts.ylabel) {
      ctx.save();
      ctx.translate(12, fr.Y((fr.ymin + fr.ymax) / 2));
      ctx.rotate(-Math.PI / 2);
      mathText(ctx, opts.ylabel, 0, 0, t.comment, 'center');
      ctx.restore();
    }
    ctx.fillStyle = t.comment;
    ctx.font = '11px sans-serif';
    if (opts.xticks) {
      ctx.textAlign = 'center';
      for (const v of opts.xticks) {
        ctx.fillText(fmt(v), fr.X(v), fr.Y(fr.ymin) + 13);
      }
    }
    if (opts.yticks) {
      ctx.textAlign = 'right';
      for (const v of opts.yticks) {
        ctx.fillText(fmt(v), fr.X(fr.xmin) - 4, fr.Y(v) + 4);
      }
    }
  }

  function fmt(v) {
    if (Math.abs(v) >= 100 || v === Math.round(v)) return String(Math.round(v));
    return v.toFixed(Math.abs(v) < 0.1 ? 2 : 1);
  }

  function line(ctx, fr, xs, ys, color, opts) {
    opts = opts || {};
    ctx.strokeStyle = color;
    ctx.lineWidth = opts.width || 1.6;
    ctx.globalAlpha = opts.alpha != null ? opts.alpha : 1;
    if (opts.dash) ctx.setLineDash(opts.dash);
    ctx.beginPath();
    for (let i = 0; i < xs.length; i++) {
      const px = fr.X(xs[i]), py = fr.Y(ys[i]);
      if (i === 0) ctx.moveTo(px, py); else ctx.lineTo(px, py);
    }
    ctx.stroke();
    ctx.setLineDash([]);
    ctx.globalAlpha = 1;
  }

  function scatter(ctx, fr, pts, color, opts) {
    opts = opts || {};
    const r = opts.r || 2;
    ctx.fillStyle = color;
    ctx.globalAlpha = opts.alpha != null ? opts.alpha : 0.8;
    for (const p of pts) {
      ctx.beginPath();
      ctx.arc(fr.X(p[0]), fr.Y(p[1]), r, 0, 2 * Math.PI);
      ctx.fill();
    }
    ctx.globalAlpha = 1;
  }

  /* Histograma normalizado como densidade (área = 1), desenhado em barras. */
  function histogram(ctx, fr, values, nbins, color, opts) {
    opts = opts || {};
    const lo = opts.range ? opts.range[0] : fr.xmin;
    const hi = opts.range ? opts.range[1] : fr.xmax;
    const bins = new Float64Array(nbins);
    const bw = (hi - lo) / nbins;
    for (const v of values) {
      const b = Math.floor((v - lo) / bw);
      if (b >= 0 && b < nbins) bins[b]++;
    }
    ctx.fillStyle = color;
    ctx.globalAlpha = opts.alpha != null ? opts.alpha : 0.35;
    for (let b = 0; b < nbins; b++) {
      const density = bins[b] / (values.length * bw);
      const x0 = fr.X(lo + b * bw), x1 = fr.X(lo + (b + 1) * bw);
      const y0 = fr.Y(Math.max(0, fr.ymin)), y1 = fr.Y(density);
      ctx.fillRect(x0, y1, x1 - x0, y0 - y1);
    }
    ctx.globalAlpha = 1;
  }

  /* Mapa de calor: avalia f(x, y) numa grade nx por ny e pinta a área do frame.
     colorFn(v) recebe o valor de f e devolve [r, g, b, a] com canais 0..255. */
  function heatmap(ctx, fr, f, nx, ny, colorFn) {
    const off = document.createElement('canvas');
    off.width = nx; off.height = ny;
    const octx = off.getContext('2d');
    const img = octx.createImageData(nx, ny);
    for (let j = 0; j < ny; j++) {
      const y = fr.ymax - (j + 0.5) / ny * (fr.ymax - fr.ymin);
      for (let i = 0; i < nx; i++) {
        const x = fr.xmin + (i + 0.5) / nx * (fr.xmax - fr.xmin);
        const c = colorFn(f(x, y));
        const k = 4 * (j * nx + i);
        img.data[k] = c[0]; img.data[k + 1] = c[1]; img.data[k + 2] = c[2]; img.data[k + 3] = c[3];
      }
    }
    octx.putImageData(img, 0, 0);
    ctx.imageSmoothingEnabled = true;
    ctx.drawImage(off, fr.X(fr.xmin), fr.Y(fr.ymax), fr.iw, fr.ih);
  }

  function label(ctx, x, y, text, color, align) {
    mathText(ctx, text, x, y, color || theme().fg, align || 'left');
  }

  /* Colormap viridis (independente do tema): t em [0,1] → [r, g, b].
     Âncoras do matplotlib interpoladas linearmente; usado nas superfícies
     de loss e densidades, onde um único matiz com alfa fica pobre. */
  const VIRIDIS = [
    [68, 1, 84], [72, 40, 120], [62, 74, 137], [49, 104, 142], [38, 130, 142],
    [31, 158, 137], [53, 183, 121], [109, 205, 89], [180, 222, 44], [253, 231, 37],
  ];
  function viridis(t) {
    t = Math.max(0, Math.min(1, t));
    const x = t * (VIRIDIS.length - 1);
    const i = Math.min(VIRIDIS.length - 2, Math.floor(x));
    const f = x - i;
    const a = VIRIDIS[i], b = VIRIDIS[i + 1];
    return [
      Math.round(a[0] + f * (b[0] - a[0])),
      Math.round(a[1] + f * (b[1] - a[1])),
      Math.round(a[2] + f * (b[2] - a[2])),
    ];
  }

  DL.plot = { theme, onRedraw, redrawAll, rgb, setup, observeResize, frame, clear, axes, line, scatter, histogram, heatmap, label, mathText, viridis };
})();
