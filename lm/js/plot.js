/* Helpers de canvas: DPR, cache de cores do tema e primitivas de desenho. */
(function () {
  'use strict';
  window.DM = window.DM || {};

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
    return { X, Y, pad, iw, ih, xmin, xmax, ymin, ymax };
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
    const lo = fr.ymin !== undefined && opts.range ? opts.range[0] : fr.xmin;
    const hi = opts.range ? opts.range[1] : fr.xmax;
    const bins = new Float64Array(nbins);
    const bw = (hi - lo) / nbins;
    let count = 0;
    for (const v of values) {
      const b = Math.floor((v - lo) / bw);
      if (b >= 0 && b < nbins) { bins[b]++; count++; }
    }
    ctx.fillStyle = color;
    ctx.globalAlpha = opts.alpha != null ? opts.alpha : 0.35;
    for (let b = 0; b < nbins; b++) {
      const density = bins[b] / (values.length * bw);
      const x0 = fr.X(lo + b * bw), x1 = fr.X(lo + (b + 1) * bw);
      const y0 = fr.Y(0), y1 = fr.Y(density);
      ctx.fillRect(x0, y1, x1 - x0, y0 - y1);
    }
    ctx.globalAlpha = 1;
  }

  function label(ctx, x, y, text, color, align) {
    mathText(ctx, text, x, y, color || theme().fg, align || 'left');
  }

  /* Gráfico de barras verticais com rótulo embaixo e valor em cima.
     values na escala do frame; colors: cor única ou array por barra. */
  function bars(ctx, fr, labels, values, colors, opts) {
    opts = opts || {};
    const n = values.length;
    const slot = fr.iw / Math.max(1, n);
    const bw = slot * (opts.barFrac || 0.62);
    for (let i = 0; i < n; i++) {
      const cx = fr.pad.l + slot * (i + 0.5);
      const h = (values[i] - fr.ymin) / (fr.ymax - fr.ymin) * fr.ih;
      const color = Array.isArray(colors) ? colors[i] : colors;
      ctx.fillStyle = color;
      ctx.globalAlpha = opts.alpha != null ? opts.alpha : 0.9;
      ctx.fillRect(cx - bw / 2, fr.Y(fr.ymin) - h, bw, h);
      ctx.globalAlpha = 1;
      if (opts.values) {
        mathText(ctx, opts.values[i], cx, fr.Y(fr.ymin) - h - 4, color, 'center', opts.valueSize || 10);
      }
      if (labels && labels[i] != null) {
        mathText(ctx, labels[i], cx, fr.Y(fr.ymin) + 13, theme().comment, 'center', opts.labelSize || 10);
      }
      if (opts.sublabels && opts.sublabels[i] != null) {
        mathText(ctx, opts.sublabels[i], cx, fr.Y(fr.ymin) + 26, opts.sublabelColor || theme().comment, 'center', 10);
      }
    }
  }

  /* Seta com cabeça triangular, em coordenadas de dados. */
  function arrow(ctx, fr, x0, y0, x1, y1, color, opts) {
    opts = opts || {};
    const px0 = fr.X(x0), py0 = fr.Y(y0), px1 = fr.X(x1), py1 = fr.Y(y1);
    const ang = Math.atan2(py1 - py0, px1 - px0);
    const head = opts.head || 9;
    ctx.strokeStyle = color;
    ctx.fillStyle = color;
    ctx.lineWidth = opts.width || 2;
    if (opts.dash) ctx.setLineDash(opts.dash);
    ctx.beginPath();
    ctx.moveTo(px0, py0);
    ctx.lineTo(px1 - head * 0.6 * Math.cos(ang), py1 - head * 0.6 * Math.sin(ang));
    ctx.stroke();
    ctx.setLineDash([]);
    ctx.beginPath();
    ctx.moveTo(px1, py1);
    ctx.lineTo(px1 - head * Math.cos(ang - 0.45), py1 - head * Math.sin(ang - 0.45));
    ctx.lineTo(px1 - head * Math.cos(ang + 0.45), py1 - head * Math.sin(ang + 0.45));
    ctx.closePath();
    ctx.fill();
  }

  DM.plot = { theme, onRedraw, redrawAll, setup, observeResize, frame, clear, axes, line, scatter, histogram, label, mathText, bars, arrow };
})();
