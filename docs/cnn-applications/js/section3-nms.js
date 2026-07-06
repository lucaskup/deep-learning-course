/* Seção 3: detecção de objetos e non-maximum suppression (NMS).
   Cena sintética com ground truth, caixas candidatas com confiança,
   limiar de confiança τ e limiar de IoU do NMS ajustáveis. */
(function () {
  'use strict';
  window.DL = window.DL || {};
  DL.sections = DL.sections || [];

  const CLASSES = ['gato', 'cachorro', 'pato'];
  const BASE_SEED = 11;
  const fm = (v, d) => v.toFixed(d == null ? 2 : d).replace('.', ',');

  function iou(a, b) {
    const ix = Math.max(0, Math.min(a.x + a.w, b.x + b.w) - Math.max(a.x, b.x));
    const iy = Math.max(0, Math.min(a.y + a.h, b.y + b.h) - Math.max(a.y, b.y));
    const inter = ix * iy;
    const union = a.w * a.h + b.w * b.h - inter;
    return union > 0 ? inter / union : 0;
  }

  function init() {
    const P = DL.plot;
    const $ = (id) => document.getElementById(id);

    const cvPre = $('s3-pre'), cvPost = $('s3-post');
    const slConf = $('s3-conf'), slIou = $('s3-iou');
    const btnRand = $('s3-rand'), btnReset = $('s3-reset');

    let seed = BASE_SEED;
    let gts = [], dets = [];

    function classColor(th, c) {
      return [th.red, th.cyan, th.green][c % 3];
    }

    function clamp01(v, lo, hi) {
      return Math.max(lo, Math.min(hi, v));
    }

    function genScene() {
      const rng = DL.utils.mulberry32(seed);
      gts = [];
      let guard = 0;
      while (gts.length < 3 && guard++ < 300) {
        const w = 0.22 + rng() * 0.14, h = 0.24 + rng() * 0.16;
        const box = { x: 0.03 + rng() * (0.94 - w), y: 0.03 + rng() * (0.94 - h), w: w, h: h, cls: gts.length };
        if (gts.every(function (g) { return iou(g, box) < 0.12; })) gts.push(box);
      }
      dets = [];
      /* Detecções redundantes em torno de cada objeto (várias caixas por objeto). */
      gts.forEach(function (g) {
        const n = 3 + Math.floor(rng() * 3);
        for (let i = 0; i < n; i++) {
          const sw = g.w * (0.82 + rng() * 0.36), sh = g.h * (0.82 + rng() * 0.36);
          const dx = (rng() - 0.5) * 0.09, dy = (rng() - 0.5) * 0.09;
          const x = clamp01(g.x + dx, 0, 1 - sw), y = clamp01(g.y + dy, 0, 1 - sh);
          const conf = i === 0 ? 0.75 + rng() * 0.22 : 0.5 + rng() * 0.42;
          dets.push({ x: x, y: y, w: sw, h: sh, cls: g.cls, conf: conf });
        }
      });
      /* Falsos positivos de baixa confiança, espalhados pela imagem. */
      const nFP = 4 + Math.floor(rng() * 3);
      for (let i = 0; i < nFP; i++) {
        const w = 0.1 + rng() * 0.16, h = 0.1 + rng() * 0.16;
        dets.push({
          x: rng() * (1 - w), y: rng() * (1 - h), w: w, h: h,
          cls: Math.floor(rng() * 3), conf: 0.06 + rng() * 0.4,
        });
      }
      dets.sort(function (a, b) { return b.conf - a.conf; });
    }

    /* NMS: filtra por confiança, percorre em ordem decrescente e suprime
       caixas da mesma classe com IoU acima do limiar. */
    function runNMS() {
      const tau = +slConf.value, thr = +slIou.value;
      const cand = dets.filter(function (d) { return d.conf >= tau; });
      const kept = [];
      for (const d of cand) {
        let ok = true;
        for (const k of kept) {
          if (k.cls === d.cls && iou(k, d) > thr) { ok = false; break; }
        }
        if (ok) kept.push(d);
      }
      return { cand: cand, kept: kept };
    }

    /* TP: caixa mantida com IoU ≥ 0,5 contra um ground truth livre da mesma classe. */
    function countTP(kept) {
      const used = gts.map(function () { return false; });
      let tp = 0;
      for (const k of kept) {
        let best = -1, bestIou = 0.5;
        for (let i = 0; i < gts.length; i++) {
          if (used[i] || gts[i].cls !== k.cls) continue;
          const v = iou(gts[i], k);
          if (v >= bestIou) { bestIou = v; best = i; }
        }
        if (best >= 0) { used[best] = true; tp++; }
      }
      return tp;
    }

    function makeFrame(ctx, w, h) {
      return P.frame(ctx, w, h, 0, 1, 0, 1, { l: 12, r: 12, t: 24, b: 12 });
    }

    function drawScene(ctx, fr, th) {
      const Yi = (v) => fr.Y(1 - v);
      ctx.fillStyle = th.line;
      ctx.globalAlpha = 0.18;
      ctx.fillRect(fr.X(0), Yi(0), fr.iw, fr.ih);
      ctx.globalAlpha = 1;
      ctx.strokeStyle = th.line;
      ctx.lineWidth = 1.5;
      ctx.strokeRect(fr.X(0), Yi(0), fr.iw, fr.ih);
      /* Legenda das classes */
      let lx = fr.X(0);
      for (let c = 0; c < CLASSES.length; c++) {
        P.mathText(ctx, CLASSES[c], lx, Yi(0) - 8, classColor(th, c), 'left', 11);
        lx += 14 + 7 * CLASSES[c].length;
      }
      return Yi;
    }

    function drawBoxRect(ctx, fr, Yi, b, color, opts) {
      const x0 = fr.X(b.x), y0 = Yi(b.y);
      const bw = fr.X(b.x + b.w) - x0, bh = Yi(b.y + b.h) - y0;
      if (opts.fill) {
        ctx.fillStyle = color;
        ctx.globalAlpha = opts.fill;
        ctx.fillRect(x0, y0, bw, bh);
        ctx.globalAlpha = 1;
      }
      ctx.strokeStyle = color;
      ctx.lineWidth = opts.width || 1.6;
      ctx.globalAlpha = opts.alpha != null ? opts.alpha : 1;
      if (opts.dash) ctx.setLineDash(opts.dash);
      ctx.strokeRect(x0, y0, bw, bh);
      ctx.setLineDash([]);
      ctx.globalAlpha = 1;
      if (opts.text) P.mathText(ctx, opts.text, x0 + 3, y0 + 12, color, 'left', 10);
    }

    function drawPre(cand) {
      const th = P.theme();
      const sz = P.setup(cvPre);
      const ctx = sz.ctx, w = sz.w, h = sz.h;
      P.clear(ctx, w, h);
      const fr = makeFrame(ctx, w, h);
      const Yi = drawScene(ctx, fr, th);
      for (const d of cand) {
        drawBoxRect(ctx, fr, Yi, d, classColor(th, d.cls), {
          width: 1.2 + 1.6 * d.conf,
          alpha: 0.35 + 0.65 * d.conf,
          text: fm(d.conf),
        });
      }
    }

    function drawPost(kept) {
      const th = P.theme();
      const sz = P.setup(cvPost);
      const ctx = sz.ctx, w = sz.w, h = sz.h;
      P.clear(ctx, w, h);
      const fr = makeFrame(ctx, w, h);
      const Yi = drawScene(ctx, fr, th);
      for (const g of gts) {
        drawBoxRect(ctx, fr, Yi, g, classColor(th, g.cls), {
          width: 1.4, alpha: 0.7, dash: [5, 4],
        });
      }
      for (const k of kept) {
        drawBoxRect(ctx, fr, Yi, k, classColor(th, k.cls), {
          width: 2.4, fill: 0.12,
          text: CLASSES[k.cls] + ' ' + fm(k.conf),
        });
      }
    }

    function redraw() {
      const r = runNMS();
      drawPre(r.cand);
      drawPost(r.kept);
      const tp = countTP(r.kept);
      const prec = r.kept.length > 0 ? tp / r.kept.length : 0;
      const rec = gts.length > 0 ? tp / gts.length : 0;
      $('s3-readout').textContent =
        'candidatas: ' + dets.length + ' · após τ: ' + r.cand.length +
        ' · mantidas pelo NMS: ' + r.kept.length + ' · TP: ' + tp +
        ' · precisão = ' + fm(prec) + ' · revocação = ' + fm(rec);
    }

    function syncLabels() {
      $('s3-conf-val').textContent = fm(+slConf.value);
      $('s3-iou-val').textContent = fm(+slIou.value);
    }

    slConf.addEventListener('input', function () { syncLabels(); redraw(); });
    slIou.addEventListener('input', function () { syncLabels(); redraw(); });
    btnRand.addEventListener('click', function () {
      seed = seed + 1;
      genScene();
      redraw();
    });
    btnReset.addEventListener('click', function () {
      seed = BASE_SEED;
      slConf.value = 0.5;
      slIou.value = 0.5;
      syncLabels();
      genScene();
      redraw();
    });

    genScene();
    syncLabels();
    redraw();
    P.onRedraw(redraw);
    P.observeResize(cvPre, redraw);
    P.observeResize(cvPost, redraw);
  }

  DL.sections.push({ name: 's3-nms', init });
})();
