/* Seção 4: classificação zero-shot.
   Uma imagem û (unitária, arrastável) e K prompts de classe v̂_k (unitários).
   s_k = û·v̂_k; probabilidades = softmax(s_k/τ); barra do argmax destacada.
   O classificador é construído na inferência a partir dos prompts. */
(function () {
  'use strict';
  window.DL = window.DL || {};
  DL.sections = DL.sections || [];

  const LABELS = ['gato', 'cachorro', 'carro', 'pássaro'];
  const K = LABELS.length;
  const IMG0 = -0.5;
  const PROMPT0 = [-Math.PI / 2, -Math.PI / 2 + 2 * Math.PI / K, -Math.PI / 2 + 4 * Math.PI / K, -Math.PI / 2 + 6 * Math.PI / K];
  const tauOf = (v) => 0.01 * Math.pow(100, v / 100);   // v∈[0,100] → τ∈[0.01,1]

  function init() {
    if (!document.getElementById('s4-circle')) return;
    const P = DL.plot;
    const $ = (id) => document.getElementById(id);
    const cvCirc = $('s4-circle'), cvBars = $('s4-bars');
    const slTau = $('s4-tau');

    let imgAng = IMG0;
    let prompts = PROMPT0.slice();
    let tau = tauOf(+slTau.value);
    let geom = null;
    let drag = null;          // 'img' ou índice de prompt (0..K-1)

    const promptColors = (th) => [th.purple, th.orange, th.pink, th.green];

    function makeGeom(w, h) {
      const pad = 26;
      const r = Math.min(w - 2 * pad, h - 2 * pad) / 2 - 18;
      return { cx: w / 2, cy: h / 2, r };
    }
    const tip = (ang, scale) => [geom.cx + geom.r * (scale || 1) * Math.cos(ang), geom.cy + geom.r * (scale || 1) * Math.sin(ang)];

    const scores = () => prompts.map((a) => Math.cos(imgAng - a));
    function softmax(s) {
      const mx = Math.max(...s);
      const e = s.map((v) => Math.exp(v / tau - mx / tau));
      const sum = e.reduce((a, b) => a + b, 0);
      return e.map((v) => v / sum);
    }

    function arrow(ctx, x0, y0, x1, y1, color, width) {
      ctx.strokeStyle = color; ctx.fillStyle = color; ctx.lineWidth = width;
      ctx.beginPath(); ctx.moveTo(x0, y0); ctx.lineTo(x1, y1); ctx.stroke();
      const ang = Math.atan2(y1 - y0, x1 - x0), hl = 11;
      ctx.beginPath();
      ctx.moveTo(x1, y1);
      ctx.lineTo(x1 - hl * Math.cos(ang - 0.4), y1 - hl * Math.sin(ang - 0.4));
      ctx.lineTo(x1 - hl * Math.cos(ang + 0.4), y1 - hl * Math.sin(ang + 0.4));
      ctx.closePath(); ctx.fill();
    }

    function drawCircle() {
      const th = P.theme();
      const { ctx, w, h } = P.setup(cvCirc);
      P.clear(ctx, w, h);
      geom = makeGeom(w, h);
      const pc = promptColors(th);
      const probs = softmax(scores());
      const arg = probs.indexOf(Math.max(...probs));

      ctx.strokeStyle = th.line; ctx.lineWidth = 1.2;
      ctx.beginPath(); ctx.arc(geom.cx, geom.cy, geom.r, 0, 2 * Math.PI); ctx.stroke();

      /* Prompts. */
      for (let k = 0; k < K; k++) {
        const t = tip(prompts[k]);
        arrow(ctx, geom.cx, geom.cy, t[0], t[1], pc[k], k === arg ? 3 : 2);
        const lp = tip(prompts[k], 1.14);
        P.mathText(ctx, LABELS[k], lp[0], lp[1] + 4, pc[k], 'center', 12);
        const dp = tip(prompts[k]);
        ctx.fillStyle = pc[k];
        ctx.beginPath(); ctx.arc(dp[0], dp[1], 5, 0, 2 * Math.PI); ctx.fill();
      }
      /* Imagem û. */
      const it = tip(imgAng);
      arrow(ctx, geom.cx, geom.cy, it[0], it[1], th.cyan, 3);
      ctx.fillStyle = th.cyan;
      ctx.beginPath(); ctx.arc(it[0], it[1], 7, 0, 2 * Math.PI); ctx.fill();
      ctx.strokeStyle = th.bg; ctx.lineWidth = 1.5; ctx.stroke();
      const il = tip(imgAng, 1.16);
      P.mathText(ctx, 'û (imagem)', il[0], il[1] + 4, th.cyan, 'center', 12);
    }

    function drawBars() {
      const th = P.theme();
      const { ctx, w, h } = P.setup(cvBars);
      P.clear(ctx, w, h);
      const pc = promptColors(th);
      const s = scores();
      const probs = softmax(s);
      const arg = probs.indexOf(Math.max(...probs));
      const padL = 34, padT = 16, padR = 12, padB = 34;
      const iw = w - padL - padR, ih = h - padT - padB;
      const base = padT + ih;
      ctx.strokeStyle = th.line; ctx.lineWidth = 1;
      ctx.beginPath(); ctx.moveTo(padL, padT); ctx.lineTo(padL, base); ctx.lineTo(padL + iw, base); ctx.stroke();
      ctx.fillStyle = th.comment; ctx.font = '10px sans-serif'; ctx.textAlign = 'right';
      for (const yv of [0, 0.5, 1]) {
        const yy = base - yv * ih;
        ctx.fillText(yv.toFixed(1), padL - 4, yy + 3);
        ctx.strokeStyle = th.line; ctx.globalAlpha = 0.5;
        ctx.beginPath(); ctx.moveTo(padL, yy); ctx.lineTo(padL + iw, yy); ctx.stroke();
        ctx.globalAlpha = 1;
      }
      const bw = iw / K;
      for (let k = 0; k < K; k++) {
        const bx = padL + k * bw + bw * 0.18;
        const bwid = bw * 0.64;
        const bh = probs[k] * ih;
        const winner = k === arg;
        ctx.fillStyle = winner ? th.green : pc[k];
        ctx.globalAlpha = winner ? 0.9 : 0.5;
        ctx.fillRect(bx, base - bh, bwid, bh);
        ctx.globalAlpha = 1;
        ctx.fillStyle = winner ? th.green : th.fg;
        ctx.font = (winner ? 'bold ' : '') + '11px sans-serif';
        ctx.textAlign = 'center';
        ctx.fillText(probs[k].toFixed(2), bx + bwid / 2, base - bh - 5);
        ctx.fillStyle = th.comment; ctx.font = '11px sans-serif';
        ctx.fillText(LABELS[k], bx + bwid / 2, base + 14);
        P.mathText(ctx, 's=' + s[k].toFixed(2), bx + bwid / 2, base + 27, th.comment, 'center', 9);
      }
    }

    function redraw() {
      drawCircle();
      drawBars();
      const probs = softmax(scores());
      const arg = probs.indexOf(Math.max(...probs));
      $('s4-readout').textContent =
        'predição: ' + LABELS[arg] + ' (p = ' + probs[arg].toFixed(3) + ')';
    }

    function pick(px, py) {
      if (!geom) return null;
      let best = null, bd = 24;
      const it = tip(imgAng);
      let d = Math.hypot(px - it[0], py - it[1]);
      if (d < bd) { bd = d; best = 'img'; }
      for (let k = 0; k < K; k++) {
        const t = tip(prompts[k]);
        d = Math.hypot(px - t[0], py - t[1]);
        if (d < bd) { bd = d; best = k; }
      }
      return best;
    }

    cvCirc.addEventListener('pointerdown', (e) => {
      const r = cvCirc.getBoundingClientRect();
      const hit = pick(e.clientX - r.left, e.clientY - r.top);
      if (hit === null) return;
      drag = hit;
      cvCirc.setPointerCapture(e.pointerId);
    });
    cvCirc.addEventListener('pointermove', (e) => {
      if (drag === null) return;
      const r = cvCirc.getBoundingClientRect();
      const a = Math.atan2(e.clientY - r.top - geom.cy, e.clientX - r.left - geom.cx);
      if (drag === 'img') imgAng = a; else prompts[drag] = a;
      redraw();
    });
    cvCirc.addEventListener('pointerup', () => { drag = null; });
    cvCirc.addEventListener('pointercancel', () => { drag = null; });

    slTau.addEventListener('input', () => {
      tau = tauOf(+slTau.value);
      $('s4-tau-val').textContent = tau.toFixed(2);
      redraw();
    });
    $('s4-reset').addEventListener('click', () => {
      imgAng = IMG0; prompts = PROMPT0.slice(); redraw();
    });

    $('s4-tau-val').textContent = tau.toFixed(2);
    redraw();
    P.onRedraw(redraw);
    P.observeResize(cvCirc, redraw);
    P.observeResize(cvBars, redraw);
  }

  DL.sections.push({ name: 's4-zeroshot', init });
})();
