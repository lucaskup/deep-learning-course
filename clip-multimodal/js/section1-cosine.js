/* Seção 1: similaridade do cosseno no espaço compartilhado.
   Dois vetores arrastáveis u, v; mostramos u·v, ‖u‖‖v‖, cosθ e θ,
   além dos normalizados û, v̂ sobre a circunferência de raio 1. */
(function () {
  'use strict';
  window.DL = window.DL || {};
  DL.sections = DL.sections || [];

  const RANGE = 2.3;          // meia-largura do plano em unidades de dados
  const U0 = [1.6, 0.7], V0 = [0.35, 1.55];

  function init() {
    if (!document.getElementById('s1-plane')) return;
    const P = DL.plot;
    const $ = (id) => document.getElementById(id);
    const cvPlane = $('s1-plane'), cvCos = $('s1-cos');

    let u = U0.slice(), v = V0.slice();
    let iso = null, drag = null;

    /* Mapa isométrico (escala igual nos dois eixos) para a circunferência ficar redonda. */
    function makeIso(w, h) {
      const pad = { l: 28, r: 14, t: 14, b: 24 };
      const iw = w - pad.l - pad.r, ih = h - pad.t - pad.b;
      const cx = pad.l + iw / 2, cy = pad.t + ih / 2;
      const s = Math.min(iw, ih) / (2 * RANGE);
      return {
        s, cx, cy,
        X: (x) => cx + x * s, Y: (y) => cy - y * s,
        invX: (px) => (px - cx) / s, invY: (py) => (cy - py) / s,
      };
    }

    function arrow(ctx, x0, y0, x1, y1, color, width) {
      ctx.strokeStyle = color; ctx.fillStyle = color; ctx.lineWidth = width;
      ctx.beginPath(); ctx.moveTo(x0, y0); ctx.lineTo(x1, y1); ctx.stroke();
      if (Math.hypot(x1 - x0, y1 - y0) < 4) return;
      const ang = Math.atan2(y1 - y0, x1 - x0), hl = 11;
      ctx.beginPath();
      ctx.moveTo(x1, y1);
      ctx.lineTo(x1 - hl * Math.cos(ang - 0.4), y1 - hl * Math.sin(ang - 0.4));
      ctx.lineTo(x1 - hl * Math.cos(ang + 0.4), y1 - hl * Math.sin(ang + 0.4));
      ctx.closePath(); ctx.fill();
    }

    const dot = () => u[0] * v[0] + u[1] * v[1];
    const norm = (a) => Math.hypot(a[0], a[1]);
    const cosT = () => dot() / (norm(u) * norm(v) || 1);

    function drawPlane() {
      const th = P.theme();
      const { ctx, w, h } = P.setup(cvPlane);
      P.clear(ctx, w, h);
      iso = makeIso(w, h);

      /* Eixos pelo centro. */
      ctx.strokeStyle = th.line; ctx.lineWidth = 1;
      ctx.beginPath();
      ctx.moveTo(iso.X(-RANGE), iso.cy); ctx.lineTo(iso.X(RANGE), iso.cy);
      ctx.moveTo(iso.cx, iso.Y(-RANGE)); ctx.lineTo(iso.cx, iso.Y(RANGE));
      ctx.stroke();

      /* Circunferência unitária (tracejada). */
      ctx.strokeStyle = th.comment; ctx.lineWidth = 1.2;
      ctx.setLineDash([4, 4]);
      ctx.beginPath(); ctx.arc(iso.cx, iso.cy, iso.s, 0, 2 * Math.PI); ctx.stroke();
      ctx.setLineDash([]);

      /* Arco do ângulo entre u e v. */
      const au = Math.atan2(u[1], u[0]), av = Math.atan2(v[1], v[0]);
      let d = av - au; while (d > Math.PI) d -= 2 * Math.PI; while (d < -Math.PI) d += 2 * Math.PI;
      const rArc = 0.55;
      ctx.strokeStyle = th.comment; ctx.lineWidth = 2;
      ctx.beginPath();
      for (let t = 0; t <= 1; t += 0.02) {
        const a = au + d * t;
        const px = iso.X(rArc * Math.cos(a)), py = iso.Y(rArc * Math.sin(a));
        t === 0 ? ctx.moveTo(px, py) : ctx.lineTo(px, py);
      }
      ctx.stroke();
      const aMid = au + d / 2;
      P.label(ctx, iso.X(0.85 * rArc * Math.cos(aMid)), iso.Y(0.85 * rArc * Math.sin(aMid)) + 4,
        'θ', th.comment, 'center');

      /* Vetores u, v. */
      arrow(ctx, iso.cx, iso.cy, iso.X(u[0]), iso.Y(u[1]), th.cyan, 2.4);
      arrow(ctx, iso.cx, iso.cy, iso.X(v[0]), iso.Y(v[1]), th.orange, 2.4);
      P.label(ctx, iso.X(u[0]) + 8, iso.Y(u[1]) - 6, 'u', th.cyan);
      P.label(ctx, iso.X(v[0]) + 8, iso.Y(v[1]) - 6, 'v', th.orange);

      /* Normalizados sobre a circunferência. */
      const uh = [u[0] / (norm(u) || 1), u[1] / (norm(u) || 1)];
      const vh = [v[0] / (norm(v) || 1), v[1] / (norm(v) || 1)];
      ctx.fillStyle = th.pink;
      ctx.beginPath(); ctx.arc(iso.X(uh[0]), iso.Y(uh[1]), 5, 0, 2 * Math.PI); ctx.fill();
      ctx.fillStyle = th.green;
      ctx.beginPath(); ctx.arc(iso.X(vh[0]), iso.Y(vh[1]), 5, 0, 2 * Math.PI); ctx.fill();
      P.label(ctx, iso.X(uh[0]) + 7, iso.Y(uh[1]) + 12, 'û', th.pink);
      P.label(ctx, iso.X(vh[0]) + 7, iso.Y(vh[1]) + 12, 'v̂', th.green);
    }

    function drawCos() {
      const th = P.theme();
      const { ctx, w, h } = P.setup(cvCos);
      P.clear(ctx, w, h);
      const c = cosT();
      const x0 = 44, x1 = w - 24, yMid = h * 0.62;
      const toX = (cc) => x0 + (cc + 1) / 2 * (x1 - x0);

      /* Trilho. */
      ctx.strokeStyle = th.line; ctx.lineWidth = 3;
      ctx.beginPath(); ctx.moveTo(x0, yMid); ctx.lineTo(x1, yMid); ctx.stroke();
      /* Ticks. */
      ctx.fillStyle = th.comment; ctx.font = '11px sans-serif'; ctx.textAlign = 'center';
      for (const tk of [-1, -0.5, 0, 0.5, 1]) {
        const tx = toX(tk);
        ctx.strokeStyle = th.line; ctx.lineWidth = 1;
        ctx.beginPath(); ctx.moveTo(tx, yMid - 5); ctx.lineTo(tx, yMid + 5); ctx.stroke();
        ctx.fillText(String(tk), tx, yMid + 20);
      }
      /* Marcador. */
      const mx = toX(c);
      ctx.fillStyle = th.purple;
      ctx.beginPath(); ctx.arc(mx, yMid, 7, 0, 2 * Math.PI); ctx.fill();
      ctx.strokeStyle = th.bg; ctx.lineWidth = 1.5; ctx.stroke();

      /* Leituras. */
      const theta = Math.acos(Math.max(-1, Math.min(1, c))) * 180 / Math.PI;
      P.mathText(ctx, 'cos θ = ' + c.toFixed(3), (x0 + x1) / 2, yMid - 34, th.purple, 'center', 14);
      P.mathText(ctx, 'θ = ' + theta.toFixed(1) + '°', (x0 + x1) / 2, yMid - 16, th.fg, 'center', 13);
    }

    function updateReadout() {
      const c = cosT();
      const theta = Math.acos(Math.max(-1, Math.min(1, c))) * 180 / Math.PI;
      $('s1-readout').textContent =
        'u·v = ' + dot().toFixed(2) +
        ' · ‖u‖‖v‖ = ' + (norm(u) * norm(v)).toFixed(2) +
        ' · cosθ = ' + c.toFixed(3) +
        ' · θ = ' + theta.toFixed(1) + '°';
    }

    function redraw() { drawPlane(); drawCos(); updateReadout(); }

    /* Arraste: pega a ponta mais próxima (u ou v). */
    function pick(px, py) {
      if (!iso) return null;
      const du = Math.hypot(px - iso.X(u[0]), py - iso.Y(u[1]));
      const dv = Math.hypot(px - iso.X(v[0]), py - iso.Y(v[1]));
      const m = Math.min(du, dv);
      if (m > 26) return null;
      return du <= dv ? 'u' : 'v';
    }
    function setVec(which, px, py) {
      let x = iso.invX(px), y = iso.invY(py);
      const lim = RANGE - 0.05;
      x = Math.max(-lim, Math.min(lim, x));
      y = Math.max(-lim, Math.min(lim, y));
      if (Math.hypot(x, y) < 0.12) { const a = Math.atan2(y, x) || 0; x = 0.12 * Math.cos(a); y = 0.12 * Math.sin(a); }
      (which === 'u' ? u : v)[0] = x;
      (which === 'u' ? u : v)[1] = y;
      redraw();
    }

    cvPlane.addEventListener('pointerdown', (e) => {
      const r = cvPlane.getBoundingClientRect();
      const tgt = pick(e.clientX - r.left, e.clientY - r.top);
      if (!tgt) return;
      drag = tgt;
      cvPlane.setPointerCapture(e.pointerId);
      setVec(drag, e.clientX - r.left, e.clientY - r.top);
    });
    cvPlane.addEventListener('pointermove', (e) => {
      if (!drag) return;
      const r = cvPlane.getBoundingClientRect();
      setVec(drag, e.clientX - r.left, e.clientY - r.top);
    });
    cvPlane.addEventListener('pointerup', () => { drag = null; });
    cvPlane.addEventListener('pointercancel', () => { drag = null; });

    $('s1-reset').addEventListener('click', () => { u = U0.slice(); v = V0.slice(); redraw(); });

    redraw();
    P.onRedraw(redraw);
    P.observeResize(cvPlane, redraw);
    P.observeResize(cvCos, redraw);
  }

  DL.sections.push({ name: 's1-cosine', init });
})();
