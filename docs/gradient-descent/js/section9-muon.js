/* Seção 9: Muon no exemplo 2×2 dos slides. Momentum M ← βM + G, normalização
   pela norma de Frobenius, valores e direções singulares de M, ortogonalização
   (forma fechada / U·Vᵀ) e a iteração de Newton-Schulz com os coeficientes do
   Muon; comparação com a direção do Adam (sinal do momentum). */
(function () {
  'use strict';
  window.DL = window.DL || {};
  DL.sections = DL.sections || [];

  const NS_A = 3.4445, NS_B = -4.7750, NS_C = 2.0315, NS_ITERS = 5;
  const CELLS = ['W11', 'W12', 'W21', 'W22', 'M11', 'M12', 'M21', 'M22', 'G11', 'G12', 'G21', 'G22'];
  const DEFAULTS = {
    eta: 0.1, beta: 0.9,
    W11: 3, W12: 5, W21: 2, W22: 4,
    M11: -1, M12: -2, M21: 3, M22: -4,
    G11: 1.9, G12: -2.2, G21: 1.3, G22: -3.4,
  };

  /* ── álgebra 2×2 (matrizes como [[a, b], [c, d]]) ── */
  const mul = (A, B) => [
    [A[0][0] * B[0][0] + A[0][1] * B[1][0], A[0][0] * B[0][1] + A[0][1] * B[1][1]],
    [A[1][0] * B[0][0] + A[1][1] * B[1][0], A[1][0] * B[0][1] + A[1][1] * B[1][1]],
  ];
  const tr = (A) => [[A[0][0], A[1][0]], [A[0][1], A[1][1]]];
  const add = (A, B, s) => [[A[0][0] + s * B[0][0], A[0][1] + s * B[0][1]], [A[1][0] + s * B[1][0], A[1][1] + s * B[1][1]]];
  const scale = (A, s) => [[A[0][0] * s, A[0][1] * s], [A[1][0] * s, A[1][1] * s]];
  const fro = (A) => Math.sqrt(A[0][0] ** 2 + A[0][1] ** 2 + A[1][0] ** 2 + A[1][1] ** 2);
  const det = (A) => A[0][0] * A[1][1] - A[0][1] * A[1][0];
  const mv = (A, v) => [A[0][0] * v[0] + A[0][1] * v[1], A[1][0] * v[0] + A[1][1] * v[1]];
  const dot4 = (A, B) => A[0][0] * B[0][0] + A[0][1] * B[0][1] + A[1][0] * B[1][0] + A[1][1] * B[1][1];

  /* SVD de uma 2×2: σ a partir de MᵀM, v_i autovetores de MᵀM, u_i = M v_i / σ_i.
     Sinal fixado para u_i ter primeira componente ≥ 0 (convenção dos slides). */
  function svd2(M) {
    const MtM = mul(tr(M), M);
    const a = MtM[0][0], b = MtM[0][1], d = MtM[1][1];
    const t = a + d, dt = a * d - b * b;
    const disc = Math.sqrt(Math.max(0, t * t / 4 - dt));
    const l1 = t / 2 + disc, l2 = Math.max(0, t / 2 - disc);
    const s1 = Math.sqrt(l1), s2 = Math.sqrt(l2);
    let v1;
    if (Math.abs(b) > 1e-12) v1 = [b, l1 - a];
    else v1 = a >= d ? [1, 0] : [0, 1];
    const n1 = Math.hypot(v1[0], v1[1]) || 1;
    v1 = [v1[0] / n1, v1[1] / n1];
    let v2 = [-v1[1], v1[0]];
    let u1, u2;
    if (s1 > 1e-12) { const t1 = mv(M, v1); u1 = [t1[0] / s1, t1[1] / s1]; } else u1 = [1, 0];
    if (s2 > 1e-12) { const t2 = mv(M, v2); u2 = [t2[0] / s2, t2[1] / s2]; } else u2 = [-u1[1], u1[0]];
    const fix = (u, v) => {
      const neg = u[0] < -1e-12 || (Math.abs(u[0]) <= 1e-12 && u[1] < 0);
      return neg ? [[-u[0], -u[1]], [-v[0], -v[1]]] : [u, v];
    };
    [u1, v1] = fix(u1, v1);
    [u2, v2] = fix(u2, v2);
    return { s1, s2, u1, u2, v1, v2 };
  }

  /* Matriz ortogonal mais próxima: U·Vᵀ = u₁v₁ᵀ + u₂v₂ᵀ. */
  function polar(M) {
    const s = svd2(M);
    return [
      [s.u1[0] * s.v1[0] + s.u2[0] * s.v2[0], s.u1[0] * s.v1[1] + s.u2[0] * s.v2[1]],
      [s.u1[1] * s.v1[0] + s.u2[1] * s.v2[0], s.u1[1] * s.v1[1] + s.u2[1] * s.v2[1]],
    ];
  }

  /* Iteração de Newton-Schulz do Muon: X ← aX + b(XXᵀ)X + c(XXᵀ)²X. */
  function newtonSchulz(M) {
    const seq = [];
    let X = scale(M, 1 / (fro(M) + 1e-12));
    seq.push(X);
    for (let i = 0; i < NS_ITERS; i++) {
      const A = mul(X, tr(X));
      const B = add(scale(A, NS_B), mul(A, A), NS_C);     /* bA + cA² */
      X = add(scale(X, NS_A), mul(B, X), 1);               /* aX + (bA + cA²)X */
      seq.push(X);
    }
    return seq;
  }

  function init() {
    const P = DL.plot;
    const $ = (id) => document.getElementById(id);

    /* Rótulo sobre fundo opaco (legível sobre heatmap, curvas e pontos);
       com fr, fica dentro do retângulo do gráfico. */
    function pill(ctx, x, y, text, color, align, fr, size) {
      const th = P.theme();
      size = size || 11;
      ctx.font = size + 'px sans-serif';
      const tw = ctx.measureText(text.replace(/[_{}]/g, '')).width;
      const lead = align === 'center' ? tw / 2 : align === 'right' ? tw : 0;
      if (fr) {
        x = Math.max(fr.X(fr.xmin) + lead + 4, Math.min(fr.X(fr.xmax) - (tw - lead) - 4, x));
        y = Math.max(fr.Y(fr.ymax) + size + 3, Math.min(fr.Y(fr.ymin) - 6, y));
      }
      ctx.fillStyle = th.card;
      ctx.globalAlpha = 0.85;
      ctx.fillRect(x - lead - 3, y - size + 1, tw + 6, size + 4);
      ctx.globalAlpha = 1;
      P.mathText(ctx, text, x, y, color, align, size);
    }
    /* Fundo opaco atrás do rótulo do eixo x que P.axes desenha dentro do
       gráfico (sobre o amarelo do viridis o texto some). */
    function xlabelPill(ctx, fr, text) {
      const th = P.theme();
      ctx.font = '11px sans-serif';
      const tw = ctx.measureText(text.replace(/[_{}]/g, '')).width;
      ctx.fillStyle = th.card;
      ctx.globalAlpha = 0.85;
      ctx.fillRect(fr.X(fr.xmax) - 7 - tw, fr.Y(fr.ymin) - 18, tw + 6, 14);
      ctx.globalAlpha = 1;
    }
    const cvGeo = $('s9-geometry'), cvNS = $('s9-ns');
    const inputs = { eta: $('s9-eta'), beta: $('s9-beta') };
    for (const c of CELLS) inputs[c] = $('s9-' + c);

    const f2 = (v) => (Math.abs(v) < 5e-4 ? '0' : (Math.round(v * 100) / 100).toString()).replace('-', '−');
    const f3 = (v) => (Math.abs(v) < 5e-4 ? '0' : (Math.round(v * 1000) / 1000).toString()).replace('-', '−');
    const matHtml = (A, f) => '<span class="s9-mat">[' + f(A[0][0]) + ' ' + f(A[0][1]) + '; ' + f(A[1][0]) + ' ' + f(A[1][1]) + ']</span>';

    function read() {
      const x = {};
      for (const k of Object.keys(inputs)) {
        const v = parseFloat(inputs[k].value);
        x[k] = isFinite(v) ? v : DEFAULTS[k];
      }
      return {
        eta: x.eta, beta: x.beta,
        W: [[x.W11, x.W12], [x.W21, x.W22]],
        M: [[x.M11, x.M12], [x.M21, x.M22]],
        G: [[x.G11, x.G12], [x.G21, x.G22]],
      };
    }

    function compute(x) {
      const M1 = add(scale(x.M, x.beta), x.G, 1);       /* M_{t+1} = βM_t + G_t */
      const nf = fro(M1);
      const Mh = scale(M1, 1 / (nf || 1));
      const s = svd2(M1);
      const O = polar(M1);
      const W1 = add(x.W, O, -x.eta);
      const dM = det(M1);
      const p = M1[0][0] + M1[1][1], q = M1[1][0] - M1[0][1];
      const sgn = (v) => (v > 0 ? 1 : v < 0 ? -1 : 0);
      const DAdam = [[sgn(M1[0][0]), sgn(M1[0][1])], [sgn(M1[1][0]), sgn(M1[1][1])]];
      const sa = svd2(DAdam);
      const cosang = dot4(DAdam, O) / ((fro(DAdam) * fro(O)) || 1);
      const ang = Math.acos(Math.max(-1, Math.min(1, cosang))) * 180 / Math.PI;
      const ns = newtonSchulz(M1);
      const nsLast = ns[ns.length - 1];
      let nsErr = 0;
      for (let i = 0; i < 2; i++) for (let j = 0; j < 2; j++) nsErr = Math.max(nsErr, Math.abs(nsLast[i][j] - O[i][j]));
      return { M1, nf, Mh, s, O, W1, dM, p, q, DAdam, sa, ang, ns, nsErr };
    }

    function renderSteps(x, r) {
      const lines = [];
      lines.push('<b>1. Momentum:</b> M<sub>t+1</sub> = ' + f2(x.beta) + '·' + matHtml(x.M, f2) + ' + ' + matHtml(x.G, f2) + ' = ' + matHtml(r.M1, f2));
      lines.push('<b>2. Normalização:</b> ‖M<sub>t+1</sub>‖<sub>F</sub> = ' + f3(r.nf) + ', M̂<sub>t+1</sub> = ' + matHtml(r.Mh, f2));
      lines.push('<b>Valores singulares de M<sub>t+1</sub>:</b> σ₁ = ' + f3(r.s.s1) + ', σ₂ = ' + f3(r.s.s2) +
        ' (razão ' + (r.s.s2 > 1e-9 ? f2(r.s.s1 / r.s.s2) : '∞') + '); u₁ = (' + f2(r.s.u1[0]) + ', ' + f2(r.s.u1[1]) + ') a ' +
        f2(Math.atan2(r.s.u1[1], r.s.u1[0]) * 180 / Math.PI) + '°, u₂ = (' + f2(r.s.u2[0]) + ', ' + f2(r.s.u2[1]) + ')');
      if (r.dM > 0) {
        lines.push('<b>3. Ortogonalização (forma fechada, det = ' + f2(r.dM) + ' &gt; 0):</b> p = a + d = ' + f2(r.p) +
          ', q = c − b = ' + f2(r.q) + ', √(p² + q²) = ' + f3(Math.hypot(r.p, r.q)) + ', O<sub>t+1</sub> = ' + matHtml(r.O, f2));
      } else {
        lines.push('<b>3. Ortogonalização (det = ' + f2(r.dM) + ' ≤ 0, a ortogonal mais próxima é uma reflexão, O = U·Vᵀ):</b> O<sub>t+1</sub> = ' + matHtml(r.O, f2));
      }
      lines.push('<b>4. Atualização:</b> W<sub>t+1</sub> = W<sub>t</sub> − ' + f2(x.eta) + '·O<sub>t+1</sub> = ' + matHtml(r.W1, f2));
      lines.push('<b>Adam com histórico estável (ṽ = m̃²):</b> D<sub>Adam</sub> = sinal(M<sub>t+1</sub>) = ' + matHtml(r.DAdam, f2) +
        ', σ = (' + f2(r.sa.s1) + ', ' + f2(r.sa.s2) + ')' + (r.sa.s2 < 1e-9 ? ', rank 1' : '') +
        '; ângulo entre D<sub>Adam</sub> e D<sub>Muon</sub> como vetores de 4 entradas: ' + f2(r.ang) + '°');
      lines.push('<b>Newton-Schulz (5 iterações a partir de M̂):</b> resultado ' + matHtml(r.ns[r.ns.length - 1], f2) +
        ', maior desvio em relação a O<sub>t+1</sub>: ' + f3(r.nsErr));
      $('s9-steps').innerHTML = lines.map((l) => '<div>' + l + '</div>').join('');
    }

    /* Três painéis: imagem do círculo unitário por M_{t+1}, D_Adam e D_Muon. */
    function drawGeometry(r) {
      const th = P.theme();
      const { ctx, w, h } = P.setup(cvGeo);
      P.clear(ctx, w, h);
      const panels = [
        { title: 'M_{t+1}', A: r.M1, s: r.s, own: true },
        { title: 'D_{Adam}', A: r.DAdam, s: r.sa, own: false },
        { title: 'D_{Muon}', A: r.O, s: svd2(r.O), own: false },
      ];
      const pw = w / 3;
      const rad = Math.min(pw, h) * 0.34;
      panels.forEach((pn, k) => {
        const cx = pw * k + pw / 2, cy = h / 2 + 8;
        /* escala: painel do momentum tem escala própria; os outros compartilham (raio 1 → rad/1.6) */
        const unit = pn.own ? rad / Math.max(pn.s.s1, 1e-9) : rad / 1.6;
        ctx.strokeStyle = th.line;
        ctx.lineWidth = 1;
        ctx.beginPath();
        ctx.moveTo(cx - rad * 1.25, cy); ctx.lineTo(cx + rad * 1.25, cy);
        ctx.moveTo(cx, cy - rad * 1.25); ctx.lineTo(cx, cy + rad * 1.25);
        ctx.stroke();
        /* círculo unitário tracejado (só onde a escala é compartilhada) */
        if (!pn.own) {
          ctx.setLineDash([3, 4]);
          ctx.strokeStyle = th.comment;
          ctx.beginPath();
          ctx.arc(cx, cy, unit, 0, 2 * Math.PI);
          ctx.stroke();
          ctx.setLineDash([]);
        }
        /* imagem do círculo */
        ctx.strokeStyle = th.orange;
        ctx.lineWidth = 2.2;
        ctx.beginPath();
        for (let i = 0; i <= 120; i++) {
          const a = 2 * Math.PI * i / 120;
          const p = mv(pn.A, [Math.cos(a), Math.sin(a)]);
          const X = cx + p[0] * unit, Y = cy - p[1] * unit;
          if (i === 0) ctx.moveTo(X, Y); else ctx.lineTo(X, Y);
        }
        ctx.stroke();
        /* direções singulares de saída σ_i u_i */
        const arrow = (vx, vy, color, label) => {
          const X = cx + vx * unit, Y = cy - vy * unit;
          ctx.strokeStyle = color; ctx.fillStyle = color; ctx.lineWidth = 1.8;
          ctx.beginPath(); ctx.moveTo(cx, cy); ctx.lineTo(X, Y); ctx.stroke();
          const ang = Math.atan2(Y - cy, X - cx), hl = 6;
          if (Math.hypot(X - cx, Y - cy) > 4) {
            ctx.beginPath(); ctx.moveTo(X, Y);
            ctx.lineTo(X - hl * Math.cos(ang - 0.45), Y - hl * Math.sin(ang - 0.45));
            ctx.lineTo(X - hl * Math.cos(ang + 0.45), Y - hl * Math.sin(ang + 0.45));
            ctx.closePath(); ctx.fill();
          }
          pill(ctx, X + 6 * Math.cos(ang), Y + 6 * Math.sin(ang) + 4, label, color, 'left', null, 10);
        };
        arrow(pn.s.s1 * pn.s.u1[0], pn.s.s1 * pn.s.u1[1], th.cyan, 'σ_1u_1');
        if (pn.s.s2 > 1e-9) arrow(pn.s.s2 * pn.s.u2[0], pn.s.s2 * pn.s.u2[1], th.green, 'σ_2u_2');
        else pill(ctx, cx + 6, cy + rad * 0.9, 'σ₂ = 0', th.green);
        /* em painéis estreitos (celular) só o nome da matriz cabe no título */
        P.mathText(ctx, (pw < 230 ? '' : 'imagem do círculo por ') + pn.title, cx, 14, th.fg, 'center', 11);
        P.label(ctx, cx, h - 6, 'σ = (' + f2(pn.s.s1) + ', ' + f2(pn.s.s2) + ')', th.comment, 'center');
      });
    }

    /* Valores singulares ao longo das iterações de Newton-Schulz. */
    function drawNS(r) {
      const th = P.theme();
      const { ctx, w, h } = P.setup(cvNS);
      P.clear(ctx, w, h);
      const s1 = r.ns.map((X) => svd2(X).s1), s2 = r.ns.map((X) => svd2(X).s2);
      const ymax = Math.max(1.4, ...s1, ...s2) * 1.08;
      const fr = P.frame(ctx, w, h, 0, NS_ITERS, 0, ymax, { l: 40, r: 10, t: 10, b: 24 });
      /* faixa em que o polinômio do Muon assenta: σ entre ≈0.7 e ≈1.2 */
      ctx.fillStyle = th.green;
      ctx.globalAlpha = 0.12;
      ctx.fillRect(fr.X(0), fr.Y(1.2), fr.iw, fr.Y(0.7) - fr.Y(1.2));
      ctx.globalAlpha = 1;
      P.axes(ctx, fr, { xlabel: 'iteração', ylabel: 'σ', xticks: [0, 1, 2, 3, 4, 5], yticks: [0, 0.5, 1, Math.round(ymax * 10) / 10] });
      P.line(ctx, fr, [0, NS_ITERS], [1, 1], th.comment, { dash: [4, 4], width: 1 });
      const ks = s1.map((_, i) => i);
      P.line(ctx, fr, ks, s1, th.cyan, { width: 1.8 });
      P.scatter(ctx, fr, ks.map((k, i) => [k, s1[i]]), th.cyan, { r: 3, alpha: 1 });
      P.line(ctx, fr, ks, s2, th.green, { width: 1.8 });
      P.scatter(ctx, fr, ks.map((k, i) => [k, s2[i]]), th.green, { r: 3, alpha: 1 });
      P.label(ctx, fr.X(0) + 8, fr.Y(ymax) + 14, 'σ₁', th.cyan);
      P.label(ctx, fr.X(0) + 30, fr.Y(ymax) + 14, 'σ₂', th.green);
      P.label(ctx, fr.X(NS_ITERS) - 4, fr.Y(1.2) - 4, 'faixa de convergência', th.green, 'right');
    }

    function redraw() {
      const x = read();
      const r = compute(x);
      renderSteps(x, r);
      drawGeometry(r);
      drawNS(r);
    }

    for (const k of Object.keys(inputs)) inputs[k].addEventListener('input', redraw);
    $('s9-reset').addEventListener('click', () => {
      for (const k of Object.keys(inputs)) inputs[k].value = DEFAULTS[k];
      redraw();
    });
    $('s9-balanced').addEventListener('click', () => {
      /* momentum com σ₁ ≈ σ₂: Muon vira (quase) o momentum normalizado */
      for (const k of Object.keys(inputs)) inputs[k].value = DEFAULTS[k];
      inputs.G11.value = 3.9; inputs.G12.value = -2.2; inputs.G21.value = 1.3; inputs.G22.value = 1.6;
      redraw();
    });
    $('s9-reflect').addEventListener('click', () => {
      /* det(M_{t+1}) < 0: a ortogonal mais próxima é uma reflexão */
      for (const k of Object.keys(inputs)) inputs[k].value = DEFAULTS[k];
      inputs.G11.value = 1.9; inputs.G12.value = -2.2; inputs.G21.value = -5.7; inputs.G22.value = -3.4;
      redraw();
    });

    redraw();
    P.onRedraw(redraw);
    P.observeResize(cvGeo, redraw);
    P.observeResize(cvNS, redraw);
  }

  DL.sections.push({ name: 's9-muon', init });
})();
