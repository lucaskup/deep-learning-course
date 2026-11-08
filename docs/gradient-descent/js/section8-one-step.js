/* Seção 8: um passo de cada otimizador no cenário numérico dos slides
   (η = 0.1, θ = (3, 5), ∇ℒ = (1, 2), v = (2, 1), G = (24, 12), E[g²] = (1, 4)).
   Os campos são editáveis; a tabela e as setas são recalculadas na hora. */
(function () {
  'use strict';
  window.DL = window.DL || {};
  DL.sections = DL.sections || [];

  const EPS = 1e-8, BETA2 = 0.999;
  const FIELDS = ['eta', 'beta', 'rho', 't1', 't2', 'g1', 'g2', 'v1', 'v2', 'gl1', 'gl2', 'G1', 'G2', 'E1', 'E2'];
  const DEFAULTS = { eta: 0.1, beta: 0.9, rho: 0.9, t1: 3, t2: 5, g1: 1, g2: 2, v1: 2, v2: 1, gl1: 0.8, gl2: 1.6, G1: 24, G2: 12, E1: 1, E2: 4 };

  const ROWS = [
    { id: 'gd', label: 'Descida de gradiente', color: 'cyan' },
    { id: 'mom', label: 'Momentum', color: 'orange' },
    { id: 'nag', label: 'Nesterov', color: 'green' },
    { id: 'norm', label: 'Normalização', color: 'comment' },
    { id: 'adagrad', label: 'AdaGrad', color: 'purple' },
    { id: 'rmsprop', label: 'RMSProp', color: 'pink' },
    { id: 'adam', label: 'Adam (t = 1)', color: 'red' },
  ];

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
    const cv = $('s8-arrows');
    const inputs = {};
    for (const f of FIELDS) inputs[f] = $('s8-' + f);

    function read() {
      const x = {};
      for (const f of FIELDS) {
        const v = parseFloat(inputs[f].value);
        x[f] = isFinite(v) ? v : DEFAULTS[f];
      }
      return x;
    }

    /* Devolve, para cada otimizador, o termo que multiplica η e o novo θ. */
    function compute(x) {
      const th = [x.t1, x.t2], g = [x.g1, x.g2];
      const out = {};
      const stepTo = (d) => [th[0] - x.eta * d[0], th[1] - x.eta * d[1]];
      out.gd = { d: g, note: 'θ − η∇ℒ' };
      const vm = [x.beta * x.v1 + (1 - x.beta) * g[0], x.beta * x.v2 + (1 - x.beta) * g[1]];
      out.mom = { d: vm, note: 'v′ = βv + (1−β)∇ℒ = (' + f3(vm[0]) + ', ' + f3(vm[1]) + ')' };
      const look = [th[0] - x.eta * x.beta * x.v1, th[1] - x.eta * x.beta * x.v2];
      const vn = [x.beta * x.v1 + (1 - x.beta) * x.gl1, x.beta * x.v2 + (1 - x.beta) * x.gl2];
      out.nag = { d: vn, note: 'ponto adiantado (' + f3(look[0]) + ', ' + f3(look[1]) + '), v′ = (' + f3(vn[0]) + ', ' + f3(vn[1]) + ')' };
      const dn = [g[0] / (Math.abs(g[0]) + EPS), g[1] / (Math.abs(g[1]) + EPS)];
      out.norm = { d: dn, note: '∇ℒ / √(∇ℒ)² = (' + f3(dn[0]) + ', ' + f3(dn[1]) + ')' };
      const Gn = [x.G1 + g[0] * g[0], x.G2 + g[1] * g[1]];
      const da = [g[0] / (Math.sqrt(Gn[0]) + EPS), g[1] / (Math.sqrt(Gn[1]) + EPS)];
      out.adagrad = { d: da, note: 'G′ = (' + f3(Gn[0]) + ', ' + f3(Gn[1]) + '), ∇ℒ/√G′ = (' + f3(da[0]) + ', ' + f3(da[1]) + ')' };
      const En = [x.rho * x.E1 + (1 - x.rho) * g[0] * g[0], x.rho * x.E2 + (1 - x.rho) * g[1] * g[1]];
      const dr = [g[0] / (Math.sqrt(En[0]) + EPS), g[1] / (Math.sqrt(En[1]) + EPS)];
      out.rmsprop = { d: dr, note: 'E[g²]′ = (' + f3(En[0]) + ', ' + f3(En[1]) + '), ∇ℒ/√E′ = (' + f3(dr[0]) + ', ' + f3(dr[1]) + ')' };
      /* Adam no primeiro passo, m₀ = v₀ = 0: a correção de viés desfaz (1−β) e (1−β₂). */
      const m1 = [(1 - x.beta) * g[0], (1 - x.beta) * g[1]];
      const s1 = [(1 - BETA2) * g[0] * g[0], (1 - BETA2) * g[1] * g[1]];
      const mh = [m1[0] / (1 - x.beta), m1[1] / (1 - x.beta)];
      const sh = [s1[0] / (1 - BETA2), s1[1] / (1 - BETA2)];
      const dad = [mh[0] / (Math.sqrt(sh[0]) + EPS), mh[1] / (Math.sqrt(sh[1]) + EPS)];
      out.adam = { d: dad, note: 'm̃₁ = (' + f3(mh[0]) + ', ' + f3(mh[1]) + '), ṽ₁ = (' + f3(sh[0]) + ', ' + f3(sh[1]) + ')' };
      for (const k of Object.keys(out)) out[k].next = stepTo(out[k].d);
      return out;
    }

    function f3(v) { return (Math.round(v * 1000) / 1000).toString().replace('-', '−'); }

    function renderTable(x, res) {
      const rows = ROWS.map((r) => {
        const o = res[r.id];
        return '<tr><td style="color: var(--' + r.color + ')">■ ' + r.label + '</td>' +
          '<td class="formula">(' + f3(o.d[0]) + ', ' + f3(o.d[1]) + ')</td>' +
          '<td class="formula">(' + f3(o.next[0]) + ', ' + f3(o.next[1]) + ')</td>' +
          '<td>' + o.note + '</td></tr>';
      }).join('');
      $('s8-table').innerHTML =
        '<table class="s8-table"><thead><tr><th>otimizador</th><th>termo que multiplica η</th>' +
        '<th>θ<sub>t+1</sub></th><th>contas intermediárias</th></tr></thead><tbody>' + rows + '</tbody></table>';
    }

    function drawArrows(x, res) {
      const th = P.theme();
      const { ctx, w, h } = P.setup(cv);
      P.clear(ctx, w, h);
      /* Janela centrada em θ_t; meia-largura proporcional ao maior passo. */
      let R = 0.05;
      for (const r of ROWS) {
        const o = res[r.id];
        R = Math.max(R, Math.abs(o.next[0] - x.t1), Math.abs(o.next[1] - x.t2));
      }
      R *= 1.35;
      const fr = P.frame(ctx, w, h, x.t1 - R, x.t1 + R, x.t2 - R, x.t2 + R, { l: 46, r: 10, t: 10, b: 24 });
      const tk = (c0) => [c0 - R / 2, c0, c0 + R / 2].map((v) => Math.round(v * 1000) / 1000);
      P.axes(ctx, fr, { xlabel: 'θ₁', ylabel: 'θ₂', xticks: tk(x.t1), yticks: tk(x.t2) });
      /* gradiente (tracejado) a partir de θ_t, só a direção */
      const gn = Math.hypot(x.g1, x.g2) || 1;
      P.line(ctx, fr, [x.t1, x.t1 + 0.9 * R * x.g1 / gn], [x.t2, x.t2 + 0.9 * R * x.g2 / gn], th.comment, { dash: [4, 4], width: 1.2 });
      pill(ctx, fr.X(x.t1 + 0.9 * R * x.g1 / gn) + 4, fr.Y(x.t2 + 0.9 * R * x.g2 / gn), '∇ℒ', th.comment, 'left', fr);
      for (const r of ROWS) {
        const o = res[r.id];
        const ax = fr.X(x.t1), ay = fr.Y(x.t2), bx = fr.X(o.next[0]), by = fr.Y(o.next[1]);
        ctx.strokeStyle = th[r.color];
        ctx.fillStyle = th[r.color];
        ctx.lineWidth = 1.8;
        ctx.beginPath();
        ctx.moveTo(ax, ay);
        ctx.lineTo(bx, by);
        ctx.stroke();
        const ang = Math.atan2(by - ay, bx - ax), hl = 6;
        ctx.beginPath();
        ctx.moveTo(bx, by);
        ctx.lineTo(bx - hl * Math.cos(ang - 0.45), by - hl * Math.sin(ang - 0.45));
        ctx.lineTo(bx - hl * Math.cos(ang + 0.45), by - hl * Math.sin(ang + 0.45));
        ctx.closePath();
        ctx.fill();
      }
      /* θ_t */
      ctx.beginPath();
      ctx.arc(fr.X(x.t1), fr.Y(x.t2), 4, 0, 2 * Math.PI);
      ctx.fillStyle = th.fg;
      ctx.fill();
      pill(ctx, fr.X(x.t1) + 7, fr.Y(x.t2) - 6, 'θ_t', th.fg);
      /* legenda (as setas podem passar por baixo dela) */
      ROWS.forEach((r, i) => pill(ctx, fr.X(fr.xmin) + 8, fr.Y(fr.ymax) + 14 + 13 * i, r.label, th[r.color]));
    }

    function redraw() {
      const x = read();
      const res = compute(x);
      renderTable(x, res);
      drawArrows(x, res);
    }

    for (const f of FIELDS) inputs[f].addEventListener('input', redraw);
    $('s8-reset').addEventListener('click', () => {
      for (const f of FIELDS) inputs[f].value = DEFAULTS[f];
      redraw();
    });

    redraw();
    P.onRedraw(redraw);
    P.observeResize(cv, redraw);
  }

  DL.sections.push({ name: 's8-one-step', init });
})();
