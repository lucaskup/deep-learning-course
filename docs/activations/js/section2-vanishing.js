/* Seção 2: dissipação (e explosão) de gradiente em uma cadeia de L camadas
   escalares idênticas a_l = φ(w·a_{l−1}). O gradiente acumulado
   ∂a_l/∂a_0 = ∏ φ′(z_k)·w é mostrado em escala log por camada. */
(function () {
  'use strict';
  window.DL = window.DL || {};
  DL.sections = DL.sections || [];

  const LOG_FLOOR = -30; /* piso do gráfico log para gradiente nulo */

  function init() {
    const U = DL.utils, P = DL.plot;
    const $ = (id) => document.getElementById(id);

    const cvGrad = $('s2-grad'), cvActs = $('s2-acts');
    const sel = $('s2-act');
    const sliderL = $('s2-L'), sliderW = $('s2-w');
    const btnResample = $('s2-resample');

    for (const a of DL.acts) {
      const o = document.createElement('option');
      o.value = a.id; o.textContent = a.nome;
      sel.appendChild(o);
    }
    sel.value = 'sigmoid';

    let L = 12, w = 1, seed = 7, z0 = 0;

    /* Entrada fixa com semente (determinística, reamostrável). */
    function resample() {
      const rng = U.mulberry32(seed);
      /* Rejeita amostras fora de [−2.5, 2.5] em vez de saturar no limite
         (um clamp deixaria a₀ = 3.00 exato com frequência visível). */
      do { z0 = 1.5 * U.randn(rng); } while (Math.abs(z0) > 2.5);
    }

    /* Propaga a cadeia: devolve ativações a_0..a_L e gradiente acumulado g_1..g_L. */
    function forward() {
      const act = DL.actById(sel.value);
      const p = act.param ? act.param.cur : 0;
      const as = [z0], gs = [];
      let a = z0, g = 1;
      for (let l = 1; l <= L; l++) {
        const pre = w * a;
        a = act.f(pre, p);
        g *= act.df(pre, p) * w;
        as.push(a);
        gs.push(g);
      }
      return { as, gs };
    }

    function layerTicks() {
      if (L <= 6) {
        const t = [];
        for (let l = 0; l <= L; l++) t.push(l);
        return t;
      }
      return [0, Math.round(L / 2), L];
    }

    /* 1.23 × 10⁻⁸ em HTML a partir de um número. */
    function sci(v) {
      if (v === 0) return '0';
      const [m, e] = v.toExponential(2).split('e');
      return m + ' × 10<sup>' + (+e) + '</sup>';
    }

    function redraw() {
      const th = P.theme();
      const { as, gs } = forward();

      /* ── |∂a_l/∂a_0| por camada, em log10 ── */
      const xs = [], lg = [];
      for (let l = 1; l <= L; l++) {
        xs.push(l);
        const m = Math.abs(gs[l - 1]);
        lg.push(m === 0 ? LOG_FLOOR : Math.max(LOG_FLOOR, Math.log10(m)));
      }
      /* referência: teto da sigmoide, (0.25·w)^l */
      const refSlope = Math.log10(0.25 * w);
      const ref = xs.map((l) => l * refSlope);

      let ymin = Math.min(0, ...lg, ref[ref.length - 1]) - 1;
      let ymax = Math.max(0, ...lg) + 1;
      ymin = Math.max(ymin, LOG_FLOOR - 2);

      let { ctx, w: cw, h: ch } = P.setup(cvGrad);
      P.clear(ctx, cw, ch);
      let fr = P.frame(ctx, cw, ch, 0, L, ymin, ymax);
      P.axes(ctx, fr, {
        xlabel: 'camada l', ylabel: 'log_{10} |∂a_l/∂a_0|',
        xticks: layerTicks(),
        yticks: [Math.ceil(ymin), 0, Math.floor(ymax)],
      });
      P.line(ctx, fr, [0, L], [0, 0], th.line, { width: 1, dash: [3, 3] });
      P.line(ctx, fr, xs, ref, th.comment, { width: 1.2, dash: [5, 4] });
      P.line(ctx, fr, xs, lg, th.purple, { width: 2 });
      P.scatter(ctx, fr, xs.map((l, i) => [l, lg[i]]), th.purple, { r: 2.5, alpha: 1 });
      /* Rótulo da referência a 60% da linha tracejada, do lado oposto ao da
         curva roxa (acima se a curva está abaixo da referência, e vice-versa),
         sobre fundo opaco e desenhado por cima. */
      {
        const mi = Math.max(0, Math.round(0.6 * L) - 1);
        const above = lg[mi] <= ref[mi];
        const lx = fr.X(xs[mi]), ly = fr.Y(ref[mi]) + (above ? -8 : 18);
        ctx.font = '11px sans-serif';
        const tw = ctx.measureText('(0.25·w)^l').width;
        ctx.fillStyle = th.card;
        ctx.globalAlpha = 0.85;
        ctx.fillRect(lx - tw / 2 - 3, ly - 10, tw + 6, 14);
        ctx.globalAlpha = 1;
        P.label(ctx, lx, ly, '(0.25·w)^l', th.comment, 'center');
      }

      const gFinal = Math.abs(gs[gs.length - 1]);
      if (gFinal === 0) {
        P.label(ctx, fr.X(L / 2), fr.Y((ymin + ymax) / 2), 'gradiente = 0 (unidade morta)', th.red, 'center');
      }

      /* ── ativação a_l por camada ── */
      const axs = [];
      for (let l = 0; l <= L; l++) axs.push(l);
      let alo = Math.min(0, ...as), ahi = Math.max(0, ...as);
      const pad = 0.1 * (ahi - alo || 1);
      alo -= pad; ahi += pad;

      ({ ctx, w: cw, h: ch } = P.setup(cvActs));
      P.clear(ctx, cw, ch);
      fr = P.frame(ctx, cw, ch, 0, L, alo, ahi);
      P.axes(ctx, fr, {
        xlabel: 'camada l', ylabel: 'a_l',
        xticks: layerTicks(),
        yticks: [Math.ceil(alo * 10) / 10, 0, Math.floor(ahi * 10) / 10],
      });
      P.line(ctx, fr, [0, L], [0, 0], th.line, { width: 1, dash: [3, 3] });
      P.line(ctx, fr, axs, as, th.cyan, { width: 2 });
      P.scatter(ctx, fr, axs.map((l) => [l, as[l]]), th.cyan, { r: 2.5, alpha: 1 });

      $('s2-readout').innerHTML =
        'a<sub>0</sub> = ' + z0.toFixed(2).replace('-', '−') +
        ' · |∂a<sub>L</sub>/∂a<sub>0</sub>| = ' + sci(gFinal) +
        (gFinal === 0 ? ' (unidade morta)' : '');
    }

    sel.addEventListener('change', redraw);
    sliderL.addEventListener('input', () => {
      L = +sliderL.value;
      $('s2-L-val').textContent = L;
      redraw();
    });
    sliderW.addEventListener('input', () => {
      w = +sliderW.value;
      $('s2-w-val').textContent = w.toFixed(2);
      redraw();
    });
    btnResample.addEventListener('click', () => {
      seed = (seed * 1664525 + 1013904223) >>> 0;
      resample();
      redraw();
    });

    resample();
    redraw();
    P.onRedraw(redraw);
    P.observeResize(cvGrad, redraw);
    P.observeResize(cvActs, redraw);
  }

  DL.sections.push({ name: 's2-vanishing', init });
})();
