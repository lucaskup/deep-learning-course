/* Seção 1: galeria das ativações ponto a ponto dos slides, com derivada,
   tangente arrastável e zonas de saturação. Também exporta DL.acts, o
   registro de ativações reutilizado pela seção 2. */
(function () {
  'use strict';
  window.DL = window.DL || {};
  DL.sections = DL.sections || [];

  const ZMIN = -6, ZMAX = 6, NPTS = 241;
  const SAT_EPS = 0.05;

  /* Constantes da SELU (slides): α ≈ 1,6733 e λ ≈ 1,0507. */
  const SELU_A = 1.6733, SELU_L = 1.0507;

  function sigm(z) { return 1 / (1 + Math.exp(-z)); }

  /* Φ(z): CDF da normal padrão via aproximação de erf (Abramowitz & Stegun 7.1.26). */
  function normCdf(z) {
    const sign = z < 0 ? -1 : 1;
    const x = Math.abs(z) / Math.SQRT2;
    const t = 1 / (1 + 0.3275911 * x);
    const poly = t * (0.254829592 + t * (-0.284496736 + t * (1.421413741 +
      t * (-1.453152027 + t * 1.061405429))));
    const erf = 1 - poly * Math.exp(-x * x);
    return 0.5 * (1 + sign * erf);
  }
  function normPdf(z) { return Math.exp(-z * z / 2) / Math.sqrt(2 * Math.PI); }

  /* Registro das ativações apresentadas nos slides (ponto a ponto).
     Cada entrada: f(z, p), df(z, p), param opcional (nome, faixa, valor atual)
     e a fórmula em HTML na notação dos slides. */
  DL.acts = [
    {
      id: 'sigmoid', nome: 'Sigmoid', param: null,
      f: (z) => sigm(z),
      df: (z) => { const s = sigm(z); return s * (1 - s); },
      formula: () => 'σ(z) = 1/(1+e<sup>−z</sup>) &nbsp;·&nbsp; σ′(z) = σ(z)(1−σ(z)) &nbsp;·&nbsp; imagem em (0, 1), derivada máxima 0.25 em z = 0',
    },
    {
      id: 'tanh', nome: 'Tanh', param: null,
      f: (z) => Math.tanh(z),
      df: (z) => { const t = Math.tanh(z); return 1 - t * t; },
      formula: () => 'tanh(z) = (e<sup>z</sup>−e<sup>−z</sup>)/(e<sup>z</sup>+e<sup>−z</sup>) &nbsp;·&nbsp; tanh′(z) = 1−tanh²(z) &nbsp;·&nbsp; imagem em (−1, 1), centrada em zero',
    },
    {
      id: 'relu', nome: 'ReLU', param: null,
      f: (z) => Math.max(0, z),
      df: (z) => (z > 0 ? 1 : 0),
      formula: () => 'ReLU(z) = max(0, z) &nbsp;·&nbsp; ReLU′(z) = 0 se z &lt; 0, 1 se z &gt; 0',
    },
    {
      id: 'lrelu', nome: 'Leaky ReLU', param: { nome: 'α', min: 0.01, max: 0.5, step: 0.01, cur: 0.1 },
      f: (z, a) => Math.max(a * z, z),
      df: (z, a) => (z > 0 ? 1 : a),
      formula: (a) => 'φ(z) = max(αz, z), α = ' + a.toFixed(2) +
        ' &nbsp;·&nbsp; φ′(z) = α se z &lt; 0, 1 se z &gt; 0 &nbsp;·&nbsp; PReLU: mesma forma com α treinável',
    },
    {
      id: 'elu', nome: 'ELU', param: { nome: 'α', min: 0.1, max: 3, step: 0.1, cur: 1 },
      f: (z, a) => (z >= 0 ? z : a * (Math.exp(z) - 1)),
      df: (z, a) => (z > 0 ? 1 : a * Math.exp(z)),
      formula: (a) => 'φ(z) = z se z ≥ 0, α(e<sup>z</sup>−1) se z &lt; 0, α = ' + a.toFixed(1) +
        ' &nbsp;·&nbsp; φ′(z) = 1 se z &gt; 0, αe<sup>z</sup> se z &lt; 0',
    },
    {
      id: 'selu', nome: 'SELU', param: null,
      f: (z) => SELU_L * (z >= 0 ? z : SELU_A * (Math.exp(z) - 1)),
      df: (z) => (z > 0 ? SELU_L : SELU_L * SELU_A * Math.exp(z)),
      formula: () => 'φ(z) = λ·[z se z ≥ 0, α(e<sup>z</sup>−1) se z &lt; 0], com α ≈ 1.6733 e λ ≈ 1.0507 (auto-normalização)',
    },
    {
      id: 'swish', nome: 'Swish', param: { nome: 'β', min: 0.1, max: 10, step: 0.1, cur: 1 },
      f: (z, b) => z * sigm(b * z),
      df: (z, b) => { const s = sigm(b * z); return s + b * z * s * (1 - s); },
      formula: (b) => 'φ(z) = z·σ(βz), β = ' + b.toFixed(1) +
        ' &nbsp;·&nbsp; φ′(z) = σ(βz) + βz·σ(βz)(1−σ(βz)) &nbsp;·&nbsp; β→0: ≈ z/2; β→∞: → ReLU(z)',
    },
    {
      id: 'gelu', nome: 'GELU', param: null,
      f: (z) => z * normCdf(z),
      df: (z) => normCdf(z) + z * normPdf(z),
      formula: () => 'φ(z) = z·Φ(z), Φ(z) = Pr(Z ≤ z), Z ∼ 𝒩(0, 1) &nbsp;·&nbsp; aproximação: φ(z) ≈ z·σ(1.702z) &nbsp;·&nbsp; φ′(z) = Φ(z) + z·𝒩(z; 0, 1)',
    },
    {
      id: 'linear', nome: 'Identity (Linear)', param: null,
      f: (z) => z,
      df: () => 1,
      formula: () => 'φ(z) = z &nbsp;·&nbsp; φ′(z) = 1 &nbsp;·&nbsp; sem não-linearidade: faz sentido apenas na camada de saída (regressão)',
    },
  ];
  DL.actById = function (id) { return DL.acts.find((a) => a.id === id); };

  function init() {
    const P = DL.plot;
    const $ = (id) => document.getElementById(id);

    const cvCurve = $('s1-curve'), cvDeriv = $('s1-deriv');
    const sel = $('s1-act');
    const paramWrap = $('s1-param-wrap'), paramSlider = $('s1-param');

    for (const a of DL.acts) {
      const o = document.createElement('option');
      o.value = a.id; o.textContent = a.nome;
      sel.appendChild(o);
    }
    sel.value = 'sigmoid';

    let x0 = 1.5;
    let frCurve = null, frDeriv = null;

    function act() { return DL.actById(sel.value); }
    function paramOf(a) { return a.param ? a.param.cur : 0; }

    /* Intervalos contíguos de saturação (|φ′| < SAT_EPS) sobre a grade. */
    function satZones(a, p) {
      const zones = [];
      let start = null;
      for (let i = 0; i < NPTS; i++) {
        const z = ZMIN + (ZMAX - ZMIN) * i / (NPTS - 1);
        const sat = Math.abs(a.df(z, p)) < SAT_EPS;
        if (sat && start === null) start = z;
        if (!sat && start !== null) { zones.push([start, z]); start = null; }
      }
      if (start !== null) zones.push([start, ZMAX]);
      return zones;
    }

    function niceRange(vals) {
      let lo = Math.min(0, ...vals), hi = Math.max(0, ...vals);
      const pad = 0.08 * (hi - lo || 1);
      return [lo - pad, hi + pad];
    }

    function shade(ctx, fr, zones, color) {
      ctx.fillStyle = color;
      ctx.globalAlpha = 0.10;
      for (const [a, b] of zones) {
        ctx.fillRect(fr.X(a), fr.Y(fr.ymax), fr.X(b) - fr.X(a), fr.ih);
      }
      ctx.globalAlpha = 1;
    }

    function redraw() {
      const th = P.theme();
      const a = act(), p = paramOf(a);
      const zs = [], ys = [], dys = [];
      for (let i = 0; i < NPTS; i++) {
        const z = ZMIN + (ZMAX - ZMIN) * i / (NPTS - 1);
        zs.push(z); ys.push(a.f(z, p)); dys.push(a.df(z, p));
      }
      const zones = satZones(a, p);
      const fx = a.f(x0, p), dfx = a.df(x0, p);

      /* ── φ(z) com tangente ── */
      let { ctx, w, h } = P.setup(cvCurve);
      P.clear(ctx, w, h);
      const [ylo, yhi] = niceRange(ys);
      frCurve = P.frame(ctx, w, h, ZMIN, ZMAX, ylo, yhi);
      shade(ctx, frCurve, zones, th.orange);
      P.axes(ctx, frCurve, { xlabel: 'z', xticks: [-6, -4, -2, 0, 2, 4, 6], yticks: [Math.ceil(ylo * 10) / 10, 0, Math.floor(yhi * 10) / 10] });
      P.line(ctx, frCurve, [ZMIN, ZMAX], [0, 0], th.line, { width: 1, dash: [3, 3] });
      P.line(ctx, frCurve, zs, ys, th.purple, { width: 2.2 });
      /* tangente em x0, recortada à área do gráfico */
      ctx.save();
      ctx.beginPath();
      ctx.rect(frCurve.X(ZMIN), frCurve.Y(yhi), frCurve.iw, frCurve.ih);
      ctx.clip();
      const tx = [x0 - 2.2, x0 + 2.2];
      P.line(ctx, frCurve, tx, tx.map((z) => fx + dfx * (z - x0)), th.green, { width: 1.6, dash: [5, 4] });
      ctx.restore();
      P.scatter(ctx, frCurve, [[x0, fx]], th.pink, { r: 5, alpha: 1 });
      P.label(ctx, frCurve.X(x0) + 8, frCurve.Y(fx) - 8, 'x_0', th.pink);

      /* ── φ′(z) ── */
      ({ ctx, w, h } = P.setup(cvDeriv));
      P.clear(ctx, w, h);
      const [dlo, dhi] = niceRange(dys);
      frDeriv = P.frame(ctx, w, h, ZMIN, ZMAX, dlo, dhi);
      shade(ctx, frDeriv, zones, th.orange);
      P.axes(ctx, frDeriv, { xlabel: 'z', xticks: [-6, -3, 0, 3, 6], yticks: [0, Math.floor(dhi * 10) / 10] });
      P.line(ctx, frDeriv, zs, dys, th.cyan, { width: 2 });
      P.line(ctx, frDeriv, [x0, x0], [dlo, dhi], th.comment, { width: 1, dash: [3, 3] });
      P.scatter(ctx, frDeriv, [[x0, dfx]], th.pink, { r: 4, alpha: 1 });

      $('s1-formula').innerHTML = a.formula(p);
      $('s1-readout').innerHTML =
        'x<sub>0</sub> = ' + x0.toFixed(2) +
        ' · φ(x<sub>0</sub>) = ' + fx.toFixed(3) +
        ' · φ′(x<sub>0</sub>) = ' + dfx.toFixed(3);
    }

    function syncParamUI() {
      const a = act();
      paramWrap.classList.toggle('hidden', !a.param);
      if (a.param) {
        $('s1-param-name').textContent = a.param.nome;
        paramSlider.min = a.param.min;
        paramSlider.max = a.param.max;
        paramSlider.step = a.param.step;
        paramSlider.value = a.param.cur;
        $('s1-param-val').textContent = a.param.cur.toFixed(2);
      }
    }

    sel.addEventListener('change', () => { syncParamUI(); redraw(); });
    paramSlider.addEventListener('input', () => {
      const a = act();
      if (a.param) {
        a.param.cur = +paramSlider.value;
        $('s1-param-val').textContent = a.param.cur.toFixed(2);
      }
      redraw();
    });

    /* Arrasto do marcador x0 (em qualquer um dos dois canvases). */
    function attachDrag(canvas, getFrame) {
      let dragging = false;
      const move = (e) => {
        const fr = getFrame();
        if (!fr) return;
        const r = canvas.getBoundingClientRect();
        x0 = Math.max(ZMIN, Math.min(ZMAX, fr.invX(e.clientX - r.left)));
        redraw();
      };
      canvas.addEventListener('pointerdown', (e) => {
        dragging = true;
        canvas.setPointerCapture(e.pointerId);
        move(e);
      });
      canvas.addEventListener('pointermove', (e) => { if (dragging) move(e); });
      canvas.addEventListener('pointerup', () => { dragging = false; });
      canvas.addEventListener('pointercancel', () => { dragging = false; });
    }
    attachDrag(cvCurve, () => frCurve);
    attachDrag(cvDeriv, () => frDeriv);

    syncParamUI();
    redraw();
    P.onRedraw(redraw);
    P.observeResize(cvCurve, redraw);
    P.observeResize(cvDeriv, redraw);
  }

  DL.sections.push({ name: 's1-gallery', init });
})();
