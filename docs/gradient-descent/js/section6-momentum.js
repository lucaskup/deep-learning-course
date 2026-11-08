/* Seção 6: vale alongado ℒ(θ₁, θ₂) = ½(θ₁² + c·θ₂²) e os otimizadores dos
   slides lado a lado: GD, momentum, Nesterov, gradiente normalizado, AdaGrad,
   RMSProp e Adam, todos partindo do mesmo θ₀ com o mesmo η. */
(function () {
  'use strict';
  window.DL = window.DL || {};
  DL.sections = DL.sections || [];

  const DOM = 2.2, MAX_ITERS = 200, EPS = 1e-8, RHO = 0.9, BETA2 = 0.999;

  /* Um otimizador por entrada: id do checkbox, rótulo e cor do tema. */
  const OPTS = [
    { id: 'gd', label: 'GD', color: 'cyan' },
    { id: 'mom', label: 'momentum', color: 'orange' },
    { id: 'nag', label: 'Nesterov', color: 'green' },
    { id: 'norm', label: 'normalizado', color: 'comment' },
    { id: 'adagrad', label: 'AdaGrad', color: 'purple' },
    { id: 'rmsprop', label: 'RMSProp', color: 'pink' },
    { id: 'adam', label: 'Adam', color: 'red' },
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

    const cvSurf = $('s6-surface'), cvLoss = $('s6-loss');
    const slC = $('s6-c'), slEta = $('s6-eta'), slBeta = $('s6-beta');
    const btnStep = $('s6-step'), btnRun = $('s6-run'), btnReset = $('s6-reset');
    const checks = {};
    for (const o of OPTS) checks[o.id] = $('s6-opt-' + o.id);

    let c = +slC.value, eta = +slEta.value, beta = +slBeta.value;
    let start = [-1.9, 1.5];
    let runs = {};
    let running = false, acc = 0;
    let frSurf = null, dragging = false;

    const L = (p) => 0.5 * (p[0] * p[0] + c * p[1] * p[1]);
    const grad = (p) => [p[0], c * p[1]];
    const enabled = () => OPTS.filter((o) => checks[o.id].checked);

    function resetTraj() {
      runs = {};
      for (const o of OPTS) {
        runs[o.id] = { traj: [start.slice()], loss: [L(start)], v: [0, 0], G: [0, 0], E: [0, 0], m: [0, 0], s: [0, 0], t: 0 };
      }
    }

    /* Regras dos slides, uma por otimizador (convenção v ← βv + (1−β)∇ℒ). */
    function update(id, r) {
      const p = r.traj[r.traj.length - 1];
      let g = grad(p), pn;
      switch (id) {
        case 'gd':
          pn = [p[0] - eta * g[0], p[1] - eta * g[1]];
          break;
        case 'mom':
          r.v = [beta * r.v[0] + (1 - beta) * g[0], beta * r.v[1] + (1 - beta) * g[1]];
          pn = [p[0] - eta * r.v[0], p[1] - eta * r.v[1]];
          break;
        case 'nag': {
          /* gradiente avaliado no ponto adiantado θ − ηβv */
          g = grad([p[0] - eta * beta * r.v[0], p[1] - eta * beta * r.v[1]]);
          r.v = [beta * r.v[0] + (1 - beta) * g[0], beta * r.v[1] + (1 - beta) * g[1]];
          pn = [p[0] - eta * r.v[0], p[1] - eta * r.v[1]];
          break;
        }
        case 'norm':
          pn = [p[0] - eta * g[0] / (Math.abs(g[0]) + EPS), p[1] - eta * g[1] / (Math.abs(g[1]) + EPS)];
          break;
        case 'adagrad':
          r.G = [r.G[0] + g[0] * g[0], r.G[1] + g[1] * g[1]];
          pn = [p[0] - eta * g[0] / (Math.sqrt(r.G[0]) + EPS), p[1] - eta * g[1] / (Math.sqrt(r.G[1]) + EPS)];
          break;
        case 'rmsprop':
          r.E = [RHO * r.E[0] + (1 - RHO) * g[0] * g[0], RHO * r.E[1] + (1 - RHO) * g[1] * g[1]];
          pn = [p[0] - eta * g[0] / (Math.sqrt(r.E[0]) + EPS), p[1] - eta * g[1] / (Math.sqrt(r.E[1]) + EPS)];
          break;
        case 'adam': {
          r.t++;
          r.m = [beta * r.m[0] + (1 - beta) * g[0], beta * r.m[1] + (1 - beta) * g[1]];
          r.s = [BETA2 * r.s[0] + (1 - BETA2) * g[0] * g[0], BETA2 * r.s[1] + (1 - BETA2) * g[1] * g[1]];
          const cm = 1 - Math.pow(beta, r.t), cs = 1 - Math.pow(BETA2, r.t);
          pn = [
            p[0] - eta * (r.m[0] / cm) / (Math.sqrt(r.s[0] / cs) + EPS),
            p[1] - eta * (r.m[1] / cm) / (Math.sqrt(r.s[1] / cs) + EPS),
          ];
          break;
        }
      }
      r.traj.push(pn);
      r.loss.push(L(pn));
    }

    function stepOnce() {
      if (done()) return;
      for (const o of enabled()) update(o.id, runs[o.id]);
    }

    function kCount() { return runs.gd.traj.length - 1; }

    function done() {
      if (kCount() >= MAX_ITERS) return true;
      const settled = (v) => !isFinite(v) || v < 1e-11 || v > 1e8;
      const en = enabled();
      if (en.length === 0) return true;
      return en.every((o) => settled(runs[o.id].loss[runs[o.id].loss.length - 1]));
    }

    function clipPlot(ctx, fr) {
      ctx.save();
      ctx.beginPath();
      ctx.rect(fr.X(fr.xmin), fr.Y(fr.ymax), fr.iw, fr.ih);
      ctx.clip();
    }

    /* Pontos divergidos são truncados para fora da janela em vez de virar NaN no path */
    const clamp = (v) => (isFinite(v) ? Math.max(-50, Math.min(50, v)) : 50);

    function drawTraj(ctx, fr, traj, color) {
      const xs = traj.map((p) => clamp(p[0]));
      const ys = traj.map((p) => clamp(p[1]));
      P.line(ctx, fr, xs, ys, color, { width: 1.6, alpha: 0.9 });
      P.scatter(ctx, fr, xs.map((x, i) => [x, ys[i]]), color, { r: 2, alpha: 0.9 });
    }

    function drawSurface() {
      const th = P.theme();
      const { ctx, w, h } = P.setup(cvSurf);
      P.clear(ctx, w, h);
      const fr = P.frame(ctx, w, h, -DOM, DOM, -DOM, DOM);
      frSurf = fr;
      const fmax = 0.5 * DOM * DOM * (1 + c);
      P.heatmap(ctx, fr, (x, y) => 0.5 * (x * x + c * y * y), 150, 150, (v) => {
        const u = Math.pow(Math.min(1, v / fmax), 0.3);
        const col = P.viridis(u);
        const lift = (u * 9) % 1 < 0.13 ? 42 : 0;
        return [Math.min(255, col[0] + lift), Math.min(255, col[1] + lift), Math.min(255, col[2] + lift), 235];
      });
      xlabelPill(ctx, fr, 'θ₁');
      P.axes(ctx, fr, { xlabel: 'θ₁', ylabel: 'θ₂', xticks: [-2, 0, 2], yticks: [-2, 0, 2] });

      const en = enabled();
      clipPlot(ctx, fr);
      for (const o of en) drawTraj(ctx, fr, runs[o.id].traj, th[o.color]);
      ctx.restore();

      /* Mínimo global em (0, 0) */
      P.scatter(ctx, fr, [[0, 0]], th.green, { r: 4, alpha: 1 });
      pill(ctx, fr.X(0) + 8, fr.Y(0) + 4, 'mínimo', th.green);
      /* Marcador do ponto inicial (arrastável) */
      ctx.beginPath();
      ctx.arc(fr.X(start[0]), fr.Y(start[1]), 6, 0, 2 * Math.PI);
      ctx.strokeStyle = th.fg;
      ctx.lineWidth = 2;
      ctx.stroke();
      /* perto da borda direita o rótulo passa para a esquerda do marcador */
      const sx = fr.X(start[0]);
      if (sx + 40 > fr.X(fr.xmax)) pill(ctx, sx - 9, fr.Y(start[1]) - 8, 'θ_0', th.fg, 'right', fr);
      else pill(ctx, sx + 9, fr.Y(start[1]) - 8, 'θ_0', th.fg, 'left', fr);

      /* Legenda sobre fundo opaco: todos os cantos da superfície são amarelos. */
      if (en.length) {
        ctx.font = '11px sans-serif';
        let lw = 0;
        for (const o of en) lw = Math.max(lw, ctx.measureText(o.label).width);
        ctx.fillStyle = th.card;
        ctx.globalAlpha = 0.85;
        ctx.fillRect(fr.X(-DOM) + 4, fr.Y(DOM) + 3, lw + 8, 14 * en.length + 4);
        ctx.globalAlpha = 1;
        en.forEach((o, i) => P.label(ctx, fr.X(-DOM) + 8, fr.Y(DOM) + 14 + 14 * i, o.label, th[o.color]));
      }
    }

    function drawLoss() {
      const th = P.theme();
      const { ctx, w, h } = P.setup(cvLoss);
      P.clear(ctx, w, h);
      const n = Math.max(20, kCount());
      const logv = (v) => (isFinite(v) && v > 0 ? Math.min(9, Math.log10(Math.max(v, 1e-12))) : (v === 0 ? -12 : 9));
      const en = enabled();
      const series = en.map((o) => runs[o.id].loss.map(logv));
      let top = 2;
      for (const s of series) for (const v of s) top = Math.max(top, v);
      const ymax = Math.min(10, Math.ceil(top) + 1);
      const fr = P.frame(ctx, w, h, 0, n, -12, ymax);
      P.axes(ctx, fr, {
        xlabel: 'k', ylabel: 'log₁₀ ℒ',
        xticks: [0, Math.round(n / 2), n],
        yticks: [-12, -6, 0, 6],
      });
      clipPlot(ctx, fr);
      series.forEach((s, i) => {
        P.line(ctx, fr, s.map((_, k) => k), s, th[en[i].color], { width: 1.6 });
      });
      ctx.restore();
    }

    function updateReadout() {
      const num = (v) => (isFinite(v) ? v.toExponential(1) : '∞');
      const parts = enabled().map((o) => o.label + ' = ' + num(runs[o.id].loss[runs[o.id].loss.length - 1]));
      $('s6-readout').textContent = 'k = ' + kCount() + (parts.length ? ' · ℒ: ' + parts.join(' · ') : '');
    }

    function redraw() {
      drawSurface();
      drawLoss();
      updateReadout();
    }

    function stopRun() {
      running = false;
      btnRun.textContent = '▶ Rodar';
      DL.stopTicker(tick);
    }

    function tick(dt) {
      acc += dt * 12;
      while (acc >= 1) { acc -= 1; stepOnce(); }
      redraw();
      if (done()) stopRun();
    }

    function setStart(x, y) {
      const lim = DOM - 0.05;
      start = [Math.max(-lim, Math.min(lim, x)), Math.max(-lim, Math.min(lim, y))];
      resetTraj();
      redraw();
    }

    cvSurf.addEventListener('pointerdown', (e) => {
      if (!frSurf) return;
      dragging = true;
      cvSurf.setPointerCapture(e.pointerId);
      stopRun();
      const r = cvSurf.getBoundingClientRect();
      setStart(frSurf.invX(e.clientX - r.left), frSurf.invY(e.clientY - r.top));
    });
    cvSurf.addEventListener('pointermove', (e) => {
      if (!dragging || !frSurf) return;
      const r = cvSurf.getBoundingClientRect();
      setStart(frSurf.invX(e.clientX - r.left), frSurf.invY(e.clientY - r.top));
    });
    cvSurf.addEventListener('pointerup', () => { dragging = false; });
    cvSurf.addEventListener('pointercancel', () => { dragging = false; });

    btnStep.addEventListener('click', () => { stepOnce(); redraw(); });
    btnRun.addEventListener('click', () => {
      if (running) { stopRun(); return; }
      if (done()) resetTraj();
      running = true;
      acc = 0;
      btnRun.textContent = '⏸ Pausar';
      DL.startTicker(tick);
    });
    btnReset.addEventListener('click', () => { stopRun(); resetTraj(); redraw(); });

    slC.addEventListener('input', () => {
      c = +slC.value;
      $('s6-c-val').textContent = c;
      resetTraj();
      redraw();
    });
    /* η e β valem a partir do próximo passo (a trajetória continua do ponto
       atual, como nas outras seções); só c redefine a paisagem e reinicia. */
    slEta.addEventListener('input', () => {
      eta = +slEta.value;
      $('s6-eta-val').textContent = eta.toFixed(3);
      redraw();
    });
    slBeta.addEventListener('input', () => {
      beta = +slBeta.value;
      $('s6-beta-val').textContent = beta.toFixed(2);
      redraw();
    });
    /* Ligar um otimizador no meio da corrida o faria partir atrasado: reinicia. */
    for (const o of OPTS) checks[o.id].addEventListener('change', () => { stopRun(); resetTraj(); redraw(); });

    resetTraj();
    redraw();
    P.onRedraw(redraw);
    P.observeResize(cvSurf, redraw);
    P.observeResize(cvLoss, redraw);
  }

  DL.sections.push({ name: 's6-momentum', init });
})();
