/* Seção 4: segmentação semântica vs instâncias. Compara a ideia ingênua
   de janela deslizante (um forward por pixel) com a predição densa do
   encoder-decoder (um forward para a máscara inteira). */
(function () {
  'use strict';
  window.DL = window.DL || {};
  DL.sections = DL.sections || [];

  /* Cinco "pessoas" como elipses, como na figura dos slides. */
  const PEOPLE = [0, 1, 2, 3, 4].map(function (i) {
    return { cx: 0.14 + 0.18 * i, cy: 0.55, rx: 0.062, ry: 0.27 };
  });
  const PATCH = 0.18; /* lado do patch da janela deslizante, em fração da imagem */
  const fmtInt = (n) => n.toLocaleString('pt-BR');

  /* Classificador "perfeito": devolve 0 (fundo) ou 1..5 (instância). */
  function classify(u, v) {
    for (let i = 0; i < PEOPLE.length; i++) {
      const p = PEOPLE[i];
      const du = (u - p.cx) / p.rx, dv = (v - p.cy) / p.ry;
      if (du * du + dv * dv <= 1) return i + 1;
    }
    return 0;
  }

  function init() {
    const P = DL.plot;
    const $ = (id) => document.getElementById(id);

    const cvImg = $('s4-img'), cvMask = $('s4-mask');
    const slGrid = $('s4-grid');
    const tabSem = $('s4-tab-sem'), tabInst = $('s4-tab-inst');
    const btnRun = $('s4-run'), btnDense = $('s4-dense'), btnReset = $('s4-reset');

    let g = +slGrid.value;
    let labels = new Int8Array(g * g).fill(-1);
    let cursor = 0;          /* próxima célula da janela deslizante */
    let forwards = 0;
    let denseUsed = false;
    let mode = 'sem';        /* 'sem' ou 'inst' */
    let running = false, acc = 0;

    function instColor(th, id) {
      return [th.red, th.orange, th.green, th.cyan, th.purple][(id - 1) % 5];
    }

    function clearMask() {
      labels = new Int8Array(g * g).fill(-1);
      cursor = 0;
      forwards = 0;
      denseUsed = false;
    }

    function makeFrame(ctx, w, h) {
      return P.frame(ctx, w, h, 0, 1, 0, 1, { l: 10, r: 10, t: 10, b: 22 });
    }

    function cellCenter(i) {
      return [((i % g) + 0.5) / g, (Math.floor(i / g) + 0.5) / g];
    }

    function drawImg() {
      const th = P.theme();
      const sz = P.setup(cvImg);
      const ctx = sz.ctx, w = sz.w, h = sz.h;
      P.clear(ctx, w, h);
      const fr = makeFrame(ctx, w, h);
      const Yi = (v) => fr.Y(1 - v);

      ctx.fillStyle = th.line;
      ctx.globalAlpha = 0.18;
      ctx.fillRect(fr.X(0), Yi(0), fr.iw, fr.ih);
      ctx.globalAlpha = 1;
      ctx.strokeStyle = th.line;
      ctx.lineWidth = 1.5;
      ctx.strokeRect(fr.X(0), Yi(0), fr.iw, fr.ih);

      /* As cinco pessoas (cada uma com sua "roupa" na imagem de entrada) */
      for (let i = 0; i < PEOPLE.length; i++) {
        const p = PEOPLE[i];
        ctx.fillStyle = instColor(th, i + 1);
        ctx.globalAlpha = 0.5;
        ctx.beginPath();
        ctx.ellipse(fr.X(p.cx), Yi(p.cy), p.rx * fr.iw, p.ry * fr.ih, 0, 0, 2 * Math.PI);
        ctx.fill();
        ctx.globalAlpha = 1;
      }
      P.mathText(ctx, '5 pessoas na cena', fr.X(0.5), Yi(0.06), th.comment, 'center', 11);

      /* Patch da janela deslizante centrado na célula atual */
      if (!denseUsed && cursor > 0 && cursor <= g * g) {
        const c = cellCenter(Math.min(cursor, g * g) - 1);
        const px = c[0] - PATCH / 2, py = c[1] - PATCH / 2;
        ctx.strokeStyle = th.orange;
        ctx.lineWidth = 2;
        ctx.strokeRect(fr.X(px), Yi(py), PATCH * fr.iw, PATCH * fr.ih);
        /* Pixel central do patch */
        ctx.fillStyle = th.orange;
        ctx.beginPath();
        ctx.arc(fr.X(c[0]), Yi(c[1]), 2.5, 0, 2 * Math.PI);
        ctx.fill();
      }
    }

    function drawMask() {
      const th = P.theme();
      const sz = P.setup(cvMask);
      const ctx = sz.ctx, w = sz.w, h = sz.h;
      P.clear(ctx, w, h);
      const fr = makeFrame(ctx, w, h);
      const Yi = (v) => fr.Y(1 - v);

      ctx.strokeStyle = th.line;
      ctx.lineWidth = 1.5;
      ctx.strokeRect(fr.X(0), Yi(0), fr.iw, fr.ih);

      const cw = fr.iw / g, ch = fr.ih / g;
      for (let i = 0; i < g * g; i++) {
        const l = labels[i];
        if (l < 0) continue; /* pixel ainda não classificado */
        const x = fr.X(0) + (i % g) * cw;
        const y = Yi(0) + Math.floor(i / g) * ch;
        if (l === 0) {
          ctx.fillStyle = th.comment;
          ctx.globalAlpha = 0.28;
        } else {
          ctx.fillStyle = mode === 'sem' ? th.red : instColor(th, l);
          ctx.globalAlpha = 0.85;
        }
        ctx.fillRect(x, y, cw + 0.5, ch + 0.5);
        ctx.globalAlpha = 1;
      }
      const cap = mode === 'sem'
        ? 'semântica: toda a classe pessoa com a mesma cor'
        : 'instâncias: cada pessoa com uma cor distinta';
      P.mathText(ctx, cap, fr.X(0.5), Yi(1) + 15, th.comment, 'center', 10);
    }

    function updateReadout() {
      const total = g * g;
      let txt;
      if (denseUsed) {
        txt = 'encoder-decoder: 1 forward classificou os ' + fmtInt(total) + ' pixels da máscara ' + g + '×' + g;
      } else if (forwards === 0) {
        txt = 'máscara ' + g + '×' + g + ' = ' + fmtInt(total) + ' pixels · nenhum forward ainda';
      } else {
        txt = 'janela deslizante: ' + fmtInt(forwards) + ' / ' + fmtInt(total) +
          ' forwards (1 pixel classificado por forward)';
      }
      $('s4-readout').textContent = txt;
    }

    function redraw() {
      drawImg();
      drawMask();
      updateReadout();
    }

    function stepCells(n) {
      const total = g * g;
      while (n-- > 0 && cursor < total) {
        const c = cellCenter(cursor);
        labels[cursor] = classify(c[0], c[1]);
        cursor++;
        forwards++;
      }
    }

    function stopRun() {
      running = false;
      btnRun.textContent = '▶ Janela deslizante';
      DL.stopTicker(tick);
    }

    function tick(dt) {
      acc += dt * Math.max(160, (g * g) / 7); /* varredura completa em ~7 s */
      const n = Math.floor(acc);
      acc -= n;
      stepCells(n);
      redraw();
      if (cursor >= g * g) stopRun();
    }

    function setMode(m) {
      mode = m;
      tabSem.classList.toggle('active', m === 'sem');
      tabInst.classList.toggle('active', m === 'inst');
      redraw();
    }

    tabSem.addEventListener('click', function () { setMode('sem'); });
    tabInst.addEventListener('click', function () { setMode('inst'); });

    slGrid.addEventListener('input', function () {
      stopRun();
      g = +slGrid.value;
      $('s4-grid-val').textContent = g + '×' + g;
      clearMask();
      redraw();
    });

    btnRun.addEventListener('click', function () {
      if (running) { stopRun(); return; }
      if (denseUsed || cursor >= g * g) clearMask();
      running = true;
      acc = 0;
      btnRun.textContent = '⏸ Pausar';
      DL.startTicker(tick);
    });

    btnDense.addEventListener('click', function () {
      stopRun();
      clearMask();
      for (let i = 0; i < g * g; i++) {
        const c = cellCenter(i);
        labels[i] = classify(c[0], c[1]);
      }
      cursor = g * g;
      forwards = 1;
      denseUsed = true;
      redraw();
    });

    btnReset.addEventListener('click', function () {
      stopRun();
      clearMask();
      redraw();
    });

    redraw();
    P.onRedraw(redraw);
    P.observeResize(cvImg, redraw);
    P.observeResize(cvMask, redraw);
  }

  DL.sections.push({ name: 's4-segmentation', init });
})();
