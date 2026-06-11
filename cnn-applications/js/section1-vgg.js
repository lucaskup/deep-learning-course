/* Seção 1: arquitetura da VGG16 camada a camada e contagem de parâmetros
   (convolução: K_h·K_w·C_in·C_out; densa: n_in·n_out; bias ignorado como nos slides). */
(function () {
  'use strict';
  window.DL = window.DL || {};
  DL.sections = DL.sections || [];

  /* Estágios da VGG16: entrada, 13 conv, 5 pool, flatten, 3 fc, softmax (24 no total). */
  const STAGES = (function () {
    const L = [{ name: 'entrada', type: 'input', s: 224, c: 3, params: 0 }];
    const blocks = [[2, 64], [2, 128], [3, 256], [3, 512], [3, 512]];
    let s = 224, cin = 3;
    blocks.forEach(function (blk, bi) {
      const n = blk[0], cout = blk[1];
      for (let i = 1; i <= n; i++) {
        L.push({ name: 'conv' + (bi + 1) + '_' + i, type: 'conv', s: s, c: cout, cin: cin, params: 9 * cin * cout });
        cin = cout;
      }
      s = s / 2;
      L.push({ name: 'pool' + (bi + 1), type: 'pool', s: s, c: cout, params: 0 });
    });
    L.push({ name: 'flatten', type: 'flat', n: 7 * 7 * 512, params: 0 });
    L.push({ name: 'fc1', type: 'fc', nin: 25088, n: 4096, params: 25088 * 4096 });
    L.push({ name: 'fc2', type: 'fc', nin: 4096, n: 4096, params: 4096 * 4096 });
    L.push({ name: 'fc3', type: 'fc', nin: 4096, n: 1000, params: 4096 * 1000 });
    L.push({ name: 'softmax', type: 'softmax', n: 1000, params: 0 });
    return L;
  })();

  /* Camadas treináveis (16: 13 conv + 3 fc), com referência ao índice do estágio. */
  const TRAINABLE = STAGES
    .map(function (st, i) { return { st: st, stage: i }; })
    .filter(function (e) { return e.st.params > 0; });

  const fmtInt = function (n) { return n.toLocaleString('pt-BR'); };
  const fmtM = function (n) { return (n / 1e6).toFixed(1).replace('.', ',') + 'M'; };

  function init() {
    const P = DL.plot;
    const $ = (id) => document.getElementById(id);

    const cvArch = $('s1-arch'), cvParams = $('s1-params');
    const slLayer = $('s1-layer');
    const btnStep = $('s1-step'), btnRun = $('s1-run'), btnReset = $('s1-reset');

    let idx = 0;
    let running = false, acc = 0;
    let slots = null; /* geometria dos estágios para clique */

    function stageColor(th, st) {
      if (st.type === 'conv') return th.red;
      if (st.type === 'pool') return th.orange;
      if (st.type === 'fc') return th.cyan;
      if (st.type === 'flat') return th.purple;
      if (st.type === 'softmax') return th.green;
      return th.comment;
    }

    function stageDims(st) {
      if (st.type === 'fc') return fmtInt(st.n);
      if (st.type === 'flat') return fmtInt(st.n);
      if (st.type === 'softmax') return '1000 classes';
      return st.s + '×' + st.s + '×' + st.c;
    }

    function drawArch() {
      const th = P.theme();
      const sz = P.setup(cvArch);
      const ctx = sz.ctx, w = sz.w, h = sz.h;
      P.clear(ctx, w, h);

      const padL = 12, padR = 12, padT = 34, padB = 36;
      const ih = h - padT - padB;
      const n = STAGES.length;
      const slotW = (w - padL - padR) / n;
      const cy = padT + ih / 2;
      slots = [];

      /* Legenda no topo */
      const legend = [
        ['conv 3×3 + ReLU', th.red], ['max pool 2×2', th.orange],
        ['densa', th.cyan], ['softmax', th.green],
      ];
      let lx = padL;
      ctx.font = '11px sans-serif';
      for (const item of legend) {
        ctx.fillStyle = item[1];
        ctx.globalAlpha = 0.75;
        ctx.fillRect(lx, 8, 10, 10);
        ctx.globalAlpha = 1;
        ctx.fillStyle = th.comment;
        ctx.textAlign = 'left';
        ctx.fillText(item[0], lx + 14, 17);
        lx += 14 + ctx.measureText(item[0]).width + 16;
      }

      for (let i = 0; i < n; i++) {
        const st = STAGES[i];
        let bw = slotW * 0.6, bh;
        if (st.type === 'flat') { bw = slotW * 0.28; bh = ih * 0.55; }
        else if (st.type === 'fc') { bh = ih * (st.n >= 4096 ? 0.18 : 0.12); }
        else if (st.type === 'softmax') { bh = ih * 0.12; }
        else { bh = Math.max(8, ih * Math.pow(st.s / 224, 0.5)); }

        const x = padL + i * slotW + (slotW - bw) / 2;
        const y = cy - bh / 2;
        slots.push({ x: x, y: y, w: bw, h: bh, slotX: padL + i * slotW, slotW: slotW });

        ctx.fillStyle = stageColor(th, st);
        let alpha = 0.7;
        if (st.type === 'conv') alpha = 0.3 + 0.5 * (st.c / 512);
        if (st.type === 'pool') alpha = 0.45;
        if (st.type === 'input') alpha = 0.35;
        ctx.globalAlpha = alpha;
        ctx.fillRect(x, y, bw, bh);
        ctx.globalAlpha = 1;

        if (i === idx) {
          ctx.strokeStyle = th.fg;
          ctx.lineWidth = 2;
          ctx.strokeRect(x - 1.5, y - 1.5, bw + 3, bh + 3);
        }
      }

      /* Rótulos do estágio atual: nome acima, dimensões abaixo */
      const cur = STAGES[idx], sl = slots[idx];
      const cx = sl.x + sl.w / 2;
      P.mathText(ctx, cur.name, cx, sl.y - 8, th.fg, 'center', 12);
      P.mathText(ctx, stageDims(cur), cx, padT + ih + 16, th.fg, 'center', 11);

      /* Marcas dos blocos convolucionais */
      ctx.fillStyle = th.comment;
      ctx.font = '10px sans-serif';
      ctx.textAlign = 'left';
      ctx.fillText('extrator de features (convoluções)', padL, h - 6);
      ctx.textAlign = 'right';
      ctx.fillText('cabeça densa', w - padR, h - 6);
    }

    function drawParams() {
      const th = P.theme();
      const sz = P.setup(cvParams);
      const ctx = sz.ctx, w = sz.w, h = sz.h;
      P.clear(ctx, w, h);

      const m = TRAINABLE.length;
      const fr = P.frame(ctx, w, h, -0.7, m - 0.3, 0, 110);
      P.axes(ctx, fr, { ylabel: 'parâmetros (M)', yticks: [0, 25, 50, 75, 100] });

      for (let i = 0; i < m; i++) {
        const e = TRAINABLE[i];
        const v = e.st.params / 1e6;
        const x0 = fr.X(i - 0.38), x1 = fr.X(i + 0.38);
        const y0 = fr.Y(0), y1 = fr.Y(Math.max(v, 0.6)); /* altura mínima visível */
        ctx.fillStyle = e.st.type === 'conv' ? th.red : th.cyan;
        ctx.globalAlpha = e.stage === idx ? 1 : 0.45;
        ctx.fillRect(x0, y1, x1 - x0, y0 - y1);
        ctx.globalAlpha = 1;
        if (e.stage === idx) {
          ctx.strokeStyle = th.fg;
          ctx.lineWidth = 1.5;
          ctx.strokeRect(x0, y1, x1 - x0, y0 - y1);
          P.mathText(ctx, fmtM(e.st.params), (x0 + x1) / 2, y1 - 5, th.fg, 'center', 10);
        }
      }
      P.mathText(ctx, 'fc1', fr.X(13), fr.Y(105), th.cyan, 'center', 10);
      P.mathText(ctx, 'convoluções: 14,7M', fr.X(6), fr.Y(0) + 14, th.red, 'center', 10);
      P.mathText(ctx, 'densas: 123,6M', fr.X(14), fr.Y(0) + 14, th.cyan, 'center', 10);
    }

    function updateReadout() {
      const st = STAGES[idx];
      let cum = 0;
      for (let i = 0; i <= idx; i++) cum += STAGES[i].params;
      let desc;
      if (st.type === 'conv') {
        desc = '3·3·' + st.cin + '·' + st.c + ' = ' + fmtInt(st.params) + ' pesos';
      } else if (st.type === 'fc') {
        desc = fmtInt(st.nin) + ' × ' + fmtInt(st.n) + ' = ' + fmtInt(st.params) + ' pesos';
      } else if (st.type === 'pool') {
        desc = 'sem parâmetros (subamostra 2×2, stride 2)';
      } else if (st.type === 'flat') {
        desc = 'sem parâmetros (7·7·512 = 25.088 valores)';
      } else if (st.type === 'softmax') {
        desc = 'sem parâmetros (normaliza os 1000 logits)';
      } else {
        desc = 'tensor RGB de entrada';
      }
      $('s1-layer-val').textContent = st.name;
      $('s1-readout').textContent =
        st.name + ' · saída ' + stageDims(st) + ' · ' + desc +
        ' · acumulado ' + fmtM(cum) + ' (total 138,3M)';
    }

    function redraw() {
      drawArch();
      drawParams();
      updateReadout();
    }

    function setIdx(i) {
      idx = Math.max(0, Math.min(STAGES.length - 1, i));
      slLayer.value = idx;
      redraw();
    }

    function stopRun() {
      running = false;
      btnRun.textContent = '▶ Percorrer';
      DL.stopTicker(tick);
    }

    function tick(dt) {
      acc += dt * 2.2;
      while (acc >= 1) {
        acc -= 1;
        if (idx >= STAGES.length - 1) { stopRun(); return; }
        setIdx(idx + 1);
      }
    }

    cvArch.addEventListener('pointerdown', function (e) {
      if (!slots) return;
      const r = cvArch.getBoundingClientRect();
      const px = e.clientX - r.left;
      for (let i = 0; i < slots.length; i++) {
        if (px >= slots[i].slotX && px < slots[i].slotX + slots[i].slotW) {
          stopRun();
          setIdx(i);
          return;
        }
      }
    });

    slLayer.addEventListener('input', function () { stopRun(); setIdx(+slLayer.value); });
    btnStep.addEventListener('click', function () {
      stopRun();
      setIdx(idx >= STAGES.length - 1 ? 0 : idx + 1);
    });
    btnRun.addEventListener('click', function () {
      if (running) { stopRun(); return; }
      if (idx >= STAGES.length - 1) setIdx(0);
      running = true;
      acc = 0;
      btnRun.textContent = '⏸ Pausar';
      DL.startTicker(tick);
    });
    btnReset.addEventListener('click', function () { stopRun(); setIdx(0); });

    redraw();
    P.onRedraw(redraw);
    P.observeResize(cvArch, redraw);
    P.observeResize(cvParams, redraw);
  }

  DL.sections.push({ name: 's1-vgg', init });
})();
