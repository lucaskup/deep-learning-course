/* Seção 3: MHA, MQA e GQA, cabeças de query compartilhando pares K,V
   e o efeito de G no tamanho do KV-cache (2·G·d_k·L floats por token). */
(function () {
  'use strict';
  window.DL = window.DL || {};
  DL.sections = DL.sections || [];

  const HS = [4, 8, 16, 32, 64, 96];
  const DK = 128;

  function divisorsOf(h) {
    const d = [];
    for (let g = 1; g <= h; g++) if (h % g === 0) d.push(g);
    return d;
  }

  function init() {
    const P = DL.plot;
    const $ = (id) => document.getElementById(id);

    const cvHeads = $('s3-heads'), cvMem = $('s3-mem');
    const slH = $('s3-h'), slG = $('s3-g'), slL = $('s3-l'), slN = $('s3-n');

    let H = HS[+slH.value];
    let divs = divisorsOf(H);
    let G = 8, L = +slL.value, n = Math.pow(2, +slN.value);

    function syncGSlider(targetG) {
      divs = divisorsOf(H);
      let best = 0;
      for (let i = 0; i < divs.length; i++) {
        if (Math.abs(divs[i] - targetG) < Math.abs(divs[best] - targetG)) best = i;
      }
      slG.max = String(divs.length - 1);
      slG.value = String(best);
      G = divs[best];
    }

    function syncLabels() {
      $('s3-h-val').textContent = H;
      $('s3-g-val').textContent = G + (G === 1 ? ' (MQA)' : (G === H ? ' (MHA)' : ' (GQA)'));
      $('s3-l-val').textContent = L;
      $('s3-n-val').textContent = n >= 1024 ? (n / 1024) + 'K' : n;
    }

    /* Memória do KV-cache em GB (fp16): n tokens × L camadas × 2 (K,V) × G × d_k × 2 bytes. */
    const cacheGB = (g) => (n * L * 2 * g * DK * 2) / Math.pow(2, 30);

    function fmtGB(v) {
      if (v >= 10) return v.toFixed(0) + ' GB';
      if (v >= 1) return v.toFixed(1) + ' GB';
      return (v * 1024).toFixed(0) + ' MB';
    }

    function roundRect(ctx, x, y, w, h, r) {
      ctx.beginPath();
      ctx.moveTo(x + r, y);
      ctx.arcTo(x + w, y, x + w, y + h, r);
      ctx.arcTo(x + w, y + h, x, y + h, r);
      ctx.arcTo(x, y + h, x, y, r);
      ctx.arcTo(x, y, x + w, y, r);
      ctx.closePath();
    }

    function drawHeads() {
      const th = P.theme();
      const { ctx, w, h } = P.setup(cvHeads);
      P.clear(ctx, w, h);
      const margin = 24;
      const usable = w - 2 * margin;
      const yQ = 36, qH = 34;
      const yKV = h - 84, kvH = 34;

      const slot = usable / H;
      const qW = Math.max(4, Math.min(44, slot - 4));
      const per = H / G;
      const kvSlot = usable / G;
      const kvW = Math.max(14, Math.min(70, kvSlot - 8));

      /* Ligações Q → par K,V do grupo */
      ctx.strokeStyle = th.comment;
      ctx.globalAlpha = 0.5;
      ctx.lineWidth = 1;
      for (let i = 0; i < H; i++) {
        const qx = margin + i * slot + slot / 2;
        const g = Math.floor(i / per);
        const kx = margin + g * kvSlot + kvSlot / 2;
        ctx.beginPath();
        ctx.moveTo(qx, yQ + qH);
        ctx.lineTo(kx, yKV);
        ctx.stroke();
      }
      ctx.globalAlpha = 1;

      /* Cabeças de query */
      for (let i = 0; i < H; i++) {
        const qx = margin + i * slot + (slot - qW) / 2;
        ctx.fillStyle = th.cyan;
        ctx.globalAlpha = 0.30;
        roundRect(ctx, qx, yQ, qW, qH, 4);
        ctx.fill();
        ctx.globalAlpha = 1;
        ctx.strokeStyle = th.cyan;
        roundRect(ctx, qx, yQ, qW, qH, 4);
        ctx.stroke();
        if (H <= 16 && qW >= 24) {
          P.mathText(ctx, 'Q_' + (i + 1), qx + qW / 2, yQ + qH / 2 + 4, th.fg, 'center', 10);
        }
      }

      /* Pares K,V */
      for (let g = 0; g < G; g++) {
        const kx = margin + g * kvSlot + (kvSlot - kvW) / 2;
        ctx.fillStyle = th.green;
        ctx.globalAlpha = 0.35;
        roundRect(ctx, kx, yKV, kvW, kvH, 4);
        ctx.fill();
        ctx.globalAlpha = 1;
        ctx.strokeStyle = th.green;
        roundRect(ctx, kx, yKV, kvW, kvH, 4);
        ctx.stroke();
        if (G <= 16 && kvW >= 28) {
          ctx.fillStyle = th.fg;
          ctx.font = '10px sans-serif';
          ctx.textAlign = 'center';
          ctx.fillText('K,V', kx + kvW / 2, yKV + kvH / 2 + 4);
        }
      }

      ctx.fillStyle = th.comment;
      ctx.font = '11px sans-serif';
      ctx.textAlign = 'left';
      ctx.fillText('H = ' + H + ' cabeças de query', margin, yQ - 10);
      ctx.fillText('G = ' + G + (G === 1 ? ' par K,V (MQA)' : (G === H ? ' pares K,V (MHA)' : ' pares K,V (GQA), ' + per + ' cabeças por grupo')), margin, yKV + kvH + 18);
    }

    function drawMem() {
      const th = P.theme();
      const { ctx, w, h } = P.setup(cvMem);
      P.clear(ctx, w, h);
      const bars = [
        { name: 'MHA', g: H, color: th.red },
        { name: 'GQA (G=' + G + ')', g: G, color: th.purple },
        { name: 'MQA', g: 1, color: th.green },
      ];
      const ymax = cacheGB(H) * 1.12;
      const fr = P.frame(ctx, w, h, 0, 3, 0, ymax, { l: 44, r: 10, t: 14, b: 30 });
      P.axes(ctx, fr, { ylabel: 'GB', yticks: [0, +(ymax / 2).toFixed(1), +ymax.toFixed(1)] });
      const bw = Math.min(70, fr.iw / 3 * 0.5);
      bars.forEach((b, k) => {
        const v = cacheGB(b.g);
        const px = fr.X(k + 0.5);
        ctx.fillStyle = b.color;
        ctx.globalAlpha = 0.75;
        ctx.fillRect(px - bw / 2, fr.Y(v), bw, fr.Y(0) - fr.Y(v));
        ctx.globalAlpha = 1;
        ctx.fillStyle = th.fg;
        ctx.font = '11px sans-serif';
        ctx.textAlign = 'center';
        ctx.fillText(fmtGB(v), px, Math.max(fr.Y(v) - 6, 12));
        ctx.fillStyle = th.comment;
        ctx.fillText(b.name, px, fr.Y(0) + 16);
      });
    }

    function updateReadout() {
      const perTok = 2 * G * DK * L;
      $('s3-readout').textContent =
        'cache por token: 2·G·d_k·L = ' + perTok.toLocaleString('pt-BR') +
        ' floats · total (n = ' + n.toLocaleString('pt-BR') + '): ' + fmtGB(cacheGB(G)) +
        ' · redução vs MHA: ' + (H / G) + '×';
    }

    function redraw() {
      syncLabels();
      drawHeads();
      drawMem();
      updateReadout();
    }

    function applyPreset(h, g, l, log2n) {
      H = h;
      slH.value = String(HS.indexOf(h));
      syncGSlider(g);
      L = l;
      slL.value = String(l);
      n = Math.pow(2, log2n);
      slN.value = String(log2n);
      redraw();
    }

    slH.addEventListener('input', () => {
      H = HS[+slH.value];
      syncGSlider(G);
      redraw();
    });
    slG.addEventListener('input', () => {
      G = divs[+slG.value];
      redraw();
    });
    slL.addEventListener('input', () => { L = +slL.value; redraw(); });
    slN.addEventListener('input', () => { n = Math.pow(2, +slN.value); redraw(); });

    $('s3-preset-gpt3').addEventListener('click', () => applyPreset(96, 96, 96, 11));
    $('s3-preset-llama').addEventListener('click', () => applyPreset(64, 8, 80, 12));
    $('s3-preset-mistral').addEventListener('click', () => applyPreset(32, 8, 32, 15));

    syncGSlider(G);
    redraw();
    P.onRedraw(redraw);
    P.observeResize(cvHeads, redraw);
    P.observeResize(cvMem, redraw);
  }

  DL.sections.push({ name: 's3-gqa', init });
})();
