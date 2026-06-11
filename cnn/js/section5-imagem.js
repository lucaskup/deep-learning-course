/* Seção 5: convolução em imagens reais. Imagem padrão (ou enviada pelo
   usuário) em tons de cinza, kernels clássicos escolhidos num dropdown,
   hiperparâmetros stride/padding/dilation e mapa de características com
   ReLU opcional. O hover na saída mostra a janela lida na entrada. */
(function () {
  'use strict';
  window.DL = window.DL || {};
  DL.sections = DL.sections || [];

  /* Lado maior da imagem de trabalho, em pixels. Mantém a convolução
     instantânea mesmo com kernel 5x5 recalculado a cada slider. */
  const MAX_SIDE = 200;

  function gauss2d(K, sigma) {
    const c = (K - 1) / 2;
    const k = [];
    let sum = 0;
    for (let i = 0; i < K; i++) {
      const row = [];
      for (let j = 0; j < K; j++) {
        const v = Math.exp(-((i - c) * (i - c) + (j - c) * (j - c)) / (2 * sigma * sigma));
        row.push(v);
        sum += v;
      }
      k.push(row);
    }
    for (let i = 0; i < K; i++) for (let j = 0; j < K; j++) k[i][j] /= sum;
    return k;
  }

  function diffOfGaussians(K, s1, s2) {
    const a = gauss2d(K, s1), b = gauss2d(K, s2);
    const k = [];
    for (let i = 0; i < K; i++) {
      const row = [];
      for (let j = 0; j < K; j++) row.push(a[i][j] - b[i][j]);
      k.push(row);
    }
    return k;
  }

  const KERNELS = {
    sobelx: {
      label: 'Sobel x (bordas verticais)',
      k: [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],
    },
    sobely: {
      label: 'Sobel y (bordas horizontais)',
      k: [[-1, -2, -1], [0, 0, 0], [1, 2, 1]],
    },
    laplaciano: {
      label: 'Laplaciano (bordas em todas as direções)',
      k: [[0, 1, 0], [1, -4, 1], [0, 1, 0]],
    },
    dog: {
      label: 'DoG 5×5 (diferença de gaussianas)',
      k: diffOfGaussians(5, 1, 2),
    },
    gauss: {
      label: 'Blur gaussiano 5×5',
      k: gauss2d(5, 1),
    },
    box: {
      label: 'Blur de média 3×3',
      k: [[1 / 9, 1 / 9, 1 / 9], [1 / 9, 1 / 9, 1 / 9], [1 / 9, 1 / 9, 1 / 9]],
    },
    sharpen: {
      label: 'Sharpen (realce)',
      k: [[0, -1, 0], [-1, 5, -1], [0, -1, 0]],
    },
    emboss: {
      label: 'Emboss (relevo)',
      k: [[-2, -1, 0], [-1, 1, 1], [0, 1, 2]],
    },
    identidade: {
      label: 'Identidade',
      k: [[0, 0, 0], [0, 1, 0], [0, 0, 0]],
    },
  };

  /* Imagem sintética usada se o JPEG não puder ser lido (ex.: página aberta
     via file://, em que getImageData no canvas é bloqueado pelo navegador). */
  function makeSynthetic() {
    const W = 200, H = 120;
    const g = new Float32Array(W * H);
    for (let i = 0; i < H; i++) {
      for (let j = 0; j < W; j++) {
        let v = 0.25 + 0.5 * (j / (W - 1));            /* gradiente horizontal */
        if (j >= 20 && j < 60 && i >= 20 && i < 60) v = 0.95;     /* quadrado */
        const dx = j - 150, dy = i - 60;
        if (dx * dx + dy * dy < 28 * 28) v = 0.05;                /* círculo */
        if (j >= 80 && j < 120 && i >= 70 && i < 110) {
          v = ((i + j) % 8 < 4) ? 0.85 : 0.15;                    /* listras */
        }
        g[i * W + j] = v;
      }
    }
    return { gray: g, W, H };
  }

  function init() {
    const P = DL.plot;
    const G = DL.grid;
    const $ = (id) => document.getElementById(id);

    const cvIn = $('s5-input'), cvOut = $('s5-output'), cvKer = $('s5-kernel');
    const selKernel = $('s5-kernel-select');
    const slS = $('s5-s'), slP = $('s5-p'), slD = $('s5-d');
    const chkRelu = $('s5-relu');
    const fileInput = $('s5-file');

    let img = makeSynthetic();          /* {gray, W, H}, valores em [0, 1] */
    let imgNote = '';
    let kernelName = 'sobelx';
    let S = 1, Pp = 0, D = 1;
    let relu = false;
    let mode = 'cinza';                 /* 'cinza' ou 'sinal' */
    let out = null;                     /* {z, Ow, Oh, vmax} ou null */
    let hover = null;                   /* {oi, oj} ou null */
    let inOff = null;                   /* offscreen com a imagem em cinza */

    function kernel() { return KERNELS[kernelName].k; }
    function ksize() { return kernel().length; }

    function buildInputOffscreen() {
      inOff = document.createElement('canvas');
      inOff.width = img.W;
      inOff.height = img.H;
      const octx = inOff.getContext('2d');
      const id = octx.createImageData(img.W, img.H);
      for (let p = 0; p < img.W * img.H; p++) {
        const v = Math.round(255 * img.gray[p]);
        id.data[4 * p] = v; id.data[4 * p + 1] = v; id.data[4 * p + 2] = v;
        id.data[4 * p + 3] = 255;
      }
      octx.putImageData(id, 0, 0);
    }

    function outSize(N) {
      const K = ksize();
      return Math.floor((N + 2 * Pp - D * (K - 1) - 1) / S) + 1;
    }

    function recompute() {
      const K = ksize(), k = kernel();
      const Ow = outSize(img.W), Oh = outSize(img.H);
      if (Ow <= 0 || Oh <= 0) { out = null; return; }
      const z = new Float32Array(Ow * Oh);
      let vmax = 1e-9;
      for (let oi = 0; oi < Oh; oi++) {
        const i0 = oi * S - Pp;
        for (let oj = 0; oj < Ow; oj++) {
          const j0 = oj * S - Pp;
          let s = 0;
          for (let m = 0; m < K; m++) {
            const ii = i0 + m * D;
            if (ii < 0 || ii >= img.H) continue;       /* linha no zero padding */
            const base = ii * img.W;
            for (let n = 0; n < K; n++) {
              const jj = j0 + n * D;
              if (jj < 0 || jj >= img.W) continue;
              s += k[m][n] * img.gray[base + jj];
            }
          }
          if (relu && s < 0) s = 0;
          z[oi * Ow + oj] = s;
          const a = Math.abs(s);
          if (a > vmax) vmax = a;
        }
      }
      out = { z, Ow, Oh, vmax };
    }

    /* Geometria comum: centraliza uma área de gw por gh pixels no canvas. */
    function geom(cv, gw, gh) {
      const { w, h } = P.setup(cv);
      const pad = 10;
      const scale = Math.min((w - 2 * pad) / gw, (h - 2 * pad) / gh);
      return { w, h, scale, ox: (w - gw * scale) / 2, oy: (h - gh * scale) / 2 };
    }

    function drawInput() {
      const th = P.theme();
      const { ctx, w, h } = P.setup(cvIn);
      P.clear(ctx, w, h);
      const Wp = img.W + 2 * Pp, Hp = img.H + 2 * Pp;
      const L = geom(cvIn, Wp, Hp);

      /* Anel de zero padding: zeros aparecem como margem preta tracejada. */
      ctx.fillStyle = '#000';
      ctx.fillRect(L.ox, L.oy, Wp * L.scale, Hp * L.scale);
      ctx.imageSmoothingEnabled = false;
      ctx.drawImage(inOff, L.ox + Pp * L.scale, L.oy + Pp * L.scale,
        img.W * L.scale, img.H * L.scale);
      if (Pp > 0) {
        ctx.setLineDash([4, 3]);
        ctx.strokeStyle = th.comment;
        ctx.lineWidth = 1;
        ctx.strokeRect(L.ox, L.oy, Wp * L.scale, Hp * L.scale);
        ctx.setLineDash([]);
      }
      ctx.strokeStyle = th.line;
      ctx.lineWidth = 1;
      ctx.strokeRect(L.ox + Pp * L.scale, L.oy + Pp * L.scale,
        img.W * L.scale, img.H * L.scale);

      /* Janela lida pelo kernel na posição da saída sob o mouse. Com
         dilation a janela cobre span = D(K−1)+1 pixels de lado. */
      if (hover && out) {
        const span = D * (ksize() - 1) + 1;
        const x = L.ox + (hover.oj * S) * L.scale;
        const y = L.oy + (hover.oi * S) * L.scale;
        ctx.strokeStyle = th.orange;
        ctx.lineWidth = 2;
        ctx.strokeRect(x, y, span * L.scale, span * L.scale);
      }
      P.label(ctx, L.ox, L.oy - 5,
        'x em cinza, ' + img.W + '×' + img.H + (Pp > 0 ? ' + padding' : ''), th.comment);
    }

    function drawOutput() {
      const th = P.theme();
      const { ctx, w, h } = P.setup(cvOut);
      P.clear(ctx, w, h);
      if (!out) {
        P.label(ctx, w / 2, h / 2, 'kernel efetivo maior que a entrada: saída vazia',
          th.comment, 'center');
        return;
      }
      const { z, Ow, Oh, vmax } = out;
      const off = document.createElement('canvas');
      off.width = Ow; off.height = Oh;
      const octx = off.getContext('2d');
      const id = octx.createImageData(Ow, Oh);
      if (mode === 'cinza') {
        for (let p = 0; p < Ow * Oh; p++) {
          const v = Math.round(255 * Math.min(1, Math.abs(z[p]) / vmax));
          id.data[4 * p] = v; id.data[4 * p + 1] = v; id.data[4 * p + 2] = v;
          id.data[4 * p + 3] = 255;
        }
      } else {
        const cardC = P.rgb(th.card), posC = P.rgb(th.green), negC = P.rgb(th.pink);
        for (let p = 0; p < Ow * Oh; p++) {
          const t = Math.min(1, Math.abs(z[p]) / vmax);
          const c = z[p] >= 0 ? posC : negC;
          id.data[4 * p] = Math.round(cardC[0] + (c[0] - cardC[0]) * t);
          id.data[4 * p + 1] = Math.round(cardC[1] + (c[1] - cardC[1]) * t);
          id.data[4 * p + 2] = Math.round(cardC[2] + (c[2] - cardC[2]) * t);
          id.data[4 * p + 3] = 255;
        }
      }
      octx.putImageData(id, 0, 0);
      const L = geom(cvOut, Ow, Oh);
      ctx.imageSmoothingEnabled = false;
      ctx.drawImage(off, L.ox, L.oy, Ow * L.scale, Oh * L.scale);
      ctx.strokeStyle = th.line;
      ctx.lineWidth = 1;
      ctx.strokeRect(L.ox, L.oy, Ow * L.scale, Oh * L.scale);

      if (hover) {
        ctx.strokeStyle = th.orange;
        ctx.lineWidth = 2;
        ctx.strokeRect(L.ox + hover.oj * L.scale, L.oy + hover.oi * L.scale,
          Math.max(3, L.scale), Math.max(3, L.scale));
      }
      P.label(ctx, L.ox, L.oy - 5,
        'z' + (relu ? ' após ReLU' : '') + ', ' + Ow + '×' + Oh, th.comment);
    }

    function drawKernel() {
      const th = P.theme();
      const { ctx, w, h } = P.setup(cvKer);
      P.clear(ctx, w, h);
      const K = ksize(), k = kernel();
      const L = G.layout(w, h, K, K, 2);
      let vmax = 1e-9;
      for (let m = 0; m < K; m++) for (let n = 0; n < K; n++) {
        vmax = Math.max(vmax, Math.abs(k[m][n]));
      }
      for (let m = 0; m < K; m++) {
        for (let n = 0; n < K; n++) {
          ctx.fillStyle = G.diverge(k[m][n], vmax, th);
          ctx.fillRect(L.x(n), L.y(m), L.cs, L.cs);
          const v = k[m][n];
          const txt = Number.isInteger(v) ? String(v)
            : (Math.abs(v) < 0.005 ? '0' : v.toFixed(2).replace(/^(-?)0\./, '$1.'));
          G.cellText(ctx, L, m, n, txt, th.fg, Math.max(8, Math.min(12, L.cs * 0.32)));
        }
      }
      G.lines(ctx, L, th.line, 1);
    }

    function updateReadout() {
      const K = ksize();
      let txt;
      if (!out) {
        txt = 'O = ⌊(N + 2·' + Pp + ' − ' + D + '·(' + K + '−1) − 1)/' + S + '⌋ + 1 ≤ 0';
      } else {
        txt = 'O = ⌊(N + 2·' + Pp + ' − ' + D + '·(' + K + '−1) − 1)/' + S + '⌋ + 1' +
          ' → saída ' + out.Ow + '×' + out.Oh;
        if (hover) {
          txt = 'z(' + (hover.oi + 1) + ', ' + (hover.oj + 1) + ') = ' +
            out.z[hover.oi * out.Ow + hover.oj].toFixed(2) + ' · ' + txt;
        }
      }
      $('s5-readout').textContent = imgNote ? imgNote + ' · ' + txt : txt;
    }

    function redraw() {
      drawInput();
      drawOutput();
      drawKernel();
      updateReadout();
    }

    function refresh() {
      hover = null;
      recompute();
      redraw();
    }

    /* Pixel da saída sob o ponteiro. */
    function outCellFromEvent(e) {
      if (!out) return null;
      const r = cvOut.getBoundingClientRect();
      const L = geom(cvOut, out.Ow, out.Oh);
      const oj = Math.floor((e.clientX - r.left - L.ox) / L.scale);
      const oi = Math.floor((e.clientY - r.top - L.oy) / L.scale);
      return oi >= 0 && oi < out.Oh && oj >= 0 && oj < out.Ow ? { oi, oj } : null;
    }

    cvOut.addEventListener('pointermove', (e) => {
      hover = outCellFromEvent(e);
      redraw();
    });
    cvOut.addEventListener('pointerleave', () => { hover = null; redraw(); });

    function setImage(gray, W, H, note) {
      img = { gray, W, H };
      imgNote = note || '';
      buildInputOffscreen();
      refresh();
    }

    /* Reduz para MAX_SIDE no lado maior e converte para luminância. */
    function loadFromImageElement(el, note) {
      const scale = Math.min(1, MAX_SIDE / Math.max(el.naturalWidth, el.naturalHeight));
      const W = Math.max(8, Math.round(el.naturalWidth * scale));
      const H = Math.max(8, Math.round(el.naturalHeight * scale));
      const off = document.createElement('canvas');
      off.width = W; off.height = H;
      const octx = off.getContext('2d', { willReadFrequently: true });
      octx.drawImage(el, 0, 0, W, H);
      let data;
      try {
        data = octx.getImageData(0, 0, W, H).data;
      } catch (e) {
        setImage(img.gray, img.W, img.H, 'imagem bloqueada pelo navegador, usando a sintética');
        return;
      }
      const gray = new Float32Array(W * H);
      for (let p = 0; p < W * H; p++) {
        gray[p] = (0.299 * data[4 * p] + 0.587 * data[4 * p + 1] + 0.114 * data[4 * p + 2]) / 255;
      }
      setImage(gray, W, H, note);
    }

    function loadDefault() {
      const el = new Image();
      el.onload = () => loadFromImageElement(el, '');
      el.onerror = () => setImage(img.gray, img.W, img.H, 'imagem padrão indisponível, usando a sintética');
      el.src = 'assets/gato.jpg';
    }

    fileInput.addEventListener('change', () => {
      const f = fileInput.files && fileInput.files[0];
      if (!f) return;
      const url = URL.createObjectURL(f);
      const el = new Image();
      el.onload = () => { loadFromImageElement(el, ''); URL.revokeObjectURL(url); };
      el.onerror = () => { URL.revokeObjectURL(url); };
      el.src = url;
    });
    $('s5-upload').addEventListener('click', () => fileInput.click());
    $('s5-default').addEventListener('click', () => { fileInput.value = ''; loadDefault(); });

    selKernel.addEventListener('change', () => { kernelName = selKernel.value; refresh(); });

    function bindSlider(sl, idVal, set) {
      sl.addEventListener('input', () => {
        $(idVal).textContent = sl.value;
        set(+sl.value);
        refresh();
      });
    }
    bindSlider(slS, 's5-s-val', (v) => { S = v; });
    bindSlider(slP, 's5-p-val', (v) => { Pp = v; });
    bindSlider(slD, 's5-d-val', (v) => { D = v; });

    chkRelu.addEventListener('change', () => { relu = chkRelu.checked; refresh(); });

    $('s5-tab-cinza').addEventListener('click', () => {
      mode = 'cinza';
      $('s5-tab-cinza').classList.add('active');
      $('s5-tab-sinal').classList.remove('active');
      redraw();
    });
    $('s5-tab-sinal').addEventListener('click', () => {
      mode = 'sinal';
      $('s5-tab-sinal').classList.add('active');
      $('s5-tab-cinza').classList.remove('active');
      redraw();
    });

    buildInputOffscreen();
    refresh();
    loadDefault();
    P.onRedraw(redraw);
    P.observeResize(cvIn, redraw);
    P.observeResize(cvOut, redraw);
  }

  DL.sections.push({ name: 's5-imagem', init });
})();
