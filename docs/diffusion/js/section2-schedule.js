/* Seção 2: explorador do noise schedule + difusão de uma imagem real. */
(function () {
  'use strict';
  window.DM = window.DM || {};
  DM.sections = DM.sections || [];

  const IMG_W = 192;

  function init() {
    const U = DM.utils, P = DM.plot;
    const $ = (id) => document.getElementById(id);

    const cvBeta = $('s2-beta'), cvAlpha = $('s2-alpha'), cvSignal = $('s2-signal'), cvImg = $('s2-image');
    const tabLinear = $('s2-linear'), tabCosine = $('s2-cosine');
    const sliderT = $('s2-T'), sliderBmax = $('s2-bmax'), slidert = $('s2-t');
    const btnPreset = $('s2-preset'), btnResample = $('s2-resample');

    let mode = 'linear';
    let T = 50, betaMin = 1e-4, betaMax = 0.2;
    let t = 0;
    let seed = 42;
    let sched = null;
    let imgX = null, imgW = 0, imgH = 0, eps = null;
    const off = document.createElement('canvas');

    /* Slider de T em escala log: [4, 1000] */
    const T_LO = 4, T_HI = 1000;
    const sliderToT = (v) => Math.round(T_LO * Math.pow(T_HI / T_LO, v / 100));
    const tToSlider = (tv) => Math.round(100 * Math.log(tv / T_LO) / Math.log(T_HI / T_LO));
    sliderT.value = tToSlider(T);

    function recompute() {
      sched = mode === 'linear'
        ? U.linearSchedule(T, betaMin, betaMax)
        : U.cosineSchedule(T);
      $('s2-T-val').textContent = T;
      $('s2-bmax-val').textContent = betaMax.toFixed(2);
      $('s2-bmax-wrap').style.opacity = mode === 'linear' ? 1 : 0.35;
      sliderBmax.disabled = mode !== 'linear';
      slidert.max = T;
      if (t > T) t = T;
      slidert.value = t;
      $('s2-t-max').textContent = T;
      redraw();
    }

    function ticksFor(T) {
      if (T <= 8) {
        const arr = [];
        for (let i = 0; i <= T; i++) arr.push(i);
        return arr;
      }
      return [0, Math.round(T / 2), T];
    }

    function drawCurves() {
      const th = P.theme();
      const xt = [], beta = [], alpha = [], sig = [], noi = [];
      for (let s = 1; s <= T; s++) { xt.push(s); beta.push(sched.beta[s]); }
      const xa = [0], aa = [1];
      for (let s = 1; s <= T; s++) { xa.push(s); aa.push(sched.alpha[s]); }
      for (let i = 0; i < xa.length; i++) { sig.push(Math.sqrt(aa[i])); noi.push(Math.sqrt(1 - aa[i])); }

      // β_t
      let { ctx, w, h } = P.setup(cvBeta);
      P.clear(ctx, w, h);
      let fr = P.frame(ctx, w, h, 0, T, 0, Math.max(0.5, betaMax * 1.1));
      P.axes(ctx, fr, { xlabel: 't', xticks: ticksFor(T), yticks: [0, fr.ymax] });
      P.line(ctx, fr, xt, beta, th.orange, { width: 2 });
      if (T <= 8) P.scatter(ctx, fr, xt.map((v, i) => [v, beta[i]]), th.orange, { r: 3, alpha: 1 });

      // α_t
      ({ ctx, w, h } = P.setup(cvAlpha));
      P.clear(ctx, w, h);
      fr = P.frame(ctx, w, h, 0, T, 0, 1.05);
      P.axes(ctx, fr, { xlabel: 't', xticks: ticksFor(T), yticks: [0, 0.5, 1] });
      P.line(ctx, fr, [0, T], [0.5, 0.5], th.line, { dash: [4, 4], width: 1 });
      P.line(ctx, fr, xa, aa, th.purple, { width: 2 });
      if (T <= 8) P.scatter(ctx, fr, xa.map((v, i) => [v, aa[i]]), th.purple, { r: 3, alpha: 1 });

      // √α (sinal) vs √(1−α) (ruído) + cruzamento
      ({ ctx, w, h } = P.setup(cvSignal));
      P.clear(ctx, w, h);
      fr = P.frame(ctx, w, h, 0, T, 0, 1.05);
      P.axes(ctx, fr, { xlabel: 't', xticks: ticksFor(T), yticks: [0, 0.5, 1] });
      P.line(ctx, fr, xa, sig, th.green, { width: 2 });
      P.line(ctx, fr, xa, noi, th.pink, { width: 2 });
      let cross = -1;
      for (let s = 1; s <= T; s++) if (sched.alpha[s] < 0.5) { cross = s; break; }
      if (cross >= 0) {
        const y = Math.sqrt(sched.alpha[cross]);
        P.scatter(ctx, fr, [[cross, y]], th.fg, { r: 4, alpha: 1 });
        P.label(ctx, fr.X(cross) + 6, fr.Y(y) - 8, 'cruzamento t = ' + cross, th.fg);
      }
      P.label(ctx, fr.X(0) + 8, fr.Y(1.0) + 12, '√α_t (sinal)', th.green);
      P.label(ctx, fr.X(0) + 8, fr.Y(1.0) + 26, '√(1−α_t) (ruído)', th.pink);
      // marca o t atual
      P.line(ctx, fr, [t, t], [0, 1.05], th.comment, { dash: [3, 3], width: 1 });
    }

    function drawImage() {
      const { ctx, w, h } = P.setup(cvImg);
      P.clear(ctx, w, h);
      if (!imgX) {
        P.label(ctx, w / 2, h / 2, 'carregando imagem…', P.theme().comment, 'center');
        return;
      }
      const a = sched.alpha[t];
      const sa = Math.sqrt(a), sn = Math.sqrt(1 - a);
      const octx = off.getContext('2d');
      const id = octx.createImageData(imgW, imgH);
      const n = imgW * imgH;
      for (let i = 0; i < n; i++) {
        for (let c = 0; c < 3; c++) {
          const z = sa * imgX[i * 3 + c] + sn * eps[i * 3 + c];
          id.data[i * 4 + c] = Math.max(0, Math.min(255, (z + 1) * 127.5));
        }
        id.data[i * 4 + 3] = 255;
      }
      octx.putImageData(id, 0, 0);
      const scale = Math.min(w / imgW, h / imgH);
      const dw = imgW * scale, dh = imgH * scale;
      ctx.imageSmoothingEnabled = true;
      ctx.drawImage(off, (w - dw) / 2, (h - dh) / 2, dw, dh);
    }

    function updateReadout() {
      const a = sched.alpha[t];
      const bt = t >= 1 ? sched.beta[t].toFixed(4) : '—';
      $('s2-t-val').textContent = t;
      $('s2-readout').innerHTML =
        'β<sub>t</sub> = ' + bt + ' · α<sub>t</sub> = ' + a.toFixed(3) +
        ' · √α<sub>t</sub> = ' + Math.sqrt(a).toFixed(3) +
        ' · √(1−α<sub>t</sub>) = ' + Math.sqrt(1 - a).toFixed(3);
    }

    function redraw() {
      drawCurves();
      drawImage();
      updateReadout();
    }

    function regenEps() {
      const rng = U.mulberry32(seed);
      eps = new Float32Array(imgW * imgH * 3);
      for (let i = 0; i < eps.length; i++) eps[i] = U.randn(rng);
    }

    function setMode(m) {
      mode = m;
      tabLinear.classList.toggle('active', m === 'linear');
      tabCosine.classList.toggle('active', m === 'cosine');
      recompute();
    }

    tabLinear.addEventListener('click', () => setMode('linear'));
    tabCosine.addEventListener('click', () => setMode('cosine'));
    /* Mover os sliders sai do preset da aula: β_min volta ao padrão DDPM */
    sliderT.addEventListener('input', () => { betaMin = 1e-4; T = sliderToT(+sliderT.value); recompute(); });
    sliderBmax.addEventListener('input', () => { betaMin = 1e-4; betaMax = +sliderBmax.value; recompute(); });
    slidert.addEventListener('input', () => { t = +slidert.value; redraw(); });
    btnResample.addEventListener('click', () => { seed = (seed * 1664525 + 1013904223) >>> 0; if (imgX) { regenEps(); drawImage(); } });

    /* Preset do exemplo numérico da aula: T = 4, β = (0.1, 0.2, 0.3, 0.4) */
    btnPreset.addEventListener('click', () => {
      mode = 'linear'; T = 4; betaMin = 0.1; betaMax = 0.4; t = Math.min(t, 4);
      tabLinear.classList.add('active');
      tabCosine.classList.remove('active');
      sliderT.value = tToSlider(T);
      sliderBmax.value = betaMax;
      recompute();
    });
    const img = new Image();
    img.onload = () => {
      imgW = IMG_W;
      imgH = Math.round(IMG_W * img.naturalHeight / img.naturalWidth);
      off.width = imgW; off.height = imgH;
      const octx = off.getContext('2d', { willReadFrequently: true });
      octx.drawImage(img, 0, 0, imgW, imgH);
      const data = octx.getImageData(0, 0, imgW, imgH).data;
      const n = imgW * imgH;
      imgX = new Float32Array(n * 3);
      for (let i = 0; i < n; i++)
        for (let c = 0; c < 3; c++)
          imgX[i * 3 + c] = data[i * 4 + c] / 127.5 - 1;
      regenEps();
      drawImage();
    };
    img.onerror = () => { console.error('Não foi possível carregar assets/gato.jpg'); };
    img.src = 'assets/gato.jpg';

    recompute();
    P.onRedraw(redraw);
    P.observeResize(cvBeta, redraw);
    P.observeResize(cvImg, drawImage);
  }

  DM.sections.push({ name: 's2-schedule', init });
})();
