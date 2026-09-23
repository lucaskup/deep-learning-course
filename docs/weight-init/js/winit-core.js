/* Núcleo numérico da demo de inicialização de pesos, sem DOM. É usado pela seção 1
   (e, via DL.winit, pela seção 2) e também por scripts/export_init_histograms.js,
   que roda este mesmo código no node para gerar os histogramas dos slides.
   Depende de DL.utils (mulberry32, randn) e, para a entrada MNIST, de DL.mnistBatch. */
(function () {
  'use strict';
  const root = typeof window !== 'undefined' ? window : globalThis;
  const DL = root.DL = root.DL || {};

  const N = 300;          // largura de todas as camadas ocultas, como nos slides
  const B = 100;          // tamanho do batch, como nos slides
  const MNIST_MEAN = 0.1307, MNIST_STD = 0.3081;   // normalização usual do MNIST

  let mnistCache = null;

  /* Batch MNIST (100 x 784) com pixels em [0, 1] normalizados para média 0 e desvio 1. */
  function mnistInput() {
    if (mnistCache) return mnistCache;
    const m = DL.mnistBatch;
    const bin = atob(m.base64);
    const X = new Float32Array(m.count * m.dim);
    for (let i = 0; i < X.length; i++) {
      X[i] = (bin.charCodeAt(i) / 255 - MNIST_MEAN) / MNIST_STD;
    }
    mnistCache = { X, dim: m.dim };
    return mnistCache;
  }

  function stdOf(arr) {
    const n = arr.length;
    let m = 0;
    for (let i = 0; i < n; i++) m += arr[i];
    m /= n;
    let v = 0;
    for (let i = 0; i < n; i++) { const d = arr[i] - m; v += d * d; }
    return Math.sqrt(v / n);
  }

  /* Amostra um peso de acordo com o esquema, para uma camada com fan-in nIn. */
  function weightSampler(cfg, nIn, rng) {
    const U = DL.utils;
    switch (cfg.init) {
      case 'zeros': return () => 0;
      case 'normal': { const s = cfg.sigma; return () => s * U.randn(rng); }
      case 'uniform': { const r = 1 / Math.sqrt(nIn); return () => r * (2 * rng() - 1); }  // σ² = 1/(3n)
      case 'xavier': { const s = Math.sqrt(1 / nIn); return () => s * U.randn(rng); }     // σ² = 1/n
      default: { const s = Math.sqrt(2 / nIn); return () => s * U.randn(rng); }           // He: σ² = 2/n
    }
  }

  /* cfg: { init, sigma, act, L, seed, input: 'gauss' | 'mnist' }.
     Devolve ativações a^(0..L), gradientes g^(1..L), desvios por camada e as
     camadas selecionadas para os histogramas. */
  function simulate(cfg) {
    const U = DL.utils;
    const L = cfg.L;
    const rng = U.mulberry32(cfg.seed);

    /* batch de entradas */
    let X, n0;
    if (cfg.input === 'mnist') {
      const m = mnistInput();
      X = m.X; n0 = m.dim;
    } else {
      n0 = N;
      X = new Float32Array(B * n0);
      for (let i = 0; i < X.length; i++) X[i] = U.randn(rng);
    }

    /* pesos θ^(l) de shape (N, fan-in), bias = 0 */
    const Ws = [], fanIn = [];
    for (let l = 0; l < L; l++) {
      const nIn = l === 0 ? n0 : N;
      const draw = weightSampler(cfg, nIn, rng);
      const W = new Float32Array(N * nIn);
      for (let i = 0; i < W.length; i++) W[i] = draw();
      Ws.push(W); fanIn.push(nIn);
    }

    /* forward: acts[l] = a^(l) (acts[0] = x), derivs[l-1] = φ'(z^(l)) */
    const acts = [X], derivs = [];
    const act = cfg.act;
    let aPrev = X;
    for (let l = 0; l < L; l++) {
      const W = Ws[l], nIn = fanIn[l];
      const a = new Float32Array(B * N), d = new Float32Array(B * N);
      for (let b = 0; b < B; b++) {
        const offIn = b * nIn, off = b * N;
        for (let j = 0; j < N; j++) {
          let s = 0;
          const wr = j * nIn;
          for (let k = 0; k < nIn; k++) s += W[wr + k] * aPrev[offIn + k];
          let av, dv;
          if (act === 'relu') { av = s > 0 ? s : 0; dv = s > 0 ? 1 : 0; }
          else if (act === 'sigmoid') { av = 1 / (1 + Math.exp(-s)); dv = av * (1 - av); }
          else { av = Math.tanh(s); dv = 1 - av * av; }
          a[off + j] = av; d[off + j] = dv;
        }
      }
      acts.push(a); derivs.push(d);
      aPrev = a;
    }

    /* backward: gradiente com desvio 1 injetado em a^(L),
       g^(l) = (θ^(l+1))^T (g^(l+1) ⊙ φ'(z^(l+1))) */
    const grads = new Array(L + 1);
    const gTop = new Float32Array(B * N);
    for (let i = 0; i < gTop.length; i++) gTop[i] = U.randn(rng);
    grads[L] = gTop;
    for (let l = L - 1; l >= 1; l--) {
      const W = Ws[l];                  // θ^(l+1), fan-in N
      const gNext = grads[l + 1], d = derivs[l];   // φ'(z^(l+1))
      const g = new Float32Array(B * N);
      for (let b = 0; b < B; b++) {
        const off = b * N;
        for (let j = 0; j < N; j++) {
          const t = gNext[off + j] * d[off + j];
          if (t === 0) continue;
          const wr = j * N;
          for (let k = 0; k < N; k++) g[off + k] += W[wr + k] * t;
        }
      }
      grads[l] = g;
    }

    /* estatísticas por camada */
    const actStd = new Array(L + 1), gradStd = new Array(L + 1);
    for (let l = 0; l <= L; l++) actStd[l] = stdOf(acts[l]);
    gradStd[0] = NaN;
    for (let l = 1; l <= L; l++) gradStd[l] = stdOf(grads[l]);

    /* camadas exibidas nos histogramas: 1, ⌈L/4⌉, ⌈L/2⌉, ⌈3L/4⌉, L (sem repetição) */
    const sel = [];
    for (const l of [1, Math.ceil(L / 4), Math.ceil(L / 2), Math.ceil(3 * L / 4), L]) {
      if (!sel.includes(l)) sel.push(l);
    }

    return { L, acts, grads, actStd, gradStd, sel };
  }

  DL.winitCore = { simulate, N, B };
})();
