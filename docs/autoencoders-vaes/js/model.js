/* Modelos minúsculos da demo de AE/VAE: MLP denso com backprop manual,
   otimizador Adam, passos de treino de autoencoder e de VAE (1 amostra de
   Monte Carlo por instância, como nos slides) e o dataset two moons. */
(function () {
  'use strict';
  window.DL = window.DL || {};

  /* ---- MLP: camadas densas, tanh nas ocultas, saída linear ---- */
  function mlp(sizes, rng) {
    const layers = [];
    for (let l = 0; l < sizes.length - 1; l++) {
      const nin = sizes[l], nout = sizes[l + 1];
      const W = new Float64Array(nin * nout);
      const scale = Math.sqrt(2 / (nin + nout));
      for (let i = 0; i < W.length; i++) W[i] = scale * DL.utils.randn(rng);
      layers.push({
        nin, nout, W, b: new Float64Array(nout),
        gW: new Float64Array(nin * nout), gb: new Float64Array(nout),
        mW: new Float64Array(nin * nout), vW: new Float64Array(nin * nout),
        mb: new Float64Array(nout), vb: new Float64Array(nout),
      });
    }
    return { layers, t: 0 };
  }

  /* Devolve a lista de ativações; acts[acts.length - 1] é a saída. */
  function forward(net, x) {
    const acts = [x];
    let a = x;
    const L = net.layers.length;
    for (let l = 0; l < L; l++) {
      const ly = net.layers[l];
      const out = new Float64Array(ly.nout);
      for (let j = 0; j < ly.nout; j++) {
        let s = ly.b[j];
        const off = j * ly.nin;
        for (let i = 0; i < ly.nin; i++) s += ly.W[off + i] * a[i];
        out[j] = l < L - 1 ? Math.tanh(s) : s;
      }
      acts.push(out);
      a = out;
    }
    return acts;
  }

  /* Acumula gradientes em gW/gb e devolve dL/d(entrada). */
  function backward(net, acts, dout) {
    const L = net.layers.length;
    let d = dout;
    for (let l = L - 1; l >= 0; l--) {
      const ly = net.layers[l];
      const aIn = acts[l], aOut = acts[l + 1];
      let dcur = d;
      if (l < L - 1) {
        dcur = new Float64Array(ly.nout);
        for (let j = 0; j < ly.nout; j++) dcur[j] = d[j] * (1 - aOut[j] * aOut[j]);
      }
      const din = new Float64Array(ly.nin);
      for (let j = 0; j < ly.nout; j++) {
        ly.gb[j] += dcur[j];
        const off = j * ly.nin;
        for (let i = 0; i < ly.nin; i++) {
          ly.gW[off + i] += dcur[j] * aIn[i];
          din[i] += ly.W[off + i] * dcur[j];
        }
      }
      d = din;
    }
    return d;
  }

  function zeroGrads(net) {
    for (const ly of net.layers) { ly.gW.fill(0); ly.gb.fill(0); }
  }

  function adamStep(net, lr, scale) {
    net.t++;
    const b1 = 0.9, b2 = 0.999, eps = 1e-8;
    const c1 = 1 - Math.pow(b1, net.t), c2 = 1 - Math.pow(b2, net.t);
    const upd = (w, g, m, v) => {
      for (let i = 0; i < w.length; i++) {
        const gi = g[i] * scale;
        m[i] = b1 * m[i] + (1 - b1) * gi;
        v[i] = b2 * v[i] + (1 - b2) * gi * gi;
        w[i] -= lr * (m[i] / c1) / (Math.sqrt(v[i] / c2) + eps);
      }
    };
    for (const ly of net.layers) {
      upd(ly.W, ly.gW, ly.mW, ly.vW);
      upd(ly.b, ly.gb, ly.mb, ly.vb);
    }
  }

  /* Um passo de treino do AE clássico: J = média de ||x − f(g(x,φ),θ)||².
     latentL2 é um decaimento leve sobre z só para manter a escala do
     latente comparável entre execuções (não muda a estrutura). */
  function aeStep(enc, dec, data, batchSize, lr, rng, latentL2) {
    latentL2 = latentL2 || 0;
    zeroGrads(enc); zeroGrads(dec);
    let loss = 0;
    for (let b = 0; b < batchSize; b++) {
      const x = data[(rng() * data.length) | 0];
      const ea = forward(enc, x);
      const z = ea[ea.length - 1];
      const da = forward(dec, z);
      const xh = da[da.length - 1];
      const dout = new Float64Array(x.length);
      for (let i = 0; i < x.length; i++) {
        const e = xh[i] - x[i];
        loss += e * e;
        dout[i] = 2 * e;
      }
      const dz = backward(dec, da, dout);
      if (latentL2 > 0) for (let i = 0; i < z.length; i++) dz[i] += 2 * latentL2 * z[i];
      backward(enc, ea, dz);
    }
    adamStep(dec, lr, 1 / batchSize);
    adamStep(enc, lr, 1 / batchSize);
    return loss / batchSize;
  }

  /* Um passo de treino do VAE (latente 1D): o encoder produz [μ, log σ²],
     amostra z = μ + σ·ε (reparametrização) e o decoder reconstrói.
     Perda por instância: reconW·||x − f(z,θ)||² + β·KL[q(z|x,φ) || N(0,1)]. */
  function vaeStep(enc, dec, data, batchSize, lr, beta, reconW, rng) {
    zeroGrads(enc); zeroGrads(dec);
    let recon = 0, kl = 0;
    for (let b = 0; b < batchSize; b++) {
      const x = data[(rng() * data.length) | 0];
      const ea = forward(enc, x);
      const out = ea[ea.length - 1];
      const mu = out[0];
      const clamped = out[1] < -8 || out[1] > 4;
      const logv = Math.max(-8, Math.min(4, out[1]));
      const sig = Math.exp(0.5 * logv);
      const eps = DL.utils.randn(rng);
      const z = mu + sig * eps;
      const da = forward(dec, [z]);
      const xh = da[da.length - 1];
      const dout = new Float64Array(x.length);
      for (let i = 0; i < x.length; i++) {
        const e = xh[i] - x[i];
        recon += reconW * e * e;
        dout[i] = reconW * 2 * e;
      }
      const dz = backward(dec, da, dout)[0];
      kl += 0.5 * (Math.exp(logv) + mu * mu - 1 - logv);
      const dEnc = new Float64Array(2);
      dEnc[0] = dz + beta * mu;
      dEnc[1] = clamped ? 0 : dz * eps * sig * 0.5 + beta * 0.5 * (Math.exp(logv) - 1);
      backward(enc, ea, dEnc);
    }
    adamStep(dec, lr, 1 / batchSize);
    adamStep(enc, lr, 1 / batchSize);
    return { recon: recon / batchSize, kl: kl / batchSize };
  }

  /* Encoder do VAE em um ponto: devolve { mu, logv, sig }. */
  function vaeEncode(enc, x) {
    const out = forward(enc, x);
    const o = out[out.length - 1];
    const logv = Math.max(-8, Math.min(4, o[1]));
    return { mu: o[0], logv, sig: Math.exp(0.5 * logv) };
  }

  function decode(dec, z) {
    const a = forward(dec, z);
    return a[a.length - 1];
  }

  function encode(enc, x) {
    const a = forward(enc, x);
    return a[a.length - 1];
  }

  /* Dataset padrão da demo: two moons padronizado (variedade 1D em 2D). */
  function makeData(n, seed) {
    const rng = DL.utils.mulberry32(seed);
    return DL.utils.twoMoons(n, 0.07, rng).pts;
  }

  DL.model = { mlp, forward, zeroGrads, adamStep, aeStep, vaeStep, vaeEncode, encode, decode, makeData };
})();
