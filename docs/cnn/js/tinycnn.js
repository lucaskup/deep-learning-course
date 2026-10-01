/* Núcleo numérico da CNN em miniatura da seção 6: dataset sintético,
   inicialização, forward com cache, backprop manual e passo de SGD.
   Sem dependência de DOM, para poder ser testado fora do navegador.

   Arquitetura (N = 6, C = 2 canais, K = 3, F1 = F2 = 2 filtros):
     x[C][N][N]
       → conv 3×3, pad 1, F1 filtros, + b, ReLU → a1[F1][N][N]
       → max pooling 2×2                        → p1[F1][N/2][N/2]
       → conv 3×3, pad 1, F2 filtros, + b, ReLU → a2[F2][N/2][N/2]
       → flatten (F2·(N/2)²)                     → v[18]
       → w·v + b                                 → logit
       → σ(logit)                                → prob(classe 1)
   Classe 0: traço vertical. Classe 1: traço horizontal. */
(function () {
  'use strict';
  window.DL = window.DL || {};

  const N = 6, C = 2, K = 3, F1 = 2, F2 = 2, M = N / 2, D = F2 * M * M;

  function zeros2(n) {
    const g = [];
    for (let i = 0; i < n; i++) g.push(new Array(n).fill(0));
    return g;
  }
  function zeros3(a, n) {
    const g = [];
    for (let i = 0; i < a; i++) g.push(zeros2(n));
    return g;
  }
  function zeros4(a, b) {
    const g = [];
    for (let i = 0; i < a; i++) {
      const row = [];
      for (let j = 0; j < b; j++) row.push(zeros2(K));
      g.push(row);
    }
    return g;
  }

  /* ── Dataset ─────────────────────────────────────────────────────── */

  /* Um exemplo: traço de comprimento 4 em posição aleatória, canal 0 mais
     intenso que o canal 1 (duas "cores"), fundo com ruído fraco. */
  function makeExample(rng, cls) {
    const X = zeros3(C, N);
    for (let c = 0; c < C; c++) {
      for (let i = 0; i < N; i++) {
        for (let j = 0; j < N; j++) X[c][i][j] = Math.round(rng() * 8) / 100;
      }
    }
    const u0 = 0.6 + 0.4 * rng();
    const u1 = 0.2 + 0.4 * rng();
    const start = Math.floor(rng() * (N - 3));
    const fixed = Math.floor(rng() * N);
    for (let t = 0; t < 4; t++) {
      const i = cls === 0 ? start + t : fixed;
      const j = cls === 0 ? fixed : start + t;
      X[0][i][j] = Math.round(u0 * 100) / 100;
      X[1][i][j] = Math.round(u1 * 100) / 100;
    }
    return { X, y: cls };
  }

  function makeDataset(n, rng) {
    const out = [];
    for (let k = 0; k < n; k++) out.push(makeExample(rng, k % 2));
    return out;
  }

  /* ── Modelo ──────────────────────────────────────────────────────── */

  function initModel(rng) {
    const rn = () => DL.utils.randn(rng);
    const m = {
      W1: zeros4(F1, C), b1: new Array(F1).fill(0),
      W2: zeros4(F2, F1), b2: new Array(F2).fill(0),
      wfc: new Array(D).fill(0), bfc: 0,
    };
    const s1 = Math.sqrt(2 / (C * K * K)), s2 = Math.sqrt(2 / (F1 * K * K)), s3 = Math.sqrt(1 / D);
    for (let f = 0; f < F1; f++) for (let c = 0; c < C; c++) for (let a = 0; a < K; a++) for (let b = 0; b < K; b++) m.W1[f][c][a][b] = s1 * rn();
    for (let f = 0; f < F2; f++) for (let c = 0; c < F1; c++) for (let a = 0; a < K; a++) for (let b = 0; b < K; b++) m.W2[f][c][a][b] = s2 * rn();
    for (let k = 0; k < D; k++) m.wfc[k] = s3 * rn();
    return m;
  }

  function zeroGrads() {
    return {
      W1: zeros4(F1, C), b1: new Array(F1).fill(0),
      W2: zeros4(F2, F1), b2: new Array(F2).fill(0),
      wfc: new Array(D).fill(0), bfc: 0,
    };
  }

  /* ── Forward ─────────────────────────────────────────────────────── */

  /* Convolução "same" com padding 1 sobre todos os canais de entrada. */
  function conv(input, W, b) {
    const F = W.length, Cin = input.length, n = input[0].length;
    const z = zeros3(F, n);
    for (let f = 0; f < F; f++) {
      for (let i = 0; i < n; i++) {
        for (let j = 0; j < n; j++) {
          let s = b[f];
          for (let c = 0; c < Cin; c++) {
            for (let a = 0; a < K; a++) {
              const ii = i + a - 1;
              if (ii < 0 || ii >= n) continue;
              for (let q = 0; q < K; q++) {
                const jj = j + q - 1;
                if (jj < 0 || jj >= n) continue;
                s += W[f][c][a][q] * input[c][ii][jj];
              }
            }
          }
          z[f][i][j] = s;
        }
      }
    }
    return z;
  }

  /* Lista de termos w·x de uma posição (para o leitor de cálculo). */
  function convTerms(input, W, f, i, j) {
    const Cin = input.length, n = input[0].length;
    const terms = [];
    for (let c = 0; c < Cin; c++) {
      for (let a = 0; a < K; a++) {
        for (let q = 0; q < K; q++) {
          const ii = i + a - 1, jj = j + q - 1;
          const inside = ii >= 0 && ii < n && jj >= 0 && jj < n;
          terms.push({ c, a, q, ii, jj, inside, w: W[f][c][a][q], x: inside ? input[c][ii][jj] : 0 });
        }
      }
    }
    return terms;
  }

  function relu3(z) {
    return z.map((g) => g.map((row) => row.map((v) => (v > 0 ? v : 0))));
  }

  /* Max pooling 2×2 com registro do argmax para o backward. */
  function maxpool(a) {
    const F = a.length;
    const p = zeros3(F, M), arg = [];
    for (let f = 0; f < F; f++) {
      const rows = [];
      for (let i = 0; i < M; i++) {
        const row = [];
        for (let j = 0; j < M; j++) {
          let best = -Infinity, bi = 0, bj = 0;
          for (let di = 0; di < 2; di++) {
            for (let dj = 0; dj < 2; dj++) {
              const v = a[f][2 * i + di][2 * j + dj];
              if (v > best) { best = v; bi = 2 * i + di; bj = 2 * j + dj; }
            }
          }
          p[f][i][j] = best;
          row.push([bi, bj]);
        }
        rows.push(row);
      }
      arg.push(rows);
    }
    return { p, arg };
  }

  function flatten(a2) {
    const v = new Array(D);
    for (let f = 0; f < F2; f++) for (let i = 0; i < M; i++) for (let j = 0; j < M; j++) v[f * M * M + i * M + j] = a2[f][i][j];
    return v;
  }

  function sigmoid(t) { return 1 / (1 + Math.exp(-t)); }

  function forward(m, X) {
    const z1 = conv(X, m.W1, m.b1);
    const a1 = relu3(z1);
    const { p: p1, arg } = maxpool(a1);
    const z2 = conv(p1, m.W2, m.b2);
    const a2 = relu3(z2);
    const v = flatten(a2);
    let logit = m.bfc;
    for (let k = 0; k < D; k++) logit += m.wfc[k] * v[k];
    return { X, z1, a1, p1, arg, z2, a2, v, logit, prob: sigmoid(logit) };
  }

  /* Entropia cruzada binária a partir do logit (numericamente estável). */
  function bceLoss(logit, y) {
    return Math.max(logit, 0) - logit * y + Math.log(1 + Math.exp(-Math.abs(logit)));
  }

  /* ── Backward ────────────────────────────────────────────────────── */

  /* Gradiente da convolução: acumula dW, db e devolve dInput. */
  function convBackward(input, W, dz, gW, gb) {
    const F = W.length, Cin = input.length, n = input[0].length;
    const dIn = zeros3(Cin, n);
    for (let f = 0; f < F; f++) {
      for (let i = 0; i < n; i++) {
        for (let j = 0; j < n; j++) {
          const g = dz[f][i][j];
          if (g === 0) continue;
          gb[f] += g;
          for (let c = 0; c < Cin; c++) {
            for (let a = 0; a < K; a++) {
              const ii = i + a - 1;
              if (ii < 0 || ii >= n) continue;
              for (let q = 0; q < K; q++) {
                const jj = j + q - 1;
                if (jj < 0 || jj >= n) continue;
                gW[f][c][a][q] += g * input[c][ii][jj];
                dIn[c][ii][jj] += g * W[f][c][a][q];
              }
            }
          }
        }
      }
    }
    return dIn;
  }

  /* Acumula em `grads` o gradiente da perda de um exemplo. Devolve a perda. */
  function backward(m, cache, y, grads) {
    const dlogit = cache.prob - y;
    grads.bfc += dlogit;
    const dv = new Array(D);
    for (let k = 0; k < D; k++) {
      grads.wfc[k] += dlogit * cache.v[k];
      dv[k] = dlogit * m.wfc[k];
    }
    const dz2 = zeros3(F2, M);
    for (let f = 0; f < F2; f++) for (let i = 0; i < M; i++) for (let j = 0; j < M; j++) {
      dz2[f][i][j] = cache.z2[f][i][j] > 0 ? dv[f * M * M + i * M + j] : 0;
    }
    const dp1 = convBackward(cache.p1, m.W2, dz2, grads.W2, grads.b2);
    const dz1 = zeros3(F1, N);
    for (let f = 0; f < F1; f++) for (let i = 0; i < M; i++) for (let j = 0; j < M; j++) {
      const [bi, bj] = cache.arg[f][i][j];
      if (cache.z1[f][bi][bj] > 0) dz1[f][bi][bj] += dp1[f][i][j];
    }
    convBackward(cache.X, m.W1, dz1, grads.W1, grads.b1);
    return bceLoss(cache.logit, y);
  }

  /* Um passo de SGD sobre um minibatch; devolve a perda média. */
  function sgdStep(m, batch, lr) {
    const g = zeroGrads();
    let loss = 0;
    for (const ex of batch) loss += backward(m, forward(m, ex.X), ex.y, g);
    const s = lr / batch.length;
    for (let f = 0; f < F1; f++) {
      m.b1[f] -= s * g.b1[f];
      for (let c = 0; c < C; c++) for (let a = 0; a < K; a++) for (let b = 0; b < K; b++) m.W1[f][c][a][b] -= s * g.W1[f][c][a][b];
    }
    for (let f = 0; f < F2; f++) {
      m.b2[f] -= s * g.b2[f];
      for (let c = 0; c < F1; c++) for (let a = 0; a < K; a++) for (let b = 0; b < K; b++) m.W2[f][c][a][b] -= s * g.W2[f][c][a][b];
    }
    for (let k = 0; k < D; k++) m.wfc[k] -= s * g.wfc[k];
    m.bfc -= s * g.bfc;
    return loss / batch.length;
  }

  function accuracy(m, data) {
    let ok = 0;
    for (const ex of data) if ((forward(m, ex.X).prob > 0.5 ? 1 : 0) === ex.y) ok++;
    return ok / data.length;
  }

  function cloneModel(m) { return JSON.parse(JSON.stringify(m)); }

  DL.tinycnn = {
    N, C, K, F1, F2, M, D,
    makeExample, makeDataset, initModel, zeroGrads, zeros3,
    conv, convTerms, forward, backward, sgdStep, bceLoss, accuracy, cloneModel,
  };
})();
