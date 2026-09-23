#!/usr/bin/env node
/* Gera os histogramas de ativações do estudo de caso da aula de inicialização de
   pesos (slides/01-mlp/07-inicializacao_de_pesos.tex) rodando no node o mesmo
   código da demo interativa (docs/weight-init/js/winit-core.js).

   Configuração: batch de 100 imagens do MNIST, 4 camadas ocultas tanh de 300
   unidades, semente 42 (a mesma da demo). Saída em slides/data/01-mlp/:
     init-hist-<esquema>.csv   bordas dos bins (x) e densidade de A^(1..4)
     init-hist-stats.tex       desvio padrão por camada e ymax por esquema

   Uso (a partir da raiz do repositório): node scripts/export_init_histograms.js */
'use strict';
const fs = require('fs');
const path = require('path');
const vm = require('vm');

const ROOT = path.resolve(__dirname, '..');
const OUT = path.join(ROOT, 'slides', 'data', '01-mlp');

/* Carrega os scripts do browser num contexto em que window = globalThis. */
const ctx = { console, Math, Float32Array, Float64Array, atob };
ctx.window = ctx;
ctx.globalThis = ctx;
vm.createContext(ctx);
for (const f of ['docs/shared/js/utils.js', 'docs/weight-init/js/mnist-batch.js',
                 'docs/weight-init/js/winit-core.js']) {
  vm.runInContext(fs.readFileSync(path.join(ROOT, f), 'utf8'), ctx, { filename: f });
}
const { simulate } = ctx.DL.winitCore;

const SCHEMES = [
  { name: 'zero', cfg: { init: 'zeros' } },
  { name: 'normal', cfg: { init: 'normal', sigma: 1 } },
  { name: 'uniform', cfg: { init: 'uniform' } },
  { name: 'xavier', cfg: { init: 'xavier' } },
  { name: 'he', cfg: { init: 'he' } },
];
const BASE = { act: 'tanh', L: 4, seed: 42, input: 'mnist', sigma: 1 };
const LO = -1.05, HI = 1.05, NBINS = 42;
const BW = (HI - LO) / NBINS;

fs.mkdirSync(OUT, { recursive: true });
const stats = [
  '% Gerado por scripts/export_init_histograms.js. Não editar à mão.',
];

for (const { name, cfg } of SCHEMES) {
  const d = simulate(Object.assign({}, BASE, cfg));
  const dens = [];
  let ymax = 0;
  for (let l = 1; l <= 4; l++) {
    const a = d.acts[l];
    const c = new Float64Array(NBINS);
    for (let i = 0; i < a.length; i++) {
      const b = Math.min(NBINS - 1, Math.max(0, Math.floor((a[i] - LO) / BW)));
      c[b]++;
    }
    for (let b = 0; b < NBINS; b++) {
      c[b] /= a.length * BW;
      if (c[b] > ymax) ymax = c[b];
    }
    dens.push(c);
    stats.push(`\\expandafter\\def\\csname inithiststd-${name}-${l}\\endcsname{${d.actStd[l].toFixed(2)}}`);
  }
  stats.push(`\\expandafter\\def\\csname inithistymax-${name}\\endcsname{${(1.08 * ymax).toFixed(3)}}`);

  /* formato "const plot": uma linha por borda esquerda, repetindo o último valor
     na borda direita final para fechar o degrau */
  const rows = ['x l1 l2 l3 l4'];
  for (let b = 0; b <= NBINS; b++) {
    const k = Math.min(b, NBINS - 1);
    rows.push([(LO + b * BW).toFixed(3), ...dens.map((c) => c[k].toFixed(4))].join(' '));
  }
  fs.writeFileSync(path.join(OUT, `init-hist-${name}.csv`), rows.join('\n') + '\n');
  console.log(name.padEnd(8), d.actStd.slice(1).map((s) => s.toFixed(3)).join(' '));
}
fs.writeFileSync(path.join(OUT, 'init-hist-stats.tex'), stats.join('\n') + '\n');
