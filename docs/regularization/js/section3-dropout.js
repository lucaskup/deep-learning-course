/* Seção 3: dropout visual em uma rede 2-6-6-1 com máscaras Bernoulli(1−p). */
(function () {
  'use strict';
  window.DL = window.DL || {};
  DL.sections = DL.sections || [];

  const LAYERS = [2, 6, 6, 1];
  const TOTAL_W = 2 * 6 + 6 * 6 + 6 * 1;   /* 54 conexões (sem contar os vieses) */
  const MASK_INTERVAL = 0.35;              /* segundos entre máscaras no modo animado */

  function init() {
    const U = DL.utils, P = DL.plot;
    const $ = (id) => document.getElementById(id);

    const cv = $('s3-net');
    const sliderP = $('s3-p');
    const btnMask = $('s3-mask'), btnPlay = $('s3-play');

    let seed = 11;
    /* Máscara apenas das camadas ocultas; entrada e saída nunca são desligadas. */
    let mask = [new Array(LAYERS[1]).fill(true), new Array(LAYERS[2]).fill(true)];
    let playing = false, acc = 0;

    function sampleMask() {
      seed = (seed * 1664525 + 1013904223) >>> 0;
      const rng = U.mulberry32(seed);
      const p = +sliderP.value;
      for (let l = 0; l < 2; l++) {
        for (let j = 0; j < mask[l].length; j++) mask[l][j] = rng() >= p;   /* m ∼ Bernoulli(1−p) */
      }
    }

    function activeOf(l) {
      /* l = índice de camada da rede (0..3); retorna vetor de atividade. */
      if (l === 0) return new Array(LAYERS[0]).fill(true);
      if (l === 3) return new Array(LAYERS[3]).fill(true);
      return mask[l - 1];
    }

    function countActive() {
      let total = 0;
      for (let l = 0; l < LAYERS.length - 1; l++) {
        const a = activeOf(l).filter(Boolean).length;
        const b = activeOf(l + 1).filter(Boolean).length;
        total += a * b;
      }
      return total;
    }

    function updateReadouts() {
      const p = +sliderP.value;
      $('s3-p-val').textContent = p.toFixed(2);
      $('s3-active').textContent = 'conexões ativas: ' + countActive() + ' / ' + TOTAL_W;
      $('s3-scale').textContent = p < 1
        ? 'escala em treino: 1/(1−p) = ' + (1 / (1 - p)).toFixed(2)
        : '';
    }

    function draw() {
      const th = P.theme();
      const { ctx, w, h } = P.setup(cv);
      P.clear(ctx, w, h);

      const padX = 70, padTop = 26, padBot = 34;
      const names = ['entrada', 'oculta 1', 'oculta 2', 'saída'];
      const pos = [];   /* pos[l][j] = [px, py] */
      for (let l = 0; l < LAYERS.length; l++) {
        const x = padX + (w - 2 * padX) * (l / (LAYERS.length - 1));
        const col = [];
        const n = LAYERS[l];
        for (let j = 0; j < n; j++) {
          const y = n === 1
            ? padTop + (h - padTop - padBot) / 2
            : padTop + (h - padTop - padBot) * (j / (n - 1));
          col.push([x, y]);
        }
        pos.push(col);
      }

      /* Conexões: só entre unidades ativas nas duas pontas. */
      ctx.lineWidth = 1;
      ctx.strokeStyle = th.comment;
      ctx.globalAlpha = 0.45;
      for (let l = 0; l < LAYERS.length - 1; l++) {
        const actA = activeOf(l), actB = activeOf(l + 1);
        for (let i = 0; i < LAYERS[l]; i++) {
          if (!actA[i]) continue;
          for (let j = 0; j < LAYERS[l + 1]; j++) {
            if (!actB[j]) continue;
            ctx.beginPath();
            ctx.moveTo(pos[l][i][0], pos[l][i][1]);
            ctx.lineTo(pos[l + 1][j][0], pos[l + 1][j][1]);
            ctx.stroke();
          }
        }
      }
      ctx.globalAlpha = 1;

      /* Unidades. */
      const fill = [th.cyan, th.purple, th.purple, th.orange];
      const R = 11;
      for (let l = 0; l < LAYERS.length; l++) {
        const act = activeOf(l);
        for (let j = 0; j < LAYERS[l]; j++) {
          const [x, y] = pos[l][j];
          ctx.beginPath();
          ctx.arc(x, y, R, 0, 2 * Math.PI);
          if (act[j]) {
            ctx.fillStyle = fill[l];
            ctx.globalAlpha = 0.85;
            ctx.fill();
            ctx.globalAlpha = 1;
            ctx.strokeStyle = th.line;
            ctx.setLineDash([]);
            ctx.lineWidth = 1;
            ctx.stroke();
          } else {
            ctx.strokeStyle = th.comment;
            ctx.setLineDash([3, 3]);
            ctx.lineWidth = 1.4;
            ctx.stroke();
            ctx.setLineDash([]);
            /* Traço de "desligada" sobre a unidade. */
            ctx.beginPath();
            ctx.moveTo(x - R * 0.55, y + R * 0.55);
            ctx.lineTo(x + R * 0.55, y - R * 0.55);
            ctx.stroke();
          }
        }
      }

      /* Rótulos das camadas. */
      for (let l = 0; l < LAYERS.length; l++) {
        P.label(ctx, pos[l][0][0], h - 10, names[l], th.comment, 'center');
      }
      P.label(ctx, 10, 16, 'm ∼ Bernoulli(1−p), unidades ocultas', th.comment);
    }

    function refresh() { updateReadouts(); draw(); }

    function tick(dt) {
      acc += dt;
      if (acc >= MASK_INTERVAL) {
        acc = 0;
        sampleMask();
        refresh();
      }
    }

    function stopPlay() {
      playing = false;
      btnPlay.textContent = '▶ Animar máscaras';
      DL.stopTicker(tick);
    }

    btnPlay.addEventListener('click', () => {
      if (playing) { stopPlay(); return; }
      playing = true;
      acc = MASK_INTERVAL;   /* primeira máscara imediata */
      btnPlay.textContent = '⏸ Pausar';
      DL.startTicker(tick);
    });
    btnMask.addEventListener('click', () => { stopPlay(); sampleMask(); refresh(); });
    sliderP.addEventListener('input', () => { sampleMask(); refresh(); });

    sampleMask();
    refresh();
    P.onRedraw(draw);
    P.observeResize(cv, draw);
  }

  DL.sections.push({ name: 's3-dropout', init });
})();
