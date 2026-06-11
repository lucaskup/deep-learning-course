/* Seção 2: Masked Language Modeling com a regra 80/10/10. Um modelo de
   coocorrência posicional treinado num corpus pt-BR faz o papel do BERT:
   p(w | contexto) na posição mascarada, perda −log p(palavra original). */
(function () {
  'use strict';
  window.DL = window.DL || {};
  DL.sections = DL.sections || [];

  const CORPUS = [
    'o gato sentou no tapete',
    'o gato dorme no tapete',
    'o gato preto sentou no sofá',
    'o gato subiu no sofá',
    'o gato preto dorme no sofá',
    'o cachorro sentou no tapete',
    'o cachorro dorme no quintal',
    'o cachorro late no quintal',
    'ele sentou no sofá da sala',
    'ela sentou no sofá da sala',
    'ela sentou no banco do parque',
    'ele sentou no banco da praça',
    'ele sacou dinheiro no banco',
    'ela sacou dinheiro no banco',
    'ele sacou dinheiro no banco da esquina',
    'ela depositou dinheiro no banco',
    'ele abriu uma conta no banco',
    'ela abriu uma conta no banco da esquina',
    'o banco abriu uma agência nova',
    'ele guardou dinheiro na conta',
    'paris é a capital da frança',
    'a capital da frança é paris',
    'berlim é a capital da alemanha',
    'a capital da alemanha é berlim',
    'lisboa é a capital de portugal',
    'a capital de portugal é lisboa',
    'brasília é a capital do brasil',
    'a capital do brasil é brasília',
  ].map((s) => s.split(' '));

  const TARGETS = [
    'o gato sentou no tapete',
    'ela sacou dinheiro no banco',
    'a capital da frança é paris',
  ].map((s) => s.split(' '));

  const WINDOW = 3; /* alcance de coocorrência em posições relativas */

  function init() {
    const P = DL.plot;
    const U = DL.utils;
    const $ = (id) => document.getElementById(id);

    const cvSent = $('s2-sent'), cvDist = $('s2-dist');
    const slRate = $('s2-rate');

    /* ── Modelo: contagens de coocorrência (palavra, contexto, offset) ── */
    const VOCAB = [];
    const counts = new Map();
    (function build() {
      const seen = new Set();
      for (const sent of CORPUS) {
        for (const w of sent) {
          if (!seen.has(w)) { seen.add(w); VOCAB.push(w); }
        }
        for (let i = 0; i < sent.length; i++) {
          for (let j = 0; j < sent.length; j++) {
            const d = j - i;
            if (d === 0 || Math.abs(d) > WINDOW) continue;
            const key = sent[i] + '|' + sent[j] + '|' + d;
            counts.set(key, (counts.get(key) || 0) + 1);
          }
        }
      }
    })();

    function cooc(w, c, d) {
      return counts.get(w + '|' + c + '|' + d) || 0;
    }

    /* Distribuição p(w | contexto visível) na posição i.
       Tokens em estado 'mask' são invisíveis; 'rand' expõe a palavra trocada. */
    function predict(i) {
      const scores = VOCAB.map((w) => {
        let s = 0;
        for (let j = 0; j < sent.length; j++) {
          const d = j - i;
          if (d === 0 || Math.abs(d) > WINDOW) continue;
          if (state[j] === 'mask') continue;
          const ctx = state[j] === 'rand' ? inputTok[j] : sent[j];
          s += (1 / Math.abs(d)) * Math.log(1 + cooc(w, ctx, d));
        }
        return 2 * s;
      });
      const mx = Math.max(...scores);
      const exps = scores.map((s) => Math.exp(s - mx));
      const z = exps.reduce((a, b) => a + b, 0);
      return exps.map((e) => e / z);
    }

    /* ── Estado da frase ── */
    let sentIdx = 0;
    let sent = TARGETS[0];
    let state = [];     /* 'visible' | 'mask' | 'rand' | 'kept' por posição */
    let inputTok = [];  /* token de entrada efetivo (após corrupção) */
    let sel = -1;       /* posição prevista exibida no gráfico */
    let seed = 11;
    let chips = [];     /* retângulos para hit-test */

    function resetSentence() {
      sent = TARGETS[sentIdx];
      state = sent.map(() => 'visible');
      inputTok = sent.slice();
      sel = -1;
    }

    function predictedPositions() {
      const out = [];
      for (let i = 0; i < sent.length; i++) if (state[i] !== 'visible') out.push(i);
      return out;
    }

    function applyRandomMask() {
      seed = (seed + 1) >>> 0;
      const rng = U.mulberry32(seed);
      const rate = +slRate.value / 100;
      resetSentence();
      const chosen = [];
      for (let i = 0; i < sent.length; i++) if (rng() < rate) chosen.push(i);
      if (chosen.length === 0) chosen.push(Math.floor(rng() * sent.length));
      for (const i of chosen) {
        const u = rng();
        if (u < 0.8) {
          state[i] = 'mask';
          inputTok[i] = '[MASK]';
        } else if (u < 0.9) {
          state[i] = 'rand';
          let w = sent[i];
          while (w === sent[i]) w = VOCAB[Math.floor(rng() * VOCAB.length)];
          inputTok[i] = w;
        } else {
          state[i] = 'kept';
          inputTok[i] = sent[i];
        }
      }
      sel = chosen[0];
    }

    /* ── Desenho da frase como chips clicáveis ── */
    function drawSentence() {
      const th = P.theme();
      const { ctx, w, h } = P.setup(cvSent);
      P.clear(ctx, w, h);
      chips = [];

      ctx.font = '13px sans-serif';
      const padX = 12, gap = 8, chipH = 32;
      const widths = inputTok.map((t) => ctx.measureText(t).width + 2 * padX);
      const total = widths.reduce((a, b) => a + b, 0) + gap * (sent.length - 1);
      let x = Math.max(8, (w - total) / 2);
      const y = h / 2 - chipH / 2 - 6;

      for (let i = 0; i < sent.length; i++) {
        const cw = widths[i];
        const st = state[i];
        let fill = th.card, stroke = th.line, txt = th.fg;
        if (st === 'mask') { fill = 'rgba(255,85,85,0.18)'; stroke = th.red; }
        if (st === 'rand') { fill = 'rgba(255,184,108,0.18)'; stroke = th.orange; }
        if (st === 'kept') { fill = 'rgba(80,250,123,0.12)'; stroke = th.green; }
        ctx.fillStyle = fill;
        ctx.strokeStyle = stroke;
        ctx.lineWidth = i === sel && st !== 'visible' ? 2.5 : 1.2;
        ctx.beginPath();
        ctx.roundRect(x, y, cw, chipH, 7);
        ctx.fill();
        ctx.stroke();
        ctx.fillStyle = txt;
        ctx.textAlign = 'center';
        ctx.font = '13px sans-serif';
        ctx.fillText(inputTok[i], x + cw / 2, y + chipH / 2 + 4);
        /* Alvo original sob o chip nas posições previstas. */
        if (st !== 'visible') {
          ctx.fillStyle = th.comment;
          ctx.font = '10px sans-serif';
          ctx.fillText('alvo: ' + sent[i], x + cw / 2, y + chipH + 14);
        }
        chips.push({ x, y, w: cw, h: chipH, i });
        x += cw + gap;
      }

      /* Legenda dos três estados da regra 80/10/10. */
      ctx.font = '10px sans-serif';
      ctx.textAlign = 'left';
      const ly = h - 8;
      ctx.fillStyle = th.red;
      ctx.fillText('■ [MASK] (80%)', 10, ly);
      ctx.fillStyle = th.orange;
      ctx.fillText('■ troca aleatória (10%)', 110, ly);
      ctx.fillStyle = th.green;
      ctx.fillText('■ mantido (10%)', 262, ly);
    }

    /* ── Distribuição prevista na posição selecionada ── */
    function drawDist() {
      const th = P.theme();
      const { ctx, w, h } = P.setup(cvDist);
      P.clear(ctx, w, h);

      if (sel < 0 || state[sel] === 'visible') {
        ctx.fillStyle = th.comment;
        ctx.font = '12px sans-serif';
        ctx.textAlign = 'center';
        ctx.fillText('clique numa palavra para mascará-la', w / 2, h / 2);
        return;
      }

      const p = predict(sel);
      const order = p.map((v, k) => k).sort((a, b) => p[b] - p[a]);
      const top = order.slice(0, 8);
      const origIdx = VOCAB.indexOf(sent[sel]);
      if (!top.includes(origIdx)) top.push(origIdx);

      const n = top.length;
      const pmax = Math.max(p[top[0]], 0.05);
      const fr = P.frame(ctx, w, h, 0, n, 0, pmax * 1.15, { l: 38, r: 10, t: 12, b: 42 });
      P.axes(ctx, fr, { ylabel: 'p(w)', yticks: [0, +(pmax / 2).toFixed(2), +pmax.toFixed(2)] });

      const bw = fr.iw / n;
      for (let k = 0; k < n; k++) {
        const idx = top[k];
        const x0 = fr.X(k) + bw * 0.12;
        const y1 = fr.Y(p[idx]);
        ctx.fillStyle = idx === origIdx ? th.green : th.cyan;
        ctx.globalAlpha = 0.85;
        ctx.fillRect(x0, y1, bw * 0.76, fr.Y(0) - y1);
        ctx.globalAlpha = 1;
        /* valor sobre a barra */
        ctx.fillStyle = th.comment;
        ctx.font = '9px sans-serif';
        ctx.textAlign = 'center';
        ctx.fillText(p[idx].toFixed(2), x0 + bw * 0.38, y1 - 3);
        /* rótulo do token, rotacionado */
        ctx.save();
        ctx.translate(x0 + bw * 0.38, fr.Y(0) + 10);
        ctx.rotate(-Math.PI / 5);
        ctx.fillStyle = idx === origIdx ? th.green : th.comment;
        ctx.textAlign = 'right';
        ctx.fillText(VOCAB[idx], 0, 6);
        ctx.restore();
      }
    }

    function updateReadout() {
      const pos = predictedPositions();
      if (pos.length === 0) {
        $('s2-readout').textContent = '|M| = 0 · mascare ao menos um token';
        return;
      }
      let loss = 0;
      for (const i of pos) {
        const p = predict(i);
        loss += -Math.log(Math.max(p[VOCAB.indexOf(sent[i])], 1e-9));
      }
      loss /= pos.length;
      $('s2-readout').textContent =
        '|M| = ' + pos.length + ' · ℒ MLM = ' + loss.toFixed(2);
    }

    function redraw() {
      drawSentence();
      drawDist();
      updateReadout();
    }

    /* ── Interações ── */
    cvSent.addEventListener('pointerdown', (e) => {
      const r = cvSent.getBoundingClientRect();
      const px = e.clientX - r.left, py = e.clientY - r.top;
      for (const c of chips) {
        if (px >= c.x && px <= c.x + c.w && py >= c.y && py <= c.y + c.h) {
          const i = c.i;
          if (state[i] === 'visible') {
            state[i] = 'mask';
            inputTok[i] = '[MASK]';
            sel = i;
          } else {
            state[i] = 'visible';
            inputTok[i] = sent[i];
            if (sel === i) {
              const pos = predictedPositions();
              sel = pos.length ? pos[0] : -1;
            }
          }
          redraw();
          return;
        }
      }
    });

    slRate.addEventListener('input', () => {
      $('s2-rate-val').textContent = slRate.value + '%';
    });
    $('s2-sample').addEventListener('click', () => {
      applyRandomMask();
      redraw();
    });
    $('s2-clear').addEventListener('click', () => {
      resetSentence();
      redraw();
    });

    const tabs = [$('s2-tab-0'), $('s2-tab-1'), $('s2-tab-2')];
    tabs.forEach((tab, k) => {
      tab.addEventListener('click', () => {
        tabs.forEach((t) => t.classList.remove('active'));
        tab.classList.add('active');
        sentIdx = k;
        resetSentence();
        redraw();
      });
    });

    resetSentence();
    /* Começa com o verbo mascarado para a demo abrir com conteúdo. */
    state[2] = 'mask';
    inputTok[2] = '[MASK]';
    sel = 2;
    redraw();
    P.onRedraw(redraw);
    P.observeResize(cvSent, redraw);
    P.observeResize(cvDist, redraw);
  }

  DL.sections.push({ name: 's2-mlm', init });
})();
