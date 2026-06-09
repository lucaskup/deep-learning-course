/* Seção 1: playground de decoding com temperatura, top-k e top-p. */
(function () {
  'use strict';
  window.DM = window.DM || {};
  DM.sections = DM.sections || [];

  const MAX_TOKENS = 20, SHOW_BARS = 15;

  function init() {
    const M = DM.model, P = DM.plot, U = DM.utils;
    const $ = (id) => document.getElementById(id);
    const f = M.fmtBR;
    const canvas = $('s1-dist');

    let mode = 'sample';
    let tau = 1, topK = Infinity, topP = 1;
    let seed = 42;
    let rng = U.mulberry32(seed);
    let generated = [];      // {token, pRaw}
    let finished = false;
    let autoTimer = null;

    const ctxTokens = () => generated.map((g) => g.token);

    function currentFiltered() {
      const dist = M.nextDist(ctxTokens(), true);
      return M.filterDist(dist, { tau, topK, topP });
    }

    function drawStrip() {
      const strip = $('s1-strip');
      strip.innerHTML = '';
      const start = document.createElement('span');
      start.className = 'token special';
      start.textContent = '⟨início⟩';
      strip.appendChild(start);
      generated.forEach((g, i) => {
        const el = document.createElement('span');
        el.className = 'token generated' + (g.token === M.EOS ? ' special' : '');
        el.textContent = g.token;
        el.title = 'p = ' + f(g.pRaw, 3) + ' · clique para voltar até aqui';
        el.addEventListener('click', () => {
          stopAuto();
          generated = generated.slice(0, i + 1);
          finished = generated[generated.length - 1].token === M.EOS;
          redraw();
        });
        strip.appendChild(el);
      });
    }

    function drawDist() {
      const th = P.theme();
      const { ctx, w, h } = P.setup(canvas);
      P.clear(ctx, w, h);
      if (finished) {
        P.label(ctx, w / 2, h / 2, 'frase encerrada em <eos>, recomece para gerar outra', th.comment, 'center');
        return;
      }
      const filtered = currentFiltered().slice(0, SHOW_BARS);
      const ymax = Math.max(0.3, Math.max(...filtered.map((d) => d.pTau)) * 1.2);
      const fr = P.frame(ctx, w, h, 0, 1, 0, ymax, { l: 34, r: 8, t: 8, b: 30 });
      P.axes(ctx, fr, { yticks: [0, Math.round(ymax * 50) / 100] });
      const labels = filtered.map((d) => d.token.length > 9 ? d.token.slice(0, 8) + '…' : d.token);
      const colors = filtered.map((d) => d.excluded ? th.comment : th.purple);
      const values = filtered.map((d) => f(d.pTau, 2));
      P.bars(ctx, fr, labels, filtered.map((d) => d.pTau), colors, { values, labelSize: 9 });
    }

    function drawReadout() {
      if (generated.length === 0) {
        $('s1-readout').innerHTML = 'nenhum token gerado ainda';
        return;
      }
      const prod = generated.reduce((a, g) => a * g.pRaw, 1);
      const slog = generated.reduce((a, g) => a + Math.log(g.pRaw), 0);
      const parts = generated.map((g) => f(g.pRaw, 2)).join('·');
      $('s1-readout').innerHTML =
        `∏ P = ${parts} = ${prod.toExponential(2).replace('.', ',')} &nbsp;·&nbsp; Σ ln P = ${f(slog, 2)}`;
    }

    function redraw() { drawStrip(); drawDist(); drawReadout(); }

    function step() {
      if (finished || generated.length >= MAX_TOKENS) return false;
      const filtered = currentFiltered();
      const choice = mode === 'greedy' ? M.argmaxOf(filtered) : M.sampleFrom(filtered, rng);
      generated.push({ token: choice.token, pRaw: choice.pRaw });
      if (choice.token === M.EOS || generated.length >= MAX_TOKENS) finished = true;
      redraw();
      return !finished;
    }

    function stopAuto() {
      if (autoTimer) { clearInterval(autoTimer); autoTimer = null; }
    }

    $('s1-next').addEventListener('click', () => { stopAuto(); step(); });
    $('s1-full').addEventListener('click', () => {
      stopAuto();
      autoTimer = setInterval(() => { if (!step()) stopAuto(); }, 160);
    });
    $('s1-restart').addEventListener('click', () => {
      stopAuto();
      generated = [];
      finished = false;
      redraw();
    });
    $('s1-reseed').addEventListener('click', () => {
      seed = (seed * 1664525 + 1013904223) >>> 0;
      rng = U.mulberry32(seed);
    });

    $('s1-tab-greedy').addEventListener('click', () => setModeUI('greedy'));
    $('s1-tab-sample').addEventListener('click', () => setModeUI('sample'));
    function setModeUI(m2) {
      mode = m2;
      $('s1-tab-greedy').classList.toggle('active', mode === 'greedy');
      $('s1-tab-sample').classList.toggle('active', mode === 'sample');
      /* greedy ignora os filtros de amostragem */
      for (const id of ['s1-topk-wrap', 's1-topp-wrap']) {
        $(id).classList.toggle('disabled', mode === 'greedy');
      }
      $('s1-topk').disabled = $('s1-topp').disabled = mode === 'greedy';
      redraw();
    }

    $('s1-tau').addEventListener('input', (e) => {
      tau = +e.target.value;
      $('s1-tau-val').textContent = f(tau, 1);
      drawDist();
    });
    $('s1-topk').addEventListener('input', (e) => {
      const v = +e.target.value;
      topK = v >= 21 ? Infinity : v;
      $('s1-topk-val').textContent = v >= 21 ? 'off' : String(v);
      drawDist();
    });
    $('s1-topp').addEventListener('input', (e) => {
      topP = +e.target.value;
      $('s1-topp-val').textContent = f(topP, 2);
      drawDist();
    });

    redraw();
    P.onRedraw(drawDist);
    P.observeResize(canvas, drawDist);
  }

  DM.sections.push({ name: 's1-playground', init });
})();
