/* Seção 7: Add & Norm. LayerNorm por token (batch antes/depois) e o efeito
   da conexão residual ao empilhar N blocos: norma do gradiente propagado
   (produto jacobiano-vetor ‖J_ℓ v‖) por profundidade, com e sem residual. */
(function () {
  'use strict';
  window.DL = window.DL || {};
  DL.sections = DL.sections || [];

  const T = 6, D = 16, MAXN = 32;

  function init() {
    const P = DL.plot;
    const U = DL.utils;
    const $ = (id) => document.getElementById(id);

    const cvLN = $('s7-ln'), cvDepth = $('s7-depth');
    const slMu = $('s7-mu'), slSigma = $('s7-sigma'), slGamma = $('s7-gamma'), slBeta = $('s7-beta');
    const slN = $('s7-N'), slGain = $('s7-gain');

    let mu = +slMu.value, sigma = +slSigma.value, gamma = +slGamma.value, beta = +slBeta.value;
    let N = +slN.value, gain = +slGain.value;
    let wseed = 3;

    /* ── base fixa do batch: ruído por célula + deslocamento/escala por token ── */
    const base = [], shift = [], scl = [];
    {
      const rng = U.mulberry32(5);
      for (let t = 0; t < T; t++) {
        const row = new Array(D);
        for (let k = 0; k < D; k++) row[k] = U.randn(rng);
        base.push(row);
        shift.push(U.randn(rng) * 1.2);
        scl.push(0.5 + rng());
      }
    }

    function lnBatch() {
      const before = [], after = [];
      for (let t = 0; t < T; t++) {
        const x = base[t].map((b) => mu * shift[t] + sigma * scl[t] * b);
        let m = 0;
        for (const v of x) m += v;
        m /= D;
        let s2 = 0;
        for (const v of x) s2 += (v - m) * (v - m);
        const sd = Math.sqrt(s2 / D) || 1;
        before.push(x);
        after.push(x.map((v) => gamma * ((v - m) / sd) + beta));
      }
      return { before, after };
    }

    function cellColor(th, v) {
      const c = P.rgb(v >= 0 ? th.cyan : th.orange);
      const a = 0.06 + 0.94 * Math.min(1, Math.abs(v) / 3);
      return 'rgba(' + c[0] + ',' + c[1] + ',' + c[2] + ',' + a + ')';
    }

    function drawMatrix(ctx, th, M_, x0, y0, cw, chh, title) {
      ctx.fillStyle = th.fg;
      ctx.font = '11px sans-serif';
      ctx.textAlign = 'left';
      ctx.fillText(title, x0, y0 - 6);
      for (let t = 0; t < T; t++) {
        for (let k = 0; k < D; k++) {
          ctx.fillStyle = cellColor(th, M_[t][k]);
          ctx.fillRect(x0 + k * cw, y0 + t * chh, cw + 0.5, chh + 0.5);
        }
      }
      ctx.strokeStyle = th.line;
      ctx.strokeRect(x0, y0, D * cw, T * chh);
      ctx.fillStyle = th.comment;
      ctx.font = '9px sans-serif';
      ctx.textAlign = 'right';
      for (let t = 0; t < T; t++) ctx.fillText('x_' + (t + 1), x0 - 3, y0 + t * chh + chh / 2 + 3);
    }

    /* Bloco de legenda com fundo semitransparente, para que o texto continue
       legível quando barras ou curvas passam por baixo. items: [[texto, cor]]. */
    function legendBox(ctx, th, x, y, items, align) {
      ctx.font = '11px sans-serif';
      let wmax = 0;
      for (const it of items) wmax = Math.max(wmax, ctx.measureText(it[0].replace(/[_{}]/g, '')).width);
      const bx = align === 'right' ? x - wmax - 6 : x - 6;
      ctx.save();
      ctx.globalAlpha = 0.85;
      ctx.fillStyle = th.card;
      ctx.fillRect(bx, y, wmax + 12, 14 * items.length + 6);
      ctx.restore();
      items.forEach((it, k) => P.label(ctx, x, y + 14 * (k + 1), it[0], it[1], align || 'left'));
    }

    function drawLN(stats) {
      const th = P.theme();
      const { ctx, w, h } = P.setup(cvLN);
      P.clear(ctx, w, h);
      const padL = 30, gap = 72, top = 26;
      const lab = 26; /* faixa do gap reservada aos rótulos x_t do painel direito */
      const panelW = (w - 2 * padL - gap) / 2;
      const cw = panelW / D;
      const chh = Math.min(24, (h - top - 26) / T);
      drawMatrix(ctx, th, stats.before, padL, top, cw, chh, 'x (antes)');
      drawMatrix(ctx, th, stats.after, padL + panelW + gap, top, cw, chh, 'LN(x) (depois)');
      /* seta entre painéis */
      const ay = top + (T * chh) / 2;
      ctx.strokeStyle = th.fg;
      ctx.lineWidth = 1.5;
      ctx.beginPath();
      ctx.moveTo(padL + panelW + 6, ay);
      ctx.lineTo(padL + panelW + gap - lab - 4, ay);
      ctx.stroke();
      ctx.fillStyle = th.fg;
      ctx.beginPath();
      ctx.moveTo(padL + panelW + gap - lab - 4, ay);
      ctx.lineTo(padL + panelW + gap - lab - 10, ay - 4);
      ctx.lineTo(padL + panelW + gap - lab - 10, ay + 4);
      ctx.fill();
      ctx.font = '10px sans-serif';
      ctx.textAlign = 'center';
      ctx.fillStyle = th.comment;
      ctx.fillText('LN', padL + panelW + (gap - lab) / 2 + 1, ay - 8);
      /* estatísticas por linha (token 1) */
      const row = stats.before[0], rowA = stats.after[0];
      const mean = (xs) => xs.reduce((a, b) => a + b, 0) / xs.length;
      const std = (xs) => {
        const m = mean(xs);
        return Math.sqrt(xs.reduce((a, b) => a + (b - m) * (b - m), 0) / xs.length);
      };
      ctx.textAlign = 'left';
      ctx.fillText(
        'token 1: μ = ' + mean(row).toFixed(2) + ', σ = ' + std(row).toFixed(2) +
        '  →  μ = ' + mean(rowA).toFixed(2) + ', σ = ' + std(rowA).toFixed(2),
        padL, top + T * chh + 16);
    }

    /* ── pilha de N blocos: F(x) = W2·relu(W1·x), pesos com ganho g ── */
    let blocks = [];
    function buildBlocks() {
      const rng = U.mulberry32(wseed);
      blocks = [];
      for (let l = 0; l < MAXN; l++) {
        const W1 = [], W2 = [];
        for (let r = 0; r < D; r++) {
          const r1 = new Array(D), r2 = new Array(D);
          for (let c = 0; c < D; c++) {
            r1[c] = U.randn(rng) / Math.sqrt(D);
            r2[c] = U.randn(rng) / Math.sqrt(D);
          }
          W1.push(r1);
          W2.push(r2);
        }
        blocks.push({ W1, W2 });
      }
    }

    function matVec(W, x) {
      const out = new Array(D).fill(0);
      for (let r = 0; r < D; r++) {
        const Wr = W[r];
        let s = 0;
        for (let c = 0; c < D; c++) s += Wr[c] * x[c];
        out[r] = s;
      }
      return out;
    }

    function F(l, x) {
      const hdn = matVec(blocks[l].W1, x).map((v) => Math.max(0, v) * Math.SQRT2);
      return matVec(blocks[l].W2, hdn).map((v) => v * gain);
    }

    const norm = (x) => Math.sqrt(x.reduce((a, b) => a + b * b, 0));

    /* Produto jacobiano-vetor de F: F(x) = g·W2·(√2·relu(W1 x)). */
    function fwdJvp(l, x, v) {
      const hpre = matVec(blocks[l].W1, x);
      const dh = matVec(blocks[l].W1, v);
      const hdn = hpre.map((u) => Math.max(0, u) * Math.SQRT2);
      const dhdn = hpre.map((u, i) => (u > 0 ? dh[i] * Math.SQRT2 : 0));
      return {
        F: matVec(blocks[l].W2, hdn).map((u) => u * gain),
        dF: matVec(blocks[l].W2, dhdn).map((u) => u * gain),
      };
    }

    /* LayerNorm e seu produto jacobiano-vetor:
       dLN = (dy − mean(dy))/σ − ŷ·mean(ŷ⊙dy)/σ. */
    function lnJvp(y, dy) {
      let m = 0;
      for (const u of y) m += u;
      m /= D;
      let s2 = 0;
      for (const u of y) s2 += (u - m) * (u - m);
      const sd = Math.sqrt(s2 / D) || 1;
      const yh = y.map((u) => (u - m) / sd);
      let dm = 0;
      for (const u of dy) dm += u;
      dm /= D;
      let dot = 0;
      for (let i = 0; i < D; i++) dot += yh[i] * dy[i];
      dot /= D;
      return { y: yh, dy: yh.map((u, i) => (dy[i] - dm) / sd - u * dot / sd) };
    }

    let x0 = [], v0 = [];
    {
      const rng = U.mulberry32(99);
      for (let i = 0; i < D; i++) x0.push(U.randn(rng));
      for (let i = 0; i < D; i++) v0.push(U.randn(rng));
      const nv = norm(v0);
      v0 = v0.map((u) => u / nv);
    }

    /* Propaga o sinal x e a direção v (JVP) pelas três pilhas; guarda ‖J_ℓ v‖. */
    function runStacks() {
      let plain = x0.slice(), res = x0.slice();
      let vP = v0.slice(), vR = v0.slice();
      const ln0 = lnJvp(x0, v0);
      let resLN = ln0.y, vL = ln0.dy;
      const jP = [norm(vP)], jR = [norm(vR)], jL = [norm(vL)];
      for (let l = 0; l < N; l++) {
        const a = fwdJvp(l, plain, vP);
        plain = a.F;
        vP = a.dF;
        const b = fwdJvp(l, res, vR);
        res = res.map((u, i) => u + b.F[i]);
        vR = vR.map((u, i) => u + b.dF[i]);
        const c = fwdJvp(l, resLN, vL);
        const ln = lnJvp(resLN.map((u, i) => u + c.F[i]), vL.map((u, i) => u + c.dF[i]));
        resLN = ln.y;
        vL = ln.dy;
        jP.push(norm(vP));
        jR.push(norm(vR));
        jL.push(norm(vL));
      }
      return { jP, jR, jL, plain, res, resLN };
    }

    function drawDepth(run) {
      const th = P.theme();
      const { ctx, w, h } = P.setup(cvDepth);
      P.clear(ctx, w, h);
      const lg = (v) => Math.max(-6, Math.min(6, v > 0 ? Math.log10(v) : -6));
      const yP = run.jP.map(lg), yR = run.jR.map(lg), yL = run.jL.map(lg);
      const all = yP.concat(yR, yL);
      const ymin = Math.min(-1, Math.floor(Math.min(...all))) - 0.3;
      const ymax = Math.max(1.5, Math.ceil(Math.max(...all))) + 0.3;
      const fr = P.frame(ctx, w, h, 0, N, ymin, ymax, { l: 44, r: 10, t: 10, b: 24 });
      P.axes(ctx, fr, {
        xlabel: 'bloco ℓ', ylabel: 'log₁₀ ‖J_ℓ v‖',
        xticks: [0, Math.round(N / 2), N],
        yticks: [Math.ceil(ymin), 0, Math.floor(ymax)],
      });
      const ls = yP.map((_, i) => i);
      P.line(ctx, fr, ls, yP, th.orange, { width: 1.8 });
      P.line(ctx, fr, ls, yR, th.green, { width: 1.8 });
      P.line(ctx, fr, ls, yL, th.cyan, { width: 1.8 });
      legendBox(ctx, th, fr.X(0) + 8, fr.Y(ymax), [
        ['F(x) puro', th.orange], ['x + F(x)', th.green], ['LN(x + F(x))', th.cyan]]);
    }

    function updateReadout(run) {
      const e = (v) => v.toExponential(1).replace('e+', 'e').replace('-', '−');
      $('s7-readout').textContent =
        '‖J_N v‖: sem residual = ' + e(run.jP[N]) + ' · com residual = ' + e(run.jR[N]) +
        ' · residual + LN = ' + e(run.jL[N]);
    }

    function redraw() {
      drawLN(lnBatch());
      const run = runStacks();
      drawDepth(run);
      updateReadout(run);
    }

    function bindSlider(sl, idVal, fmt, set) {
      sl.addEventListener('input', () => {
        set(+sl.value);
        $(idVal).textContent = fmt(+sl.value);
        redraw();
      });
    }
    bindSlider(slMu, 's7-mu-val', (v) => v.toFixed(1), (v) => { mu = v; });
    bindSlider(slSigma, 's7-sigma-val', (v) => v.toFixed(1), (v) => { sigma = v; });
    bindSlider(slGamma, 's7-gamma-val', (v) => v.toFixed(1), (v) => { gamma = v; });
    bindSlider(slBeta, 's7-beta-val', (v) => v.toFixed(1), (v) => { beta = v; });
    bindSlider(slN, 's7-N-val', (v) => String(v), (v) => { N = v; });
    bindSlider(slGain, 's7-gain-val', (v) => v.toFixed(2), (v) => { gain = v; });

    $('s7-resample').addEventListener('click', () => {
      wseed = (wseed * 1664525 + 1013904223) >>> 0;
      buildBlocks();
      redraw();
    });

    buildBlocks();
    redraw();
    P.onRedraw(redraw);
    P.observeResize(cvLN, redraw);
    P.observeResize(cvDepth, redraw);
  }

  DL.sections.push({ name: 's7-addnorm-depth', init });
})();
