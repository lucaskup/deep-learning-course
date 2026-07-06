/* Seção 2: sinal no backward. Lê o estado compartilhado DL.winit
   (configuração e rede da seção 1) e mostra os gradientes por camada. */
(function () {
  'use strict';
  window.DL = window.DL || {};
  DL.sections = DL.sections || [];

  const INIT_NAMES = {
    zeros: 'zeros (θ = 0)',
    normal: 'normal 𝒩(0, σ²)',
    xavier: 'Xavier (1/n)',
    he: 'He (2/n)',
  };
  const ACT_NAMES = { tanh: 'tanh', sigmoid: 'sigmoide', relu: 'ReLU' };

  function init() {
    const P = DL.plot, S = DL.winit;
    const $ = (id) => document.getElementById(id);

    const cvHist = $('s2-hist'), cvStd = $('s2-std');

    function sciHtml(x) {
      if (x === 0) return '0';
      const e = Math.floor(Math.log10(Math.abs(x)));
      const m = x / Math.pow(10, e);
      return m.toFixed(2) + ' × 10<sup>' + e + '</sup>';
    }

    function updateReadouts() {
      const cfg = S.cfg, d = S.data;
      let label = INIT_NAMES[cfg.init];
      if (cfg.init === 'normal') label += ', σ = ' + cfg.sigma.toFixed(2);
      $('s2-config').textContent =
        'configuração atual: ' + label + ' · ' + ACT_NAMES[cfg.act] + ' · L = ' + cfg.L;
      const ratio = d.gradStd[d.L] > 0 ? d.gradStd[1] / d.gradStd[d.L] : 0;
      $('s2-ratio').innerHTML = ratio === 0
        ? 'σ(g<sub>1</sub>) / σ(g<sub>L</sub>) = 0 (o gradiente não chega)'
        : 'σ(g<sub>1</sub>) / σ(g<sub>L</sub>) = ' + sciHtml(ratio);
    }

    function redraw() {
      const d = S.data;
      if (!d) return;
      const th = P.theme();
      let m = 0;
      for (const l of d.sel) if (d.gradStd[l] > m) m = d.gradStd[l];
      const hi = Math.max(3.5 * m, 1e-6);
      S.drawHistRow(cvHist, d.sel, (l) => d.grads[l], (l) => d.gradStd[l], -hi, hi, th.orange);
      S.drawStdCurve(cvStd, d.gradStd, 1, d.sel, th.orange, 'σ injetado na saída = 1');
      updateReadouts();
    }

    S.listeners.push(redraw);
    redraw();
    P.onRedraw(redraw);
    P.observeResize(cvHist, redraw);
    P.observeResize(cvStd, redraw);
  }

  DL.sections.push({ name: 's2-backward', init });
})();
