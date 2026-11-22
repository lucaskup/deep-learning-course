/* Cola geral: laço de animação compartilhado e inicialização das seções. */
(function () {
  'use strict';
  window.DM = window.DM || {};

  /* Laço rAF compartilhado: roda apenas enquanto houver tickers ativos. */
  const tickers = new Set();
  let rafId = null;
  let lastTime = 0;

  function loop(now) {
    /* Limita dt para que uma aba inativa não avance a simulação de uma vez. */
    const dt = lastTime ? Math.min(0.1, (now - lastTime) / 1000) : 0;
    lastTime = now;
    for (const fn of Array.from(tickers)) fn(dt);
    if (tickers.size > 0) rafId = requestAnimationFrame(loop);
    else { rafId = null; lastTime = 0; }
  }

  DM.startTicker = function (fn) {
    tickers.add(fn);
    if (rafId === null) rafId = requestAnimationFrame(loop);
  };
  DM.stopTicker = function (fn) { tickers.delete(fn); };

  /* O botão de tema e o seletor de paleta ficam em shared/js/theme.js. */

  /* Inicializa as seções registradas (scripts defer rodam em ordem). */
  DM.sections = DM.sections || [];
  for (const s of DM.sections) {
    try { s.init(); } catch (e) { console.error('Falha ao iniciar seção', s.name, e); }
  }
})();
