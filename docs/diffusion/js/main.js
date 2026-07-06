/* Cola geral: tema, laço de animação compartilhado e inicialização das seções. */
(function () {
  'use strict';
  window.DM = window.DM || {};

  /* Laço rAF compartilhado: roda apenas enquanto houver tickers ativos. */
  const tickers = new Set();
  let rafId = null;
  let lastTime = 0;

  function loop(now) {
    const dt = lastTime ? (now - lastTime) / 1000 : 0;
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

  /* Botão de tema (mesmo mecanismo da página inicial do curso). */
  const root = document.documentElement;
  const btn = document.getElementById('theme-toggle');
  function syncIcon() { btn.textContent = root.getAttribute('data-theme') === 'light' ? '🌙' : '☀'; }
  btn.addEventListener('click', function () {
    const next = root.getAttribute('data-theme') === 'light' ? 'dark' : 'light';
    root.setAttribute('data-theme', next);
    localStorage.setItem('theme', next);
    syncIcon();
  });
  syncIcon();

  /* Inicializa as seções registradas (scripts defer rodam em ordem). */
  DM.sections = DM.sections || [];
  for (const s of DM.sections) {
    try { s.init(); } catch (e) { console.error('Falha ao iniciar seção', s.name, e); }
  }
})();
