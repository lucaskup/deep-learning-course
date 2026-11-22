/* Tema compartilhado por todas as páginas do site (inicial e demos):
   botão claro/escuro da navegação, seletor de paleta (Dracula / Omarchy)
   no rodapé e o helper DL.btnLabel dos botões com ícone. O script inline no <head> de cada página aplica data-theme e
   data-palette antes da primeira pintura; aqui ficam só os controles.
   As chaves de localStorage ('theme', 'palette') são as mesmas do site
   lucaskup.github.io, servido na mesma origem. */
(function () {
  'use strict';

  var ICON_MOON = '<svg class="icon" viewBox="0 0 24 24" aria-hidden="true"><path d="M12 3a6 6 0 0 0 9 9 9 9 0 1 1-9-9Z"/></svg>';
  var ICON_SUN = '<svg class="icon" viewBox="0 0 24 24" aria-hidden="true"><circle cx="12" cy="12" r="4"/><path d="M12 2v2"/><path d="M12 20v2"/><path d="m4.93 4.93 1.41 1.41"/><path d="m17.66 17.66 1.41 1.41"/><path d="M2 12h2"/><path d="M20 12h2"/><path d="m6.34 17.66-1.41 1.41"/><path d="m19.07 4.93-1.41 1.41"/></svg>';

  /* Rótulo de botão com ícone: os botões Rodar/Pausar das demos trocam de
     ícone e de texto a cada clique. Uso: DL.btnLabel(btn, 'pause', 'Pausar').
     Fica aqui, e não em plot.js, porque lm e diffusion têm plot.js próprio
     (namespace DM) e carregam este arquivo; o helper é publicado nos dois. */
  var BTN_ICONS = {
    play: '<svg class="icon icon-inline" viewBox="0 0 24 24" aria-hidden="true"><path d="M6 4l14 8-14 8z"/></svg>',
    pause: '<svg class="icon icon-inline" viewBox="0 0 24 24" aria-hidden="true"><rect x="6" y="4" width="4" height="16" rx="1"/><rect x="14" y="4" width="4" height="16" rx="1"/></svg>'
  };
  function btnLabel(btn, icon, text) {
    btn.innerHTML = BTN_ICONS[icon] || '';
    btn.appendChild(document.createTextNode(text));
  }
  window.DL = window.DL || {};
  window.DL.btnLabel = btnLabel;
  if (window.DM) window.DM.btnLabel = btnLabel;

  var root = document.documentElement;
  function save(key, value) {
    try { localStorage.setItem(key, value); } catch (e) { /* armazenamento bloqueado */ }
  }

  /* Botão claro/escuro: o ícone mostra o tema em uso. */
  (function initTheme() {
    var btn = document.getElementById('theme-toggle');
    if (!btn) return;
    function sync() {
      var light = root.getAttribute('data-theme') === 'light';
      btn.innerHTML = light ? ICON_SUN : ICON_MOON;
      btn.setAttribute('aria-label', light ? 'Mudar para o tema escuro' : 'Mudar para o tema claro');
    }
    btn.addEventListener('click', function () {
      var next = root.getAttribute('data-theme') === 'light' ? 'dark' : 'light';
      root.setAttribute('data-theme', next);
      save('theme', next);
      sync();
    });
    sync();
  })();

  /* Seletor de paleta: par de botões de texto acrescentado ao rodapé.
     Dracula é o padrão (sem atributo); Omarchy grava data-palette. */
  (function initPalette() {
    var footer = document.querySelector('footer');
    if (!footer) return;
    var PALETTES = ['dracula', 'omarchy'];
    function current() { return root.getAttribute('data-palette') === 'omarchy' ? 'omarchy' : 'dracula'; }

    var wrap = document.createElement('p');
    wrap.className = 'palette-switch';
    wrap.setAttribute('role', 'group');
    wrap.setAttribute('aria-label', 'Tema de cores');
    wrap.textContent = 'tema:';

    var buttons = PALETTES.map(function (name) {
      var b = document.createElement('button');
      b.type = 'button';
      b.textContent = name;
      b.addEventListener('click', function () {
        if (name === 'omarchy') root.setAttribute('data-palette', name);
        else root.removeAttribute('data-palette');
        save('palette', name);
        sync();
      });
      wrap.append(' ', b);
      return b;
    });

    /* O ícone da aba acompanha a paleta; o prefixo do caminho (../ nas demos)
       vem do próprio <link> da página. */
    var favicon = document.querySelector('link[rel="icon"]');

    function sync() {
      buttons.forEach(function (b) { b.setAttribute('aria-pressed', String(b.textContent === current())); });
      if (favicon) {
        var file = current() === 'omarchy' ? 'favicon-omarchy.svg' : 'favicon.svg';
        favicon.setAttribute('href', favicon.getAttribute('href').replace(/favicon(-omarchy)?\.svg/, file));
      }
    }

    sync();
    footer.appendChild(wrap);
  })();
})();
