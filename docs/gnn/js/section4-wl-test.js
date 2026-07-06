/* Seção 4: teste de Weisfeiler-Lehman (1-WL) passo a passo.
   Dois presets: o par dos slides (caminho vs triângulo+pendente, que o
   teste separa em 1 iteração) e o par em que o 1-WL falha (C6 vs dois
   triângulos). O refinamento usa um dicionário de assinaturas
   compartilhado entre os dois grafos para que as cores sejam comparáveis. */
(function () {
  'use strict';
  window.DL = window.DL || {};
  DL.sections = DL.sections || [];

  const PRESETS = {
    a: {
      name: 'caminho vs triângulo+pendente',
      g1: {
        pos: [[1, 3], [3.5, 3], [6, 3], [8.5, 3]],
        edges: [[0, 1], [1, 2], [2, 3]],
      },
      g2: {
        pos: [[1.5, 3], [4.5, 4.6], [4.5, 1.4], [7.5, 4.6]],
        edges: [[0, 1], [0, 2], [1, 2], [1, 3]],
      },
    },
    b: {
      name: 'C6 vs dois triângulos',
      g1: {
        pos: [[4.75, 5.4], [7.1, 4.2], [7.1, 1.8], [4.75, 0.6], [2.4, 1.8], [2.4, 4.2]],
        edges: [[0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [5, 0]],
      },
      g2: {
        pos: [[2.4, 4.9], [3.9, 2.4], [0.9, 2.4], [7.1, 4.9], [8.6, 2.4], [5.6, 2.4]],
        edges: [[0, 1], [1, 2], [2, 0], [3, 4], [4, 5], [5, 3]],
      },
    },
  };

  const state = { preset: 'a', iter: 0, lab1: [], lab2: [], stable: false };

  function adjacency(g) {
    const adj = g.pos.map(() => []);
    for (const [a, b] of g.edges) { adj[a].push(b); adj[b].push(a); }
    return adj;
  }

  function reset() {
    const p = PRESETS[state.preset];
    state.lab1 = p.g1.pos.map(() => 0);
    state.lab2 = p.g2.pos.map(() => 0);
    state.iter = 0;
    state.stable = false;
  }

  /* Uma iteração de refinamento, com mapa de assinaturas compartilhado. */
  function step() {
    const p = PRESETS[state.preset];
    const map = new Map();
    let next = 0;
    function refine(labels, adj) {
      return labels.map((c, i) => {
        const sig = c + '|' + adj[i].map((j) => labels[j]).sort().join(',');
        if (!map.has(sig)) map.set(sig, next++);
        return map.get(sig);
      });
    }
    const n1 = refine(state.lab1, adjacency(p.g1));
    const n2 = refine(state.lab2, adjacency(p.g2));
    /* Estabilizou se as partições não mudaram (mesmo número de classes
       e mesma atribuição relativa). */
    const sameSplit = (oldL, newL) =>
      new Set(oldL.map((c, i) => c + ':' + newL[i])).size === new Set(newL).size &&
      new Set(oldL).size === new Set(newL).size;
    state.stable = sameSplit(state.lab1, n1) && sameSplit(state.lab2, n2);
    state.lab1 = n1;
    state.lab2 = n2;
    state.iter++;
  }

  function multiset(labels) {
    const letters = 'ABCDEFGHIJ';
    return labels.map((c) => letters[c] || '?').sort();
  }

  function init() {
    const cv1 = document.getElementById('s4-g1');
    const cv2 = document.getElementById('s4-g2');
    const ms1 = document.getElementById('s4-ms1');
    const ms2 = document.getElementById('s4-ms2');
    const readout = document.getElementById('s4-readout');

    function palette(t) {
      return [t.cyan, t.orange, t.green, t.purple, t.pink, t.red, t.comment];
    }

    function drawGraph(cv, g, labels) {
      const { ctx, w, h } = DL.plot.setup(cv);
      const t = DL.plot.theme();
      DL.plot.clear(ctx, w, h);
      const fr = DL.plot.frame(ctx, w, h, 0, 9.6, 0, 6, { l: 8, r: 8, t: 8, b: 8 });
      const cols = palette(t);
      const letters = 'ABCDEFGHIJ';

      ctx.strokeStyle = t.line;
      ctx.lineWidth = 1.6;
      for (const [a, b] of g.edges) {
        ctx.beginPath();
        ctx.moveTo(fr.X(g.pos[a][0]), fr.Y(g.pos[a][1]));
        ctx.lineTo(fr.X(g.pos[b][0]), fr.Y(g.pos[b][1]));
        ctx.stroke();
      }
      g.pos.forEach((p, i) => {
        const c = labels[i];
        ctx.fillStyle = cols[c % cols.length];
        ctx.beginPath();
        ctx.arc(fr.X(p[0]), fr.Y(p[1]), 13, 0, 2 * Math.PI);
        ctx.fill();
        ctx.fillStyle = t.bg;
        ctx.font = 'bold 12px sans-serif';
        ctx.textAlign = 'center';
        ctx.textBaseline = 'middle';
        ctx.fillText(letters[c] || '?', fr.X(p[0]), fr.Y(p[1]) + 0.5);
        ctx.textBaseline = 'alphabetic';
      });
    }

    function redraw() {
      const p = PRESETS[state.preset];
      drawGraph(cv1, p.g1, state.lab1);
      drawGraph(cv2, p.g2, state.lab2);
      const m1 = multiset(state.lab1), m2 = multiset(state.lab2);
      ms1.textContent = '{' + m1.join(', ') + '}';
      ms2.textContent = '{' + m2.join(', ') + '}';
      const differ = m1.join() !== m2.join();
      let msg = 'iteração ' + state.iter + ' · ';
      if (differ) {
        msg += '✗ multisets diferem → G₁ e G₂ NÃO são isomorfos';
      } else if (state.stable) {
        msg += '◼ multisets iguais e o refinamento estabilizou → o 1-WL FALHA: inconclusivo';
      } else {
        msg += 'multisets iguais até aqui → continue iterando';
      }
      readout.textContent = msg;
    }

    function setPreset(key) {
      state.preset = key;
      reset();
      redraw();
    }

    document.getElementById('s4-preset-a').addEventListener('click', () => setPreset('a'));
    document.getElementById('s4-preset-b').addEventListener('click', () => setPreset('b'));
    document.getElementById('s4-step').addEventListener('click', () => { step(); redraw(); });
    document.getElementById('s4-reset').addEventListener('click', () => { reset(); redraw(); });

    reset();
    DL.plot.observeResize(cv1, redraw);
    DL.plot.observeResize(cv2, redraw);
    DL.plot.onRedraw(redraw);
    redraw();
  }

  DL.sections.push({ name: 'wl-test', init });
})();
