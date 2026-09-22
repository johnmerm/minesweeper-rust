/*
 * Browser front-end for the Rust minesweeper engine.
 *
 * The .wasm module is built from the `minesweeper_wasm` crate with plain
 * `cargo build --target wasm32-unknown-unknown` — no wasm-bindgen, no bundler,
 * so this file is the entire glue layer. See wasm/src/lib.rs for the ABI.
 *
 * Two loading modes:
 *   - index.html      fetches minesweeper.wasm next to this script (needs http/https)
 *   - standalone.html defines MINESWEEPER_WASM_BASE64 first, so it also works
 *                     from a file:// URL with no network at all
 */
(function () {
  'use strict';

  // Cell byte encodings, mirroring `cell_code` in wasm/src/lib.rs.
  var VISIBLE_MINE = 9, HIDDEN = 10, FLAGGED = 11;
  // Indices into the u32 stats array, mirroring the STAT_* constants.
  var STAT_MC_VALID = 0, STAT_MC_ATTEMPTS = 1, STAT_MC_MEMORY = 2,
      STAT_CS_VALID = 3, STAT_CS_ATTEMPTS = 4, STAT_CS_MEMORY = 5, STAT_USED = 6;
  var MODE_AUTO = 0, MODE_MC = 1, MODE_CS = 2;

  var wasm = null;          // the module's exports
  var cellEls = [];         // one DOM node per board cell, rebuilt on new game
  var width = 0, height = 0;
  var showProbs = true, autoReveal = false, flagMode = false;
  var pendingCompute = null;
  var startedAt = 0, timerId = 0;

  var el = {
    grid: document.getElementById('grid'),
    status: document.getElementById('status'),
    minesLeft: document.getElementById('mines-left'),
    timer: document.getElementById('timer'),
    hover: document.getElementById('hover'),
    sim: document.getElementById('sim'),
    error: document.getElementById('error'),
    width: document.getElementById('in-width'),
    height: document.getElementById('in-height'),
    mines: document.getElementById('in-mines'),
    strategy: document.getElementById('in-strategy')
  };

  /* ---------------------------------------------------------------- loading */

  function base64ToBytes(b64) {
    var bin = atob(b64);
    var bytes = new Uint8Array(bin.length);
    for (var i = 0; i < bin.length; i++) bytes[i] = bin.charCodeAt(i);
    return bytes;
  }

  function loadWasm() {
    if (typeof MINESWEEPER_WASM_BASE64 === 'string') {
      return WebAssembly.instantiate(base64ToBytes(MINESWEEPER_WASM_BASE64).buffer);
    }
    // Deliberately not instantiateStreaming: some static hosts serve .wasm with
    // the wrong Content-Type, which streaming instantiation rejects outright.
    return fetch('minesweeper.wasm')
      .then(function (r) {
        if (!r.ok) throw new Error('HTTP ' + r.status + ' fetching minesweeper.wasm');
        return r.arrayBuffer();
      })
      .then(function (buf) { return WebAssembly.instantiate(buf); });
  }

  function fail(err) {
    el.error.hidden = false;
    el.error.textContent = 'Could not start the WebAssembly module: ' + err.message +
      '. Serving the page over file:// blocks the .wasm fetch — use standalone.html instead.';
    el.status.textContent = '';
  }

  /* ------------------------------------------------- reading module memory */

  // Typed-array views must be rebuilt after every call into the module: growing
  // the linear memory detaches every view onto the old ArrayBuffer.
  function cells() {
    return new Uint8Array(wasm.memory.buffer, wasm.ms_cells_ptr(), width * height);
  }
  function probs() {
    return new Float32Array(wasm.memory.buffer, wasm.ms_probs_ptr(), width * height);
  }
  function stats() {
    return new Uint32Array(wasm.memory.buffer, wasm.ms_stats_ptr(), wasm.ms_stats_len());
  }

  /* ------------------------------------------------------------- rendering */

  function buildGrid() {
    width = wasm.ms_width();
    height = wasm.ms_height();
    el.grid.style.gridTemplateColumns = 'repeat(' + width + ', var(--cell))';
    el.grid.textContent = '';
    cellEls = new Array(width * height);

    var frag = document.createDocumentFragment();
    for (var i = 0; i < width * height; i++) {
      var cell = document.createElement('div');
      cell.className = 'cell hidden';
      cell.dataset.i = i;
      var label = document.createElement('span');
      label.className = 'prob';
      cell.appendChild(label);
      cellEls[i] = cell;
      frag.appendChild(cell);
    }
    el.grid.appendChild(frag);
  }

  /** Grey → red tint, matching the CLI, Qt and Actix front-ends. */
  function probColor(p) {
    var r = Math.round(204 + 51 * p);
    var gb = Math.round(204 * (1 - p));
    return 'rgb(' + r + ',' + gb + ',' + gb + ')';
  }

  function render() {
    var c = cells(), p = probs();
    var over = wasm.ms_state() !== 0;

    for (var i = 0; i < cellEls.length; i++) {
      var node = cellEls[i], code = c[i], label = node.lastChild;
      var text = '', cls = 'cell', bg = '', pct = '';

      if (code === HIDDEN || code === FLAGGED) {
        cls += code === FLAGGED ? ' flagged' : ' hidden';
        text = code === FLAGGED ? '⚑' : '';
        bg = probColor(p[i]);
        pct = Math.round(p[i] * 100) + '%';
      } else if (code === VISIBLE_MINE) {
        cls += ' visible mine';
        text = '✹';
      } else {
        cls += ' visible' + (code > 0 ? ' n' + code : '');
        text = code > 0 ? String(code) : '';
      }

      node.className = cls;
      node.style.backgroundColor = bg;
      node.title = pct ? 'Mine: ' + pct : '';
      // firstChild is the text node we manage; the trailing span is the label.
      if (node.firstChild !== label) node.removeChild(node.firstChild);
      if (text) node.insertBefore(document.createTextNode(text), label);
      label.textContent = over ? '' : pct;
    }

    el.grid.classList.toggle('no-prob', !showProbs);
    renderStatus();
  }

  function renderStatus() {
    var state = wasm.ms_state();
    el.status.className = 'status' + (state === 1 ? ' won' : state === 2 ? ' lost' : '');
    el.status.textContent = state === 1 ? 'You won!' : state === 2 ? 'Boom — game over' : 'Playing';
    el.minesLeft.textContent = wasm.ms_mines() - wasm.ms_flags();
    if (state !== 0) stopTimer();
  }

  function formatBytes(b) {
    if (b < 1024) return b + ' B';
    if (b < 1048576) return (b / 1024).toFixed(1) + ' KB';
    return (b / 1048576).toFixed(1) + ' MB';
  }

  function renderSim() {
    var s = stats();
    var lines = [];
    if (s[STAT_CS_ATTEMPTS] > 0 || s[STAT_CS_VALID] > 0) {
      lines.push({
        used: s[STAT_USED] === MODE_CS,
        text: 'exact: ' + s[STAT_CS_VALID].toLocaleString() + ' layouts / ' +
              s[STAT_CS_ATTEMPTS].toLocaleString() + ' steps [' + formatBytes(s[STAT_CS_MEMORY]) + ']'
      });
    }
    if (s[STAT_MC_ATTEMPTS] > 0 || s[STAT_MC_VALID] > 0) {
      lines.push({
        used: s[STAT_USED] === MODE_MC,
        text: 'sampled: ' + s[STAT_MC_VALID].toLocaleString() + ' valid / ' +
              s[STAT_MC_ATTEMPTS].toLocaleString() + ' draws [' + formatBytes(s[STAT_MC_MEMORY]) + ']'
      });
    }
    el.sim.textContent = '';
    lines.forEach(function (line) {
      var div = document.createElement('div');
      if (line.used) div.className = 'used';
      div.textContent = line.text;
      el.sim.appendChild(div);
    });
  }

  /* ----------------------------------------------------------- game driving */

  /**
   * Recompute probabilities off the click handler.
   *
   * The estimators are synchronous and can take a moment on a large board, so
   * the board repaint happens first and the numbers land a frame later rather
   * than freezing the click.
   */
  function scheduleCompute() {
    if (pendingCompute) clearTimeout(pendingCompute);
    el.sim.textContent = 'calculating…';
    pendingCompute = setTimeout(function () {
      pendingCompute = null;
      wasm.ms_compute(Number(el.strategy.value));
      if (autoReveal) wasm.ms_auto_reveal(Number(el.strategy.value));
      render();
      renderSim();
    }, 0);
  }

  function newGame(w, h, m) {
    wasm.ms_new(w, h, m);
    el.width.value = wasm.ms_width();
    el.height.value = wasm.ms_height();
    el.mines.value = wasm.ms_mines();
    buildGrid();
    stopTimer();
    el.timer.textContent = '0:00';
    startedAt = 0;
    render();
    el.sim.textContent = '';
  }

  function onCell(index, flag) {
    if (wasm.ms_state() !== 0) return;
    var x = index % width, y = Math.floor(index / width);
    if (flag) {
      wasm.ms_flag(x, y);
    } else {
      if (cells()[index] === FLAGGED) return; // never blow up a flagged cell
      wasm.ms_reveal(x, y);
      startTimer();
    }
    render();
    scheduleCompute();
  }

  /* ----------------------------------------------------------------- timer */

  function startTimer() {
    if (timerId) return;
    startedAt = Date.now();
    timerId = setInterval(function () {
      var secs = Math.floor((Date.now() - startedAt) / 1000);
      el.timer.textContent = Math.floor(secs / 60) + ':' + String(secs % 60).padStart(2, '0');
    }, 500);
  }

  function stopTimer() {
    if (timerId) clearInterval(timerId);
    timerId = 0;
  }

  /* ----------------------------------------------------------------- events */

  function cellIndexFrom(target) {
    var node = target.closest ? target.closest('.cell') : null;
    return node ? Number(node.dataset.i) : -1;
  }

  function wireEvents() {
    el.grid.addEventListener('click', function (e) {
      var i = cellIndexFrom(e.target);
      if (i >= 0) onCell(i, flagMode);
    });
    el.grid.addEventListener('contextmenu', function (e) {
      e.preventDefault();
      var i = cellIndexFrom(e.target);
      if (i >= 0) onCell(i, true);
    });
    el.grid.addEventListener('mouseover', function (e) {
      var i = cellIndexFrom(e.target);
      if (i < 0) return;
      var c = cells()[i];
      el.hover.textContent = (c === HIDDEN || c === FLAGGED)
        ? 'Mine probability: ' + (probs()[i] * 100).toFixed(1) + '%'
        : '';
    });
    el.grid.addEventListener('mouseleave', function () { el.hover.textContent = ''; });

    document.getElementById('btn-new').addEventListener('click', function () {
      newGame(Number(el.width.value), Number(el.height.value), Number(el.mines.value));
    });

    Array.prototype.forEach.call(document.querySelectorAll('.preset'), function (btn) {
      btn.addEventListener('click', function () {
        newGame(Number(btn.dataset.w), Number(btn.dataset.h), Number(btn.dataset.m));
      });
    });

    var probsBtn = document.getElementById('btn-probs');
    probsBtn.addEventListener('click', function () {
      showProbs = !showProbs;
      probsBtn.classList.toggle('on', showProbs);
      el.grid.classList.toggle('no-prob', !showProbs);
    });

    var autoBtn = document.getElementById('btn-auto');
    autoBtn.addEventListener('click', function () {
      autoReveal = !autoReveal;
      autoBtn.classList.toggle('on', autoReveal);
      if (autoReveal) scheduleCompute();
    });

    var flagBtn = document.getElementById('btn-flagmode');
    flagBtn.addEventListener('click', function () {
      flagMode = !flagMode;
      flagBtn.classList.toggle('on', flagMode);
    });

    el.strategy.addEventListener('change', scheduleCompute);
  }

  /* ------------------------------------------------------------------ start */

  loadWasm().then(function (result) {
    wasm = result.instance.exports;

    // The module has no imports at all, so entropy comes in from here.
    var seed = new Uint32Array(2);
    if (window.crypto && window.crypto.getRandomValues) {
      window.crypto.getRandomValues(seed);
    } else {
      seed[0] = (Math.random() * 0xffffffff) >>> 0;
      seed[1] = Date.now() >>> 0;
    }
    wasm.ms_seed(seed[0], seed[1]);

    wireEvents();
    newGame(10, 10, 10);
  }).catch(fail);
})();
