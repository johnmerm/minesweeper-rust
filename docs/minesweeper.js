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
/**
 * Which build this *script* is, stamped by wasm/bundle.py.
 *
 * Global, and outside the closure, on purpose: index.html carries the same id
 * and checks it against this one. They are separate files a host can serve from
 * different generations, and when that happens the page half-works or dies on a
 * control the other half removed — a symptom that points nowhere near the cause.
 *
 * The check lives in the page rather than here, because the stale half cannot be
 * relied on to notice anything and the page is the half that arrived fresh.
 */
var MINESWEEPER_SCRIPT_BUILD = 'd34ae2ff';

(function () {
  'use strict';

  // Cell byte encodings, mirroring `cell_code` in wasm/src/lib.rs.
  var VISIBLE_MINE = 9, HIDDEN = 10, FLAGGED = 11;
  // Indices into the u32 stats array, mirroring the STAT_* constants.
  var STAT_VALID = 0, STAT_NODES = 1, STAT_MEMORY = 2, STAT_SOLVED = 3,
      STAT_CACHE_HITS = 4, STAT_CACHE_MISSES = 5;

  var wasm = null;          // the module's exports
  var cellEls = [];         // one DOM node per board cell, rebuilt on new game
  var painted = [];         // what each cell currently shows, to skip no-op writes
  var width = 0, height = 0;
  var showProbs = true, autoReveal = false, flagMode = false;
  var pendingCompute = null;
  var startedAt = 0, timerId = 0;

  // Which estimate is on screen: 'both', 'exact' or 'neural'. In 'neural' the
  // solver's numbers are not merely hidden — nothing on the page reports them,
  // hover and tooltips included — because the whole use of that mode is to watch
  // the network unaided, and a proved value visible in a tooltip is a cheat.
  var showMode = 'both';
  // Set once the player picks a mode. Until then the choice is the page's, and
  // the page declines to score a board too big to be worth doing unasked.
  var showChosen = false;
  var neuralReady = true;    // false once the build turns out to carry no weights
  var neuralFrame = null;
  var neuralLeft = 0, neuralTotal = 0;
  var neuralChunk = 4;        // cells per step call, adapted to the frame budget
  var neuralError = null;     // mean |network - exact| at the last correction
  var neuralOpened = 0, neuralFlagged = 0;   // what the network has played, this game
  // How much of a frame the network may take. The exact values are already on
  // screen by then, so this only decides how fast the overlay fills in; anything
  // much larger and the board stops responding while it does.
  var FRAME_BUDGET_MS = 8;
  // `render` walks every cell on the board, which on a 120x120 grid costs more
  // than the handful of network passes a frame fits. Repainting a few times a
  // second still reads as filling in, and leaves the frame to the work.
  var REPAINT_EVERY_MS = 120;
  // The overlay comes on by itself up to this many cells. A full pass is a
  // forward pass per cell and restarts on every move, so on a 200x200 board it
  // would keep a core busy for minutes between clicks without being asked. Above
  // the cap the button still turns it on.
  var NEURAL_AUTO_MAX_CELLS = 4096;
  // What the network has to say before auto-play acts on it. The solver answers
  // with proof; the network never returns exactly 0 or 1, so the question has to
  // be asked with a threshold.
  var NEURAL_OPEN_BELOW = 50, NEURAL_FLAG_ABOVE = 950;   // per-mille

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
    neuralNote: document.getElementById('neural-note'),
    show: document.getElementById('in-show')
  };

  /* ---------------------------------------------------------------- loading */

  /**
   * Which build this is, stamped into index.html by wasm/bundle.py.
   *
   * Reported on the page because these files are served from a CDN: when a fix
   * appears not to have landed, the first question is always whether the browser
   * is even running the new build, and this answers it without guesswork.
   */
  function buildId() {
    return typeof MINESWEEPER_BUILD === 'string' && MINESWEEPER_BUILD !== 'dev'
      ? MINESWEEPER_BUILD
      : '';
  }

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
    // The build id keeps the module in step with this script — without it a CDN
    // can serve a cached script beside a fresh module, or the reverse.
    return fetch('minesweeper.wasm' + (buildId() ? '?v=' + buildId() : ''))
      .then(function (r) {
        if (!r.ok) throw new Error('HTTP ' + r.status + ' fetching minesweeper.wasm');
        return r.arrayBuffer();
      })
      .then(function (buf) { return WebAssembly.instantiate(buf); });
  }

  function fail(err) {
    if (el.error) {
      el.error.hidden = false;
      // Only blame file:// when the failure was actually the fetch. This used to
      // say it whatever went wrong, which sent one real bug off in exactly the
      // wrong direction.
      var fetchFailed = /fetch|HTTP \d|Failed to|NetworkError|Load failed/i.test(err.message);
      el.error.textContent = 'Could not start: ' + err.message +
        (fetchFailed
          ? '. Serving the page over file:// blocks the .wasm fetch — use standalone.html instead.'
          : '. Try a hard reload (Ctrl/Cmd-Shift-R); a cached script from an older build can do this.');
    }
    if (el.status) el.status.textContent = '';
    if (window.console) console.error(err);
  }

  /**
   * Attach a handler, tolerating a control that is not on the page.
   *
   * The script and the markup are two files a host can serve from different
   * generations, and when that happened a missing control threw out of
   * `wireEvents` and took the whole game with it — a dead board because a
   * <select> had been removed. A control that is not there is a control that
   * does nothing.
   */
  function on(node, event, handler) {
    if (node) node.addEventListener(event, handler);
    return node;
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
  function neuralProbs() {
    return new Float32Array(wasm.memory.buffer, wasm.ms_neural_probs_ptr(), width * height);
  }

  /* ------------------------------------------------------------- rendering */

  function buildGrid() {
    width = wasm.ms_width();
    height = wasm.ms_height();
    el.grid.style.gridTemplateColumns = 'repeat(' + width + ', var(--cell))';
    el.grid.textContent = '';
    cellEls = new Array(width * height);
    painted = new Array(width * height);

    var frag = document.createDocumentFragment();
    for (var i = 0; i < width * height; i++) {
      var cell = document.createElement('div');
      cell.className = 'cell hidden';
      cell.dataset.i = i;
      var label = document.createElement('span');
      label.className = 'prob';
      cell.appendChild(label);
      var guess = document.createElement('span');
      guess.className = 'guess';
      cell.appendChild(guess);
      cellEls[i] = cell;
      frag.appendChild(cell);
    }
    el.grid.appendChild(frag);
  }

  /** A cell on a board the solver could not finish. Deliberately not the grey
   *  that probColor(0) produces, which is what "certainly safe" looks like. */
  var UNKNOWN = 'rgb(198,203,214)';

  /** Grey → red tint, matching the CLI, Qt and Actix front-ends. */
  function probColor(p) {
    var r = Math.round(204 + 51 * p);
    var gb = Math.round(204 * (1 - p));
    return 'rgb(' + r + ',' + gb + ',' + gb + ')';
  }

  function render() {
    var c = cells(), p = probs();
    // Without a finished solve the buffer holds nothing, and probColor(0) is the
    // very grey that means "safe". An unsolved board therefore gets a colour of
    // its own and a '?', or it would read as a board with no mines on it.
    var known = solved();
    // -1 marks a cell the network has not reached yet, which is why the buffer
    // cannot simply start at zero: nearly-zero is a real answer here.
    var g = neuralOn() ? neuralProbs() : null;
    var exact = showMode !== 'neural';
    var over = wasm.ms_state() !== 0;

    for (var i = 0; i < cellEls.length; i++) {
      var code = c[i];
      var text = '', cls = 'cell', bg = '', pct = '';

      if (code === HIDDEN || code === FLAGGED) {
        cls += code === FLAGGED ? ' flagged' : ' hidden';
        text = code === FLAGGED ? '⚑' : '';
        // The tint follows whichever estimate is being shown, so in 'neural' the
        // colour is the network's opinion and not a proof wearing its clothes.
        var tint = exact ? (known ? p[i] : -1) : (g && g[i] >= 0 ? g[i] : -1);
        bg = tint >= 0 ? probColor(tint) : (exact && !known ? UNKNOWN : '');
        pct = exact ? (known ? Math.round(p[i] * 100) + '%' : '?') : '';
      } else if (code === VISIBLE_MINE) {
        cls += ' visible mine';
        text = '✹';
      } else {
        cls += ' visible' + (code > 0 ? ' n' + code : '');
        text = code > 0 ? String(code) : '';
      }

      // Touching the DOM for a cell that already looks right is what made a big
      // board crawl: a move changes a handful of cells, but repainting all of
      // them cost seconds on a 120x120 grid — far more than the estimator did.
      var guessed = '';
      if (g && !over && (code === HIDDEN || code === FLAGGED) && g[i] >= 0) {
        guessed = Math.round(g[i] * 100) + '%';
      }

      var shown = cls + '\u0000' + bg + '\u0000' + text + '\u0000' + (over ? '' : pct) +
                  '\u0000' + guessed;
      if (painted[i] === shown) {
        continue;
      }
      painted[i] = shown;

      var node = cellEls[i], guessLabel = node.lastChild, label = guessLabel.previousSibling;
      guessLabel.textContent = guessed;
      node.className = cls;
      node.style.backgroundColor = bg;
      node.title = pct
        ? 'Mine: ' + pct + (guessed ? '  network: ' + guessed : '')
        : (guessed ? 'Network: ' + guessed : '');
      // firstChild is the text node we manage; the two trailing spans are labels.
      if (node.firstChild !== label && node.firstChild !== guessLabel) {
        node.removeChild(node.firstChild);
      }
      if (text) node.insertBefore(document.createTextNode(text), label);
      label.textContent = over ? '' : pct;
    }

    el.grid.classList.toggle('no-prob', !showProbs);
    el.grid.classList.toggle('show-exact', exact);
    el.grid.classList.toggle('show-neural', !!g);
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

  /** Share of region solves answered from the cache, since the page loaded. */
  function reuse(s) {
    var looked = s[STAT_CACHE_HITS] + s[STAT_CACHE_MISSES];
    if (!looked) return '';
    return ' · ' + Math.round(100 * s[STAT_CACHE_HITS] / looked) + '% reused';
  }

  function solved() {
    return stats()[STAT_SOLVED] === 1;
  }

  function renderSim() {
    var s = stats();
    var div = document.createElement('div');
    if (solved()) {
      div.className = 'used';
      // "layouts" is the sum over independent regions, not the product: the
      // solver splits the board and never enumerates the whole cross-product.
      div.textContent = 'exact: ' + s[STAT_VALID].toLocaleString() + ' region layouts / ' +
          s[STAT_NODES].toLocaleString() + ' nodes [' + formatBytes(s[STAT_MEMORY]) + ']' + reuse(s);
    } else {
      // There is nothing to fall back to, and that is deliberate: a sampled
      // number reads exactly like a proved one once it is in a cell.
      div.className = 'unsolved';
      div.textContent = 'no exact answer within ' + s[STAT_NODES].toLocaleString() +
          ' nodes — showing none rather than guessing';
    }
    el.sim.textContent = '';
    el.sim.appendChild(div);
  }

  /* --------------------------------------------------------- neural overlay */

  function cancelNeural() {
    if (neuralFrame !== null) cancelAnimationFrame(neuralFrame);
    neuralFrame = null;
  }

  function neuralNote(text) {
    if (el.neuralNote) el.neuralNote.textContent = text;
  }

  /**
   * Score every unopened cell, a few per animation frame.
   *
   * One forward pass per cell is not cheap and there are up to 40 000 of them, so
   * doing it in one go would freeze the page for seconds. Instead each frame
   * spends a fixed slice of time on it and gives the rest back, and `render`
   * paints whatever has arrived — cells the pass has not reached yet hold -1 and
   * simply show nothing. The chunk size is measured rather than guessed, because
   * a cell costs an order of magnitude more on a phone than on a laptop.
   */
  function scheduleNeural() {
    cancelNeural();
    if (!neuralOn()) return;
    if (wasm.ms_state() !== 0) {
      // Nothing left to score, but the tally of what the network played is the
      // point of the exercise and must survive the end of the game.
      neuralNote(describeNeural());
      return;
    }

    neuralTotal = wasm.ms_neural_begin(learningRate());
    neuralLeft = neuralTotal;
    if (!neuralTotal) {
      render();
      neuralNote(describeNeural());
      return;
    }

    var painted_at = 0;
    var tick = function () {
      neuralFrame = null;
      var deadline = performance.now() + FRAME_BUDGET_MS;
      do {
        var before = performance.now();
        neuralLeft = wasm.ms_neural_step(neuralChunk);
        var per = (performance.now() - before) / neuralChunk;
        if (per > 0) {
          neuralChunk = Math.max(1, Math.min(512, Math.round(FRAME_BUDGET_MS / per)));
        }
      } while (neuralLeft > 0 && performance.now() < deadline);

      var now = performance.now();
      if (neuralLeft === 0 || now - painted_at > REPAINT_EVERY_MS) {
        painted_at = now;
        var scaled = wasm.ms_neural_error();
        neuralError = scaled ? scaled / 10000 : null;
        render();
        neuralNote(describeNeural());
      }
      if (neuralLeft > 0) {
        neuralFrame = requestAnimationFrame(tick);
      } else if (autoReveal && showMode === 'neural') {
        neuralAutoPlay();
      }
    };
    neuralFrame = requestAnimationFrame(tick);
  }

  function describeNeural() {
    if (!neuralReady) {
      return 'network: this build carries no usable weights — rebuild with wasm/build.sh';
    }
    if (showMode !== 'exact' && !neuralOn()) {
      return 'network: not scored automatically on a board this large — pick a mode to ask for it';
    }
    if (!neuralOn()) return '';
    var done = neuralTotal - neuralLeft;
    var text = neuralLeft > 0
      ? 'network: ' + done.toLocaleString() + ' / ' + neuralTotal.toLocaleString() + ' cells…'
      : 'network: ' + neuralTotal.toLocaleString() + ' cells';
    if (neuralError !== null) {
      text += ' · off by ' + (neuralError * 100).toFixed(1) + ' points, corrected';
    } else if (showMode === 'neural') {
      text += ' · uncorrected';
    }
    if (neuralOpened || neuralFlagged) {
      text += ' · it has opened ' + neuralOpened + ' and flagged ' + neuralFlagged;
      if (wasm.ms_state() === 2) text += ', then hit a mine';
      if (wasm.ms_state() === 1) text += ', and won';
    }
    return text;
  }

  /**
   * How hard to correct the network during the next scoring pass, or 0.
   *
   * Two gates. Only after a solve that finished: an unfinished one leaves the
   * buffer empty, which would teach the network that everything is safe. And only in the
   * modes where the solver is on screen — in 'neural' the whole point is to see
   * what the network does unaided, and a network being corrected by the solver
   * mid-run is not the thing being measured.
   */
  function learningRate() {
    if (showMode === 'neural') return 0;
    return solved() ? 20 : 0;   // rate 0.02
  }

  /**
   * Whether the network should be scoring this board.
   *
   * The weights live inside the module, so there is nothing to fetch and nothing
   * to wait for: `ms_model_load` parses them the first time and says whether it
   * worked. It can only fail if the build is broken, which is worth saying out
   * loud rather than leaving the overlay quietly dead.
   *
   * The size guard applies only while the mode is still the page's own choice. A
   * pass is one forward pass per cell and starts again after every move, so a
   * 200x200 board would keep a core busy between clicks that nobody asked for —
   * but once a player picks a mode, that *is* the asking, and it is honoured.
   */
  function neuralOn() {
    if (showMode === 'exact' || !neuralReady) return false;
    if (!showChosen && width * height > NEURAL_AUTO_MAX_CELLS) return false;
    if (!wasm.ms_model_load()) {
      neuralReady = false;
      return false;
    }
    return true;
  }

  /**
   * Let the network play, now that the whole board is scored.
   *
   * The solver's auto-play runs to a fixpoint inside the module because each
   * step is cheap. This one cannot: every pass needs the board rescored, which
   * is seconds of work spread over frames. So one pass is applied here and the
   * next arrives the ordinary way — `scheduleCompute` recomputes, `scheduleNeural`
   * rescores, and this runs again off the end of it. It stops when a pass changes
   * nothing, or when the network opens a mine, which it eventually will.
   */
  function neuralAutoPlay() {
    var acted = wasm.ms_neural_auto(NEURAL_OPEN_BELOW, NEURAL_FLAG_ABOVE);
    var opened = acted >>> 16, flagged = acted & 0xffff;
    if (!opened && !flagged) {
      neuralNote(describeNeural() + ' · nothing it is sure enough about');
      return;
    }
    neuralOpened += opened;
    neuralFlagged += flagged;
    render();
    scheduleCompute();
  }

  function setShowMode(mode) {
    showMode = mode;
    showChosen = true;
    // Whatever the last correction measured describes a mode we may have just
    // left, so it stops being said the moment it stops being true.
    neuralError = null;
    cancelNeural();
    render();
    neuralNote(describeNeural());
    scheduleNeural();
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
    // Whatever the network was scoring describes a board that no longer exists.
    cancelNeural();
    // ...and so does the count beside it, which would otherwise sit there
    // claiming the last board's cells until the next pass starts.
    if (neuralOn()) neuralNote('network: …');
    el.sim.textContent = 'calculating…';
    pendingCompute = setTimeout(function () {
      pendingCompute = null;
      wasm.ms_compute();
      // Proof-driven auto-play only when a proof is what is on screen. In
      // 'neural' the network drives instead, which it cannot do until it has
      // scored the board — so that runs off the end of the scoring pass.
      if (autoReveal && showMode !== 'neural') wasm.ms_auto_reveal();
      render();
      renderSim();
      // After the exact numbers, never instead of them.
      scheduleNeural();
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
    cancelNeural();
    neuralError = null;
    neuralOpened = neuralFlagged = 0;
    render();
    // A fresh board still has a probability: mines / cells, the same for every
    // square. Without this the grid would read 0% until the first click.
    scheduleCompute();
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
      if (c !== HIDDEN && c !== FLAGGED) {
        el.hover.textContent = '';
        return;
      }
      var parts = [];
      if (showMode !== 'neural') {
        parts.push(solved()
          ? 'Mine probability: ' + (probs()[i] * 100).toFixed(1) + '%'
          : 'Mine probability: not known — the search did not finish');
      }
      if (neuralOn()) {
        var guess = neuralProbs()[i];
        parts.push('network: ' + (guess >= 0 ? (guess * 100).toFixed(1) + '%' : '…'));
      }
      el.hover.textContent = parts.join(' · ');
    });
    el.grid.addEventListener('mouseleave', function () { el.hover.textContent = ''; });

    on(document.getElementById('btn-new'), 'click', function () {
      newGame(Number(el.width.value), Number(el.height.value), Number(el.mines.value));
    });

    Array.prototype.forEach.call(document.querySelectorAll('.preset'), function (btn) {
      btn.addEventListener('click', function () {
        newGame(Number(btn.dataset.w), Number(btn.dataset.h), Number(btn.dataset.m));
      });
    });

    var probsBtn = on(document.getElementById('btn-probs'), 'click', function () {
      showProbs = !showProbs;
      if (probsBtn) probsBtn.classList.toggle('on', showProbs);
      el.grid.classList.toggle('no-prob', !showProbs);
    });

    var autoBtn = on(document.getElementById('btn-auto'), 'click', function () {
      autoReveal = !autoReveal;
      if (autoBtn) autoBtn.classList.toggle('on', autoReveal);
      if (autoReveal) scheduleCompute();
    });

    var flagBtn = on(document.getElementById('btn-flagmode'), 'click', function () {
      flagMode = !flagMode;
      if (flagBtn) flagBtn.classList.toggle('on', flagMode);
    });

    on(el.show, 'change', function () { setShowMode(el.show.value); });

  }

  /* ------------------------------------------------------------------ start */

  function showBuild() {
    var stamp = document.getElementById('build');
    if (stamp) stamp.textContent = buildId() ? 'build ' + buildId() : 'unversioned build';

  }

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
    showBuild();
    newGame(10, 10, 10);
  }).catch(fail);
})();
