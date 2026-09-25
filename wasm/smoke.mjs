/*
 * Smoke test for the built WebAssembly artifact.
 *
 *     node wasm/smoke.mjs            # tests docs/minesweeper.wasm
 *
 * Needs nothing but a Node with WebAssembly support — no npm install. It drives
 * the same ABI docs/minesweeper.js uses, so it catches a stale or broken
 * docs/minesweeper.wasm before it is committed.
 */
import fs from 'fs';
import path from 'path';
import { fileURLToPath } from 'url';

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const wasmPath = process.argv[2] || path.join(root, 'docs', 'minesweeper.wasm');
const bytes = fs.readFileSync(wasmPath);

const HIDDEN = 10, FLAGGED = 11, VISIBLE_MINE = 9;
let failures = 0;

function check(label, ok, detail) {
  console.log((ok ? 'ok   ' : 'FAIL ') + label + (detail === undefined ? '' : '  → ' + detail));
  if (!ok) failures++;
}

async function load(seedHi, seedLo) {
  const { instance } = await WebAssembly.instantiate(bytes);
  const e = instance.exports;
  e.ms_seed(seedHi, seedLo);
  return {
    e,
    cells: () => new Uint8Array(e.memory.buffer, e.ms_cells_ptr(), e.ms_width() * e.ms_height()),
    probs: () => new Float32Array(e.memory.buffer, e.ms_probs_ptr(), e.ms_width() * e.ms_height()),
    stats: () => new Uint32Array(e.memory.buffer, e.ms_stats_ptr(), e.ms_stats_len()),
    neural: () => new Float32Array(e.memory.buffer, e.ms_neural_probs_ptr(), e.ms_width() * e.ms_height()),
  };
}

// The module must stay import-free: that is what lets a plain page instantiate it.
check('module declares no imports', WebAssembly.Module.imports(new WebAssembly.Module(bytes)).length === 0);

const g = await load(0x1234, 0x5678);
g.e.ms_new(10, 10, 10);
check('board dimensions', g.e.ms_width() === 10 && g.e.ms_height() === 10 && g.e.ms_mines() === 10);
check('starts fully hidden', [...g.cells()].every((c) => c === HIDDEN));
check('starts playing', g.e.ms_state() === 0);

g.e.ms_compute(0);
check('untouched board is uniform', Math.abs(g.probs()[0] - 0.1) < 1e-6, g.probs()[0]);

g.e.ms_reveal(4, 4);
check('first click is never a mine', g.cells()[44] !== VISIBLE_MINE && g.e.ms_state() !== 2);

g.e.ms_flag(0, 0);
check('flag toggles on', g.cells()[0] === FLAGGED && g.e.ms_flags() === 1);
g.e.ms_flag(0, 0);
check('flag toggles off', g.cells()[0] === HIDDEN && g.e.ms_flags() === 0);

// Probabilities over the unopened cells must account for exactly the mines left.
for (const [w, h, m] of [[9, 9, 10], [16, 16, 40], [30, 16, 99]]) {
  const t0 = Date.now();
  g.e.ms_new(w, h, m);
  g.e.ms_reveal(w >> 1, h >> 1);
  g.e.ms_compute(0);
  const c = g.cells(), p = g.probs();
  let sum = 0;
  for (let i = 0; i < w * h; i++) if (c[i] === HIDDEN || c[i] === FLAGGED) sum += p[i];
  check(`probabilities sum to the mine count on ${w}x${h}/${m}`,
        Math.abs(sum - m) < 0.05, `${sum.toFixed(3)} vs ${m}, ${Date.now() - t0}ms`);
}

// Auto-reveal must only ever open cells the estimator calls certainly safe, so a
// full solve driven by it may stall or win, but must never lose on a 0% cell.
g.e.ms_new(16, 16, 40);
g.e.ms_reveal(8, 8);
let rounds = 0;
while (g.e.ms_state() === 0 && rounds++ < 300) {
  g.e.ms_compute(0);
  if (g.e.ms_auto_reveal(0) > 0) continue;
  const c = g.cells(), p = g.probs();
  let best = -1;
  for (let i = 0; i < c.length; i++) if (c[i] === HIDDEN && (best < 0 || p[i] < p[best])) best = i;
  if (best < 0) break;
  g.e.ms_reveal(best % 16, Math.floor(best / 16));
}
check('a full game reaches a terminal state', g.e.ms_state() !== 0 || rounds >= 300,
      'state ' + g.e.ms_state() + ' after ' + rounds + ' rounds');

// Same seed, same board; different seeds, different boards.
async function mineLayout(hi, lo) {
  const s = await load(hi, lo);
  s.e.ms_new(8, 8, 10);
  s.e.ms_reveal(3, 3);
  for (let y = 0; y < 8; y++) for (let x = 0; x < 8; x++) s.e.ms_reveal(x, y);
  return [...s.cells()].map((v) => (v === VISIBLE_MINE ? '*' : '.')).join('');
}
const [a, b, c] = await Promise.all([mineLayout(1, 2), mineLayout(1, 2), mineLayout(42, 7)]);
check('seeding is deterministic', a === b);
check('different seeds give different boards', a !== c);
check('mine count is honoured', [...a].filter((ch) => ch === '*').length === 10);

// Nonsense dimensions are clamped rather than rejected: 200 a side, and never
// so many mines that the board has no safe cell left.
g.e.ms_new(1, 9999, 99999);
check('board size is clamped', g.e.ms_width() === 3 && g.e.ms_height() === 200 && g.e.ms_mines() === 599,
      `${g.e.ms_width()}x${g.e.ms_height()}/${g.e.ms_mines()}`);

// A board well past the old 50x50 ceiling has to stay playable.
{
  const big = await load(11, 13);
  big.e.ms_new(120, 120, 1800);
  const started = Date.now();
  big.e.ms_reveal(60, 60);
  big.e.ms_compute(0);
  const took = Date.now() - started;
  const cells = big.cells(), probs = big.probs();
  let sum = 0;
  for (let i = 0; i < 120 * 120; i++) if (cells[i] === HIDDEN || cells[i] === FLAGGED) sum += probs[i];
  check('a 120x120 board works', big.e.ms_width() === 120 && Math.abs(sum - 1800) < 1,
        `first move ${took}ms, probabilities sum to ${sum.toFixed(1)} vs 1800`);
}

// Auto-reveal only ever opens cells it claims are *proven* safe, so it must never
// end a game. This is the whole-pipeline check: it covers the estimator, the
// propagation fixpoint, and the rule that a sampled 0% is not proof of anything.
// It has already caught one bug that unit tests could not see, where a scaling
// error made every cell on a sparse board read 0% and auto-reveal walked onto a
// mine. It also bounds how long a single click may take.
{
  let games = 0, opened = 0, detonations = 0, slowest = 0, slowestOn = '';
  for (const [w, h, m] of [[30, 30, 250], [30, 16, 99], [50, 50, 400], [16, 16, 40], [50, 50, 150]]) {
    for (let seed = 1; seed <= 4; seed++) {
      const s = await load(seed, seed * 977);
      s.e.ms_new(w, h, m);
      s.e.ms_reveal(w >> 1, h >> 1);
      games++;
      for (let move = 1; move <= 20 && s.e.ms_state() === 0; move++) {
        s.e.ms_compute(0);
        const started = Date.now();
        opened += s.e.ms_auto_reveal(0);
        const took = Date.now() - started;
        if (took > slowest) { slowest = took; slowestOn = `${w}x${h}/${m} seed ${seed}`; }
        if (s.e.ms_state() === 2) { detonations++; break; }
        if (s.e.ms_state() !== 0) break;
        const cells = s.cells(), probs = s.probs();
        let best = -1;
        for (let i = 0; i < w * h; i++) if (cells[i] === HIDDEN && (best < 0 || probs[i] < probs[best])) best = i;
        if (best < 0) break;
        s.e.ms_reveal(best % w, Math.floor(best / w));
      }
    }
  }
  check('auto-reveal never opens a mine', detonations === 0,
        `${games} games, ${opened} cells opened, ${detonations} detonations`);
  check('no single auto-reveal stalls the page', slowest < 5000, `worst ${slowest}ms on ${slowestOn}`);
}

// Auto-play also flags the cells it proves are mines. Checking that is easy once
// the game is lost: losing reveals every mine, turning a correctly flagged cell
// into a revealed mine. Any cell still showing a flag was flagged wrongly.
{
  let flagged = 0, wrong = 0, games = 0;
  for (const [w, h, m] of [[16, 16, 40], [30, 16, 99], [30, 30, 250]]) {
    for (let seed = 1; seed <= 4; seed++) {
      const s = await load(seed * 7, seed);
      s.e.ms_new(w, h, m);
      s.e.ms_reveal(w >> 1, h >> 1);
      games++;
      for (let move = 1; move <= 12 && s.e.ms_state() === 0; move++) {
        s.e.ms_compute(0);
        s.e.ms_auto_reveal(0);
        if (s.e.ms_state() !== 0) break;
        const cells = s.cells(), probs = s.probs();
        let best = -1;
        for (let i = 0; i < w * h; i++) if (cells[i] === HIDDEN && (best < 0 || probs[i] < probs[best])) best = i;
        if (best < 0) break;
        s.e.ms_reveal(best % w, Math.floor(best / w));
      }
      flagged += s.e.ms_flags();
      // Force the game to end so every mine is revealed.
      for (let i = 0; i < w * h && s.e.ms_state() === 0; i++) {
        if (s.cells()[i] === HIDDEN) s.e.ms_reveal(i % w, Math.floor(i / w));
      }
      if (s.e.ms_state() === 2) {
        for (const code of s.cells()) if (code === FLAGGED) wrong++;
      }
    }
  }
  check('auto-play only flags actual mines', wrong === 0,
        `${games} games, ${flagged} flags placed, ${wrong} on non-mines`);
}

// A grid that reads 0% everywhere would mean "all safe"; the estimates must
// always account for exactly the mines that are left.
{
  const s = await load(1, 2);
  s.e.ms_new(50, 50, 150);
  s.e.ms_reveal(25, 25);
  s.e.ms_compute(0);
  const cells = s.cells(), probs = s.probs();
  let sum = 0;
  for (let i = 0; i < 2500; i++) if (cells[i] === HIDDEN || cells[i] === FLAGGED) sum += probs[i];
  check('a large sparse board stays calibrated', Math.abs(sum - 150) < 0.5, `sum ${sum.toFixed(2)} vs 150`);
}

// The neural overlay: the page loads the weights itself, drives the scoring a
// few cells per frame, and shows -1 as "not reached yet". All of that is ABI the
// page depends on and nothing else exercises.
{
  const s = await load(7, 11);
  {
    s.e.ms_new(16, 16, 40);
    check('nothing is parsed until the overlay is used', s.e.ms_model_ready() === 0);
    check('scoring without a model does nothing', s.e.ms_neural_begin() === 0);

    // The weights ship inside the module, so this is the whole loading story —
    // no fetch, no second asset, nothing a host can serve wrongly.
    check('the built-in weights parse', s.e.ms_model_load() === 1 && s.e.ms_model_ready() === 1);

    s.e.ms_reveal(8, 8);
    s.e.ms_compute(0);
    const total = s.e.ms_neural_begin();
    const hidden = [...s.cells()].filter((c) => c === HIDDEN || c === FLAGGED).length;
    check('every unopened cell is scheduled', total === hidden, `${total} vs ${hidden}`);
    check('nothing is scored before the first step',
          [...s.neural()].every((v) => v === -1));

    // Drive it the way the page does: a bounded number of cells at a time.
    let left = total, steps = 0;
    const t0 = Date.now();
    while (left > 0 && steps < 10000) { left = s.e.ms_neural_step(32); steps++; }
    check('stepping finishes the board', left === 0, `${steps} steps, ${Date.now() - t0}ms`);

    const guess = s.neural(), cells = s.cells();
    let scored = 0, bad = 0;
    for (let i = 0; i < 256; i++) {
      if (cells[i] === HIDDEN || cells[i] === FLAGGED) {
        scored++;
        if (!(guess[i] >= 0 && guess[i] <= 1)) bad++;
      } else if (guess[i] !== -1) {
        bad++;   // an open cell must never be given a guess
      }
    }
    check('every guess is a probability', bad === 0, `${scored} scored, ${bad} bad`);

    // A network that has learned nothing answers the same everywhere, which is
    // also exactly what a patch layout mismatch looks like.
    const values = [...guess].filter((v) => v >= 0);
    const spread = Math.max(...values) - Math.min(...values);
    check('the network distinguishes cells', spread > 0.01, `spread ${spread.toFixed(3)}`);

    // Correcting from the exact solve must move it towards those numbers.
    const probs = s.probs();
    const errorOf = (g) => {
      let sum = 0, n = 0;
      for (let i = 0; i < 256; i++) {
        if (cells[i] === HIDDEN || cells[i] === FLAGGED) { sum += Math.abs(g[i] - probs[i]); n++; }
      }
      return sum / n;
    };
    const before = errorOf(guess);
    for (let i = 0; i < 25; i++) s.e.ms_neural_learn(100);
    s.e.ms_neural_begin();
    while (s.e.ms_neural_step(64) > 0) { /* score the same board again */ }
    const after = errorOf(s.neural());
    check('learning moves the network towards the exact answer', after < before,
          `${before.toFixed(4)} → ${after.toFixed(4)}`);

    // Last, because ms_new reallocates and every view above it goes stale.
    //
    // A new board must not throw the weights away — they describe the game, not
    // the position — and must not leave the buffer reading 0, which every other
    // buffer here uses to mean "proven safe". A whole board nobody has scored
    // showing 0% is the exact shape of that mistake, and it is what the page did.
    s.e.ms_new(20, 20, 60);
    check('a new game keeps the model', s.e.ms_model_ready() === 1);
    check('a new game scores nothing yet',
          [...s.neural()].every((v) => v === -1),
          `${[...s.neural()].filter((v) => v !== -1).length} cells claim an answer`);
    check('a new game can still be scored', s.e.ms_neural_begin() === 400);

    // Letting the network play is a separate export from ms_auto_reveal on
    // purpose: it acts on an estimate, so it can open a mine. What it must never
    // do is act on a cell nobody scored — NOT_SCORED is -1, which is below every
    // threshold, so a half-scored board would look like a field of certainties.
    check('it will not play a board it has not scored', s.e.ms_neural_auto(50, 950) === 0);
  }
}

// The network playing for itself.
//
// An untouched board is uniform: nothing on it is under 5% or over 95%, and
// declining to act there is correct, not a failure. So the board gets one
// opening click to cascade, and the network takes over from there — which is
// what the page does, since in network-only mode the solver's auto-play is not
// run at all. Running it first would be the wrong test: once the solver has
// taken every proven move, what is left is precisely the set of positions
// nothing can be certain about, and the network is right to decline those too.
{
  const [W, H, M] = [30, 16, 99];
  let s = null;
  for (let seed = 1; seed <= 8 && !s; seed++) {
    const t = await load(seed, seed * 7919);
    t.e.ms_model_load();
    t.e.ms_new(W, H, M);
    t.e.ms_reveal(W >> 1, H >> 1);
    const opened = [...t.cells()].filter((c) => c !== HIDDEN).length;
    if (t.e.ms_state() === 0 && opened > 10) s = t;
  }

  if (!s) {
    check('a mid-game board to hand the network', false, 'no seed produced one');
  } else {
    s.e.ms_compute(0);
    s.e.ms_neural_begin();
    while (s.e.ms_neural_step(256) > 0) { /* finish scoring */ }

    let opened = 0, flagged = 0, passes = 0;
    while (s.e.ms_state() === 0 && passes < 40) {
      const acted = s.e.ms_neural_auto(50, 950);
      if (!acted) break;
      opened += acted >>> 16;
      flagged += acted & 0xffff;
      passes++;
      s.e.ms_compute(0);
      s.e.ms_neural_begin();
      while (s.e.ms_neural_step(256) > 0) { /* rescore */ }
    }
    check('the network can play a board', opened + flagged > 0,
          `${opened} opened, ${flagged} flagged over ${passes} passes, state ${s.e.ms_state()}`);

    // However it ended, the board must be left consistent: the reported flag
    // count has to match the grid it just wrote.
    check('the board is left consistent',
          [...s.cells()].filter((c) => c === FLAGGED).length === s.e.ms_flags());
  }
}

console.log(failures ? `\n${failures} check(s) failed` : '\nall checks passed');
process.exit(failures ? 1 : 0);
