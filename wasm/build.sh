#!/usr/bin/env bash
#
# Build the WebAssembly front-end and refresh the static site in docs/.
#
# Requires only a Rust toolchain with the wasm target:
#     rustup target add wasm32-unknown-unknown
#
# No wasm-bindgen, no wasm-pack, no npm.
set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
out="$root/docs"
artifact="$root/target/wasm32-unknown-unknown/release/minesweeper_wasm.wasm"

RUSTFLAGS="${RUSTFLAGS:-} -C strip=symbols" \
  cargo build --manifest-path "$root/Cargo.toml" \
              -p minesweeper_wasm --target wasm32-unknown-unknown --release

cp "$artifact" "$out/minesweeper.wasm"

# standalone.html inlines the module as base64 so the page also runs straight
# from a file:// URL, where fetching the .wasm would be blocked.
python3 "$root/wasm/bundle.py" "$out"

# The notes are served from docs/ too, and a stale rendering of them drifts from
# its source exactly the way a stale .wasm drifts from the engine.
python3 "$root/notes/render.py" > /dev/null

printf 'wrote %s (%s bytes)\n' "$out/minesweeper.wasm" "$(wc -c < "$out/minesweeper.wasm")"
printf 'wrote %s (%s bytes)\n' "$out/standalone.html" "$(wc -c < "$out/standalone.html")"
