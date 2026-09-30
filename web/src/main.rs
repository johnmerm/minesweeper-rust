//! Web front-end: one shared game, served to a page that looks and behaves like
//! the WebAssembly one in `docs/`.
//!
//! The server owns the game. The exact search and the network run on a
//! [`session`] thread, exactly as they do behind the Qt GUI, and their answers
//! land in the shared [`Controller`] as they arrive — the network scores a few
//! cells at a time, and a whole board can take seconds. The page therefore does
//! not reload per move: it posts moves as JSON, gets the board back, and asks
//! again (`GET /state?since=`) for as long as the server says answers are still
//! on their way.
//!
//! The same server can also hand out the WebAssembly build in `docs/`, where the
//! game runs in the browser and the server only delivers files: mounted at
//! `/wasm/` beside the server-side game, or on its own with `--wasm`. And
//! `/terminal` runs the console front-end itself in the browser — see [`terminal`].

mod terminal;

use actix_web::{get, post, web, App, HttpResponse, HttpServer, Responder};
use minesweeper_core::session::{self, Controller, Show, NOT_SCORED};
use minesweeper_core::{CellContent, CellState, GameState};
use serde::{Deserialize, Serialize};
use std::sync::{Arc, Mutex, MutexGuard};
use tera::{Context, Tera};

/// The game, plus a counter the page uses to ask whether anything changed.
struct Shared {
    ctl: Controller,
    /// Bumped on every change a client could see.
    revision: u64,
}

struct AppState {
    shared: Arc<Mutex<Shared>>,
    tera: Tera,
}

impl AppState {
    /// The shared game. A panic while it was held leaves a board that is still
    /// a board, so a poisoned lock is recovered rather than taking the server down.
    fn lock(&self) -> MutexGuard<'_, Shared> {
        self.shared.lock().unwrap_or_else(|poisoned| poisoned.into_inner())
    }
}

/// Everything the page draws, in one response.
///
/// Cells use the wasm crate's byte encoding — `0..=8` a visible count, `9` a
/// visible mine, `10` hidden, `11` flagged — so the page's drawing code reads the
/// same values whichever back end it is talking to.
#[derive(Serialize)]
struct StateView {
    revision: u64,
    width: usize,
    height: usize,
    mines: usize,
    flags: usize,
    /// 0 playing, 1 won, 2 lost.
    state: u8,
    cells: Vec<u8>,
    /// `null` while the solve is still running.
    exact: Option<ExactView>,
    neural: NeuralView,
    show: i32,
    auto_play: bool,
    elapsed_ms: u64,
    timing: bool,
    /// More is coming; ask again.
    busy: bool,
}

#[derive(Serialize)]
struct ExactView {
    /// False when the search gave up. `probs` is then empty — never zeros, which
    /// every front-end reads as proof that a cell is safe.
    solved: bool,
    probs: Vec<f32>,
    layouts: usize,
    nodes: usize,
    memory_bytes: usize,
    cache_hits: u32,
    cache_misses: u32,
}

#[derive(Serialize)]
struct NeuralView {
    /// Whether the network is scoring this board at all.
    on: bool,
    /// Whether it would, but the board is past the size guard until a mode is picked.
    held_back: bool,
    broken: bool,
    /// `-1` where it has not looked yet; empty when it is off.
    probs: Vec<f32>,
    total: usize,
    remaining: usize,
    error: Option<f32>,
    stuck: bool,
    opened: u32,
    flagged: u32,
}

impl StateView {
    fn of(shared: &Shared) -> Self {
        let ctl = &shared.ctl;
        let game = &ctl.game;
        let cells: Vec<u8> = game.grid.iter().flatten().map(encode_cell).collect();
        let flags = cells.iter().filter(|&&c| c == FLAGGED).count();
        let exact = ctl.exact.as_ref().map(|e| ExactView {
            solved: e.probs.is_some(),
            probs: e.probs.clone().unwrap_or_default(),
            layouts: e.layouts,
            nodes: e.nodes,
            memory_bytes: e.memory_bytes,
            cache_hits: e.cache_hits,
            cache_misses: e.cache_misses,
        });
        let on = ctl.neural_wanted();
        let n = &ctl.neural;
        StateView {
            revision: shared.revision,
            width: game.width,
            height: game.height,
            mines: game.mines_count,
            flags,
            state: match game.state {
                GameState::Playing => 0,
                GameState::Won => 1,
                GameState::Lost => 2,
            },
            cells,
            exact,
            neural: NeuralView {
                on,
                held_back: !on && !n.broken && ctl.show != Show::Exact,
                broken: n.broken,
                probs: if on {
                    // Four decimals is finer than anything drawn, and a quarter
                    // of the bytes on a big board polled several times a second.
                    n.probs.iter().map(|&p| if p < 0.0 { NOT_SCORED } else { round4(p) }).collect()
                } else {
                    Vec::new()
                },
                total: n.total,
                remaining: n.remaining,
                error: n.error,
                stuck: n.stuck,
                opened: n.opened,
                flagged: n.flagged,
            },
            show: ctl.show.index(),
            auto_play: ctl.auto_play,
            elapsed_ms: ctl.elapsed().as_millis() as u64,
            timing: ctl.timing(),
            busy: ctl.busy(),
        }
    }
}

fn round4(p: f32) -> f32 {
    (p * 10_000.0).round() / 10_000.0
}

const VISIBLE_MINE: u8 = 9;
const HIDDEN: u8 = 10;
const FLAGGED: u8 = 11;

fn encode_cell(cell: &minesweeper_core::Cell) -> u8 {
    match cell.state {
        CellState::Hidden => HIDDEN,
        CellState::Flagged => FLAGGED,
        CellState::Visible => match cell.content {
            CellContent::Mine => VISIBLE_MINE,
            CellContent::Empty(n) => n.min(8),
        },
    }
}

/// Answer with the board after a change.
fn changed(mut shared: MutexGuard<'_, Shared>) -> HttpResponse {
    shared.revision += 1;
    HttpResponse::Ok().json(StateView::of(&shared))
}

#[derive(Deserialize)]
struct MoveParams {
    x: usize,
    y: usize,
}

#[derive(Deserialize)]
struct NewGameParams {
    width: usize,
    height: usize,
    mines: usize,
}

#[derive(Deserialize)]
struct SettingsParams {
    auto_play: Option<bool>,
    show: Option<i32>,
}

#[derive(Deserialize)]
struct Since {
    since: Option<u64>,
}

#[get("/")]
async fn index(data: web::Data<AppState>) -> HttpResponse {
    let initial = serde_json_string(&StateView::of(&data.lock()));
    let mut context = Context::new();
    context.insert("initial", &initial);
    match data.tera.render("index.html", &context) {
        Ok(page) => HttpResponse::Ok().content_type("text/html; charset=utf-8").body(page),
        Err(error) => HttpResponse::InternalServerError().body(format!("template error: {error}")),
    }
}

/// The board, or `204 No Content` when nothing changed since `since`.
#[get("/state")]
async fn state(data: web::Data<AppState>, query: web::Query<Since>) -> HttpResponse {
    let shared = data.lock();
    if query.since == Some(shared.revision) {
        return HttpResponse::NoContent().finish();
    }
    HttpResponse::Ok().json(StateView::of(&shared))
}

#[post("/reveal")]
async fn reveal(data: web::Data<AppState>, params: web::Json<MoveParams>) -> impl Responder {
    let mut shared = data.lock();
    shared.ctl.reveal(params.x, params.y);
    changed(shared)
}

#[post("/flag")]
async fn flag(data: web::Data<AppState>, params: web::Json<MoveParams>) -> impl Responder {
    let mut shared = data.lock();
    shared.ctl.flag(params.x, params.y);
    changed(shared)
}

#[post("/new")]
async fn new_game(data: web::Data<AppState>, params: web::Json<NewGameParams>) -> impl Responder {
    let mut shared = data.lock();
    shared.ctl.new_game(params.width, params.height, params.mines);
    changed(shared)
}

#[post("/settings")]
async fn settings(data: web::Data<AppState>, params: web::Json<SettingsParams>) -> impl Responder {
    let mut shared = data.lock();
    if let Some(on) = params.auto_play {
        shared.ctl.set_auto_play(on);
    }
    if let Some(show) = params.show {
        shared.ctl.set_show(Show::from_index(show));
    }
    changed(shared)
}

/// JSON for embedding in a `<script>`. Serialising plain data cannot fail, and
/// if it somehow did the page fetches `/state` on load anyway. `</` is escaped so
/// nothing in it can ever close the script element early.
fn serde_json_string(view: &StateView) -> String {
    serde_json::to_string(view)
        .unwrap_or_else(|_| "null".into())
        .replace("</", "<\\/")
}

/// The static site `wasm/build.sh` writes: `index.html`, `minesweeper.js`, the
/// `.wasm`, `standalone.html` and the rendered notes.
const DOCS_DIR: &str = "docs";

/// `docs/` as a static site under `mount`.
///
/// Read from disk per request rather than embedded, so a `./wasm/build.sh`
/// shows up on the next reload without restarting the server — the same reason
/// the site is served straight from the repository. The trailing-slash redirect
/// matters: `index.html` refers to `minesweeper.js` and `notes/` relatively, and
/// without the slash they would resolve against the server root.
fn docs_site(mount: &str) -> actix_files::Files {
    actix_files::Files::new(mount, DOCS_DIR)
        .index_file("index.html")
        .redirect_to_slash_directory()
}

/// What to run, from the command line: `web [--wasm] [--port N]`.
struct Options {
    /// Serve only the WebAssembly site, with no server-side game.
    wasm: bool,
    port: u16,
}

fn parse_options() -> std::io::Result<Options> {
    let mut wasm = false;
    let mut port = None;
    let mut args = std::env::args().skip(1);
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--wasm" => wasm = true,
            "--port" => {
                let value = args.next().unwrap_or_default();
                port = Some(value.parse().map_err(|_| {
                    std::io::Error::other(format!("--port wants a number, not {value:?}"))
                })?);
            }
            other => {
                return Err(std::io::Error::other(format!(
                    "unknown argument {other:?}; usage: web [--wasm] [--port N]"
                )))
            }
        }
    }
    // Different defaults, so the two can run side by side.
    Ok(Options { wasm, port: port.unwrap_or(if wasm { 8081 } else { 8080 }) })
}

/// Serve only `docs/`: the game is entirely in the browser.
async fn serve_wasm(port: u16) -> std::io::Result<()> {
    if !std::path::Path::new(DOCS_DIR).join("minesweeper.wasm").is_file() {
        return Err(std::io::Error::other(
            "docs/minesweeper.wasm not found — run from the workspace root, after ./wasm/build.sh",
        ));
    }
    println!("Serving the WebAssembly build at http://127.0.0.1:{port}/");
    println!("  (single-file version: http://127.0.0.1:{port}/standalone.html)");
    HttpServer::new(|| App::new().service(docs_site("/")))
        .bind(("127.0.0.1", port))?
        .run()
        .await
}

#[actix_web::main]
async fn main() -> std::io::Result<()> {
    let options = parse_options()?;
    if options.wasm {
        return serve_wasm(options.port).await;
    }

    let tera = Tera::new("web/templates/**/*").map_err(|error| {
        std::io::Error::other(format!(
            "{error} — run from the workspace root, so web/templates/ resolves"
        ))
    })?;

    let (jobs, inbox) = std::sync::mpsc::channel();
    let shared = Arc::new(Mutex::new(Shared { ctl: Controller::new(jobs), revision: 0 }));
    let sink = Arc::clone(&shared);
    std::thread::spawn(move || {
        session::run(inbox, move |reply| {
            let mut shared = sink.lock().unwrap_or_else(|poisoned| poisoned.into_inner());
            if shared.ctl.on_reply(reply) {
                shared.revision += 1;
            }
        })
    });

    let templates = web::Data::new(tera.clone());
    let consoles = web::Data::new(Arc::new(terminal::Sessions::default()));
    let app_data = web::Data::new(AppState { shared, tera });
    let port = options.port;
    println!("Starting web server at http://127.0.0.1:{port}");
    println!("  (the WebAssembly build is at http://127.0.0.1:{port}/wasm/)");
    println!("  (the console, in a browser terminal, is at http://127.0.0.1:{port}/terminal)");

    HttpServer::new(move || {
        App::new()
            .app_data(app_data.clone())
            .app_data(templates.clone())
            .app_data(consoles.clone())
            .service(index)
            .service(state)
            .service(reveal)
            .service(flag)
            .service(new_game)
            .service(settings)
            .service(docs_site("/wasm"))
            .service(terminal::page)
            .service(terminal::socket)
    })
    .bind(("127.0.0.1", port))?
    .run()
    .await
}
