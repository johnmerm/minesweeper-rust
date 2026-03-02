use actix_web::{get, post, web, App, HttpResponse, HttpServer, Responder};
use minesweeper_core::{Minesweeper, CellState, CellContent};
use minesweeper_core::probability::{MonteCarlo, ConstraintSearch, SimUpdate};
use std::sync::{Mutex, mpsc::Sender};
use tera::{Tera, Context};
use serde::{Deserialize, Serialize};

#[derive(Serialize)]
struct CellView {
    state: CellState,
    content: CellContent,
    prob_color: String,
    prob_pct: String,
}

struct AppState {
    game: Mutex<Minesweeper>,
    /// Last-used board settings, shown as defaults in the New Game form.
    settings: Mutex<(usize, usize, usize)>,
    tera: Tera,
}

#[derive(Deserialize)]
struct MoveParams {
    x: usize,
    y: usize,
}

#[derive(Deserialize)]
struct NewGameParams {
    width:  Option<usize>,
    height: Option<usize>,
    mines:  Option<usize>,
}

/// Run a strategy synchronously, returning (probs, valid, attempts, memory_bytes).
fn run_sync(run: impl FnOnce(Sender<SimUpdate>)) -> (Vec<Vec<f64>>, usize, usize, usize) {
    let (tx, rx) = std::sync::mpsc::channel();
    run(tx);
    let mut probs = Vec::new();
    let mut valid = 0;
    let mut attempts = 0;
    let mut mem = 0;
    while let Ok(update) = rx.recv() {
        if let SimUpdate::Done { probs: p, valid: v, attempts: a, memory_bytes: m, .. } = update {
            probs = p;
            valid = v;
            attempts = a;
            mem = m;
            break;
        }
    }
    (probs, valid, attempts, mem)
}

fn fmt_memory(bytes: usize) -> String {
    match bytes {
        b if b < 1_024          => format!("{} B", b),
        b if b < 1_024 * 1_024  => format!("{:.1} KB", b as f64 / 1_024.0),
        b                       => format!("{:.1} MB", b as f64 / 1_048_576.0),
    }
}

#[get("/")]
async fn index(data: web::Data<AppState>) -> impl Responder {
    let game = data.game.lock().unwrap();
    let (sw, sh, sm) = *data.settings.lock().unwrap();

    let (mc_probs, mc_valid, mc_attempts, mc_mem) =
        run_sync(|tx| MonteCarlo::new().calculate_with_progress(&game, tx));
    let (cs_probs, cs_valid, cs_attempts, cs_mem) =
        run_sync(|tx| ConstraintSearch::new().calculate_with_progress(&game, tx));

    let probs = if cs_valid > 0 { cs_probs } else { mc_probs };

    let mc_status = format!(
        "MC: {} valid / {} sampled  [{}]",
        mc_valid, mc_attempts, fmt_memory(mc_mem)
    );
    let cs_status = format!(
        "CS: {} layouts / {} steps  [{}]",
        cs_valid, cs_attempts, fmt_memory(cs_mem)
    );

    let grid_view: Vec<Vec<CellView>> = (0..game.height)
        .map(|y| {
            (0..game.width)
                .map(|x| {
                    let p = probs[y][x];
                    let r = (204.0 + 51.0 * p).round() as u8;
                    let g = (204.0 * (1.0 - p)).round() as u8;
                    CellView {
                        state: game.grid[y][x].state,
                        content: game.grid[y][x].content,
                        prob_color: format!("rgb({},{},{})", r, g, g),
                        prob_pct: format!("{:.0}%", p * 100.0),
                    }
                })
                .collect()
        })
        .collect();

    let mut context = Context::new();
    context.insert("width", &game.width);
    context.insert("height", &game.height);
    context.insert("mines_count", &game.mines_count);
    context.insert("state", &game.state);
    context.insert("grid", &grid_view);
    context.insert("mc_status", &mc_status);
    context.insert("cs_status", &cs_status);
    context.insert("settings_width",  &sw);
    context.insert("settings_height", &sh);
    context.insert("settings_mines",  &sm);

    let rendered = data.tera.render("index.html", &context).unwrap();
    HttpResponse::Ok().body(rendered)
}

#[post("/reveal")]
async fn reveal(data: web::Data<AppState>, params: web::Form<MoveParams>) -> impl Responder {
    let mut game = data.game.lock().unwrap();
    game.reveal(params.x, params.y);
    HttpResponse::SeeOther().insert_header(("Location", "/")).finish()
}

#[post("/flag")]
async fn flag(data: web::Data<AppState>, params: web::Form<MoveParams>) -> impl Responder {
    let mut game = data.game.lock().unwrap();
    game.toggle_flag(params.x, params.y);
    HttpResponse::SeeOther().insert_header(("Location", "/")).finish()
}

#[post("/new")]
async fn new_game(data: web::Data<AppState>, params: web::Form<NewGameParams>) -> impl Responder {
    let (cur_w, cur_h, cur_m) = *data.settings.lock().unwrap();
    let w = params.width .unwrap_or(cur_w).clamp(3, 50);
    let h = params.height.unwrap_or(cur_h).clamp(3, 50);
    let m = params.mines .unwrap_or(cur_m).clamp(1, w * h - 1);

    *data.settings.lock().unwrap() = (w, h, m);
    *data.game.lock().unwrap() = Minesweeper::new(w, h, m);

    HttpResponse::SeeOther().insert_header(("Location", "/")).finish()
}

#[actix_web::main]
async fn main() -> std::io::Result<()> {
    let tera = Tera::new("web/templates/**/*").unwrap();
    let game = Mutex::new(Minesweeper::new(10, 10, 10));
    let settings = Mutex::new((10usize, 10usize, 10usize));
    let app_data = web::Data::new(AppState { game, settings, tera });

    println!("Starting web server at http://127.0.0.1:8080");

    HttpServer::new(move || {
        App::new()
            .app_data(app_data.clone())
            .service(index)
            .service(reveal)
            .service(flag)
            .service(new_game)
    })
    .bind(("127.0.0.1", 8080))?
    .run()
    .await
}
