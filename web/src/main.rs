use actix_web::{get, post, web, App, HttpResponse, HttpServer, Responder};
use minesweeper_core::Minesweeper;
use std::sync::Mutex;
use tera::{Tera, Context};
use serde::Deserialize;

struct AppState {
    game: Mutex<Minesweeper>,
    tera: Tera,
}

#[derive(Deserialize)]
struct MoveParams {
    x: usize,
    y: usize,
}

#[get("/")]
async fn index(data: web::Data<AppState>) -> impl Responder {
    let game = data.game.lock().unwrap();
    let mut context = Context::new();
    
    // Convert grid to a serializable format if needed, but Minesweeper should be fine if we wrapped it or just pass fields.
    // Since Minesweeper struct itself isn't Serialize (I didn't derive it on the main struct, only sub-parts),
    // I will construct the context manually.
    
    context.insert("width", &game.width);
    context.insert("height", &game.height);
    context.insert("mines_count", &game.mines_count);
    context.insert("state", &game.state);
    context.insert("grid", &game.grid);

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
async fn new_game(data: web::Data<AppState>) -> impl Responder {
    let mut game = data.game.lock().unwrap();
    *game = Minesweeper::new(10, 10, 10);
    HttpResponse::SeeOther().insert_header(("Location", "/")).finish()
}

#[actix_web::main]
async fn main() -> std::io::Result<()> {
    let tera = Tera::new("web/templates/**/*").unwrap();
    let game = Mutex::new(Minesweeper::new(10, 10, 10));
    let app_data = web::Data::new(AppState { game, tera });

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