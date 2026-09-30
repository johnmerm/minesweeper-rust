//! The console front-end, in a browser tab.
//!
//! `GET /terminal` is a page running xterm.js; `GET /terminal/ws` upgrades to a
//! WebSocket, and the server runs the real `cli` binary on a pseudo-terminal at
//! the other end of it. Keystrokes go down as binary frames, output comes back
//! the same way, and a text frame `{"cols":…,"rows":…}` resizes the terminal. So
//! what the tab shows is the terminal front-end itself, colours and all — not a
//! port of it.
//!
//! A page that starts processes is worth being careful with, so:
//! - it only ever runs the `cli` binary built beside this one — never a shell,
//!   and the only thing the browser chooses is a board size, parsed as numbers;
//! - the upgrade is refused unless `Origin` names this server, so another site
//!   open in the same browser cannot drive it (a WebSocket ignores CORS);
//! - there is a cap on how many run at once, and closing the tab kills the process.

use actix_web::{get, web, HttpRequest, HttpResponse};
use actix_ws::Message;
use portable_pty::{native_pty_system, ChildKiller, CommandBuilder, PtySize};
use serde::Deserialize;
use std::io::{Read, Write};
use std::path::PathBuf;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

/// How many consoles may run at once.
const MAX_SESSIONS: usize = 8;

/// Consoles currently running.
#[derive(Default)]
pub struct Sessions(AtomicUsize);

/// Held for as long as a console runs; gives its slot back when dropped.
struct Slot(Arc<Sessions>);

impl Slot {
    fn take(sessions: &Arc<Sessions>) -> Option<Self> {
        let taken = sessions
            .0
            .fetch_update(Ordering::SeqCst, Ordering::SeqCst, |n| (n < MAX_SESSIONS).then_some(n + 1))
            .is_ok();
        taken.then(|| Slot(Arc::clone(sessions)))
    }
}

impl Drop for Slot {
    fn drop(&mut self) {
        self.0 .0.fetch_sub(1, Ordering::SeqCst);
    }
}

/// The `cli` binary cargo built next to this one.
///
/// Located rather than configured: `cargo build` puts every workspace binary in
/// the same `target/<profile>/` directory, and there is then no path for anyone
/// to point somewhere else.
fn cli_binary() -> Option<PathBuf> {
    let exe = std::env::current_exe().ok()?;
    let cli = exe.with_file_name(format!("cli{}", std::env::consts::EXE_SUFFIX));
    cli.is_file().then_some(cli)
}

#[get("/terminal")]
pub async fn page(tera: web::Data<tera::Tera>) -> HttpResponse {
    match tera.render("terminal.html", &tera::Context::new()) {
        Ok(page) => HttpResponse::Ok().content_type("text/html; charset=utf-8").body(page),
        Err(error) => HttpResponse::InternalServerError().body(format!("template error: {error}")),
    }
}

/// Board size, as the `cli` binary takes it on its command line.
#[derive(Deserialize)]
pub struct Board {
    width: Option<u16>,
    height: Option<u16>,
    mines: Option<u32>,
    cols: Option<u16>,
    rows: Option<u16>,
}

/// Whether the request's `Origin` is this server.
///
/// Browsers send `Origin` on every WebSocket handshake, and a page from anywhere
/// can open one to `127.0.0.1` — the same-origin policy does not apply to them.
/// A handshake without the header did not come from a browser page at all.
fn same_origin(req: &HttpRequest) -> bool {
    let header = |name| req.headers().get(name).and_then(|v| v.to_str().ok());
    let (Some(origin), Some(host)) = (header("origin"), header("host")) else {
        return header("origin").is_none();
    };
    origin
        .strip_prefix("http://")
        .or_else(|| origin.strip_prefix("https://"))
        .is_some_and(|rest| rest == host)
}

#[get("/terminal/ws")]
pub async fn socket(
    req: HttpRequest,
    body: web::Payload,
    board: web::Query<Board>,
    sessions: web::Data<Arc<Sessions>>,
) -> Result<HttpResponse, actix_web::Error> {
    if !same_origin(&req) {
        return Ok(HttpResponse::Forbidden().body("cross-origin terminal connections are refused"));
    }
    let Some(cli) = cli_binary() else {
        return Ok(HttpResponse::ServiceUnavailable()
            .body("the cli binary is not built — run `cargo build -p cli`, then reload"));
    };
    let Some(slot) = Slot::take(&sessions) else {
        return Ok(HttpResponse::ServiceUnavailable()
            .body(format!("{MAX_SESSIONS} consoles are already running; close one and reload")));
    };

    let size = PtySize {
        cols: board.cols.unwrap_or(100).clamp(20, 500),
        rows: board.rows.unwrap_or(40).clamp(10, 300),
        pixel_width: 0,
        pixel_height: 0,
    };
    let pty = native_pty_system()
        .openpty(size)
        .map_err(|e| actix_web::error::ErrorInternalServerError(format!("no pseudo-terminal: {e}")))?;

    let mut command = CommandBuilder::new(cli);
    // Numbers only: `Query` has already refused anything that is not one, so
    // nothing the browser sends becomes an argument the binary does not expect.
    if let (Some(w), Some(h), Some(m)) = (board.width, board.height, board.mines) {
        command.args([w.to_string(), h.to_string(), m.to_string()]);
    }
    command.env("TERM", "xterm-256color");
    command.env("COLORTERM", "truecolor");
    let mut child = pty
        .slave
        .spawn_command(command)
        .map_err(|e| actix_web::error::ErrorInternalServerError(format!("could not start cli: {e}")))?;
    // The child holds its own end now; keeping ours open would stop the reader
    // ever seeing end-of-file when the child exits.
    drop(pty.slave);
    let mut killer = child.clone_killer();

    let reader = pty.master.try_clone_reader();
    let writer = pty.master.take_writer();
    let (mut reader, mut writer) = match (reader, writer) {
        (Ok(r), Ok(w)) => (r, w),
        _ => {
            let _ = killer.kill();
            return Ok(HttpResponse::InternalServerError().body("could not attach to the pseudo-terminal"));
        }
    };

    let (response, mut session, mut messages) = actix_ws::handle(&req, body)?;

    // The pseudo-terminal is blocking I/O, so it gets threads of its own; the
    // async side only ever touches the channels.
    let (out_tx, mut out_rx) = tokio::sync::mpsc::channel::<Vec<u8>>(64);
    std::thread::spawn(move || {
        let mut buf = [0u8; 8192];
        while let Ok(n @ 1..) = reader.read(&mut buf) {
            if out_tx.blocking_send(buf[..n].to_vec()).is_err() {
                break;
            }
        }
    });
    let (in_tx, in_rx) = std::sync::mpsc::channel::<Vec<u8>>();
    std::thread::spawn(move || {
        for bytes in in_rx {
            if writer.write_all(&bytes).and_then(|()| writer.flush()).is_err() {
                break;
            }
        }
    });
    std::thread::spawn(move || {
        let _ = child.wait();
    });

    // One task per direction; whichever ends first — the console quitting, or
    // the tab going away — ends the other, and the process with them.
    let killer = Arc::new(std::sync::Mutex::new(killer));
    let kill_on_exit = Arc::clone(&killer);
    let mut to_browser = session.clone();
    actix_web::rt::spawn(async move {
        while let Some(bytes) = out_rx.recv().await {
            if to_browser.binary(bytes).await.is_err() {
                break;
            }
        }
        // The console quit (`q`), or died: say so by closing the socket.
        let _ = to_browser.close(None).await;
        kill(&kill_on_exit);
    });

    let master = pty.master;
    actix_web::rt::spawn(async move {
        let _slot = slot;
        while let Some(message) = messages.recv().await {
            match message {
                Ok(Message::Binary(bytes)) => {
                    if in_tx.send(bytes.to_vec()).is_err() {
                        break;
                    }
                }
                Ok(Message::Text(text)) => {
                    if let Ok(resize) = serde_json::from_str::<Resize>(&text) {
                        let _ = master.resize(PtySize {
                            cols: resize.cols.clamp(20, 500),
                            rows: resize.rows.clamp(10, 300),
                            pixel_width: 0,
                            pixel_height: 0,
                        });
                    }
                }
                Ok(Message::Ping(bytes)) => {
                    if session.pong(&bytes).await.is_err() {
                        break;
                    }
                }
                Ok(Message::Close(_)) | Err(_) => break,
                Ok(_) => {}
            }
        }
        // However it ended, the process does not outlive the tab.
        kill(&killer);
        let _ = session.close(None).await;
    });

    Ok(response)
}

fn kill(killer: &std::sync::Mutex<Box<dyn ChildKiller + Send + Sync>>) {
    // An error means it had already exited, which is the outcome wanted.
    if let Ok(mut killer) = killer.lock() {
        let _ = killer.kill();
    }
}

#[derive(Deserialize)]
struct Resize {
    cols: u16,
    rows: u16,
}
