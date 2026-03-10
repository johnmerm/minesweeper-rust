use qmetaobject::prelude::*;
use qmetaobject::{QVariantList, QVariantMap};
use minesweeper_core::{Minesweeper, CellState, CellContent, GameState};
use minesweeper_core::probability::{MonteCarlo, ConstraintSearch, SimUpdate, Strategy};
#[cfg(feature = "neural")]
use minesweeper_core::probability::NeuralNetwork;
use minesweeper_core::probability::monte_carlo::combinations;
use cstr::cstr;
use std::collections::HashSet;
use std::sync::mpsc::Receiver;

#[derive(QObject, Default)]
struct MinesweeperGui {
    base: qt_base_class!(trait QObject),

    board_width: qt_property!(i32; NOTIFY boardChanged),
    board_height: qt_property!(i32; NOTIFY boardChanged),

    cells: qt_property!(QVariantList; NOTIFY boardChanged),
    status_text: qt_property!(QString; NOTIFY boardChanged),
    /// Status line for the Monte Carlo sampling strategy.
    sim_status: qt_property!(QString; NOTIFY boardChanged),
    /// Status line for the Constraint Search (DFS) strategy.
    cs_status: qt_property!(QString; NOTIFY boardChanged),
    /// Status line for the Neural Network strategy.
    nn_status: qt_property!(QString; NOTIFY boardChanged),
    layout_count: qt_property!(QString; NOTIFY boardChanged),
    /// When true, cells whose mine probability is exactly 0 are revealed automatically.
    auto_reveal: qt_property!(bool; NOTIFY boardChanged),

    boardChanged: qt_signal!(),

    init: qt_method!(fn(&mut self)),
    reveal: qt_method!(fn(&mut self, index: i32)),
    flag: qt_method!(fn(&mut self, index: i32)),
    reset: qt_method!(fn(&mut self, w: i32, h: i32, m: i32)),
    check_prob_update: qt_method!(fn(&mut self)),

    game: Option<Minesweeper>,
    /// Best-priority probs — used for cell background colour.
    probs: Vec<Vec<f64>>,
    /// Per-strategy probability grids, shown independently in each cell.
    mc_probs: Vec<Vec<f64>>,
    cs_probs: Vec<Vec<f64>>,
    #[cfg(feature = "neural")] nn_probs: Vec<Vec<f64>>,
    mc_has_data: bool,
    cs_has_data: bool,
    #[cfg(feature = "neural")] nn_has_data: bool,
    /// Path to ONNX model file; loaded once on first use.
    #[cfg(feature = "neural")] nn_model_path: String,
    prob_rx: Option<Receiver<SimUpdate>>,
    /// Strategies that have sent their Done message.
    done_strategies: HashSet<Strategy>,
    /// Priority level of the strategy whose data is currently in `probs`.
    probs_priority: u8,
}

impl MinesweeperGui {
    fn init(&mut self) {
        let game = Minesweeper::new(10, 10, 10);
        self.board_width = 10;
        self.board_height = 10;
        let up = uniform_probs(&game);
        self.probs = up.clone();
        self.mc_probs = up.clone();
        self.cs_probs = up.clone();
        #[cfg(feature = "neural")] { self.nn_probs = up; }
        #[cfg(feature = "neural")] { self.nn_has_data = false; }
        #[cfg(not(feature = "neural"))] { let _ = up; }
        self.mc_has_data = false;
        self.cs_has_data = false;
        #[cfg(feature = "neural")] {
            self.nn_model_path = std::env::var("NN_MODEL_PATH")
                .unwrap_or_else(|_| "neural/onnx/model.onnx".to_string());
        }
        self.game = Some(game);
        self.render_cells();
        self.boardChanged();
    }

    fn reveal(&mut self, index: i32) {
        if let Some(game) = &mut self.game {
            let x = (index % self.board_width) as usize;
            let y = (index / self.board_width) as usize;
            game.reveal(x, y);
            self.update_view();
        }
    }

    fn flag(&mut self, index: i32) {
        if let Some(game) = &mut self.game {
            let x = (index % self.board_width) as usize;
            let y = (index / self.board_width) as usize;
            game.toggle_flag(x, y);
            self.update_view();
        }
    }

    fn reset(&mut self, w: i32, h: i32, m: i32) {
        let w = (w as usize).clamp(3, 50);
        let h = (h as usize).clamp(3, 50);
        let m = (m as usize).clamp(1, w * h - 1);
        self.prob_rx = None;
        self.done_strategies.clear();
        self.probs_priority = 0;
        let game = Minesweeper::new(w, h, m);
        self.board_width = w as i32;
        self.board_height = h as i32;
        let up = uniform_probs(&game);
        self.probs = up.clone();
        self.mc_probs = up.clone();
        self.cs_probs = up.clone();
        #[cfg(feature = "neural")] { self.nn_probs = up; }
        #[cfg(not(feature = "neural"))] { let _ = up; }
        self.mc_has_data = false;
        self.cs_has_data = false;
        #[cfg(feature = "neural")] { self.nn_has_data = false; }
        self.game = Some(game);
        self.sim_status = QString::default();
        self.cs_status = QString::default();
        self.nn_status = QString::default();
        self.render_cells();
        self.boardChanged();
    }

    /// Called by the QML Timer every 100 ms. Drains the channel and applies updates.
    /// Strategies with higher priority override lower-priority probs once they
    /// have valid data (exact search beats random sampling).
    fn check_prob_update(&mut self) {
        let updates: Vec<SimUpdate> = if let Some(rx) = &self.prob_rx {
            let mut v = Vec::new();
            while let Ok(u) = rx.try_recv() {
                v.push(u);
            }
            v
        } else {
            return;
        };

        if updates.is_empty() {
            return;
        }

        let mut any_change = false;

        for update in updates {
            match update {
                SimUpdate::Progress {
                    strategy: Strategy::MonteCarlo,
                    valid,
                    attempts,
                    max_attempts,
                    memory_bytes,
                    probs,
                } => {
                    if valid > 0 {
                        self.mc_probs = probs.clone();
                        self.mc_has_data = true;
                        if self.try_update_probs(Strategy::MonteCarlo, probs) {
                            any_change = true;
                        }
                    }
                    self.sim_status = QString::from(format!(
                        "MC: {} valid  /  {} of {} sampled  [{}]",
                        valid, attempts, max_attempts, fmt_memory(memory_bytes)
                    ));
                }
                SimUpdate::Done {
                    strategy: Strategy::MonteCarlo,
                    valid,
                    attempts,
                    memory_bytes,
                    probs,
                } => {
                    if valid > 0 {
                        self.mc_probs = probs.clone();
                        self.mc_has_data = true;
                        if self.try_update_probs(Strategy::MonteCarlo, probs) {
                            any_change = true;
                        }
                    }
                    self.done_strategies.insert(Strategy::MonteCarlo);
                    self.sim_status = QString::from(format!(
                        "✓ MC: {} valid  /  {} sampled  [{}]",
                        valid, attempts, fmt_memory(memory_bytes)
                    ));
                }
                SimUpdate::Progress {
                    strategy: Strategy::ConstraintSearch,
                    valid,
                    attempts,
                    memory_bytes,
                    probs,
                    ..
                } => {
                    if valid > 0 {
                        self.cs_probs = probs.clone();
                        self.cs_has_data = true;
                        if self.try_update_probs(Strategy::ConstraintSearch, probs) {
                            any_change = true;
                        }
                    }
                    self.cs_status = if valid > 0 {
                        QString::from(format!(
                            "CS: {} layouts / {} steps  [{}]",
                            valid, attempts, fmt_memory(memory_bytes)
                        ))
                    } else {
                        QString::from(format!("CS: searching…  [{}]", fmt_memory(memory_bytes)))
                    };
                }
                SimUpdate::Done {
                    strategy: Strategy::ConstraintSearch,
                    valid,
                    attempts,
                    memory_bytes,
                    probs,
                } => {
                    if valid > 0 {
                        self.cs_probs = probs.clone();
                        self.cs_has_data = true;
                        if self.try_update_probs(Strategy::ConstraintSearch, probs) {
                            any_change = true;
                        }
                    }
                    self.done_strategies.insert(Strategy::ConstraintSearch);
                    self.cs_status = QString::from(format!(
                        "✓ CS: {} layouts / {} steps  [{}]",
                        valid, attempts, fmt_memory(memory_bytes)
                    ));
                }
                #[cfg(feature = "neural")]
                SimUpdate::Done {
                    strategy: Strategy::NeuralNetwork,
                    valid,
                    memory_bytes,
                    probs,
                    ..
                } => {
                    if valid > 0 {
                        self.nn_probs = probs.clone();
                        self.nn_has_data = true;
                        if self.try_update_probs(Strategy::NeuralNetwork, probs) {
                            any_change = true;
                        }
                    }
                    self.done_strategies.insert(Strategy::NeuralNetwork);
                    self.nn_status = QString::from(format!(
                        "✓ NN: {} cells  [{}]",
                        valid, fmt_memory(memory_bytes)
                    ));
                }
                // NeuralNetwork does not send Progress updates (single-shot).
                #[cfg(feature = "neural")]
                SimUpdate::Progress {
                    strategy: Strategy::NeuralNetwork,
                    ..
                } => {}
            }
        }

        // Close channel once all active strategies have finished.
        #[cfg(not(feature = "neural"))]
        let all_active = [Strategy::MonteCarlo, Strategy::ConstraintSearch];
        #[cfg(feature = "neural")]
        let all_active = [Strategy::MonteCarlo, Strategy::ConstraintSearch, Strategy::NeuralNetwork];
        if all_active.iter().all(|s| self.done_strategies.contains(s)) {
            self.prob_rx = None;
        }

        if any_change {
            self.maybe_auto_reveal();
            self.render_cells();
            self.boardChanged();
        }
    }

    /// If auto-reveal is on, reveal every hidden cell whose best-estimate mine
    /// probability is exactly 0. Uses the highest-priority strategy's probs
    /// (`self.probs`), so CS results (exact) take precedence over MC ones.
    /// Reveals are done in one pass; the resulting `update_view` call re-launches
    /// the simulation on the new board state, which may expose further safe cells
    /// on the next timer tick.
    fn maybe_auto_reveal(&mut self) {
        if !self.auto_reveal {
            return;
        }
        // Only act when the game is running and mines are already placed.
        let should_run = self.game.as_ref()
            .map(|g| g.state == GameState::Playing && g.mines_generated)
            .unwrap_or(false);
        if !should_run {
            return;
        }

        let game = self.game.as_ref().unwrap();
        let to_reveal: Vec<(usize, usize)> = (0..game.height)
            .flat_map(|y| (0..game.width).map(move |x| (x, y)))
            .filter(|&(x, y)| {
                game.grid[y][x].state == CellState::Hidden
                    && self.probs.get(y).and_then(|r| r.get(x)).copied().unwrap_or(1.0) < 1e-9
            })
            .collect();

        if to_reveal.is_empty() {
            return;
        }

        let game = self.game.as_mut().unwrap();
        for (x, y) in to_reveal {
            game.reveal(x, y);
        }
        // Re-run the simulation on the updated board; the timer will call
        // check_prob_update → maybe_auto_reveal again if new safe cells appear.
        self.update_view();
    }

    /// Update `self.probs` with `new_probs` if `strategy` has higher or equal
    /// priority than whoever last wrote `self.probs`. Returns true when updated.
    fn try_update_probs(&mut self, strategy: Strategy, new_probs: Vec<Vec<f64>>) -> bool {
        if strategy.priority() >= self.probs_priority {
            self.probs = new_probs;
            self.probs_priority = strategy.priority();
            true
        } else {
            false
        }
    }

    /// Render immediately with cached probs, then start all background strategies.
    fn update_view(&mut self) {
        self.render_cells();
        self.boardChanged();

        if let Some(game) = &self.game {
            if game.state == GameState::Playing && game.mines_generated {
                let game_clone = game.clone();
                let game_clone2 = game_clone.clone();
                #[cfg(feature = "neural")]
                let game_clone3 = game_clone.clone();

                let (tx, rx) = std::sync::mpsc::channel();
                self.prob_rx = Some(rx);
                self.done_strategies.clear();
                self.probs_priority = 0;
                self.mc_has_data = false;
                self.cs_has_data = false;
                #[cfg(feature = "neural")] { self.nn_has_data = false; }

                let mc_tx = tx.clone();
                let cs_tx = tx.clone();
                #[cfg(feature = "neural")]
                let nn_tx = tx;
                #[cfg(not(feature = "neural"))]
                drop(tx);

                std::thread::spawn(move || {
                    MonteCarlo::new().calculate_with_progress(&game_clone, mc_tx);
                });
                std::thread::spawn(move || {
                    ConstraintSearch::new().calculate_with_progress(&game_clone2, cs_tx);
                });

                #[cfg(feature = "neural")]
                {
                    let model_path = self.nn_model_path.clone();
                    std::thread::spawn(move || {
                        match NeuralNetwork::new(&model_path) {
                            Ok(nn) => nn.calculate_with_progress(&game_clone3, nn_tx),
                            Err(e) => {
                                eprintln!("NeuralNetwork load error: {e}");
                            }
                        }
                    });
                }
            }
        }
    }

    fn render_cells(&mut self) {
        if let Some(game) = &self.game {
            let mut new_cells = QVariantList::default();

            // Count hidden + flagged cells and remaining mines for layout count.
            let n: usize = (0..game.height)
                .flat_map(|y| (0..game.width).map(move |x| (x, y)))
                .filter(|&(x, y)| matches!(game.grid[y][x].state, CellState::Hidden | CellState::Flagged))
                .count();
            let k = game.mines_count;
            let cs_done = self.done_strategies.contains(&Strategy::ConstraintSearch);
            self.layout_count = if game.state == GameState::Playing {
                let tick = if cs_done { "✓ " } else { "" };
                QString::from(format!("{}{} possible layouts", tick, fmt_count(combinations(n, k))))
            } else {
                QString::default()
            };

            for y in 0..game.height {
                for x in 0..game.width {
                    let cell = &game.grid[y][x];
                    let mut map = QVariantMap::default();
                    let p = self.probs.get(y).and_then(|row| row.get(x)).copied().unwrap_or(0.0);

                    let prob_bg = |p: f64| -> String {
                        let r = (204.0 + 51.0 * p).round() as u8;
                        let g = (204.0 * (1.0 - p)).round() as u8;
                        format!("#{:02x}{:02x}{:02x}", r, g, g)
                    };

                    let (color, bg_color): (&str, String) = match cell.state {
                        CellState::Hidden => ("black", prob_bg(p)),
                        CellState::Flagged => ("red", prob_bg(p)),
                        CellState::Visible => match cell.content {
                            CellContent::Mine => ("white", "red".to_string()),
                            CellContent::Empty(0) => ("black", "#eee".to_string()),
                            CellContent::Empty(n) => {
                                let c = match n {
                                    1 => "blue",
                                    2 => "green",
                                    3 => "red",
                                    4 => "darkblue",
                                    _ => "black"
                                };
                                (c, "#eee".to_string())
                            }
                        },
                    };

                    let final_text = if cell.state == CellState::Visible {
                        if let CellContent::Empty(n) = cell.content {
                            if n > 0 { n.to_string() } else { "".to_string() }
                        } else if let CellContent::Mine = cell.content {
                            "*".to_string()
                        } else { "".to_string() }
                    } else if cell.state == CellState::Flagged {
                        "F".to_string()
                    } else {
                        "".to_string()
                    };

                    let is_hidden = matches!(cell.state, CellState::Hidden | CellState::Flagged);
                    let mc_p = self.mc_probs.get(y).and_then(|r| r.get(x)).copied().unwrap_or(0.0);
                    let cs_p = self.cs_probs.get(y).and_then(|r| r.get(x)).copied().unwrap_or(0.0);
                    let mc_prob_text = if is_hidden && self.mc_has_data {
                        format!("{:.0}%", mc_p * 100.0)
                    } else { String::new() };
                    let cs_prob_text = if is_hidden && self.cs_has_data {
                        format!("{:.0}%", cs_p * 100.0)
                    } else { String::new() };
                    #[cfg(feature = "neural")]
                    let nn_prob_text = {
                        let nn_p = self.nn_probs.get(y).and_then(|r| r.get(x)).copied().unwrap_or(0.0);
                        if is_hidden && self.nn_has_data {
                            format!("{:.0}%", nn_p * 100.0)
                        } else { String::new() }
                    };
                    #[cfg(not(feature = "neural"))]
                    let nn_prob_text = String::new();

                    // A hidden cell is a "border" cell if it is adjacent to at least one
                    // visible numbered cell — i.e. it is directly constrained.
                    let is_border = matches!(cell.state, CellState::Hidden | CellState::Flagged)
                        && (-1isize..=1)
                            .flat_map(|dy| (-1isize..=1).map(move |dx| (dx, dy)))
                            .filter(|&(dx, dy)| dx != 0 || dy != 0)
                            .any(|(dx, dy)| {
                                let nx = x as isize + dx;
                                let ny = y as isize + dy;
                                nx >= 0
                                    && nx < game.width as isize
                                    && ny >= 0
                                    && ny < game.height as isize
                                    && game.grid[ny as usize][nx as usize].state
                                        == CellState::Visible
                                    && matches!(
                                        game.grid[ny as usize][nx as usize].content,
                                        CellContent::Empty(n) if n > 0
                                    )
                            });

                    map.insert(QString::from("text"), QString::from(final_text).into());
                    map.insert(QString::from("color"), QString::from(color).into());
                    map.insert(QString::from("bgColor"), QString::from(bg_color).into());
                    map.insert(QString::from("mcProbText"), QString::from(mc_prob_text).into());
                    map.insert(QString::from("csProbText"), QString::from(cs_prob_text).into());
                    map.insert(QString::from("nnProbText"), QString::from(nn_prob_text).into());
                    map.insert(QString::from("isBorder"), is_border.into());
                    new_cells.push(map.into());
                }
            }
            self.cells = new_cells;

            self.status_text = match game.state {
                GameState::Playing => QString::from(format!("Mines: {}", game.mines_count)),
                GameState::Won => QString::from("YOU WON!"),
                GameState::Lost => QString::from("GAME OVER"),
            };
        }
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn uniform_probs(game: &Minesweeper) -> Vec<Vec<f64>> {
    let p = game.mines_count as f64 / (game.width * game.height) as f64;
    vec![vec![p; game.width]; game.height]
}

/// Human-readable memory size (B / KB / MB).
fn fmt_memory(bytes: usize) -> String {
    match bytes {
        b if b < 1_024             => format!("{} B", b),
        b if b < 1_024 * 1_024    => format!("{:.1} KB", b as f64 / 1_024.0),
        b                          => format!("{:.1} MB", b as f64 / 1_048_576.0),
    }
}

/// Human-readable scale suffix (K / M / B / T / scientific).
fn fmt_count(v: f64) -> String {
    match v {
        v if v < 1_000.0          => format!("{:.0}", v),
        v if v < 1_000_000.0      => format!("{:.1}K", v / 1e3),
        v if v < 1_000_000_000.0  => format!("{:.1}M", v / 1e6),
        v if v < 1e12             => format!("{:.1}B", v / 1e9),
        v if v < 1e15             => format!("{:.1}T", v / 1e12),
        v                         => format!("{:.2e}", v),
    }
}

const QML: &str = r##"
import QtQuick 2.0
import QtQuick.Controls 2.0
import QtQuick.Layouts 1.0
import Minesweeper 1.0

ApplicationWindow {
    id: root
    visible: true

    // Approximate height / width consumed by non-grid UI elements (margins, labels, controls).
    property int uiPadH: 210
    property int uiPadW: 20

    // Cell size fills available window space, clamped to a sensible minimum.
    property int cellSize: Math.max(10, Math.floor(
        Math.min(
            (root.width  - uiPadW) / minesweeper.board_width,
            (root.height - uiPadH) / minesweeper.board_height
        )
    ))

    width:  320
    height: 520
    minimumWidth:  minesweeper.board_width  * 12 + uiPadW
    minimumHeight: minesweeper.board_height * 12 + uiPadH
    title: "Rust Minesweeper"

    property string hoveredProb: ""
    property int prevBoardW: -1
    property int prevBoardH: -1

    // Resize the window to a sensible default whenever the board dimensions change
    // (i.e. a new game with different W/H). Moves don't change dimensions so the
    // window stays at whatever size the user last dragged it to.
    Connections {
        target: minesweeper
        function onBoardChanged() {
            if (minesweeper.board_width !== prevBoardW || minesweeper.board_height !== prevBoardH) {
                root.width  = minesweeper.board_width  * 32 + root.uiPadW
                root.height = minesweeper.board_height * 32 + root.uiPadH
                prevBoardW = minesweeper.board_width
                prevBoardH = minesweeper.board_height
            }
        }
    }

    MinesweeperGame {
        id: minesweeper
        Component.onCompleted: init()
    }

    Timer {
        interval: 100
        running: true
        repeat: true
        onTriggered: minesweeper.check_prob_update()
    }

    ColumnLayout {
        anchors.fill: parent
        anchors.margins: 10
        spacing: 4

        Text {
            text: root.hoveredProb !== "" ? root.hoveredProb : minesweeper.status_text
            font.pixelSize: 20
            Layout.alignment: Qt.AlignHCenter
        }

        Text {
            visible: minesweeper.layout_count !== ""
            text: minesweeper.layout_count
            font.pixelSize: 11
            color: "#555"
            Layout.alignment: Qt.AlignHCenter
        }

        Text {
            visible: minesweeper.sim_status !== ""
            text: minesweeper.sim_status
            font.pixelSize: 11
            color: "#888"
            Layout.alignment: Qt.AlignHCenter
        }

        Text {
            visible: minesweeper.cs_status !== ""
            text: minesweeper.cs_status
            font.pixelSize: 11
            color: "#668"
            Layout.alignment: Qt.AlignHCenter
        }

        Text {
            visible: minesweeper.nn_status !== ""
            text: minesweeper.nn_status
            font.pixelSize: 11
            color: "#468"
            Layout.alignment: Qt.AlignHCenter
        }

        GridLayout {
            columns: minesweeper.board_width
            columnSpacing: 2
            rowSpacing: 2
            Layout.alignment: Qt.AlignHCenter

            Repeater {
                model: minesweeper.cells
                delegate: Rectangle {
                    width: root.cellSize
                    height: root.cellSize
                    color: modelData.bgColor
                    border.color: modelData.isBorder ? "#5599ff" : "#999"
                    border.width: modelData.isBorder ? 2 : 1

                    Text {
                        anchors.centerIn: parent
                        text: modelData.text
                        color: modelData.color
                        font.bold: true
                        font.pixelSize: Math.max(8, root.cellSize - 10)
                    }

                    Text {
                        visible: modelData.mcProbText !== "" && root.cellSize >= 14
                        text: modelData.mcProbText
                        font.pixelSize: 7
                        color: "#888"
                        anchors.top: parent.top
                        anchors.right: parent.right
                        anchors.margins: 1
                    }

                    Text {
                        visible: modelData.csProbText !== "" && root.cellSize >= 14
                        text: modelData.csProbText
                        font.pixelSize: 7
                        color: "#558"
                        anchors.bottom: parent.bottom
                        anchors.right: parent.right
                        anchors.margins: 1
                    }

                    Text {
                        visible: modelData.nnProbText !== "" && root.cellSize >= 14
                        text: modelData.nnProbText
                        font.pixelSize: 7
                        color: "#468"
                        anchors.bottom: parent.bottom
                        anchors.left: parent.left
                        anchors.margins: 1
                    }

                    MouseArea {
                        anchors.fill: parent
                        hoverEnabled: true
                        acceptedButtons: Qt.LeftButton | Qt.RightButton
                        onEntered: {
                            var mc = modelData.mcProbText
                            var cs = modelData.csProbText
                            var nn = modelData.nnProbText
                            if (mc !== "" || cs !== "" || nn !== "") {
                                root.hoveredProb = "MC: " + (mc !== "" ? mc : "?") + "  |  CS: " + (cs !== "" ? cs : "?") + "  |  NN: " + (nn !== "" ? nn : "?")
                            } else {
                                root.hoveredProb = ""
                            }
                        }
                        onExited: root.hoveredProb = ""
                        onClicked: {
                            if (mouse.button === Qt.RightButton) {
                                minesweeper.flag(index)
                            } else {
                                minesweeper.reveal(index)
                            }
                        }
                    }
                }
            }
        }

        // New-game settings row
        RowLayout {
            Layout.alignment: Qt.AlignHCenter
            spacing: 6

            Text { text: "W:"; font.pixelSize: 12 }
            SpinBox { id: wSpin; from: 3; to: 50; value: 10; implicitWidth: 75 }

            Text { text: "H:"; font.pixelSize: 12 }
            SpinBox { id: hSpin; from: 3; to: 50; value: 10; implicitWidth: 75 }

            Text { text: "M:"; font.pixelSize: 12 }
            SpinBox { id: mSpin; from: 1; to: 999; value: 10; implicitWidth: 80 }
        }

        RowLayout {
            Layout.alignment: Qt.AlignHCenter
            spacing: 10

            CheckBox {
                text: "Auto-reveal safe cells"
                checked: minesweeper.auto_reveal
                onCheckedChanged: minesweeper.auto_reveal = checked
                font.pixelSize: 12
            }

            Button {
                text: "New Game"
                onClicked: minesweeper.reset(wSpin.value, hSpin.value, mSpin.value)
            }
        }

        Item { Layout.fillHeight: true }
    }
}
"##;

fn main() {
    qml_register_type::<MinesweeperGui>(cstr!("Minesweeper"), 1, 0, cstr!("MinesweeperGame"));
    let mut engine = QmlEngine::new();
    engine.load_data(QML.into());
    engine.exec();
}
