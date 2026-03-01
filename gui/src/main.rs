use qmetaobject::prelude::*;
use qmetaobject::{QVariantList, QVariantMap};
use minesweeper_core::{Minesweeper, CellState, CellContent, GameState};
use minesweeper_core::probability::{MonteCarlo, ConstraintSearch, SimUpdate, Strategy};
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
    layout_count: qt_property!(QString; NOTIFY boardChanged),

    boardChanged: qt_signal!(),

    init: qt_method!(fn(&mut self)),
    reveal: qt_method!(fn(&mut self, index: i32)),
    flag: qt_method!(fn(&mut self, index: i32)),
    reset: qt_method!(fn(&mut self)),
    check_prob_update: qt_method!(fn(&mut self)),

    game: Option<Minesweeper>,
    /// Currently displayed mine probabilities (best available from any strategy).
    probs: Vec<Vec<f64>>,
    prob_rx: Option<Receiver<SimUpdate>>,
    /// Strategies that have sent their Done message.
    done_strategies: HashSet<Strategy>,
    /// Priority level of the strategy whose data is currently in `probs`.
    /// Higher priority strategies (exact > sampling) override lower ones.
    probs_priority: u8,
}

impl MinesweeperGui {
    fn init(&mut self) {
        let game = Minesweeper::new(10, 10, 10);
        self.board_width = 10;
        self.board_height = 10;
        self.probs = uniform_probs(&game);
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

    fn reset(&mut self) {
        self.prob_rx = None;
        self.done_strategies.clear();
        self.probs_priority = 0;
        let game = Minesweeper::new(10, 10, 10);
        self.probs = uniform_probs(&game);
        self.game = Some(game);
        self.sim_status = QString::default();
        self.cs_status = QString::default();
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
                    probs,
                } => {
                    if valid > 0 && self.try_update_probs(Strategy::MonteCarlo, probs) {
                        any_change = true;
                    }
                    self.sim_status = QString::from(format!(
                        "{} valid  /  {} of {} sampled",
                        valid, attempts, max_attempts
                    ));
                }
                SimUpdate::Done {
                    strategy: Strategy::MonteCarlo,
                    valid,
                    attempts,
                    probs,
                } => {
                    if valid > 0 && self.try_update_probs(Strategy::MonteCarlo, probs) {
                        any_change = true;
                    }
                    self.done_strategies.insert(Strategy::MonteCarlo);
                    self.sim_status = QString::from(format!(
                        "✓ {} valid  /  {} sampled",
                        valid, attempts
                    ));
                }
                SimUpdate::Progress {
                    strategy: Strategy::ConstraintSearch,
                    valid,
                    attempts,
                    probs,
                    ..
                } => {
                    if valid > 0 && self.try_update_probs(Strategy::ConstraintSearch, probs) {
                        any_change = true;
                    }
                    self.cs_status = if valid > 0 {
                        QString::from(format!("CS: {} layouts / {} steps", valid, attempts))
                    } else {
                        QString::from("CS: searching…")
                    };
                }
                SimUpdate::Done {
                    strategy: Strategy::ConstraintSearch,
                    valid,
                    attempts,
                    probs,
                } => {
                    if valid > 0 && self.try_update_probs(Strategy::ConstraintSearch, probs) {
                        any_change = true;
                    }
                    self.done_strategies.insert(Strategy::ConstraintSearch);
                    self.cs_status = QString::from(format!(
                        "✓ CS: {} layouts / {} steps",
                        valid, attempts
                    ));
                }
            }
        }

        // Close channel once all active strategies have finished.
        let all_active = [Strategy::MonteCarlo, Strategy::ConstraintSearch];
        if all_active.iter().all(|s| self.done_strategies.contains(s)) {
            self.prob_rx = None;
        }

        if any_change {
            self.render_cells();
            self.boardChanged();
        }
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

                let (tx, rx) = std::sync::mpsc::channel();
                self.prob_rx = Some(rx);
                self.done_strategies.clear();
                self.probs_priority = 0;

                let mc_tx = tx.clone();
                let cs_tx = tx;

                std::thread::spawn(move || {
                    MonteCarlo::new().calculate_with_progress(&game_clone, mc_tx);
                });
                std::thread::spawn(move || {
                    ConstraintSearch::new().calculate_with_progress(&game_clone2, cs_tx);
                });
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

                    let prob_text = if matches!(cell.state, CellState::Hidden | CellState::Flagged) {
                        format!("{:.0}%", p * 100.0)
                    } else {
                        String::new()
                    };

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
                    map.insert(QString::from("probText"), QString::from(prob_text).into());
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
    width: 340
    height: 500
    title: "Rust Minesweeper"

    property string hoveredProb: ""

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

        GridLayout {
            columns: minesweeper.board_width
            columnSpacing: 2
            rowSpacing: 2
            Layout.alignment: Qt.AlignHCenter

            Repeater {
                model: minesweeper.cells
                delegate: Rectangle {
                    width: 30
                    height: 30
                    color: modelData.bgColor
                    border.color: modelData.isBorder ? "#5599ff" : "#999"
                    border.width: modelData.isBorder ? 2 : 1

                    Text {
                        anchors.centerIn: parent
                        text: modelData.text
                        color: modelData.color
                        font.bold: true
                    }

                    Text {
                        visible: modelData.probText !== ""
                        text: modelData.probText
                        font.pixelSize: 8
                        color: "#333"
                        anchors.bottom: parent.bottom
                        anchors.right: parent.right
                        anchors.margins: 1
                    }

                    MouseArea {
                        anchors.fill: parent
                        hoverEnabled: true
                        acceptedButtons: Qt.LeftButton | Qt.RightButton
                        onEntered: root.hoveredProb = modelData.probText !== "" ? "Mine probability: " + modelData.probText : ""
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

        Button {
            text: "New Game"
            Layout.alignment: Qt.AlignHCenter
            onClicked: minesweeper.reset()
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
