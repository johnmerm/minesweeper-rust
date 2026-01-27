use qmetaobject::prelude::*;
use qmetaobject::{QVariantList, QVariantMap};
use minesweeper_core::{Minesweeper, CellState, CellContent, GameState};
use cstr::cstr;

#[derive(QObject, Default)]
struct MinesweeperGui {
    base: qt_base_class!(trait QObject),
    
    board_width: qt_property!(i32; NOTIFY boardChanged),
    board_height: qt_property!(i32; NOTIFY boardChanged),
    
    cells: qt_property!(QVariantList; NOTIFY boardChanged),
    status_text: qt_property!(QString; NOTIFY boardChanged),

    boardChanged: qt_signal!(),
    
    init: qt_method!(fn(&mut self)),
    reveal: qt_method!(fn(&mut self, index: i32)),
    flag: qt_method!(fn(&mut self, index: i32)),
    reset: qt_method!(fn(&mut self)),

    game: Option<Minesweeper>, 
}

impl MinesweeperGui {
    fn init(&mut self) {
        self.game = Some(Minesweeper::new(10, 10, 10));
        self.board_width = 10;
        self.board_height = 10;
        self.update_view();
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
        self.game = Some(Minesweeper::new(10, 10, 10));
        self.update_view();
    }

    fn update_view(&mut self) {
        if let Some(game) = &self.game {
            let mut new_cells = QVariantList::default();
            
            for y in 0..game.height {
                for x in 0..game.width {
                    let cell = &game.grid[y][x];
                    let mut map = QVariantMap::default();
                    
                    let (color, bg_color) = match cell.state {
                        CellState::Hidden => ("black", "#ccc"),
                        CellState::Flagged => ("red", "#ccc"),
                        CellState::Visible => match cell.content {
                            CellContent::Mine => ("white", "red"),
                            CellContent::Empty(0) => ("black", "#eee"),
                            CellContent::Empty(n) => {
                                let c = match n {
                                    1 => "blue",
                                    2 => "green",
                                    3 => "red",
                                    4 => "darkblue",
                                    _ => "black"
                                };
                                (c, "#eee")
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

                    map.insert(QString::from("text"), QString::from(final_text).into());
                    map.insert(QString::from("color"), QString::from(color).into());
                    map.insert(QString::from("bgColor"), QString::from(bg_color).into());
                    new_cells.push(map.into());
                }
            }
            self.cells = new_cells;

            self.status_text = match game.state {
                GameState::Playing => QString::from(format!("Mines: {}", game.mines_count)),
                GameState::Won => QString::from("YOU WON!"),
                GameState::Lost => QString::from("GAME OVER"),
            };
            
            self.boardChanged();
        }
    }
}

const QML: &str = r##"
import QtQuick 2.0
import QtQuick.Controls 2.0
import QtQuick.Layouts 1.0
import Minesweeper 1.0

ApplicationWindow {
    visible: true
    width: 340
    height: 400
    title: "Rust Minesweeper"

    MinesweeperGame {
        id: minesweeper
        Component.onCompleted: init()
    }

    ColumnLayout {
        anchors.fill: parent
        anchors.margins: 10
        spacing: 10

        Text {
            text: minesweeper.status_text
            font.pixelSize: 20
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
                    border.color: "#999"
                    border.width: 1

                    Text {
                        anchors.centerIn: parent
                        text: modelData.text
                        color: modelData.color
                        font.bold: true
                    }

                    MouseArea {
                        anchors.fill: parent
                        acceptedButtons: Qt.LeftButton | Qt.RightButton
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
        
        Item { Layout.fillHeight: true } // Spacer
    }
}
"##;

fn main() {
    qml_register_type::<MinesweeperGui>(cstr!("Minesweeper"), 1, 0, cstr!("MinesweeperGame"));
    let mut engine = QmlEngine::new();
    engine.load_data(QML.into());
    engine.exec();
}