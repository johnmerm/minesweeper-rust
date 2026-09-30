import QtQuick 2.15
import QtQuick.Controls 2.15
import QtQuick.Layouts 1.15
import QtQuick.Window 2.15
import Minesweeper 1.0

// Laid out after docs/index.html, so the two front-ends read the same way.
ApplicationWindow {
    id: root
    visible: true
    title: "Minesweeper"
    color: "#f0f0f0"
    // The page's cell size, shrinking with a narrow window down to a floor.
    // Past that the board scrolls — at 200 a side it cannot all fit, and
    // unreadable cells are worse than a scroll bar.
    readonly property int gap: 2
    readonly property int cellSize: Math.max(22, Math.min(30, Math.floor(
        (root.width - 44) / Math.max(1, minesweeper.board_width)) - gap))
    readonly property int pitch: cellSize + gap
    // The board's full size, frame included.
    readonly property int boardWidth: minesweeper.board_width * pitch + 20
    readonly property int boardHeight: minesweeper.board_height * pitch + 20

    // Open at a size that shows the whole page, and again whenever a new game
    // changes the board's shape; otherwise leave the window where the player put it.
    property int shapeW: -1
    property int shapeH: -1
    function fitWindow() {
        if (minesweeper.board_width === shapeW && minesweeper.board_height === shapeH) return
        shapeW = minesweeper.board_width
        shapeH = minesweeper.board_height
        var natural = minesweeper.board_width * 32 + 20
        width = Math.min(Screen.desktopAvailableWidth - 40, Math.max(720, natural + 24))
        var chrome = page.implicitHeight - frame.height
        height = Math.min(Screen.desktopAvailableHeight - 60,
                          chrome + minesweeper.board_height * 32 + 20 + 24)
    }
    Connections {
        target: minesweeper
        function onBoard_changed() { Qt.callLater(root.fitWindow) }
    }

    property int hoverIndex: -1

    MinesweeperGame {
        id: minesweeper
        Component.onCompleted: init()
    }

    Timer {
        interval: 500
        repeat: true
        running: minesweeper.timer_running
        onTriggered: minesweeper.tick()
    }

    // The page scrolls when the window is shorter than it, as the HTML page
    // does, rather than squeezing the board into whatever height is left over.
    Flickable {
        id: pageScroll
        anchors.fill: parent
        contentWidth: width
        contentHeight: page.implicitHeight + 24
        boundsBehavior: Flickable.StopAtBounds
        interactive: contentHeight > height
        ScrollBar.vertical: ScrollBar { policy: pageScroll.interactive ? ScrollBar.AlwaysOn : ScrollBar.AlwaysOff }

    ColumnLayout {
        id: page
        x: 12
        y: 12
        width: pageScroll.width - 24
        spacing: 6

        Text {
            text: "Minesweeper"
            font.pixelSize: 22
            font.weight: Font.DemiBold
            color: "#222"
            Layout.alignment: Qt.AlignHCenter
        }
        Text {
            text: "Rust engine · exact solver and neural network on a background thread"
            font.pixelSize: 12
            color: "#777"
            Layout.alignment: Qt.AlignHCenter
        }

        Text {
            text: minesweeper.status_text
            font.pixelSize: 18
            font.weight: Font.DemiBold
            color: minesweeper.status_kind === 1 ? "#1a7f37"
                 : minesweeper.status_kind === 2 ? "#c1121f" : "#222"
            Layout.alignment: Qt.AlignHCenter
        }

        RowLayout {
            Layout.alignment: Qt.AlignHCenter
            spacing: 18
            Text {
                textFormat: Text.RichText
                font.pixelSize: 15
                text: "<span style='color:#777'>Mines</span> <b>" + minesweeper.mines_left + "</b>"
            }
            Text {
                textFormat: Text.RichText
                font.pixelSize: 15
                text: "<span style='color:#777'>Time</span> <b>" + minesweeper.timer_text + "</b>"
            }
        }

        // The board, framed as on the page, scrolling when it outgrows the window.
        Item {
            id: boardArea
            Layout.fillWidth: true
            Layout.preferredHeight: frame.height

            Rectangle {
                id: frame
                anchors.horizontalCenter: parent.horizontalCenter
                // The whole board when it fits; otherwise as much as the window
                // shows, scrolling inside.
                width: Math.min(parent.width, root.boardWidth)
                height: Math.min(root.height - 40, root.boardHeight)
                color: "#999"
                border.color: "#666"
                border.width: 5

                // Horizontal scrolling outside, vertical inside: GridView only
                // creates the rows that are on screen, which is what keeps a
                // 200x200 board from building 40 000 delegates at once.
                Flickable {
                    id: hscroll
                    anchors.fill: parent
                    anchors.margins: 7
                    clip: true
                    contentWidth: grid.width
                    flickableDirection: Flickable.HorizontalFlick
                    boundsBehavior: Flickable.StopAtBounds
                    interactive: contentWidth > width
                    ScrollBar.horizontal: ScrollBar { policy: hscroll.interactive ? ScrollBar.AlwaysOn : ScrollBar.AlwaysOff }

                    GridView {
                        id: grid
                        width: minesweeper.board_width * root.pitch
                        height: hscroll.height
                        cellWidth: root.pitch
                        cellHeight: root.pitch
                        boundsBehavior: Flickable.StopAtBounds
                        interactive: contentHeight > height
                        clip: true
                        model: minesweeper.cells
                        ScrollBar.vertical: ScrollBar { policy: grid.interactive ? ScrollBar.AlwaysOn : ScrollBar.AlwaysOff }

                        delegate: Rectangle {
                            width: root.cellSize
                            height: root.cellSize
                            color: model.bg
                            // Raised for unopened cells, flat for opened ones; the
                            // blue edge marks the cells a number speaks about.
                            border.width: model.border ? 2 : 1
                            border.color: model.border ? "#5599ff" : (model.raised ? "#eeeeee" : "#999999")
                            opacity: mouse.containsMouse && model.raised ? 0.85 : 1.0

                            Text {
                                anchors.centerIn: parent
                                text: model.text
                                color: model.fg
                                font.bold: true
                                font.pixelSize: Math.round(root.cellSize * 0.56)
                            }
                            // The exact value, bottom-right.
                            Text {
                                visible: minesweeper.show_exact && model.prob !== ""
                                text: model.prob
                                font.pixelSize: Math.max(8, Math.round(root.cellSize * 0.27))
                                font.weight: Font.DemiBold
                                color: "#444"
                                anchors.right: parent.right
                                anchors.bottom: parent.bottom
                                anchors.rightMargin: 2
                            }
                            // The network's guess: top-left and purple beside the
                            // proof, taking its corner when it is shown alone. It is
                            // an approximation of the other number, not a peer.
                            Text {
                                id: guessLabel
                                readonly property bool alone: !minesweeper.show_exact
                                visible: minesweeper.show_neural && model.guess !== ""
                                text: model.guess
                                font.pixelSize: Math.max(8, Math.round(root.cellSize * 0.27))
                                font.weight: Font.DemiBold
                                color: "#6a4fb6"
                                x: alone ? parent.width - width - 2 : 2
                                y: alone ? parent.height - height : 0
                            }

                            MouseArea {
                                id: mouse
                                anchors.fill: parent
                                hoverEnabled: true
                                acceptedButtons: Qt.LeftButton | Qt.RightButton
                                cursorShape: model.raised ? Qt.PointingHandCursor : Qt.ArrowCursor
                                onEntered: root.hoverIndex = index
                                onExited: if (root.hoverIndex === index) root.hoverIndex = -1
                                onClicked: mouse.button === Qt.RightButton
                                    ? minesweeper.flag(index)
                                    : minesweeper.reveal(index)
                            }
                        }
                    }
                }
            }
        }

        Text {
            id: hoverLine
            // Re-read whenever the numbers behind the cell change, not only on entry.
            text: {
                minesweeper.sim_text; minesweeper.neural_note; minesweeper.show_mode
                return root.hoverIndex >= 0 ? minesweeper.hover_text(root.hoverIndex) : ""
            }
            font.pixelSize: 14
            color: "#555"
            Layout.alignment: Qt.AlignHCenter
            Layout.preferredHeight: 18
        }
        Text {
            text: minesweeper.sim_text
            font.pixelSize: 11
            font.weight: minesweeper.sim_kind === 2 ? Font.Normal : Font.DemiBold
            color: minesweeper.sim_kind === 0 ? "#335" : minesweeper.sim_kind === 1 ? "#a3521b" : "#888"
            Layout.alignment: Qt.AlignHCenter
        }
        Text {
            text: minesweeper.neural_note
            font.pixelSize: 11
            color: "#6a4fb6"
            Layout.alignment: Qt.AlignHCenter
            Layout.preferredHeight: 14
        }

        Pane {
            Layout.alignment: Qt.AlignHCenter
            background: Rectangle { color: "white"; border.color: "#ddd"; radius: 8 }
            padding: 8
            Flow {
                width: Math.min(page.width - 16, 780)
                spacing: 8
                Label { text: "W"; height: wSpin.height; verticalAlignment: Text.AlignVCenter }
                SpinBox { id: wSpin; from: 3; to: 200; value: 10; editable: true }
                Label { text: "H"; height: wSpin.height; verticalAlignment: Text.AlignVCenter }
                SpinBox { id: hSpin; from: 3; to: 200; value: 10; editable: true }
                Label { text: "Mines"; height: wSpin.height; verticalAlignment: Text.AlignVCenter }
                SpinBox { id: mSpin; from: 1; to: 39999; value: 10; editable: true }
                Button { text: "New game"; onClicked: newGame(wSpin.value, hSpin.value, mSpin.value) }
            }
        }
        RowLayout {
            Layout.alignment: Qt.AlignHCenter
            spacing: 6
            Button { text: "Beginner"; flat: true; onClicked: newGame(9, 9, 10) }
            Button { text: "Intermediate"; flat: true; onClicked: newGame(16, 16, 40) }
            Button { text: "Expert"; flat: true; onClicked: newGame(30, 16, 99) }
        }

        Pane {
            Layout.alignment: Qt.AlignHCenter
            background: Rectangle { color: "white"; border.color: "#ddd"; radius: 8 }
            padding: 8
            Flow {
                width: Math.min(page.width - 16, 780)
                spacing: 8
                Button {
                    text: "Show probabilities"
                    checkable: true
                    checked: minesweeper.show_probs
                    onToggled: minesweeper.show_probs = checked
                }
                Button {
                    text: "Auto-play deductions"
                    checkable: true
                    checked: minesweeper.auto_play
                    onToggled: minesweeper.auto_play = checked
                }
                Button {
                    text: "Flag mode"
                    checkable: true
                    checked: minesweeper.flag_mode
                    onToggled: minesweeper.flag_mode = checked
                }
                Label { text: "Show"; height: 40; verticalAlignment: Text.AlignVCenter }
                ComboBox {
                    model: ["Both estimates", "Constraint search only", "Neural network only"]
                    currentIndex: minesweeper.show_mode
                    implicitWidth: 240
                    onActivated: minesweeper.show_mode = index
                }
            }
        }

        Text {
            // A fixed width, not fillWidth: a wrapped text's height depends on
            // its width, and letting the layout choose both is a binding loop.
            Layout.preferredWidth: Math.min(pageScroll.width - 24, 640)
            Layout.alignment: Qt.AlignHCenter
            horizontalAlignment: Text.AlignHCenter
            wrapMode: Text.WordWrap
            textFormat: Text.StyledText
            font.pixelSize: 12
            color: "#777"
            text: "Left click reveals · right click (or <i>flag mode</i>) flags. "
                + "Cells are tinted grey to red by the chance they hide a mine.<br>"
                + "Every percentage is exact — the fraction of consistent layouts with a mine there. "
                + "If the search cannot finish, cells read <i>?</i> rather than a guess.<br>"
                + "<i>Show</i> picks the solver's proof, the network's guess (purple, top-left), or both.<br>"
                + "<i>Auto-play deductions</i> opens every cell proven safe and flags every proven mine. "
                + "On <i>neural network only</i> it follows the network instead — under 5% opens, "
                + "over 95% flags — and will eventually open a mine."
        }
    }
    }

    function newGame(w, h, m) {
        minesweeper.reset(w, h, m)
        // The engine clamps; show what it actually made.
        wSpin.value = minesweeper.board_width
        hSpin.value = minesweeper.board_height
        mSpin.value = minesweeper.mines_left
        root.hoverIndex = -1
    }
}
