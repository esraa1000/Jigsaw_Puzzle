import os
import cv2

from PyQt6.QtWidgets import (
    QWidget, QPushButton, QLabel, QLineEdit, QFrame,
    QVBoxLayout, QHBoxLayout, QMessageBox, QSizePolicy,
    QProgressBar
)
from PyQt6.QtGui import QPixmap, QFont, QImage
from PyQt6.QtCore import Qt, QThread, pyqtSignal

from pipeline.utils import split_grid
from pipeline.solver_2x2 import JigsawSolverHybrid
from pipeline.solver_4x4 import JigsawSolverV7_HybridGUI
from pipeline.solver_8x8 import JigsawSolver8x8Clean


# ============================
# CONFIG
# ============================
DATASET_BASE = r"D:\College\Junior\Semester 1\Image Processing\Project\dataset"


# ==========================================================
# WORKER THREAD (SOLVER)
# ==========================================================
class SolverWorker(QThread):
    finished = pyqtSignal(object)

    def __init__(self, image, dimension):
        super().__init__()
        self.image = image
        self.dimension = dimension

    def run(self):
        N = int(self.dimension[0])  # "2x2" → 2
        pieces = split_grid(self.image, N, N)

        if self.dimension == "2x2":
            solver = JigsawSolverHybrid(pieces)
        elif self.dimension == "4x4":
            solver = JigsawSolverV7_HybridGUI(pieces)
        else:  # 8x8
            solver = JigsawSolver8x8Clean(pieces)

        result = solver.solve()
        self.finished.emit(result)


# ==========================================================
# MAIN GUI
# ==========================================================
class PuzzleGUI(QWidget):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("Jigsaw Puzzle Solver – Classical CV")
        self.setMinimumSize(1000, 700)

        self.selected_dimension = None
        self.current_pixmap = None
        self.worker = None

        # ============================
        # STYLE
        # ============================
        self.setStyleSheet("""
            QWidget {
                background-color: #f0f0f0;
                font-family: 'Segoe UI';
                color: black;
            }
            QPushButton {
                background-color: #4a90e2;
                color: white;
                padding: 10px 18px;
                border-radius: 8px;
                font-size: 14px;
            }
            QPushButton:hover {
                background-color: #3b79bf;
            }
            QPushButton:disabled {
                background-color: #9bbdde;
                color: #eeeeee;
            }
            QLineEdit {
                padding: 8px 12px;
                font-size: 14px;
                border-radius: 8px;
                border: 1px solid #cccccc;
            }
            QFrame#TopBar {
                background-color: white;
                border-radius: 12px;
                border: 1px solid #cccccc;
            }
            QFrame#ImageFrame {
                background-color: white;
                border-radius: 12px;
                border: 1px solid #cccccc;
            }
        """)

        # ============================
        # DIMENSION BUTTONS
        # ============================
        self.dim_2x2_btn = QPushButton("2x2")
        self.dim_4x4_btn = QPushButton("4x4")
        self.dim_8x8_btn = QPushButton("8x8")

        for btn in (self.dim_2x2_btn, self.dim_4x4_btn, self.dim_8x8_btn):
            btn.setCheckable(True)
            btn.clicked.connect(self.select_dimension)

        # ============================
        # INPUT
        # ============================
        self.puzzle_input = QLineEdit()
        self.puzzle_input.setPlaceholderText("Puzzle number (0–109)")

        # ============================
        # BUTTONS
        # ============================
        self.show_puzzle_btn = QPushButton("Show Puzzle")
        self.show_puzzle_btn.clicked.connect(self.show_puzzle)

        self.solve_btn = QPushButton("Solve")
        self.solve_btn.clicked.connect(self.solve_puzzle)

        # ============================
        # PROGRESS BAR
        # ============================
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        self.progress_bar.setRange(0, 0)  # Indeterminate

        # ============================
        # IMAGE LABEL
        # ============================
        self.image_label = QLabel("No puzzle selected")
        self.image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.image_label.setFont(QFont("Segoe UI", 14))
        self.image_label.setStyleSheet("background-color: #fafafa;")
        self.image_label.setSizePolicy(
            QSizePolicy.Policy.Expanding,
            QSizePolicy.Policy.Expanding
        )

        # ============================
        # TOP BAR
        # ============================
        top_bar_frame = QFrame()
        top_bar_frame.setObjectName("TopBar")
        top_bar_frame.setFixedHeight(100)

        top_bar = QHBoxLayout()
        top_bar.setAlignment(Qt.AlignmentFlag.AlignCenter)
        top_bar.setSpacing(12)

        top_bar.addWidget(self.dim_2x2_btn)
        top_bar.addWidget(self.dim_4x4_btn)
        top_bar.addWidget(self.dim_8x8_btn)
        top_bar.addWidget(self.puzzle_input)
        top_bar.addWidget(self.show_puzzle_btn)
        top_bar.addWidget(self.solve_btn)

        top_bar_frame.setLayout(top_bar)

        # ============================
        # IMAGE FRAME
        # ============================
        image_frame = QFrame()
        image_frame.setObjectName("ImageFrame")

        image_layout = QVBoxLayout()
        image_layout.addWidget(self.image_label)
        image_layout.addWidget(self.progress_bar)
        image_frame.setLayout(image_layout)

        # ============================
        # MAIN LAYOUT
        # ============================
        main_layout = QVBoxLayout()
        main_layout.setContentsMargins(20, 20, 20, 20)
        main_layout.setSpacing(15)
        main_layout.addWidget(top_bar_frame)
        main_layout.addWidget(image_frame)

        self.setLayout(main_layout)

    # ==========================================================
    # LOGIC
    # ==========================================================
    def select_dimension(self):
        sender = self.sender()
        self.selected_dimension = sender.text()
        for btn in (self.dim_2x2_btn, self.dim_4x4_btn, self.dim_8x8_btn):
            if btn != sender:
                btn.setChecked(False)

    def show_puzzle(self):
        if not self.selected_dimension:
            QMessageBox.warning(self, "Error", "Select a dimension.")
            return

        if not self.puzzle_input.text().isdigit():
            QMessageBox.warning(self, "Error", "Enter a valid puzzle number.")
            return

        puzzle_number = int(self.puzzle_input.text())
        folder = os.path.join(DATASET_BASE, f"puzzle_{self.selected_dimension}")

        for ext in ("jpg", "png", "jpeg"):
            path = os.path.join(folder, f"{puzzle_number}.{ext}")
            if os.path.exists(path):
                self.current_pixmap = QPixmap(path)
                self._update_image()
                return

        QMessageBox.warning(self, "Error", "Puzzle not found.")

    def solve_puzzle(self):
        if not self.selected_dimension:
            QMessageBox.warning(self, "Error", "Select a dimension.")
            return

        if not self.puzzle_input.text().isdigit():
            QMessageBox.warning(self, "Error", "Enter a valid puzzle number.")
            return

        puzzle_number = int(self.puzzle_input.text())
        folder = os.path.join(DATASET_BASE, f"puzzle_{self.selected_dimension}")

        for ext in ("jpg", "png", "jpeg"):
            path = os.path.join(folder, f"{puzzle_number}.{ext}")
            if os.path.exists(path):
                break
        else:
            QMessageBox.warning(self, "Error", "Puzzle not found.")
            return

        image = cv2.imread(path)

        self.solve_btn.setEnabled(False)
        self.progress_bar.setVisible(True)

        self.worker = SolverWorker(image, self.selected_dimension)
        self.worker.finished.connect(self.on_solve_finished)
        self.worker.start()

    def on_solve_finished(self, result):
        h, w, ch = result.shape
        qimg = QImage(result.data, w, h, ch * w, QImage.Format.Format_BGR888)
        self.current_pixmap = QPixmap.fromImage(qimg)

        self._update_image()
        self.progress_bar.setVisible(False)
        self.solve_btn.setEnabled(True)

    # ==========================================================
    # IMAGE RESIZE HANDLING
    # ==========================================================
    def _update_image(self):
        if self.current_pixmap is None:
            return

        self.image_label.setPixmap(
            self.current_pixmap.scaled(
                self.image_label.size(),
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation
            )
        )

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._update_image()
