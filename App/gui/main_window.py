from PyQt6.QtWidgets import (
    QWidget, QPushButton, QLabel, QFileDialog, QVBoxLayout, QHBoxLayout,
    QLineEdit, QFrame, QMessageBox
)
from PyQt6.QtGui import QPixmap, QFont
from PyQt6.QtCore import Qt
from pipeline.preprocessing import run_preprocessing
from gui.steps_window import StepsWindow
import os


class PuzzleGUI(QWidget):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("Jigsaw Puzzle Solver - Classical CV")
        self.setMinimumSize(1000, 700)

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
                padding: 10px;
            }

            QFrame#ImageFrame {
                background-color: white;
                border-radius: 12px;
                border: 1px solid #cccccc;
            }
        """)

        self.image_path = None
        self.steps = None
        self.selected_dimension = None
        self.puzzle_number = None

        # Buttons
        # self.upload_btn = QPushButton("Upload Image")
        # self.upload_btn.clicked.connect(self.load_image)

        # Dimension buttons
        self.dim_2x2_btn = QPushButton("2x2")
        self.dim_4x4_btn = QPushButton("4x4")
        self.dim_8x8_btn = QPushButton("8x8")
        for btn in [self.dim_2x2_btn, self.dim_4x4_btn, self.dim_8x8_btn]:
            btn.setCheckable(True)
            btn.clicked.connect(self.select_dimension)

        # Puzzle number input
        self.puzzle_input = QLineEdit()
        self.puzzle_input.setPlaceholderText("Puzzle number (0–109)")

        self.run_btn = QPushButton("Run Preprocessing")
        self.run_btn.clicked.connect(self.run_pipeline)

        self.show_puzzle_btn = QPushButton("Show Puzzle")
        self.show_puzzle_btn.clicked.connect(self.show_puzzle)


        self.show_steps_btn = QPushButton("Show Steps")
        self.show_steps_btn.setEnabled(False)
        self.show_steps_btn.clicked.connect(self.open_steps_window)

        # Image preview
        self.image_label = QLabel("No puzzle chosen")
        self.image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.image_label.setFont(QFont("Segoe UI", 14))
        self.image_label.setStyleSheet("color: #555555; padding: 30px;")

        # Top bar frame
        top_bar_frame = QFrame()
        top_bar_frame.setObjectName("TopBar")
        top_bar_frame.setFixedHeight(100)

        top_bar = QHBoxLayout()
        top_bar.setContentsMargins(10, 5, 10, 5)
        top_bar.setSpacing(12)
        top_bar.setAlignment(Qt.AlignmentFlag.AlignCenter)

        #top_bar.addWidget(self.upload_btn)
        top_bar.addWidget(self.dim_2x2_btn)
        top_bar.addWidget(self.dim_4x4_btn)
        top_bar.addWidget(self.dim_8x8_btn)
        top_bar.addWidget(self.puzzle_input)
        top_bar.addWidget(self.run_btn)
        top_bar.addWidget(self.show_puzzle_btn)  # <-- ADDED: Show Puzzle button
        top_bar.addWidget(self.show_steps_btn)


        top_bar_frame.setLayout(top_bar)

        # Image frame
        image_frame = QFrame()
        image_frame.setObjectName("ImageFrame")
        image_layout = QVBoxLayout()
        image_layout.setContentsMargins(10, 10, 10, 10)
        image_layout.addWidget(self.image_label)
        image_frame.setLayout(image_layout)

        # Main layout
        main_layout = QVBoxLayout()
        main_layout.setContentsMargins(20, 20, 20, 20)
        main_layout.setSpacing(15)
        main_layout.addWidget(top_bar_frame)
        main_layout.addWidget(image_frame)

        self.setLayout(main_layout)

    def load_image(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Select Puzzle Image", "",
            "Images (*.png *.jpg *.jpeg)"
        )
        if path:
            self.image_path = path
            pixmap = QPixmap(path)
            self.image_label.setPixmap(
                pixmap.scaled(600, 600, Qt.AspectRatioMode.KeepAspectRatio,
                              Qt.TransformationMode.SmoothTransformation)
            )

    def select_dimension(self):
        sender = self.sender()
        self.selected_dimension = sender.text()

        # Uncheck other buttons
        for btn in [self.dim_2x2_btn, self.dim_4x4_btn, self.dim_8x8_btn]:
            if btn != sender:
                btn.setChecked(False)


    def show_puzzle(self):
        
        if not self.selected_dimension or self.puzzle_input.text() == "":
            QMessageBox.warning(self, "Error", "Select dimension and enter puzzle number first.")
            return
        # Determine folder based on selected dimension
        if self.selected_dimension == "2x2":
            folder = "C:\\Users\\Nada Serour\\Jigsaw_Puzzle\\dataset\\puzzle_2x2"
        elif self.selected_dimension == "4x4":
            folder = "C:\\Users\\Nada Serour\\Jigsaw_Puzzle\\dataset\\puzzle_4x4"
        elif self.selected_dimension == "8x8":
            folder = "C:\\Users\\Nada Serour\\Jigsaw_Puzzle\\dataset\\puzzle_8x8"
        else:
            QMessageBox.warning(self, "Error", "Invalid dimension selected.")
            return

         # Validate puzzle number
        puzzle_number = self.puzzle_input.text()
        if not puzzle_number.isdigit():
            QMessageBox.warning(self, "Error", "Enter a valid puzzle number.")
            return
        puzzle_number = int(puzzle_number)
         # Find the file with supported extensions
        for ext in ["jpg", "png", "jpeg"]:
            potential_path = os.path.join(folder, f"{puzzle_number}.{ext}")
            if os.path.exists(potential_path):
                pixmap = QPixmap(potential_path)
                self.image_label.setPixmap(
                    pixmap.scaled(600, 600, Qt.AspectRatioMode.KeepAspectRatio,
                                  Qt.TransformationMode.SmoothTransformation)
             )
            return

        QMessageBox.warning(self, "Error", f"Puzzle {puzzle_number} not found in {folder}.")

    def run_pipeline(self):
        # ----------------- ADD THIS -----------------
        # Automatically determine image path based on dimension and puzzle number
        if self.selected_dimension == "2x2":
            folder = "C:\\Users\\Nada Serour\\Jigsaw_Puzzle\\dataset\\puzzle_2x2"
        elif self.selected_dimension == "4x4":
            folder = "C:\\Users\\Nada Serour\\Jigsaw_Puzzle\\dataset\\puzzle_4x4"
        elif self.selected_dimension == "8x8":
            folder = "C:\\Users\\Nada Serour\\Jigsaw_Puzzle\\dataset\\puzzle_8x8"
        else:
            QMessageBox.warning(self, "Error", "Invalid dimension selected.")
            return
        # Build the full path (accept jpg, png, jpeg)
        for ext in ["jpg", "png", "jpeg"]:
            potential_path = os.path.join(folder, f"{self.puzzle_number}.{ext}")
            if os.path.exists(potential_path):
                self.image_path = potential_path
                break
        else:
            QMessageBox.warning(self, "Error", f"Image {self.puzzle_number} not found in {folder}.")
            return
        # Display the original image first
        pixmap = QPixmap(self.image_path)
        self.image_label.setPixmap(
            pixmap.scaled(600, 600, Qt.AspectRatioMode.KeepAspectRatio,
                        Qt.TransformationMode.SmoothTransformation)
        )
# --------------------------------------------

        if not self.image_path:
            QMessageBox.warning(self, "Error", "Please upload an image first.")
            return

        puzzle_text = self.puzzle_input.text()
        if not puzzle_text.isdigit() or not (0 <= int(puzzle_text) <= 109):
            QMessageBox.warning(self, "Error", "Enter a valid puzzle number (0–109).")
            return
        self.puzzle_number = int(puzzle_text)

        if not self.selected_dimension:
            QMessageBox.warning(self, "Error", "Select a puzzle dimension (2x2, 4x4, 8x8).")
            return

        # Run preprocessing
        final_edges, self.steps = run_preprocessing(self.image_path)

        # Display edges
        pixmap = QPixmap("./output/steps/Edges.png")
        self.image_label.setPixmap(
            pixmap.scaled(600, 600, Qt.AspectRatioMode.KeepAspectRatio,
                          Qt.TransformationMode.SmoothTransformation)
        )

        self.show_steps_btn.setEnabled(True)

    def open_steps_window(self):
        if self.steps is None:
            return
        self.steps_window = StepsWindow(self.steps)
        self.steps_window.show()
