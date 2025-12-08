from PyQt6.QtWidgets import (
    QWidget, QPushButton, QLabel, QFileDialog, QVBoxLayout, QHBoxLayout, QComboBox, QFrame
)
from PyQt6.QtGui import QPixmap, QFont
from PyQt6.QtCore import Qt
from pipeline.preprocessing import run_preprocessing
from gui.steps_window import StepsWindow


class PuzzleGUI(QWidget):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("Jigsaw Puzzle Solver - Classical CV")
        self.setMinimumSize(1000, 700)

              # Apply window stylesheet (clean modern theme)
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

            /* ---- Modern, clean, consistent ComboBox styling ---- */

            QComboBox {
                padding: 8px 12px;
                font-size: 14px;
                border-radius: 8px;
                background-color: white;
                border: 1px solid #cccccc;
                color: #333333;
            }

            QComboBox:hover {
                border: 1px solid #4a90e2;
            }

            /* Remove old drop-down button and replace with a cleaner one */
            QComboBox::drop-down {
                border: none;
                width: 32px;
            }

            /* Custom arrow (simple modern ▼ triangle) */
            QComboBox::down-arrow {
                width: 0;
                height: 0;
                border-left: 6px solid transparent;
                border-right: 6px solid transparent;
                border-top: 8px solid #777777;
                margin-right: 8px;
            }

            QComboBox::down-arrow:hover {
                border-top: 8px solid #4a90e2;
            }

            /* Dropdown list */
            QComboBox QAbstractItemView {
                background-color: white;
                border: 1px solid #cccccc;
                padding: 6px;
                outline: 0;
                selection-background-color: #4a90e2;
                selection-color: white;
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

        # Buttons
        self.upload_btn = QPushButton("Upload Image")
        self.upload_btn.clicked.connect(self.load_image)

        self.dimension_box = QComboBox()
        self.dimension_box.addItems(["2x2", "4x4", "8x8"])

        self.run_btn = QPushButton("Run Preprocessing")
        self.run_btn.clicked.connect(self.run_pipeline)

        self.show_steps_btn = QPushButton("Show Steps")
        self.show_steps_btn.setEnabled(False)
        self.show_steps_btn.clicked.connect(self.open_steps_window)

        # Image preview area
        self.image_label = QLabel("No image uploaded")
        self.image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.image_label.setFont(QFont("Segoe UI", 14))
        self.image_label.setStyleSheet("color: #555555; padding: 30px;")

        # Top bar container (styled frame)
        top_bar_frame = QFrame()
        top_bar_frame.setObjectName("TopBar")

        # Make top bar only as tall as necessary
        top_bar_frame.setFixedHeight(70)

        # Top bar layout
        top_bar = QHBoxLayout()
        top_bar.setContentsMargins(10, 5, 10, 5)   # small padding
        top_bar.setSpacing(12)                     # spacing between widgets
        top_bar.setAlignment(Qt.AlignmentFlag.AlignCenter)

        top_bar.addWidget(self.upload_btn)
        top_bar.addWidget(self.dimension_box)
        top_bar.addWidget(self.run_btn)
        top_bar.addWidget(self.show_steps_btn)

        top_bar_frame.setLayout(top_bar)


        # Image frame for main display
        image_frame = QFrame()
        image_frame.setObjectName("ImageFrame")
        image_layout = QVBoxLayout()
        image_layout.setContentsMargins(10, 10, 10, 10)
        image_layout.addWidget(self.image_label)
        image_frame.setLayout(image_layout)


        # Main layout
        main_layout = QVBoxLayout()
        main_layout.setContentsMargins(20, 20, 20, 20)
        main_layout.setSpacing(15)     # reduced spacing between top bar and image frame
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

    def run_pipeline(self):
        if not self.image_path:
            return

        # Run preprocessing (Milestone 1)
        final_edges, self.steps = run_preprocessing(self.image_path)

        # Display the edges result
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
