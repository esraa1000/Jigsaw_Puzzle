from PyQt6.QtWidgets import (
    QWidget, QLabel, QVBoxLayout, QScrollArea, QFrame, QHBoxLayout
)
from PyQt6.QtGui import QPixmap, QFont
from PyQt6.QtCore import Qt


class StepsWindow(QWidget):
    def __init__(self, steps):
        super().__init__()
        self.setWindowTitle("Processing Steps")
        self.setMinimumSize(900, 700)

        # Main layout
        main_layout = QVBoxLayout()
        main_layout.setContentsMargins(10, 10, 10, 10)

        # Scroll area
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setStyleSheet("background-color: #f0f0f0; border: none;")

        content = QWidget()
        content_layout = QVBoxLayout()
        content_layout.setSpacing(20)

        # Create a stylized card for each step
        for name in steps.keys():

            # Outer frame (card)
            frame = QFrame()
            frame.setFrameShape(QFrame.Shape.StyledPanel)
            frame.setStyleSheet("""
                QFrame {
                    background-color: white;
                    border-radius: 12px;
                    padding: 15px;
                    border: 1px solid #cccccc;
                }
            """)
            frame_layout = QVBoxLayout()
            frame_layout.setSpacing(12)

            # Step title
            title_label = QLabel(name)
            title_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            title_label.setFont(QFont("Segoe UI", 14, QFont.Weight.Bold))
            title_label.setStyleSheet("color: #333333;")

            # Image
            img_path = f"./output/steps/{name}.png"
            pixmap = QPixmap(img_path)

            img_label = QLabel()
            img_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            img_label.setPixmap(
                pixmap.scaled(650, 650, Qt.AspectRatioMode.KeepAspectRatio,
                              Qt.TransformationMode.SmoothTransformation)
            )

            # Add to frame
            frame_layout.addWidget(title_label)
            frame_layout.addWidget(img_label)
            frame.setLayout(frame_layout)

            # Add card to scrollable content
            content_layout.addWidget(frame)

        content_layout.addStretch()
        content.setLayout(content_layout)
        scroll.setWidget(content)

        main_layout.addWidget(scroll)
        self.setLayout(main_layout)
