from gui.main_window import PuzzleGUI
from PyQt6.QtWidgets import QApplication
import sys

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = PuzzleGUI()
    window.show()
    sys.exit(app.exec())
