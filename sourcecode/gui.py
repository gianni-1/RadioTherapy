# gui.py

import sys
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QPushButton, QVBoxLayout, QWidget, QFileDialog
)
from PySide6.QtCore import Qt
from PySide6.QtGui import QAction

class MainWindow(QMainWindow):
    """
    MainWindow is the primary GUI window for the RadioTherapy project.
    It provides a menu bar with file actions and a central widget containing an upload button.
    """
    def __init__(self):
        super().__init__()
        self.setWindowTitle("RadioTherapy GUI")
        self.setMinimumSize(600, 400)

        # Create the menu bar and add the File menu
        self._create_menu_bar()

        # Create the central widget with an Upload button
        self._create_central_widget()

    def _create_menu_bar(self):
        """
        Creates the menu bar with a 'File' menu and adds actions such as 'Open' and 'Exit'.
        """
        menubar = self.menuBar()
        file_menu = menubar.addMenu("File")

        # Create 'Open' action to open a file dialog
        open_action = QAction("Open", self)
        open_action.setStatusTip("Open a CT Scan file")
        open_action.triggered.connect(self.open_file_dialog)
        file_menu.addAction(open_action)

        # Create 'Exit' action to close the application
        exit_action = QAction("Exit", self)
        exit_action.setStatusTip("Exit application")
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)

    def _create_central_widget(self):
        """
        Sets up the central widget with a vertical layout and adds an upload button.
        """
        central_widget = QWidget(self)
        self.setCentralWidget(central_widget)

        layout = QVBoxLayout()
        layout.setAlignment(Qt.AlignCenter)

        upload_button = QPushButton("Upload CT Scan", self)
        upload_button.setToolTip("Click to upload a CT scan (NIfTI file)")
        upload_button.clicked.connect(self.open_file_dialog)
        layout.addWidget(upload_button)

        central_widget.setLayout(layout)

    def open_file_dialog(self):
        """
        Opens a file dialog for selecting a CT scan file in NIfTI format.
        If a file is selected, its path is printed to the console.
        """
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Select CT Scan", "", "NIfTI Files (*.nii *.nii.gz)"
        )
        if file_path:
            print(f"Selected file: {file_path}")
            # Here, you can add additional logic to pass the file to your data loader

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())