import sys
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QPushButton, QVBoxLayout, QWidget, QFileDialog, QMessageBox
)
from PySide6.QtCore import Qt
from PySide6.QtGui import QAction
import visualization
import nibabel as nib  # for handling NIfTI files
import inference_module as inference  # for dose calculation
import numpy as np  # for numerical operations
import os  # for file path operations

class MainWindow(QMainWindow):
    """
    MainWindow is the primary GUI window for the RadioTherapy project.
    It provides a menu bar with file actions and a central widget containing buttons for CT upload, dose calculation, and visualization.
    """
    def __init__(self):
        super().__init__()
        self.setWindowTitle("RadioTherapy GUI")
        self.setMinimumSize(600, 400)

        self.ct_file = None  # store imported CT file

        # Create the menu bar and add the File menu
        self._create_menu_bar()

        # Create the central widget with buttons
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
        Sets up the central widget with a vertical layout and adds buttons for CT upload, dose calculation, and visualization.
        """
        central_widget = QWidget(self)
        self.setCentralWidget(central_widget)

        layout = QVBoxLayout()
        layout.setAlignment(Qt.AlignmentFlag.AlignCenter)

        # Upload button for CT scan
        upload_button = QPushButton("Upload CT Scan", self)
        upload_button.setToolTip("Click to upload a CT scan (NIfTI file)")
        upload_button.clicked.connect(self.open_file_dialog)
        layout.addWidget(upload_button)

        # Dose calculation button (disabled until CT file is uploaded)
        self.dose_button = QPushButton("Calculate Dose", self)
        self.dose_button.setToolTip("Calculate dose distribution (requires CT scan)")
        self.dose_button.setEnabled(False)
        self.dose_button.clicked.connect(self.calculate_dose)
        layout.addWidget(self.dose_button)

        # Visualization button for inference results
        self.visualize_button = QPushButton("Visualize Dose Distribution", self)
        self.visualize_button.setToolTip("Visualize the calculated dose distribution")
        self.visualize_button.setEnabled(False)  # Disabled until inference is complete
        self.visualize_button.clicked.connect(self.visualize_inference_results)
        layout.addWidget(self.visualize_button)

        central_widget.setLayout(layout)

    def open_file_dialog(self):
        """
        Opens a file dialog for selecting a CT scan file in NIfTI format.
        If a file is selected, its path is stored and the dose calculation button is enabled.
        """
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Select CT Scan", "", "NIfTI Files (*.nii *.nii.gz)"
        )
        if file_path:
            self.ct_file = file_path  # store the imported CT file
            print(f"Selected file: {file_path}")
            try:
                # Load the NIfTI file to ensure it's valid
                img = nib.load(file_path) 
                data = np.asarray(img)
                self.dose_button.setEnabled(True)  # enable dose calculation button, after successful load
                print(f"Image shape: {data.shape}")
            except Exception as e:
                print(f"Error loading file: {e}")
                QMessageBox.critical(self, "Error", f"Failed to load the selected file: {e}")

    def calculate_dose(self):
        """
        Placeholder for dose calculation logic. Requires a CT file to be uploaded first.
        """
        if not self.ct_file:
            QMessageBox.warning(self, "Error", "Please upload a CT scan first.")
            return

        try:
            # Run inference
            inference.run_inference(self.ct_file)
            QMessageBox.information(self, "Success", "Dose distribution calculated successfully.")
            self.visualize_button.setEnabled(True)  # Enable visualization button
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to calculate dose distribution: {e}")

    def visualize_inference_results(self):
        """
        Visualizes the calculated dose distribution from inference results and embeds it in the GUI window.
        """
        try:
            # TODO: Impelement variable for dose_result_path
            if not hasattr(self, 'dose_result_path') or not self.dose_result_path:
                QMessageBox.warning(self, "Error", "No dose calculation results available. Please calculate dose first.")
                return
                
            # Load the dose distribution from the result file
            fig = visualization.load_and_visualize(self.dose_result_path, title="Calculated Dose Distribution")
            
            # Create a Qt widget from the matplotlib figure
            canvas = visualization.get_visualization_widget(fig)
            
            # Create a new window to display the visualization
            visualization_window = QWidget()
            layout = QVBoxLayout()
            layout.addWidget(canvas)
            visualization_window.setLayout(layout)
            visualization_window.setWindowTitle("Dose Visualization")
            visualization_window.setMinimumSize(600, 500)
            visualization_window.show()
            
            # Store a reference to prevent garbage collection
            self._visualization_window = visualization_window
            
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to visualize dose distribution: {e}")

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())