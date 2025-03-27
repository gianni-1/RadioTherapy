from PySide6.QtWidgets import (
    QApplication, QMainWindow, QPushButton, QFileDialog, QVBoxLayout, QWidget, QMenuBar, QMenu, QWidgetAction
)
import sys 

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("RadioTherapy App")

        #Menuleiste
        menu_bar = self.menuBar()
        file_menu = menu_bar.addMenu("Datei")

        open_action = QWidgetAction(self)
        open_action.triggered.connect(self.open_file_dialog)
        file_menu.addAction(open_action)

        exit_action = QWidgetAction(self)
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)


        #Zentrales Widget mit Button 
        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        layout = QVBoxLayout()
        upload_button = QPushButton("CT-Scan hochladen")
        upload_button.clicked.connect(self.open_file_dialog)

        layout.addWidget(upload_button)
        central_widget.setLayout(layout)

    def open_file_dialog(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Datei auswählen", "", "Nifty Files (*.nii *.nii.gz)") 
        if file_path:
            print("Datei ausgewählt: ", file_path)

app = QApplication(sys.argv)
window = MainWindow()
window.show()
app.exec()