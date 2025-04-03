from PySide6.QtWidgets import (
    QApplication, QMainWindow, QPushButton, QFileDialog, QVBoxLayout, QWidget, QMenuBar, QMenu, QWidgetAction, QMessageBox
)
import sys 
import nibabel as nib

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

        # Zeige die Einleitung beim Start
        self.show_introduction()

    def open_file_dialog(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Nifty-Datei auswählen", "", "Nifty Files (*.nii *.nii.gz)") 
        if file_path:
            print("Datei ausgewählt: ", file_path)

            #Laden der Nifty-Datei
            try: 
                img = nib.load(file_path)
                data = img.get_fdata()
                print("Bildgröße: ", data.shape)
            except Exception as e:
                print("Fehler beim Laden der Datei: ", e)

    def show_introduction(self):
        intro_text = (
            "Willkommen bei der RadioTherapy App\n\n"
            "Im Folgenden werden Sie aufgefordert, die Nifty-Daten ihrer Patienten auszuwählen.\n"
            "Außerdem können Sie Informationen über Partikel und Energieniveaus eingeben.\n"
            "Die App wird Ihnen dann die Möglichkeit geben, die Daten zu analysieren und zu visualisieren.\n\n"
                    )
        QMessageBox.information(self, "Einleitung", intro_text)

app = QApplication(sys.argv)
window = MainWindow()
window.show()
app.exec()