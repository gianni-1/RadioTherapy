from PySide6.QtWidgets import (
    QApplication, QMainWindow, QPushButton, QFileDialog, QVBoxLayout, QWidget, QMenuBar, QMenu, QWidgetAction, QMessageBox, QInputDialog
)
import sys 
import nibabel as nib
import dose_prediction
import visualization  # visualization module for dose visualization
import dummy_dose  # remove this line when you have a real dose

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("RadioTherapy App")

        self.ct_file = None  # store imported CT file

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
        
        self.dose_button = QPushButton("Dosisverteilung berechnen")
        self.dose_button.setEnabled(False)  # disabled until CT file is imported
        self.dose_button.clicked.connect(self.calculate_dose)
        layout.addWidget(self.dose_button)

        # New button for dummy dose visualization
        visualize_dose_button = QPushButton("Dosis visualisieren")
        visualize_dose_button.clicked.connect(self.visualize_dummy_dose)
        layout.addWidget(visualize_dose_button)

        central_widget.setLayout(layout)

        # Zeige die Einleitung beim Start
        self.show_introduction()

    def open_file_dialog(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Nifty-Datei auswählen", "", "Nifty Files (*.nii *.nii.gz)") 
        if file_path:
            self.ct_file = file_path  # store the imported CT file
            self.dose_button.setEnabled(True)  # enable dose calc button
            print("Datei ausgewählt: ", file_path)

            #Laden der Nifty-Datei
            try: 
                img = nib.load(file_path)
                data = img.get_fdata()
                print("Bildgröße: ", data.shape)
            except Exception as e:
                print("Fehler beim Laden der Datei: ", e)

    def calculate_dose(self):
        if not self.ct_file:
            QMessageBox.warning(self, "Fehler", "Bitte importieren Sie zuerst die CT-Daten.")
            return
        input_path = self.ct_file  # use the previously imported CT file

        # Let the user select the trained DoseNet model (.ckpt file)
        model_path, _ = QFileDialog.getOpenFileName(
            self, "DoseNet Modell auswählen", "", "Checkpoint Files (*.ckpt);;All Files (*)"
        )
        if not model_path:
            return

        # Let the user choose the output path for the dose distribution.
        output_path, _ = QFileDialog.getSaveFileName(
            self, "Speicherort für Dosisverteilung", "", "Nifty Files (*.nii *.nii.gz)"
        )
        if not output_path:
            return

        try:
            dose = dose_prediction.predict_dose(input_path, output_path, model_path)
            QMessageBox.information(self, "Erfolg", "Dosisverteilung wurde berechnet und gespeichert.")
            visualization.visualize_dose(dose)  # visualize the dose result
        except Exception as e:
            QMessageBox.critical(self, "Fehler", f"Fehler bei der Dosisberechnung: {e}")

    def visualize_dummy_dose(self):
        try:
            dose = dummy_dose.generate_dummy_dose() # change this, when you have a real dose
            visualization.visualize_dose(dose, title="Dosisverteilung")
        except Exception as e:
            QMessageBox.critical(self, "Fehler", f"Fehler bei Dummy-Visualisierung: {e}")

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