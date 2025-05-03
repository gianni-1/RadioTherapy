# Add glob import for file searching
import os
import sys
import traceback
import glob
# ensure the project root (parent of sourcecode/) is on the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from monai.data.image_reader import NibabelReader
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QPushButton, QVBoxLayout, QWidget, QFileDialog,
    QMessageBox, QGroupBox, QToolButton,
    QLabel, QSpinBox, QDoubleSpinBox, QProgressDialog
)
from PySide6.QtCore import Qt, QObject, Signal, QThread, Slot
from PySide6.QtGui import QAction, QPixmap
import visualization
import nibabel as nib  # for handling NIfTI files
import numpy as np  # for numerical operations
from system_manager import SystemManager
from parameter_manager import ParameterManager
import torch
from monai.transforms import (
    Compose, LoadImaged, EnsureChannelFirstd, Lambdad,
    EnsureTyped, Orientationd, Spacingd, SpatialPadd,
    CenterSpatialCropd, ScaleIntensityRangePercentilesd, ToTensord,
)
from monai.data import NumpyReader
def handle_exception(exc_type, exc_value, exc_tb):
    # Print full traceback for uncaught exceptions
    traceback.print_exception(exc_type, exc_value, exc_tb)

sys.excepthook = handle_exception

class TrainingWorker(QObject):
    """
    Worker object to run the training in a separate thread.
    """
    finished = Signal()
    error = Signal(str)
    def __init__(self, manager):
        super().__init__()
        self.manager = manager
    def run(self):
        try:
            self.manager.run_training()
            self.finished.emit()
        except Exception as ex:
            traceback.print_exc()
            # Emit error signal instead of showing QMessageBox in worker thread
            self.error.emit(str(ex))

class MainWindow(QMainWindow):
    """
    Main entry point for the RadioTherapy project.
    
    This function sets up the configuration parameters, initializes the SystemManager,
    and triggers the training and inference processes.
    
    MainWindow is the primary GUI window for the RadioTherapy project.
    It provides a menu bar with file actions and a central widget containing buttons for CT upload, dose calculation, and visualization.
    """
    def __init__(self):
        super().__init__()
        self.setWindowTitle("RadioTherapy Project")
        self.setMinimumSize(800, 800)

        self.input_dir = None  # store the input directory for training inputs
        self.output_dir = None # store the output directory for training outputs

        self.ct_file = None  # store imported CT file (inference)

        # prepare parameters and transforms with default values
        # These values will be updated based on user input in the GUI
        self.pm = ParameterManager(
            energies=[0], batch_size=2, cube_size=64,                  
            num_epochs=5, learning_rate=1e-4, patience=3
        )
        transforms_chain = Compose([
            LoadImaged(keys=["input", "target"], reader=NumpyReader),
            EnsureChannelFirstd(keys=["input", "target"]),
            EnsureTyped(keys=["input", "target"]),
            Orientationd(keys=["input", "target"], axcodes="RAS"),
            Spacingd(keys=["input", "target"], pixdim=(2.4, 2.4, 2.4), mode= ("bilinear", "nearest")[1]),
            SpatialPadd(keys=["input", "target"], spatial_size=self.pm.cube_size, method="symmetric"),
            CenterSpatialCropd(keys=["input", "target"], roi_size=self.pm.cube_size),
            ScaleIntensityRangePercentilesd(
                keys="input", lower=0, upper=99.5, b_min=0, b_max=1
            ),
            ToTensord(keys=["input", "target"])
        ])
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # instantiate SystemManager here – no main.py needed
        self.system_manager = SystemManager(
            root_dir="", transforms=transforms_chain,
            resolutions=self.pm.resolutions, energies=self.pm.energies,
            batch_size=self.pm.batch_size, device=device,
            num_epochs=self.pm.num_epochs, learning_rate=self.pm.learning_rate, patience=self.pm.patience,
            cube_size= self.pm.cube_size,
            seed=42
        )

        # Create the menu bar and add the File menu
        self._create_menu_bar()

        # Create the central widget with buttons
        self._create_central_widget()
        # adjust window size to fit expanded widgets
        self.adjustSize()
        # optionally enforce minimum size to current content
        self.setMinimumSize(self.size())

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
        layout.setAlignment(Qt.AlignmentFlag.AlignTop)

        # Training section
        training_group = QGroupBox("Training", self)
        training_layout = QVBoxLayout()
        training_layout.setAlignment(Qt.AlignmentFlag.AlignTop)
        training_group.setLayout(training_layout)
        layout.addWidget(training_group)

        # Paremeter inputs for training 
        # Advanced training parameters in checkable group box
        advanced_group = QGroupBox("Advanced Settings", self)
        advanced_group.setCheckable(True)
        advanced_group.setChecked(False)
        adv_layout = QVBoxLayout()
        advanced_group.setLayout(adv_layout)
        training_layout.addWidget(advanced_group)

        # Advanced training parameters
        batch_label = QLabel("Batch Size:", self)
        self.batch_spin = QSpinBox(self)
        self.batch_spin.setRange(1, 128)
        self.batch_spin.setValue(2)
        self.batch_spin.setSingleStep(1)
        adv_layout.addWidget(batch_label)
        adv_layout.addWidget(self.batch_spin)
        epochs_label = QLabel("Number of Epochs:", self)
        self.epochs_spin = QSpinBox(self)
        self.epochs_spin.setRange(1, 1000)
        self.epochs_spin.setValue(5)
        self.epochs_spin.setSingleStep(1)
        adv_layout.addWidget(epochs_label)
        adv_layout.addWidget(self.epochs_spin)
        patience_label = QLabel("Early stopping patience:", self)
        self.patience_spin = QSpinBox(self)
        self.patience_spin.setRange(1, 100)
        self.patience_spin.setValue(3)
        adv_layout.addWidget(patience_label)
        adv_layout.addWidget(self.patience_spin)
        learning_rate_label = QLabel("Learning Rate:", self)
        self.learning_rate_spin = QDoubleSpinBox(self)
        self.learning_rate_spin.setRange(0.0, 1.0)
        self.learning_rate_spin.setSingleStep(1e-5)
        self.learning_rate_spin.setDecimals(6)
        self.learning_rate_spin.setValue(1e-4)
        adv_layout.addWidget(learning_rate_label)
        adv_layout.addWidget(self.learning_rate_spin)

        # Add a spacer to the advanced layout to push the widgets to the top
        for idx in range(adv_layout.count()):
            adv_layout.itemAt(idx).widget().setVisible(False)

        # toggle visibility of advanced parameters when the group box is checked/unchecked
        advanced_group.toggled.connect(
            lambda chk: (
                [adv_layout.itemAt(i).widget().setVisible(chk)
                 for i in range(adv_layout.count())],
                self.adjustSize()
            )
        )

        # Inference section
        inference_group = QGroupBox("Inference", self)
        inference_layout = QVBoxLayout()
        inference_layout.setAlignment(Qt.AlignmentFlag.AlignTop)
        inference_group.setLayout(inference_layout)
        layout.addWidget(inference_group)

        #Energy spectrum inputs(eV)
        energy_min_label = QLabel("Energy Min (eV):", self)
        self.energy_min_spin = QDoubleSpinBox(self)
        self.energy_min_spin.setRange(0.0, 10000.0)
        self.energy_min_spin.setDecimals(2)
        self.energy_min_spin.setValue(0.0)
        inference_layout.addWidget(energy_min_label)
        inference_layout.addWidget(self.energy_min_spin)

        energy_max_label= QLabel("Energy Max (eV):", self)
        self.energy_max_spin = QDoubleSpinBox(self)
        self.energy_max_spin.setRange(0.0, 10000.0)
        self.energy_max_spin.setDecimals(2)
        self.energy_max_spin.setValue(50.0)
        inference_layout.addWidget(energy_max_label)
        inference_layout.addWidget(self.energy_max_spin)

        # Upload button for CT scan
        upload_button = QPushButton("Upload CT Scan", self)
        upload_button.setToolTip("Click to upload a CT scan (NIfTI file)")
        upload_button.clicked.connect(self.open_file_dialog)
        inference_layout.addWidget(upload_button)

        # Dose calculation button (disabled until CT file is uploaded)
        self.dose_button = QPushButton("Calculate Dose", self)
        self.dose_button.setToolTip("Calculate dose distribution (requires CT scan)")
        self.dose_button.setEnabled(False)
        self.dose_button.clicked.connect(self.calculate_dose)
        inference_layout.addWidget(self.dose_button)

        # Dummy dose visualization button
        visualize_dose_button = QPushButton("Visualize Dummy Dose", self)
        visualize_dose_button.setToolTip("Visualize a dummy dose distribution")
        visualize_dose_button.clicked.connect(self.visualize_inference_results)
        inference_layout.addWidget(visualize_dose_button)

        #Training folder selection buttons
        self.input_button = QPushButton("Select Energy Folder for Training", self)
        self.input_button.setToolTip("Select the folder containing input cubes for training")
        self.input_button.clicked.connect(self.select_input_folder)
        training_layout.addWidget(self.input_button)
        # Input path label
        self.input_label = QLabel("No input folder selected", self)
        training_layout.addWidget(self.input_label)

        self.train_button = QPushButton("Train Model", self)
        self.train_button.setToolTip("Train the model with the selected input and output folders")
        self.train_button.setEnabled(False)  # Initially disabled
        self.train_button.clicked.connect(self.train_model)
        training_layout.addWidget(self.train_button)
        # Button to show training graphs in-app
        self.show_plots_button = QPushButton("Show Training Graphs", self)
        self.show_plots_button.setToolTip("Display loss and adversarial curves inside GUI")
        self.show_plots_button.setEnabled(False)  # Initially disabled
        self.show_plots_button.clicked.connect(self.show_training_plots)
        training_layout.addWidget(self.show_plots_button)

        # Visualization button for inference results
        self.visualize_button = QPushButton("Visualize Dose Distribution", self)
        self.visualize_button.setToolTip("Visualize the calculated dose distribution")
        self.visualize_button.setEnabled(False)  # Disabled until inference is complete
        self.visualize_button.clicked.connect(self.visualize_inference_results)
        inference_layout.addWidget(self.visualize_button)

        central_widget.setLayout(layout)
        
    
    # Select input and output folders for training
    def select_input_folder(self):
        """
        Opens a directory dialog to select the output cubes folder for training.
        Enables the Train button if both folders are selected.
        """
        folder = QFileDialog.getExistingDirectory(self, "Select Input Cubes Directory", "", QFileDialog.Option.ShowDirsOnly)
        if folder:
            # If the user selected an energy folder containing both 'inputcube' and 'outputcube', derive both paths
            energy_folder = folder
            # Parse and propagate selected energy value
            energy_value = float(os.path.basename(energy_folder).replace("_", "."))
            # Update GUI parameter manager and system manager
            self.pm.energies = [energy_value]
            self.system_manager.energies = [energy_value]
            input_cube_path = os.path.join(energy_folder, "inputcube")
            output_cube_path = os.path.join(energy_folder, "outputcube")
            if os.path.isdir(input_cube_path) and os.path.isdir(output_cube_path):
                # Use the energy-specific subfolders directly
                self.input_dir = input_cube_path
                self.input_label.setText(f"Input cube folder: {self.input_dir}")
                self.output_dir = output_cube_path
 
                # Determine cube size from one of the input .npy files
                files = glob.glob(os.path.join(self.input_dir, "*.npy"))
                if files:
                    arr = np.load(files[0])
                    self.pm.cube_size = arr.shape
                    transforms_chain = Compose([
                        LoadImaged(keys=["input", "target"], reader=NumpyReader),
                        EnsureChannelFirstd(keys=["input", "target"]),
                        EnsureTyped(keys=["input", "target"]),
                        Orientationd(keys=["input", "target"], axcodes="RAS"),
                        Spacingd(keys=["input", "target"], pixdim=(2.4, 2.4, 2.4), mode=("bilinear", "nearest")),
                        SpatialPadd(keys=["input", "target"], spatial_size=self.pm.cube_size, method="symmetric"),
                        CenterSpatialCropd(keys=["input", "target"], roi_size=self.pm.cube_size),
                        ScaleIntensityRangePercentilesd(keys="input", lower=0, upper=99.5, b_min=0, b_max=1),
                        ToTensord(keys=["input", "target"])
                    ])
                    self.system_manager.cube_size = self.pm.cube_size
                    self.system_manager.transforms = transforms_chain
                self.update_train_button_state()
                return

            # Otherwise treat folder as the direct input-cube directory
            self.input_dir = folder
            self.input_label.setText(f"Input folder: {folder}")
            # determine cube size from first cube file in folder
            files = glob.glob(os.path.join(folder, "*.npy")) + glob.glob(os.path.join(folder, "*.nii")) + glob.glob(os.path.join(folder, "*.nii.gz"))
            if files:
                sample = files[0]
                try:
                    if sample.lower().endswith(".npy"):
                        arr = np.load(sample)
                    else:
                        arr = np.asarray(nib.load(sample).dataobj)
                    size = arr.shape
                    self.pm.cube_size = size
                    # update transforms in system_manager
                    transforms_chain = Compose([
                        LoadImaged(keys=["input", "target"], reader=NumpyReader),
                        EnsureChannelFirstd(keys=["input", "target"]),
                        EnsureTyped(keys=["input", "target"]),
                        Orientationd(keys=["input", "target"], axcodes="RAS"),
                        Spacingd(keys=["input", "target"], pixdim=(2.4, 2.4, 2.4), mode=("bilinear", "nearest")),
                        SpatialPadd(keys=["input", "target"], spatial_size=self.pm.cube_size, method="symmetric"),
                        CenterSpatialCropd(keys=["input", "target"], roi_size=self.pm.cube_size),
                        ScaleIntensityRangePercentilesd(
                            keys=["input"], lower=0, upper=99.5, b_min=0, b_max=1
                        ),
                        ToTensord(keys=["input", "target"])
                    ])
                    self.system_manager.cube_size = self.pm.cube_size
                    self.system_manager.transforms = transforms_chain
                except Exception as e:
                    QMessageBox.warning(self, "Warning", f"Failed to determine cube size: {e}")
                print(f"Cube size set to: {self.pm.cube_size}")
            self.update_train_button_state()

    
    # Update the state of the Train button based on folder selection
    def update_train_button_state(self):
        """
        Enables the Train button if both input and output directories are selected.
        """
        if self.input_dir:
            self.train_button.setEnabled(True)
        else:
            self.train_button.setEnabled(False)
    
    # Train the model using the selected input and output folders
    def train_model(self):
        """
        Calls the training pipeline using the selected input and output folders.
        Expects both folders to be subdirectories of the same parent (root) folder.
        """
        if not self.input_dir or not self.output_dir:
            QMessageBox.warning(self, "Error", "Please select both input and output folders.")
            return
        
        #Check that both directories are subdirectories of the same parent (root) folder
        parent_in = os.path.dirname(self.input_dir)
        parent_out = os.path.dirname(self.output_dir)
        if parent_in != parent_out:
            QMessageBox.warning(self, "Error", "Input and output folders must be subdirectories of the same parent folder.")
            return

        self.pm.batch_size = self.batch_spin.value()
        self.pm.num_epochs = self.epochs_spin.value()
        self.pm.patience = self.patience_spin.value()
        self.pm.learning_rate = self.learning_rate_spin.value()

        energy_folder = parent_in
        # parent of energy_folder is the dataset root containing all energy subfolders
        dataset_root = os.path.dirname(energy_folder)
        self.system_manager.root_dir = dataset_root
        #update training parameters from GUI
        self.system_manager.batch_size = self.pm.batch_size
        self.system_manager.num_epochs = self.pm.num_epochs
        self.system_manager.patience = self.pm.patience
        self.system_manager.learning_rate = self.pm.learning_rate

        # run training in background thread to avoid freezing GUI
        progress = QProgressDialog("Training model...", None, 0, 0, self)
        progress.setWindowModality(Qt.ApplicationModal)
        progress.setCancelButton(None)
        progress.setMinimumDuration(0)
        progress.show()
        # create worker and thread
        thread = QThread(self)
        worker = TrainingWorker(self.system_manager)
        worker.moveToThread(thread)
        thread.started.connect(worker.run)
        worker.finished.connect(thread.quit)
        worker.finished.connect(worker.deleteLater)
        thread.finished.connect(thread.deleteLater)
        # handle completion and errors on main thread
        worker.finished.connect(self.on_training_finished)
        worker.error.connect(self.on_training_error)
        # keep references to prevent garbage collection
        self._train_thread = thread
        self._train_worker = worker
        self._train_progress = progress
        thread.start()

    # Open a file dialog to select a CT scan file (inference)
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

    # Calculate the dose distribution using the selected CT scan file(inference)
    def calculate_dose(self):
        """
        Placeholder for dose calculation logic. Requires a CT file to be uploaded first.
        """
        if not self.ct_file:
            QMessageBox.warning(self, "Error", "Please upload a CT scan first.")
            return

        min_e = self.energy_min_spin.value()
        max_e = self.energy_max_spin.value()
        #validate energy range
        if min_e > max_e:
            QMessageBox.warning(self, "Error", "Minimum energy must be less than maximum energy.")
            return   
        self.system_manager.energies = [min_e, max_e]  # Update the energy range for inference

        try:
            # Run inference
            self.system_manager.run_inference(self.ct_file)
            QMessageBox.information(self, "Success", "Dose distribution calculated successfully.")
            self.visualize_button.setEnabled(True)  # Enable visualization button
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to calculate dose distribution: {e}")

    # Visualize the dose distribution from inference results
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
            
    @Slot()
    def on_training_finished(self):
        """Slot to close progress dialog and show success message on main thread"""
        if hasattr(self, '_train_progress'):
            self._train_progress.close()
        QMessageBox.information(self, "Success", "Training completed.")
        # show plots in-app
        self.show_training_plots()
        # enable the button to show training plots
        self.show_plots_button.setEnabled(True)

    @Slot(str)
    def on_training_error(self, msg):
        """Slot to close progress dialog and show error message on main thread"""
        if hasattr(self, '_train_progress'):
            self._train_progress.close()
        QMessageBox.critical(self, "Error", f"Failed to train the model: {msg}")
    
    @Slot()
    def show_training_plots(self):
        """Open a window showing the three training plot images together."""
        import glob
        from PySide6.QtWidgets import QScrollArea
        from PySide6.QtGui import QGuiApplication
        # compute screen geometry
        screen_rect = QGuiApplication.primaryScreen().availableGeometry()
        # use up to 80% of screen
        max_w_screen = int(screen_rect.width() * 0.8)
        max_h_screen = int(screen_rect.height() * 0.8)
        # divide screen into two rows: top row height, bottom row height
        row_height = max_h_screen // 2
        # divide width: top two share row, bottom spans
        top_w = max_w_screen // 2
        bottom_w = max_w_screen

        patterns = ["loss_curves_res*.png", "learning_curve.png", "adv_curves.png"]
        plots_window = QWidget()
        plots_window.setWindowTitle("Training Graphs")
        labels = []
        idx = 0
        for pattern in patterns:
            for fname in glob.glob(pattern):
                lbl = QLabel()
                pix = QPixmap(fname)
                # pick target dims by position
                if idx < 2:
                    target_w, target_h = top_w, row_height
                else:
                    target_w, target_h = bottom_w, row_height
                pix = pix.scaled(target_w, target_h, Qt.KeepAspectRatio, Qt.SmoothTransformation)
                lbl.setPixmap(pix)
                lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
                labels.append(lbl)
                idx += 1
        from PySide6.QtWidgets import QGridLayout
        grid = QGridLayout()
        if len(labels) >= 2:
            grid.addWidget(labels[0], 0, 0)
            grid.addWidget(labels[1], 0, 1)
            if len(labels) >= 3:
                grid.addWidget(labels[2], 1, 0, 1, 2)
        else:
            for i, lbl in enumerate(labels):
                grid.addWidget(lbl, i, 0)
        scroll = QScrollArea()
        container = QWidget()
        container.setLayout(grid)
        scroll.setWidget(container)
        scroll.setWidgetResizable(True)
        main_layout = QVBoxLayout()
        main_layout.addWidget(scroll)
        plots_window.setLayout(main_layout)
        plots_window.resize(max_w_screen, max_h_screen)
        plots_window.show()
        self._plots_window = plots_window

if __name__ == "__main__":
    try:
        app = QApplication(sys.argv)
        window = MainWindow()
        window.show()
        sys.exit(app.exec())
    except Exception:
        traceback.print_exc()
        raise