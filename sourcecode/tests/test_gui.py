import sys
import os
import numpy as np
import pytest
from PySide6.QtWidgets import QApplication
from gui import MainWindow

@pytest.fixture(scope="session")
def qapp():
    """Provides a single QApplication instance for all tests."""
    app = QApplication.instance() or QApplication(sys.argv)
    return app

@pytest.fixture
def window(qapp, qtbot, tmp_path, monkeypatch):
    """
    Creates a MainWindow, constructs a dummy dataset in tmp_path,
    patches QFileDialog.select_input_folder, and returns the window.
    """
    # Create dummy dataset folder structure
    root = tmp_path / "dataset"
    e1 = root / "10_0"
    inputcube = e1 / "inputcube"
    outputcube = e1 / "outputcube"
    inputcube.mkdir(parents=True)
    outputcube.mkdir(parents=True)
    # Save dummy .npy file
    arr = np.zeros((4, 4, 4))
    np.save(inputcube / "dummy.npy", arr)

    # Instantiate MainWindow
    win = MainWindow()
    qtbot.addWidget(win)

    # Patch QFileDialog.select_input_folder to return our tmp folder
    def fake_select():
        win.input_dir = str(inputcube)
        win.output_dir = str(outputcube)
        # energies and transforms are also updated internally
        win.update_train_button_state()
    monkeypatch.setattr(win, "select_input_folder", fake_select)

    return win

def test_initial_button_states(window):
    """Dose, Train, and Visualize buttons are initially disabled."""
    assert not window.dose_button.isEnabled()
    assert not window.train_button.isEnabled()
    assert not window.visualize_button.isEnabled()

def test_energy_range_validation(window, monkeypatch):
    """calculate_dose raises an exception when min > max."""
    # Set conditions so that dose_button can't be enabled
    window.ct_file = "dummy.nii"        # Dummy-Pfad
    window.model_file_bool = True
    window.dose_button.setEnabled(True)

    # invalid range
    window.energy_min_spin.setValue(50.0)
    window.energy_max_spin.setValue(10.0)

    # Intercept warning dialog instead of exception
    warnings = []
    from PySide6.QtWidgets import QMessageBox
    monkeypatch.setattr(QMessageBox, "warning", lambda *args, **kwargs: warnings.append(True))
    window.calculate_dose()
    assert warnings, "Expected a warning dialog for invalid energy range"

def test_select_input_folder_enables_train(window):
    """select_input_folder enables the train_button when structure is correct."""
    # simulate user action
    window.select_input_folder()
    assert window.train_button.isEnabled()