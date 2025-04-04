import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np

def visualize_dose(dose, title="Dosisverteilung - Axiale Ansicht"):
    # Visualizes the middle axial slice of a 3D dose distribution.
    if dose.ndim != 3:
        raise ValueError("Dose distribution must be a 3D array.")
    mid_slice = dose[dose.shape[0] // 2]
    plt.figure()
    plt.imshow(mid_slice, cmap="hot")
    plt.title(title)
    plt.colorbar()
    plt.show()

def load_and_visualize(nifti_path, title="Dosisverteilung - Axiale Ansicht"):
    # Loads a Nifti dose distribution file and visualizes its middle axial slice.
    img = nib.load(nifti_path)
    dose = img.get_fdata()
    visualize_dose(dose, title)
