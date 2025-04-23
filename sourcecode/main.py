# main.py braucht man nicht mehr weil alles in gui.py integriert ist
"""
import torch
from monai.transforms import (
    Compose, LoadImaged, EnsureChannelFirstd, Spacingd,
    SpatialPadd, CenterSpatialCropd, ToTensord
)
from monai.data import NibabelReader

from system_manager import SystemManager
from parameter_manager import ParameterManager

def main(root_dir):
    
    Main entry point for the RadioTherapy project.
    
    This function sets up the configuration parameters, initializes the SystemManager,
    and triggers the training and inference processes.
    
    # Set the device (prefer GPU if available, otherwise CPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Define the root directory for your dataset.
    # Update this path to point to where your dataset is stored.
    root_dir = "/Users/giannigagliardi/Documents/Git/RadioTherapy/data"  

    # Define the training configurations:
    # - Resolutions: a list of tuples representing different spatial resolutions (e.g., 64x64x64 and 32x32x32).
    # - Energies: a list of energy values (e.g., in keV) for which separate trainings will be executed.
    energies = [11.5] # Wurde aus Datei rausgezogen
    batch_size = 2
    cube_size = (64, 64, 64)  # Cube size to which CT data will be resized.
    n_epochs = 5
    lr = 1e-4  # Learning rate for the optimizer.
    patience = 3

    # Initialize the ParameterManager with the defined configurations.
    # This class centralizes the configuration parameters for the project.
    param_manager = ParameterManager(
        energies=energies,
        batch_size=batch_size,
        cube_size=cube_size,
        n_epochs=n_epochs,
        lr=lr,
        patience=patience
    )

    # Define a transformation pipeline for loading and preprocessing NIfTI images.
    # In this case, we load the image using NibabelReader, ensure the channel dimension is first, and convert to tensor.
    transforms_chain = Compose([
        LoadImaged(keys=["image"], reader=NibabelReader),
        EnsureChannelFirstd(keys=["image"]),
        Spacingd(keys=["image"], pixdim=(2.4, 2.4, 2.4), mode=("bilinear")),
        SpatialPadd(keys=["image"], spatial_size=param_manager.cube_size, method="symmetric"),
        CenterSpatialCropd(keys=["image"], roi_size=param_manager.cube_size),
        ToTensord(keys=["image"])
    ])

    # Initialize the SystemManager with the given configuration.
    # SystemManager orchestrates the entire workflow: loading data, training, and inference.
    system_manager = SystemManager(
        root_dir=root_dir,
        transforms=transforms_chain,
        resolutions=param_manager.resolutions,
        energies=param_manager.energies,
        batch_size=param_manager.batch_size,
        device=device,
        n_epochs=param_manager.n_epochs,
        lr=param_manager.lr,
        patience=param_manager.patience,
        seed=42
    )

    # This function should check that the data is complete and correctly formatted.
    system_manager.validiereDaten()

    # Run the training process for all resolution and energy combinations.
    system_manager.run_training()

    # After training is complete, run inference.
    system_manager.run_inference()


if __name__ == "__main__":
    main()
"""