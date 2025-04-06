# main.py

import torch
from monai.transforms import Compose, LoadImaged, EnsureChannelFirstd, ToTensord
from monai.data import NibabelReader

# Import the SystemManager from your modularized project
from system_manager import SystemManager

def main():
    """
    Main entry point for the RadioTherapy project.
    
    This function sets up the configuration parameters, initializes the SystemManager,
    and triggers the training and inference processes.
    """
    # Set the device (prefer GPU if available, otherwise CPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Define the root directory for your dataset.
    # Update this path to point to where your dataset is stored.
    root_dir = "/Users/giannigagliardi/Documents/Git/RadioTherapy/data"  

    # Define a transformation pipeline for loading and preprocessing NIfTI images.
    # In this case, we load the image using NibabelReader, ensure the channel dimension is first, and convert to tensor.
    transforms_chain = Compose([
        LoadImaged(keys=["image"], reader=NibabelReader),
        EnsureChannelFirstd(keys=["image"]),
        ToTensord(keys=["image"])
    ])

    # Define the training configurations:
    # - Resolutions: a list of tuples representing different spatial resolutions (e.g., 64x64x64 and 32x32x32).
    # - Energies: a list of energy values (e.g., in keV) for which separate trainings will be executed.
    resolutions = [(64, 64, 64), (32, 32, 32)]
    energies = [62, 75, 90]
    batch_size = 2

    # Initialize the SystemManager with the given configuration.
    # SystemManager orchestrates the entire workflow: loading data, training, and inference.
    system_manager = SystemManager(
        root_dir=root_dir,
        transforms=transforms_chain,
        resolutions=resolutions,
        energies=energies,
        batch_size=batch_size,
        device=device,
        seed=42
    )

    # (Optional) Validate the dataset before training begins.
    # This function should check that the data is complete and correctly formatted.
    system_manager.validiereDaten()

    # Run the training process for all resolution and energy combinations.
    system_manager.starteTraining()

    # After training is complete, run inference.
    system_manager.starteInferenz()


if __name__ == "__main__":
    main()