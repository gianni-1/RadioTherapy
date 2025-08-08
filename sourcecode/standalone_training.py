#!/usr/bin/env python3
"""
Standalone Training Script for RadioTherapy Project

This script allows you to run training without the GUI. It provides the same
functionality as the training feature in the GUI but can be run from command line.

Usage:
    python standalone_training.py --input_folder /path/to/energy/folder [options]

Example:
    python standalone_training.py --input_folder /path/to/dataset/62_0 --epochs 10 --batch_size 4
"""

import os
import sys
import argparse
import logging
import traceback
import glob
import numpy as np
import torch
from pathlib import Path

# Add project root to Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

# Import project modules
import log_config  # Initialize logging config
from system_manager import SystemManager
from parameter_manager import ParameterManager
from monai.transforms import (
    Compose, LoadImaged, EnsureChannelFirstd, Lambdad,
    EnsureTyped, Orientationd, Spacingd, SpatialPadd,
    CenterSpatialCropd, ScaleIntensityRangePercentilesd, ToTensord,
)
from monai.data import NumpyReader

logger = logging.getLogger(__name__)


def parse_arguments():
    """Parse command line arguments for training configuration."""
    parser = argparse.ArgumentParser(
        description="Standalone training script for RadioTherapy project",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Required arguments
    parser.add_argument(
        '--input_folder', '-i', 
        type=str, 
        required=True,
        help='Path to energy folder containing inputcube and outputcube subdirectories'
    )
    
    # Training parameters
    parser.add_argument(
        '--batch_size', '-b',
        type=int,
        default=2,
        help='Batch size for training'
    )
    
    parser.add_argument(
        '--epochs', '-e',
        type=int,
        default=5,
        help='Number of training epochs'
    )
    
    parser.add_argument(
        '--learning_rate', '-lr',
        type=float,
        default=1e-5,
        help='Learning rate for training'
    )
    
    parser.add_argument(
        '--patience', '-p',
        type=int,
        default=20,
        help='Early stopping patience'
    )
    
    parser.add_argument(
        '--device',
        type=str,
        default='auto',
        choices=['auto', 'cpu', 'cuda'],
        help='Device to use for training (auto selects GPU if available)'
    )
    
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed for reproducibility'
    )
    
    return parser.parse_args()


def validate_input_folder(input_folder):
    """
    Validate that the input folder structure is correct.
    
    Args:
        input_folder (str): Path to the energy folder
        
    Returns:
        tuple: (input_cube_path, output_cube_path, dataset_root)
        
    Raises:
        ValueError: If folder structure is invalid
    """
    input_folder = Path(input_folder)
    
    if not input_folder.exists():
        raise ValueError(f"Input folder does not exist: {input_folder}")
    
    # Check if it's an energy folder with inputcube and outputcube subdirectories
    input_cube_path = input_folder / "inputcube"
    output_cube_path = input_folder / "outputcube"
    
    if input_cube_path.exists() and output_cube_path.exists():
        # This is an energy folder
        dataset_root = input_folder.parent
        logger.info(f"Found energy folder structure: {input_folder}")
        logger.info(f"Input cubes: {input_cube_path}")
        logger.info(f"Output cubes: {output_cube_path}")
        return str(input_cube_path), str(output_cube_path), str(dataset_root)
    else:
        # Treat as direct input cube directory
        logger.info(f"Treating as direct input cube directory: {input_folder}")
        return str(input_folder), None, str(input_folder.parent)


def determine_cube_size(input_folder):
    """
    Determine cube size from the first available cube file.
    
    Args:
        input_folder (str): Path to input cube folder
        
    Returns:
        tuple: Cube size dimensions
    """
    # Look for .npy, .nii, or .nii.gz files
    patterns = ["*.npy", "*.nii", "*.nii.gz"]
    files = []
    
    for pattern in patterns:
        files.extend(glob.glob(os.path.join(input_folder, pattern)))
    
    if not files:
        raise ValueError(f"No cube files found in {input_folder}")
    
    sample_file = files[0]
    logger.info(f"Determining cube size from: {sample_file}")
    
    try:
        if sample_file.lower().endswith(".npy"):
            arr = np.load(sample_file)
        else:
            import nibabel as nib
            arr = np.asarray(nib.load(sample_file).dataobj)
        
        cube_size = arr.shape
        logger.info(f"Detected cube size: {cube_size}")
        return cube_size[0]
        
    except Exception as e:
        raise ValueError(f"Failed to determine cube size from {sample_file}: {e}")


def get_energy_list(dataset_root):
    """
    Get list of available energies from dataset root directory.
    
    Args:
        dataset_root (str): Path to dataset root containing energy folders
        
    Returns:
        list: List of energy values
    """
    try:
        # List all energy subfolder names (skip hidden files)
        energy_names = [
            d for d in os.listdir(dataset_root)
            if os.path.isdir(os.path.join(dataset_root, d)) and not d.startswith('.')
        ]
        
        # Parse numeric energy values from folder names and sort numerically
        energies_list = sorted([float(name.replace("_", ".")) for name in energy_names])
        logger.info(f"Found energies: {energies_list}")
        return energies_list
        
    except Exception as e:
        logger.warning(f"Could not determine energies from dataset root: {e}")
        return [0.0]  # Default fallback


def setup_device(device_arg):
    """Setup compute device based on argument."""
    if device_arg == 'auto':
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_arg)
    
    logger.info(f"Using device: {device}")
    return device


def main():
    """Main training function."""
    try:
        # Parse command line arguments
        args = parse_arguments()
        
        logger.info("=== RadioTherapy Standalone Training ===")
        logger.info(f"Input folder: {args.input_folder}")
        logger.info(f"Batch size: {args.batch_size}")
        logger.info(f"Epochs: {args.epochs}")
        logger.info(f"Learning rate: {args.learning_rate}")
        logger.info(f"Patience: {args.patience}")
        logger.info(f"Seed: {args.seed}")
        
        # Validate input folder structure
        input_cube_path, output_cube_path, dataset_root = validate_input_folder(args.input_folder)
        
        # Determine cube size
        cube_size = determine_cube_size(input_cube_path)
        
        # Get ALL available energies for proper training (not just folder energy)
        energies = get_energy_list(dataset_root)
        logger.info(f"Training with ALL energies: {energies} (proper multi-energy training)")
        
        # Setup device
        device = setup_device(args.device)
        
        # Initialize parameter manager
        pm = ParameterManager(
            energies=energies,
            batch_size=args.batch_size,
            cube_size=cube_size,
            num_epochs=args.epochs,
            learning_rate=args.learning_rate,
            patience=args.patience
        )
        
        # Setup transforms
        transforms_chain = Compose([
            LoadImaged(keys=["input", "target"], reader=NumpyReader),
            EnsureChannelFirstd(keys=["input", "target"]),
            EnsureTyped(keys=["input", "target"]),
            Orientationd(keys=["input", "target"], axcodes="RAS"),
            Spacingd(keys=["input", "target"], pixdim=(2.4, 2.4, 2.4), mode=("bilinear", "nearest")),
            SpatialPadd(keys=["input", "target"], spatial_size=pm.cube_size, method="symmetric"),
            CenterSpatialCropd(keys=["input", "target"], roi_size=pm.cube_size),
            ScaleIntensityRangePercentilesd(
                keys="input", lower=0, upper=99.5, b_min=0, b_max=1
            ),
            ToTensord(keys=["input", "target"])
        ])
        logger.info("Resolutions: {}".format(pm.resolutions))
        # Initialize SystemManager
        system_manager = SystemManager(
            root_dir=dataset_root,
            transforms=transforms_chain,
            resolutions=pm.resolutions,
            energies=pm.energies,
            energy_min=pm.energy_min,
            energy_max=pm.energy_max,
            quad_energies=pm.quad_energies,
            quad_weights=pm.quad_weights,
            batch_size=pm.batch_size,
            device=device,
            num_epochs=pm.num_epochs,
            learning_rate=pm.learning_rate,
            patience=pm.patience,
            cube_size=pm.cube_size,
            seed=args.seed
        )
        
        # Start training
        logger.info("Starting training...")
        system_manager.run_training()
        
        logger.info("=== Training completed successfully! ===")
        
        # Look for generated plots
        plot_patterns = ["loss_curves_res*.png", "learning_curve.png", "adv_curves.png"]
        for pattern in plot_patterns:
            plots = glob.glob(pattern)
            if plots:
                logger.info(f"Generated plots: {plots}")
        
    except KeyboardInterrupt:
        logger.info("Training interrupted by user.")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Training failed: {e}")
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
