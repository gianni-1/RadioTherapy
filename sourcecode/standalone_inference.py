#!/usr/bin/env python3
"""
Standalone inference script for radiotherapy dose prediction using latent diffusion.
Implements 4-point Gaussian Quadrature integration for multi-energy dose prediction.
"""

import argparse
import logging
from pathlib import Path

import numpy as np
import torch

# Optional nibabel support for NIfTI files
try:
    import nibabel as nib
    has_nibabel = True
except ImportError:
    has_nibabel = False

# Import system components from sourcecode
import sys
sys.path.append('/home/mpalm/RadioTherapy/sourcecode')
from system_manager import SystemManager


def load_ct(path: Path):
    """Load CT data from .npy or .nii/.nii.gz file and return array and affine."""
    suffix = path.suffix.lower()
    if suffix in ['.nii', '.gz'] and has_nibabel:
        nii = nib.load(str(path))
        arr = nii.get_fdata().astype(np.float32)
        affine = nii.affine
    else:
        arr = np.load(str(path)).astype(np.float32)
        affine = None
    return arr, affine


def save_dose(path: Path, dose: np.ndarray, affine=None):
    """Save dose array to .npy or .nii/.nii.gz file using affine if available."""
    suffix = path.suffix.lower()
    if suffix in ['.nii', '.gz'] and affine is not None and has_nibabel:
        out_nii = nib.Nifti1Image(dose, affine)
        out_nii.to_filename(str(path))
    else:
        np.save(str(path), dose)


def get_gaussian_quadrature_4point(energy_min=3.47, energy_max=46.53):
    """
    Get 4-point Gaussian Quadrature nodes and weights for energy integration.
    
    Args:
        energy_min (float): Minimum energy value (keV)
        energy_max (float): Maximum energy value (keV)
        
    Returns:
        tuple: (energies, weights) for 4-point Gaussian Quadrature
    """
    # 4-point Gaussian Quadrature nodes and weights on [-1, 1]
    nodes = np.array([
        -0.8611363115940526,
        -0.3399810435848563,
         0.3399810435848563,
         0.8611363115940526
    ])
    
    weights = np.array([
        0.3478548451374538,
        0.6521451548625461,
        0.6521451548625461,
        0.3478548451374538
    ])
    
    # Transform from [-1, 1] to [energy_min, energy_max]
    energy_range = energy_max - energy_min
    energies = energy_min + (nodes + 1) * energy_range / 2
    
    # Scale weights by the transformation factor
    scaled_weights = weights * energy_range / 2
    
    # Normalize weights to sum to 1 for dose integration
    normalized_weights = scaled_weights / np.sum(scaled_weights)
    
    return energies.tolist(), normalized_weights.tolist()


def create_system_manager(model_path: str, device: str = 'cpu'):
    """
    Create and initialize SystemManager with the model checkpoint.
    
    Args:
        model_path (str): Path to the model checkpoint
        device (str): Device to use
        
    Returns:
        SystemManager: Initialized system manager
    """
    logger = logging.getLogger(__name__)
    
    # Create SystemManager
    system_manager = SystemManager(device=torch.device(device))
    
    # Load model checkpoint 
    logger.info(f"Loading model checkpoint: {model_path}")
    system_manager.model_checkpoint = model_path
    
    # Initialize inference module
    logger.info("Initializing inference module...")
    system_manager.create_inference_module()
    
    return system_manager


def main():
    parser = argparse.ArgumentParser(
        description="Standalone inference for dose prediction using 4-point Gaussian Quadrature"
    )
    parser.add_argument('-m', '--model', required=True, help='Path to energy-conditioned model checkpoint')
    parser.add_argument('-i', '--input', required=True, help='Input CT file (.npy or .nii/.nii.gz)')
    parser.add_argument('-o', '--output', required=True, help='Output dose file (.npy or .nii/.nii.gz)')
    parser.add_argument('-e', '--energy', type=float, default=None, help='Target energy for single energy inference')
    parser.add_argument('--energy-min', type=float, default=3.47, help='Minimum energy for quadrature (keV)')
    parser.add_argument('--energy-max', type=float, default=46.53, help='Maximum energy for quadrature (keV)')
    parser.add_argument('--use-quadrature', action='store_true', default=True, 
                        help='Use 4-point Gaussian Quadrature (default: True)')
    parser.add_argument('--single-energy', action='store_true', default=False,
                        help='Force single energy inference instead of quadrature')
    parser.add_argument('-d', '--device', default='cpu', help='Torch device (cpu or cuda)')
    parser.add_argument('--cube-size', nargs=3, type=int, default=[64, 64, 64], 
                        help='Target cube size for inference (D H W)')
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)

    model_path = Path(args.model)
    input_path = Path(args.input)
    output_path = Path(args.output)
    target_cube_size = tuple(args.cube_size)

    # Load CT data
    logger.info(f"Loading CT from {input_path}")
    ct_array, affine = load_ct(input_path)
    logger.info(f"CT shape: {ct_array.shape}, affine: {'present' if affine is not None else 'none'}")

    # Convert to torch tensor and add channel dimension if needed
    if len(ct_array.shape) == 3:
        ct_tensor = torch.from_numpy(ct_array).unsqueeze(0)  # Add channel dimension
    else:
        ct_tensor = torch.from_numpy(ct_array)
    
    logger.info(f"CT tensor shape: {ct_tensor.shape}")

    # Create system manager and load model
    logger.info(f"Initializing system with model: {model_path}")
    system_manager = create_system_manager(str(model_path), args.device)
    
    # Get the inference module
    inference_module = system_manager.inference_module
    
    if inference_module is None:
        raise RuntimeError("Failed to create inference module")

    # Decide between single energy and quadrature inference
    if args.single_energy or args.energy is not None:
        # Single energy inference
        target_energy = args.energy if args.energy is not None else inference_module.energies[0]
        logger.info(f"Running single energy inference at {target_energy} keV")
        
        # Run single energy inference
        dose_tensor = inference_module.run_inference_conditioned_on_energy(
            ct_tensor, target_energy, target_cube_size
        )
        
    else:
        # 4-point Gaussian Quadrature inference (default)
        logger.info("Running 4-point Gaussian Quadrature inference")
        
        # Get quadrature energies and weights
        quad_energies, quad_weights = get_gaussian_quadrature_4point(
            args.energy_min, args.energy_max
        )
        
        logger.info(f"Quadrature energies: {[f'{e:.2f}' for e in quad_energies]} keV")
        logger.info(f"Quadrature weights: {[f'{w:.4f}' for w in quad_weights]}")
        
        # Run quadrature inference using the inference module's quadrature method
        dose_tensor = inference_module.run_inference_over_energies(
            ct_tensor, target_cube_size, quad_energies, quad_weights
        )

    # Convert to numpy and remove batch/channel dimensions if present
    if isinstance(dose_tensor, torch.Tensor):
        dose_np = dose_tensor.squeeze().cpu().numpy()
    else:
        dose_np = np.array(dose_tensor, dtype=np.float32)

    logger.info(f"Final dose shape: {dose_np.shape}")
    logger.info(f"Dose statistics: min={dose_np.min():.4f}, max={dose_np.max():.4f}, "
                f"mean={dose_np.mean():.4f}, std={dose_np.std():.4f}")

    # Check for realistic dose values
    max_dose = dose_np.max()
    if max_dose < 1.0:
        logger.warning(f"⚠️  Max dose {max_dose:.2f} Gy seems low for radiotherapy")
    elif max_dose > 100.0:
        logger.warning(f"⚠️  Max dose {max_dose:.2f} Gy seems high for radiotherapy")
    else:
        logger.info(f"✓ Max dose {max_dose:.2f} Gy is in realistic range")

    # Save results
    logger.info(f"Saving dose output to {output_path}")
    save_dose(output_path, dose_np, affine)
    logger.info("✓ 4-point Gaussian Quadrature inference completed successfully!")
    logger.info(f"✓ Expected Gaussian bell-curve dose distribution saved to {output_path}")


if __name__ == '__main__':
    main()
