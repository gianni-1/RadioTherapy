#!/usr/bin/env python3
"""
Gaussian Quadrature Inference für Radiotherapie Dose Prediction.
Implementiert korrekte 4-Punkt Gaussian Quadrature Integration über Energien.
"""

import argparse
import logging
from pathlib import Path
import numpy as np
import torch
import matplotlib.pyplot as plt
import os
import sys

# Add sourcecode to path
sys.path.append('sourcecode')

from inference_module import InferenceModule

# Optional nibabel support for NIfTI files
try:
    import nibabel as nib
    has_nibabel = True
except ImportError:
    has_nibabel = False


def setup_logging():
    """Setup logging configuration."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(name)s - %(message)s'
    )
    return logging.getLogger(__name__)


def load_combined_checkpoint(checkpoint_path: str):
    """Load combined checkpoint with all energy models."""
    logger = logging.getLogger(__name__)
    logger.info(f"Loading combined checkpoint from: {checkpoint_path}")
    
    try:
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        logger.info(f"✓ Checkpoint loaded successfully")
        
        if 'models_by_energy' in checkpoint:
            models_by_energy = checkpoint['models_by_energy']
            logger.info(f"Found {len(models_by_energy)} energy-specific models")
            for key in models_by_energy.keys():
                logger.info(f"  - {key}")
                
        if 'dose_normalization_params' in checkpoint:
            dose_params = checkpoint['dose_normalization_params']
            logger.info(f"✓ Found dose normalization parameters for {len(dose_params)} energy/resolution combinations")
            for key, params in dose_params.items():
                logger.info(f"  {key}: clip_min={params.get('clip_min', 0.0):.6f}, clip_max={params.get('clip_max', 1.0):.6f}")
        else:
            logger.warning("No dose normalization parameters found in checkpoint")
            dose_params = {}
            
        return checkpoint, models_by_energy, dose_params
        
    except Exception as e:
        logger.error(f"Failed to load checkpoint: {e}")
        raise


def load_ct(path: Path):
    """Load CT data from .npy or .nii/.nii.gz file."""
    logger = logging.getLogger(__name__)
    suffix = path.suffix.lower()
    
    if suffix in ['.nii', '.gz'] and has_nibabel:
        nii = nib.load(str(path))
        arr = nii.get_fdata().astype(np.float32)
        affine = nii.affine
        logger.info(f"Loaded NIfTI: shape={arr.shape}")
    else:
        arr = np.load(str(path)).astype(np.float32)
        affine = None
        logger.info(f"Loaded NPY: shape={arr.shape}")
        
    logger.info(f"CT range: {arr.min():.2f} to {arr.max():.2f}")
    return arr, affine


def extract_energy_models(models_by_energy: dict, target_energies: list):
    """Extract models for specific energies."""
    logger = logging.getLogger(__name__)
    extracted_models = {}
    available_energies = []
    
    for key, model_data in models_by_energy.items():
        try:
            # Parse energy from key (e.g., "res64x64x64_e46.53")
            if '_e' in key:
                energy_str = key.split('_e')[1]
                energy = float(energy_str)
                available_energies.append(energy)
                
                # Check if this energy is in our target list
                for target_energy in target_energies:
                    if abs(energy - target_energy) < 0.01:  # Close enough
                        extracted_models[target_energy] = {
                            'key': key,
                            'energy': energy,
                            'models': model_data
                        }
                        logger.info(f"✓ Found model for energy {target_energy:.4f} keV (key: {key})")
                        break
        except Exception as e:
            logger.warning(f"Could not parse energy from key {key}: {e}")
    
    logger.info(f"Available energies: {sorted(available_energies)}")
    logger.info(f"Extracted models for {len(extracted_models)} target energies")
    
    missing_energies = [e for e in target_energies if e not in extracted_models]
    if missing_energies:
        logger.error(f"Missing models for energies: {missing_energies}")
        raise ValueError(f"Models not found for energies: {missing_energies}")
    
    return extracted_models


def run_gaussian_quadrature_inference(ct_array, checkpoint_path, quad_energies, quad_weights, 
                                     device='cpu', cube_size=(64, 64, 64)):
    """
    Run 4-point Gaussian Quadrature Inference.
    
    Args:
        ct_array: Input CT array
        checkpoint_path: Path to combined checkpoint
        quad_energies: List of 4 quadrature energies [x1, x2, x3, x4]
        quad_weights: List of 4 quadrature weights [w1, w2, w3, w4] 
        device: PyTorch device
        cube_size: Target cube size
    
    Returns:
        integrated_dose: Final integrated dose distribution
    """
    logger = logging.getLogger(__name__)
    
    logger.info("=" * 60)
    logger.info("GAUSSIAN QUADRATURE INFERENCE")
    logger.info("=" * 60)
    logger.info(f"Quadrature energies: {quad_energies}")
    logger.info(f"Quadrature weights: {quad_weights}")
    logger.info(f"Device: {device}")
    logger.info(f"Target cube size: {cube_size}")
    
    # Load checkpoint
    checkpoint, models_by_energy, dose_params = load_combined_checkpoint(checkpoint_path)
    
    # Extract models for quadrature energies
    extracted_models = extract_energy_models(models_by_energy, quad_energies)
    
    # Initialize results storage
    energy_results = {}
    
    # Run inference for each quadrature energy
    for i, (energy, weight) in enumerate(zip(quad_energies, quad_weights)):
        logger.info("-" * 40)
        logger.info(f"QUADRATURE POINT {i+1}/4: Energy {energy:.4f} keV, Weight {weight:.6f}")
        logger.info("-" * 40)
        
        model_info = extracted_models[energy]
        model_key = model_info['key']
        
        # Create energy-specific models dict
        energy_models = {energy: extracted_models[energy]['models']}
        
        # Get scale factor
        scale_factor = extracted_models[energy]['models'].get('scale_factor', 1.0)
        logger.info(f"Using scale factor: {scale_factor}")
        
        # Create InferenceModule for this energy
        inference_module = InferenceModule(
            models_by_energy=energy_models,
            energies=[energy],
            energy_weights=[1.0],  # Single energy, so weight=1
            device=device,
            scale_factor=scale_factor,
            dose_normalization_params=dose_params
        )
        
        # Run inference
        logger.info(f"Running inference for energy {energy:.4f} keV...")
        dose_result = inference_module.run_inference_conditioned_on_energy(
            ct_tensor=torch.from_numpy(ct_array).unsqueeze(0),  # Add batch dimension
            energy_value=energy
        )
        
        # Store result
        if isinstance(dose_result, torch.Tensor):
            dose_np = dose_result.cpu().numpy()
        else:
            dose_np = np.array(dose_result, dtype=np.float32)
            
        # Remove batch dimension if present
        if dose_np.ndim == 5 and dose_np.shape[0] == 1:
            dose_np = dose_np[0, 0]  # Remove batch and channel dims
        elif dose_np.ndim == 4 and dose_np.shape[0] == 1:
            dose_np = dose_np[0]  # Remove batch dim
            
        energy_results[energy] = {
            'dose': dose_np,
            'weight': weight
        }
        
        logger.info(f"✓ Energy {energy:.4f}: dose shape={dose_np.shape}, "
                   f"range=[{dose_np.min():.6f}, {dose_np.max():.6f}], weight={weight:.6f}")
    
    # Gaussian Quadrature Integration
    logger.info("=" * 40)
    logger.info("GAUSSIAN QUADRATURE INTEGRATION")
    logger.info("=" * 40)
    
    integrated_dose = None
    total_weight = 0.0
    
    for energy, weight in zip(quad_energies, quad_weights):
        dose = energy_results[energy]['dose']
        
        if integrated_dose is None:
            integrated_dose = weight * dose
        else:
            integrated_dose += weight * dose
            
        total_weight += weight
        logger.info(f"Added energy {energy:.4f} with weight {weight:.6f}")
    
    logger.info(f"Total weight sum: {total_weight:.6f}")
    logger.info(f"Final integrated dose: shape={integrated_dose.shape}, "
               f"range=[{integrated_dose.min():.6f}, {integrated_dose.max():.6f}]")
    
    # Normalize by total weight (should be close to 1 for proper quadrature)
    if abs(total_weight - 1.0) > 0.001:
        logger.warning(f"Total weight {total_weight:.6f} != 1.0, normalizing...")
        integrated_dose /= total_weight
    
    return integrated_dose, energy_results


def save_dose(path: Path, dose: np.ndarray, affine=None):
    """Save dose array to file."""
    logger = logging.getLogger(__name__)
    suffix = path.suffix.lower()
    
    if suffix in ['.nii', '.gz'] and affine is not None and has_nibabel:
        out_nii = nib.Nifti1Image(dose, affine)
        out_nii.to_filename(str(path))
        logger.info(f"✓ Saved NIfTI: {path}")
    else:
        np.save(str(path), dose)
        logger.info(f"✓ Saved NPY: {path}")


def create_visualization(integrated_dose, energy_results, output_dir):
    """Create visualization of quadrature integration results."""
    logger = logging.getLogger(__name__)
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Gaussian Quadrature Dose Integration', fontsize=16)
    
    # Individual energy results
    for i, (energy, data) in enumerate(energy_results.items()):
        if i >= 4:  # Only show first 4
            break
        row = i // 2
        col = i % 2
        
        dose = data['dose']
        weight = data['weight']
        
        # Show middle slice
        mid_slice = dose.shape[2] // 2
        slice_data = dose[:, :, mid_slice]
        
        im = axes[row, col].imshow(slice_data, cmap='hot', origin='lower')
        axes[row, col].set_title(f'Energy {energy:.2f} keV\nWeight: {weight:.4f}')
        axes[row, col].axis('off')
        plt.colorbar(im, ax=axes[row, col], shrink=0.8)
    
    # Integrated result
    mid_slice = integrated_dose.shape[2] // 2
    integrated_slice = integrated_dose[:, :, mid_slice]
    
    im = axes[0, 2].imshow(integrated_slice, cmap='hot', origin='lower')
    axes[0, 2].set_title('Integrated Dose\n(Gaussian Quadrature)')
    axes[0, 2].axis('off')
    plt.colorbar(im, ax=axes[0, 2], shrink=0.8)
    
    # Statistics
    axes[1, 2].axis('off')
    stats_text = f"""Quadrature Integration Stats:
    
    Final dose range: [{integrated_dose.min():.3f}, {integrated_dose.max():.3f}]
    Final dose mean: {integrated_dose.mean():.3f}
    Final dose std: {integrated_dose.std():.3f}
    
    Individual contributions:"""
    
    for energy, data in energy_results.items():
        dose = data['dose']
        weight = data['weight']
        contribution = (weight * dose).max()
        stats_text += f"\n    E={energy:.2f}: max_contrib={contribution:.3f}"
    
    axes[1, 2].text(0.1, 0.9, stats_text, transform=axes[1, 2].transAxes, 
                   fontsize=10, verticalalignment='top', fontfamily='monospace')
    
    plt.tight_layout()
    viz_path = os.path.join(output_dir, 'gaussian_quadrature_results.png')
    plt.savefig(viz_path, dpi=300, bbox_inches='tight')
    plt.close()
    logger.info(f"✓ Visualization saved: {viz_path}")


def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description="Gaussian Quadrature Inference for Radiotherapy Dose Prediction"
    )
    parser.add_argument('-m', '--model', required=True, 
                       help='Path to combined model checkpoint')
    parser.add_argument('-i', '--input', required=True, 
                       help='Input CT file (.npy or .nii/.nii.gz)')
    parser.add_argument('-o', '--output', required=True, 
                       help='Output dose file (.npy or .nii/.nii.gz)')
    parser.add_argument('--energies', nargs=4, type=float, 
                       default=[3.4716, 15.7505, 34.2495, 46.5284],
                       help='4 Gaussian quadrature energies')
    parser.add_argument('--weights', nargs=4, type=float,
                       default=[8.6964, 16.3036, 16.3036, 8.6964],
                       help='4 Gaussian quadrature weights (will be normalized)')
    parser.add_argument('-d', '--device', default='cpu', 
                       help='PyTorch device (cpu or cuda)')
    parser.add_argument('--cube-size', nargs=3, type=int, default=[64, 64, 64],
                       help='Target cube size [D, H, W]')
    parser.add_argument('--visualize', action='store_true',
                       help='Create visualization of results')
    
    args = parser.parse_args()
    
    logger = setup_logging()
    
    # Normalize weights to sum to 1
    weights = np.array(args.weights)
    weights = weights / weights.sum()
    
    logger.info("=" * 60)
    logger.info("GAUSSIAN QUADRATURE INFERENCE PIPELINE")
    logger.info("=" * 60)
    logger.info(f"Model: {args.model}")
    logger.info(f"Input: {args.input}")
    logger.info(f"Output: {args.output}")
    logger.info(f"Energies: {args.energies}")
    logger.info(f"Weights (normalized): {weights.tolist()}")
    logger.info(f"Device: {args.device}")
    
    # Load CT
    input_path = Path(args.input)
    ct_array, affine = load_ct(input_path)
    
    # Run Gaussian Quadrature Inference
    integrated_dose, energy_results = run_gaussian_quadrature_inference(
        ct_array=ct_array,
        checkpoint_path=args.model,
        quad_energies=args.energies,
        quad_weights=weights.tolist(),
        device=args.device,
        cube_size=tuple(args.cube_size)
    )
    
    # Save result
    output_path = Path(args.output)
    save_dose(output_path, integrated_dose, affine)
    
    # Create visualization if requested
    if args.visualize:
        output_dir = output_path.parent
        create_visualization(integrated_dose, energy_results, output_dir)
    
    logger.info("=" * 60)
    logger.info("INFERENCE COMPLETED SUCCESSFULLY")
    logger.info("=" * 60)
    logger.info(f"Final integrated dose saved to: {output_path}")
    logger.info(f"Dose statistics:")
    logger.info(f"  Shape: {integrated_dose.shape}")
    logger.info(f"  Range: [{integrated_dose.min():.6f}, {integrated_dose.max():.6f}]")
    logger.info(f"  Mean: {integrated_dose.mean():.6f}")
    logger.info(f"  Std: {integrated_dose.std():.6f}")


if __name__ == '__main__':
    main()
