#!/usr/bin/env python3
"""
Emergency fallback: Extract dose normalization parameters from Ground Truth data
when training-based parameters are missing or incorrect.
"""

import numpy as np
import glob
import os
import logging
import torch

logger = logging.getLogger(__name__)

def extract_gt_normalization_params(gt_dose_path, energy, resolution=100):
    """
    Extract normalization parameters directly from Ground Truth dose files.
    
    Args:
        gt_dose_path (str): Path to ground truth dose directory
        energy (float): Energy in keV (e.g., 46.53)
        resolution (int): Spatial resolution
        
    Returns:
        dict: Dictionary with clip_min, clip_max, etc.
    """
    logger.info(f"Extracting GT normalization parameters for energy {energy} keV")
    
    # Find all ground truth files for this energy
    energy_str = str(energy).replace('.', '_')
    
    # Search patterns for ground truth files
    search_patterns = [
        f"{gt_dose_path}/**/outputcube/*_{int(resolution)}.npy",
        f"{gt_dose_path}/**/*{energy_str}*/*.npy",
        f"{gt_dose_path}/**/*{energy}*/*.npy",
        f"{gt_dose_path}/**/outputcube/*.npy"
    ]
    
    gt_files = []
    for pattern in search_patterns:
        found_files = glob.glob(pattern, recursive=True)
        gt_files.extend(found_files)
    
    gt_files = list(set(gt_files))  # Remove duplicates
    logger.info(f"Found {len(gt_files)} potential ground truth files")
    
    if not gt_files:
        logger.error(f"No ground truth files found for energy {energy} keV")
        return None
    
    # Process ground truth files
    all_doses = []
    valid_files = 0
    
    for file_path in gt_files:
        try:
            dose = np.load(file_path)
            
            # Handle different array shapes
            if dose.ndim == 5:  # [1, 1, D, H, W]
                dose = dose[0, 0]
            elif dose.ndim == 4:  # [1, D, H, W]
                dose = dose[0]
            
            # Only use voxels with actual dose (>0)
            dose_nonzero = dose[dose > 1e-6]
            
            if len(dose_nonzero) > 0:
                all_doses.append(dose_nonzero)
                valid_files += 1
                logger.debug(f"✓ File {os.path.basename(file_path)}: {len(dose_nonzero)} dose voxels, max={dose_nonzero.max():.3f}")
            else:
                logger.debug(f"⚠ File {os.path.basename(file_path)}: no dose voxels found")
                
        except Exception as e:
            logger.warning(f"Failed to load {file_path}: {e}")
            continue
    
    if not all_doses:
        logger.error(f"No valid dose data found in {len(gt_files)} files")
        return None
    
    # Combine all dose values
    all_doses = np.concatenate(all_doses)
    logger.info(f"✓ Combined dose data from {valid_files} files: {len(all_doses)} dose voxels")
    
    # Calculate normalization parameters
    clip_min = 0.0  # Dose is always >= 0
    clip_max = np.percentile(all_doses, 99.5)  # Use 99.5th percentile as max
    
    # Calculate additional statistics
    dose_mean = np.mean(all_doses)
    dose_std = np.std(all_doses)
    dose_median = np.median(all_doses)
    
    params = {
        'clip_min': float(clip_min),
        'clip_max': float(clip_max),
        'dose_mean': float(dose_mean),
        'dose_std': float(dose_std),
        'dose_median': float(dose_median),
        'dose_count': int(len(all_doses)),
        'energy': energy,
        'resolution': resolution,
        'source': 'ground_truth',
        'valid_files': valid_files,
        'total_files': len(gt_files)
    }
    
    logger.info(f"✓ GT normalization parameters for energy {energy} keV:")
    logger.info(f"  clip_min: {clip_min:.6f}")
    logger.info(f"  clip_max: {clip_max:.6f}")
    logger.info(f"  dose_mean: {dose_mean:.6f}")
    logger.info(f"  dose_std: {dose_std:.6f}")
    logger.info(f"  dose_count: {len(all_doses)}")
    
    return params

def create_gt_dose_normalization_params(traindata_path, energies):
    """
    Create complete dose normalization parameters from ground truth for all energies.
    
    Args:
        traindata_path (str): Path to training data directory
        energies (list): List of energy values in keV
        
    Returns:
        dict: Complete dose normalization parameters
    """
    dose_normalization_params = {}
    
    for energy in energies:
        # Try different energy folder naming conventions
        energy_folders = [
            f"{traindata_path}/{energy}",
            f"{traindata_path}/{str(energy).replace('.', '_')}",
            f"{traindata_path}/{int(energy)}_{int((energy % 1) * 100):02d}",
        ]
        
        params = None
        for energy_folder in energy_folders:
            if os.path.exists(energy_folder):
                logger.info(f"Using energy folder: {energy_folder}")
                params = extract_gt_normalization_params(energy_folder, energy)
                if params:
                    break
        
        if params:
            # Create entries for all resolutions
            for resolution in [25, 50, 100]:
                key = f'res{resolution}_e{energy:.2f}'
                dose_normalization_params[key] = params.copy()
                dose_normalization_params[key]['resolution'] = resolution
        else:
            logger.error(f"Failed to extract GT parameters for energy {energy} keV")
    
    return dose_normalization_params

def apply_gt_normalization_fix(checkpoint_path, traindata_path, output_path=None):
    """
    Fix checkpoint by adding ground truth based dose normalization parameters.
    
    Args:
        checkpoint_path (str): Path to existing checkpoint
        traindata_path (str): Path to training data directory
        output_path (str): Output path for fixed checkpoint (optional)
        
    Returns:
        str: Path to fixed checkpoint
    """
    logger.info("=== APPLYING GROUND TRUTH NORMALIZATION FIX ===")
    
    # Load existing checkpoint
    logger.info(f"Loading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    
    # Extract available energies from checkpoint
    models_by_energy = checkpoint.get('models_by_energy', {})
    energies = []
    
    for key in models_by_energy.keys():
        if '_e' in key:
            energy_str = key.split('_e')[1]
            try:
                energy_val = float(energy_str)
                if energy_val not in energies:
                    energies.append(energy_val)
            except ValueError:
                continue
    
    logger.info(f"Found energies in checkpoint: {energies}")
    
    # Extract GT-based normalization parameters
    gt_dose_params = create_gt_dose_normalization_params(traindata_path, energies)
    
    if gt_dose_params:
        # Update checkpoint with GT parameters
        checkpoint['dose_normalization_params'] = gt_dose_params
        
        # Save fixed checkpoint
        if output_path is None:
            base_name = os.path.splitext(checkpoint_path)[0]
            output_path = f"{base_name}_GT_FIXED.ckpt"
        
        torch.save(checkpoint, output_path)
        logger.info(f"✓ Fixed checkpoint saved: {output_path}")
        logger.info(f"✓ Added GT-based dose normalization for {len(gt_dose_params)} energy/resolution combinations")
        
        return output_path
    else:
        logger.error("Failed to extract any GT normalization parameters")
        return None

if __name__ == "__main__":
    import argparse
    import log_config
    
    parser = argparse.ArgumentParser(description="Extract dose normalization from Ground Truth")
    parser.add_argument('--checkpoint', '-c', required=True, help='Path to model checkpoint')
    parser.add_argument('--traindata', '-t', required=True, help='Path to training data directory')
    parser.add_argument('--output', '-o', help='Output path for fixed checkpoint')
    
    args = parser.parse_args()
    
    # Apply the fix
    fixed_checkpoint = apply_gt_normalization_fix(
        args.checkpoint,
        args.traindata,
        args.output
    )
    
    if fixed_checkpoint:
        print(f"✓ Fixed checkpoint created: {fixed_checkpoint}")
    else:
        print("✗ Failed to create fixed checkpoint")
