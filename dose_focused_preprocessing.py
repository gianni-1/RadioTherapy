#!/usr/bin/env python3
"""
Dose-focused data preprocessing to fix ultra-sparse training data problem.
Crops training samples to dose regions and filters out unusable samples.
"""

import numpy as np
import torch
import os
import glob
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

def find_dose_bounding_box(dose_array, threshold=1e-6, margin=10):
    """
    Find minimal bounding box around dose regions with margin.
    
    Args:
        dose_array: 3D numpy array with dose values
        threshold: Minimum dose value to consider as "dose region"
        margin: Voxels to add around dose region
        
    Returns:
        tuple: (min_coords, max_coords) or None if no dose found
    """
    dose_mask = dose_array > threshold
    
    if dose_mask.sum() == 0:
        return None
    
    # Find bounding box coordinates
    coords = np.where(dose_mask)
    
    min_coords = []
    max_coords = []
    
    for i, coord_list in enumerate(coords):
        axis_min = max(0, coord_list.min() - margin)
        axis_max = min(dose_array.shape[i], coord_list.max() + margin + 1)
        min_coords.append(axis_min)
        max_coords.append(axis_max)
    
    return tuple(min_coords), tuple(max_coords)

def crop_to_dose_region(ct_array, dose_array, threshold=1e-6, margin=10):
    """
    Crop CT and dose arrays to focus on dose region.
    
    Args:
        ct_array: 3D CT data
        dose_array: 3D dose data
        threshold: Minimum dose value to consider
        margin: Voxels margin around dose region
        
    Returns:
        tuple: (ct_cropped, dose_cropped, crop_info) or (None, None, None)
    """
    bbox = find_dose_bounding_box(dose_array, threshold, margin)
    
    if bbox is None:
        logger.warning("No dose region found - skipping sample")
        return None, None, None
    
    min_coords, max_coords = bbox
    
    # Crop both arrays
    ct_cropped = ct_array[
        min_coords[0]:max_coords[0],
        min_coords[1]:max_coords[1], 
        min_coords[2]:max_coords[2]
    ]
    
    dose_cropped = dose_array[
        min_coords[0]:max_coords[0],
        min_coords[1]:max_coords[1],
        min_coords[2]:max_coords[2]
    ]
    
    crop_info = {
        'original_shape': dose_array.shape,
        'cropped_shape': dose_cropped.shape,
        'min_coords': min_coords,
        'max_coords': max_coords,
        'dose_voxel_count': (dose_cropped > threshold).sum(),
        'dose_max': dose_cropped.max(),
        'dose_mean': dose_cropped[dose_cropped > threshold].mean() if (dose_cropped > threshold).sum() > 0 else 0.0
    }
    
    return ct_cropped, dose_cropped, crop_info

def validate_sample_quality(dose_array, min_dose_voxels=50, min_max_dose=0.01):
    """
    Check if a sample is suitable for training.
    
    Args:
        dose_array: 3D dose array
        min_dose_voxels: Minimum number of dose voxels required
        min_max_dose: Minimum maximum dose value required
        
    Returns:
        dict: Validation results
    """
    dose_mask = dose_array > 1e-6
    dose_voxel_count = dose_mask.sum()
    max_dose = dose_array.max()
    
    is_valid = (dose_voxel_count >= min_dose_voxels) and (max_dose >= min_max_dose)
    
    validation_info = {
        'is_valid': is_valid,
        'dose_voxel_count': int(dose_voxel_count),
        'max_dose': float(max_dose),
        'reason': []
    }
    
    if dose_voxel_count < min_dose_voxels:
        validation_info['reason'].append(f'Too few dose voxels: {dose_voxel_count} < {min_dose_voxels}')
    
    if max_dose < min_max_dose:
        validation_info['reason'].append(f'Max dose too low: {max_dose:.6f} < {min_max_dose}')
    
    return validation_info

def process_training_sample(ct_path, dose_path, output_dir, sample_id):
    """
    Process a single training sample: validate, crop, and save.
    
    Args:
        ct_path: Path to CT file
        dose_path: Path to dose file  
        output_dir: Output directory
        sample_id: Unique sample identifier
        
    Returns:
        dict: Processing results
    """
    try:
        # Load data
        ct_array = np.load(ct_path)
        dose_array = np.load(dose_path)
        
        # Validate sample quality
        validation = validate_sample_quality(dose_array)
        
        if not validation['is_valid']:
            logger.info(f"Sample {sample_id} REJECTED: {', '.join(validation['reason'])}")
            return {
                'sample_id': sample_id,
                'status': 'rejected',
                'validation': validation
            }
        
        # Crop to dose region
        ct_cropped, dose_cropped, crop_info = crop_to_dose_region(ct_array, dose_array)
        
        if ct_cropped is None:
            logger.warning(f"Sample {sample_id} FAILED: Could not crop to dose region")
            return {
                'sample_id': sample_id,
                'status': 'failed',
                'validation': validation
            }
        
        # Save cropped data
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        ct_output_path = output_dir / f"{sample_id}_ct_cropped.npy"
        dose_output_path = output_dir / f"{sample_id}_dose_cropped.npy"
        
        np.save(ct_output_path, ct_cropped.astype(np.float32))
        np.save(dose_output_path, dose_cropped.astype(np.float32))
        
        logger.info(f"Sample {sample_id} PROCESSED: {crop_info['original_shape']} → {crop_info['cropped_shape']}, {crop_info['dose_voxel_count']} dose voxels")
        
        return {
            'sample_id': sample_id,
            'status': 'success',
            'validation': validation,
            'crop_info': crop_info,
            'output_paths': {
                'ct': str(ct_output_path),
                'dose': str(dose_output_path)
            }
        }
        
    except Exception as e:
        logger.error(f"Sample {sample_id} ERROR: {e}")
        return {
            'sample_id': sample_id,
            'status': 'error',
            'error': str(e)
        }

def process_energy_dataset(energy_dir, output_dir):
    """
    Process all samples in an energy directory.
    
    Args:
        energy_dir: Path to energy directory (e.g., traindata/46_53)
        output_dir: Output directory for processed data
        
    Returns:
        dict: Processing summary
    """
    energy_dir = Path(energy_dir)
    
    if not energy_dir.exists():
        raise ValueError(f"Energy directory does not exist: {energy_dir}")
    
    input_dir = energy_dir / "inputcube"
    output_dose_dir = energy_dir / "outputcube"
    
    if not input_dir.exists() or not output_dose_dir.exists():
        raise ValueError(f"inputcube or outputcube directory missing in {energy_dir}")
    
    # Find all CT files
    ct_files = list(input_dir.glob("*.npy"))
    
    if not ct_files:
        logger.warning(f"No CT files found in {input_dir}")
        return {'processed': 0, 'rejected': 0, 'failed': 0, 'errors': 0}
    
    results = {
        'processed': 0,
        'rejected': 0, 
        'failed': 0,
        'errors': 0,
        'samples': []
    }
    
    for ct_file in ct_files:
        # Find corresponding dose file
        sample_name = ct_file.stem
        dose_file = output_dose_dir / f"{sample_name}.npy"
        
        if not dose_file.exists():
            logger.warning(f"No corresponding dose file for {ct_file}")
            continue
        
        # Process sample
        result = process_training_sample(
            ct_path=ct_file,
            dose_path=dose_file,
            output_dir=output_dir,
            sample_id=sample_name
        )
        
        results['samples'].append(result)
        results[result['status']] += 1
    
    logger.info(f"Energy processing complete: {results['processed']} processed, {results['rejected']} rejected, {results['failed']} failed, {results['errors']} errors")
    
    return results

if __name__ == "__main__":
    import argparse
    import log_config
    
    parser = argparse.ArgumentParser(description="Process training data with dose-focused cropping")
    parser.add_argument('--traindata', required=True, help='Path to training data directory')
    parser.add_argument('--output', required=True, help='Output directory for processed data')
    parser.add_argument('--energy', help='Specific energy to process (e.g., 46_53)')
    
    args = parser.parse_args()
    
    traindata_dir = Path(args.traindata)
    output_dir = Path(args.output)
    
    if args.energy:
        # Process single energy
        energy_dir = traindata_dir / args.energy
        energy_output_dir = output_dir / args.energy
        
        logger.info(f"Processing energy {args.energy}")
        results = process_energy_dataset(energy_dir, energy_output_dir)
        
        print(f"✓ Energy {args.energy}: {results['processed']} samples processed")
        
    else:
        # Process all energies
        energy_dirs = [d for d in traindata_dir.iterdir() if d.is_dir()]
        
        total_results = {'processed': 0, 'rejected': 0, 'failed': 0, 'errors': 0}
        
        for energy_dir in energy_dirs:
            logger.info(f"Processing energy {energy_dir.name}")
            
            energy_output_dir = output_dir / energy_dir.name
            results = process_energy_dataset(energy_dir, energy_output_dir)
            
            for key in total_results:
                total_results[key] += results[key]
            
            print(f"✓ Energy {energy_dir.name}: {results['processed']} samples processed")
        
        print(f"\n=== TOTAL RESULTS ===")
        print(f"Processed: {total_results['processed']}")
        print(f"Rejected: {total_results['rejected']}")
        print(f"Failed: {total_results['failed']}")
        print(f"Errors: {total_results['errors']}")
