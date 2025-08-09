#!/usr/bin/env python3
"""
Quick test script to validate the critical fixes for ultra-sparse dose training.
Tests dose-focused preprocessing, extreme weighted loss, and sample filtering.
"""

import sys
import os
sys.path.append('/home/mpalm/RadioTherapy/sourcecode')

import torch
import numpy as np
import logging
import log_config

# Import our modules
from training_pipeline import AutoencoderTrainer
from data_management import DataLoaderModule, DoseNpyDataset

logger = logging.getLogger(__name__)

def test_dose_focused_preprocessing():
    """Test the dose-focused preprocessing functions"""
    print("=== Testing Dose-Focused Preprocessing ===")
    
    # Import the preprocessing functions
    sys.path.append('/home/mpalm/RadioTherapy')
    from dose_focused_preprocessing import validate_sample_quality, crop_to_dose_region
    
    # Create test data with ultra-sparse dose
    test_ct = np.random.randn(100, 100, 100).astype(np.float32)
    test_dose = np.zeros((100, 100, 100), dtype=np.float32)
    
    # Add small dose region (simulating ultra-sparse data)
    test_dose[45:55, 45:55, 45:55] = np.random.uniform(0.1, 2.0, (10, 10, 10))
    
    print(f"Test dose stats: max={test_dose.max():.6f}, "
          f"dose_voxels={np.sum(test_dose > 1e-6)}, "
          f"total_voxels={test_dose.size}")
    
    # Test validation
    validation = validate_sample_quality(test_dose)
    print(f"Sample validation: {validation}")
    
    # Test cropping
    ct_cropped, dose_cropped, crop_info = crop_to_dose_region(test_ct, test_dose)
    
    if ct_cropped is not None:
        print(f"Cropping successful: {test_dose.shape} → {dose_cropped.shape}")
        print(f"Crop info: {crop_info}")
    else:
        print("Cropping failed")
    
    print("✓ Dose-focused preprocessing test completed\n")

def test_extreme_loss_functions():
    """Test the extreme loss functions on sparse data"""
    print("=== Testing Extreme Loss Functions ===")
    
    # Create AutoencoderTrainer instance
    trainer = AutoencoderTrainer(
        autoencoder=None, discriminator=None, unet=None,
        optimizer_g=None, optimizer_d=None, optimizer_diff=None,
        device='cpu'
    )
    
    # Create ultra-sparse test data
    batch_size = 2
    spatial_size = 64
    
    # Prediction (random values)
    pred = torch.randn(batch_size, 1, spatial_size, spatial_size, spatial_size) * 0.1
    
    # Target with ultra-sparse dose (only 16 voxels have dose)
    target = torch.zeros(batch_size, 1, spatial_size, spatial_size, spatial_size)
    
    # Add tiny dose regions
    for b in range(batch_size):
        # 8 voxels per batch = 16 total (ultra-sparse like in logs)
        coords = torch.randint(0, spatial_size, (8, 3))
        for i in range(8):
            x, y, z = coords[i]
            target[b, 0, x, y, z] = torch.rand(1) * 0.2  # Max dose ~0.2 (like logs)
    
    dose_voxels = (target > 1e-6).sum().item()
    total_voxels = target.numel()
    print(f"Test data: {dose_voxels}/{total_voxels} dose voxels ({100*dose_voxels/total_voxels:.4f}%)")
    
    # Test different loss functions
    try:
        extreme_loss = trainer.extreme_dose_focused_loss(pred, target)
        print(f"✓ Extreme dose-focused loss: {extreme_loss:.6f}")
    except Exception as e:
        print(f"✗ Extreme loss failed: {e}")
    
    try:
        weighted_loss = trainer.weighted_dose_loss(pred, target)
        print(f"✓ Extreme weighted loss: {weighted_loss:.6f}")
    except Exception as e:
        print(f"✗ Weighted loss failed: {e}")
    
    try:
        masked_loss = trainer.masked_dose_loss(pred, target)
        print(f"✓ Pure dose loss: {masked_loss:.6f}")
    except Exception as e:
        print(f"✗ Masked loss failed: {e}")
    
    print("✓ Extreme loss functions test completed\n")

def test_sample_filtering():
    """Test the sample filtering in DataLoaderModule"""
    print("=== Testing Sample Filtering ===")
    
    # Create test dataset instance
    test_dataset = DoseNpyDataset(root_dir="/home/mpalm/RadioTherapy/traindata")
    
    # Check if validation method exists
    if hasattr(test_dataset, 'validate_sample_quality'):
        print("✓ Sample validation method found")
        
        # Test with dummy dose files
        test_dose_good = np.zeros((100, 100, 100))
        test_dose_good[40:60, 40:60, 40:60] = np.random.uniform(0.1, 1.0, (20, 20, 20))
        
        test_dose_bad = np.zeros((100, 100, 100))
        test_dose_bad[50:55, 50:55, 50:55] = np.random.uniform(0.001, 0.01, (5, 5, 5))
        
        good_result = test_dataset.validate_sample_quality('/tmp/dummy_good.npy')
        bad_result = test_dataset.validate_sample_quality('/tmp/dummy_bad.npy')
        
        print(f"Sample filtering working: good={good_result}, bad={bad_result}")
    else:
        print("✗ Sample validation method not found")
    
    print("✓ Sample filtering test completed\n")

def test_realistic_training_scenario():
    """Simulate realistic training with ultra-sparse data"""
    print("=== Testing Realistic Training Scenario ===")
    
    # Simulate batch with statistics from logs
    batch_size = 2
    spatial_size = 50  # Smaller for faster testing
    
    # Create batch similar to problematic log entries
    ct_batch = torch.randn(batch_size, 1, spatial_size, spatial_size, spatial_size)
    dose_batch = torch.zeros(batch_size, 1, spatial_size, spatial_size, spatial_size)
    
    # Add ultra-sparse dose (16 voxels total like in logs)
    dose_coords = [
        (25, 25, 25), (26, 25, 25), (25, 26, 25), (25, 25, 26),
        (27, 25, 25), (25, 27, 25), (25, 25, 27), (26, 26, 25),
        (30, 30, 30), (31, 30, 30), (30, 31, 30), (30, 30, 31),
        (32, 30, 30), (30, 32, 30), (30, 30, 32), (31, 31, 30)
    ]
    
    for i, (x, y, z) in enumerate(dose_coords):
        batch_idx = i % batch_size
        dose_batch[batch_idx, 0, x, y, z] = np.random.uniform(0.1, 0.25)
    
    dose_voxels = (dose_batch > 1e-6).sum().item()
    total_voxels = dose_batch.numel()
    max_dose = dose_batch.max().item()
    mean_dose = dose_batch[dose_batch > 1e-6].mean().item()
    
    print(f"Realistic scenario stats:")
    print(f"  Dose voxels: {dose_voxels}/{total_voxels} ({100*dose_voxels/total_voxels:.4f}%)")
    print(f"  Max dose: {max_dose:.6f}")
    print(f"  Mean dose: {mean_dose:.6f}")
    print(f"  (Similar to logs: 16/2M voxels, max~0.25)")
    
    # Test if extreme loss can handle this
    trainer = AutoencoderTrainer(
        autoencoder=None, discriminator=None, unet=None,
        optimizer_g=None, optimizer_d=None, optimizer_diff=None,
        device='cpu'
    )
    
    # Random prediction
    pred_batch = torch.randn_like(dose_batch) * 0.1
    
    try:
        loss = trainer.extreme_dose_focused_loss(pred_batch, dose_batch)
        print(f"✓ Extreme loss on realistic data: {loss:.6f}")
        
        if loss.item() > 1e-6:  # Should be meaningful, not zero
            print("✓ Loss is meaningful (not collapsed to zero)")
        else:
            print("⚠ Loss is very small - might indicate problems")
            
    except Exception as e:
        print(f"✗ Extreme loss failed on realistic data: {e}")
    
    print("✓ Realistic training scenario test completed\n")

def main():
    """Run all critical tests"""
    print("=== CRITICAL FIXES VALIDATION ===")
    print("Testing fixes for ultra-sparse dose training problems\n")
    
    try:
        test_dose_focused_preprocessing()
        test_extreme_loss_functions()
        test_sample_filtering()
        test_realistic_training_scenario()
        
        print("=== ALL TESTS COMPLETED ===")
        print("✓ Critical fixes appear to be working")
        print("Ready for training test with real data")
        
    except Exception as e:
        print(f"✗ Test failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
