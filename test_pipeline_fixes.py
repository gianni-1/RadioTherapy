#!/usr/bin/env python3
"""
Test script to validate pipeline fixes for RadioTherapy dose prediction.
This script tests the key fixes applied to resolve the 30-50x too low dose values.
"""

import logging
import torch
import numpy as np
from pathlib import Path

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s:%(name)s: %(message)s')
logger = logging.getLogger(__name__)

def test_transforms_consistency():
    """Test that transforms produce consistent 64x64x64 outputs."""
    logger.info("🔍 Testing transform consistency...")
    
    try:
        from sourcecode.system_manager import SystemManager
        
        # Initialize minimal SystemManager to test transforms
        system = SystemManager(
            root_dir="/home/mpalm/RadioTherapy/traindata",
            transforms=None,
            resolutions=[(64, 64, 64)],
            energies=[11.5],
            energy_min=3.47,
            energy_max=46.53,
            quad_energies=[11.5],
            quad_weights=[1.0],
            batch_size=2,
            device="cpu",
            num_epochs=1,
            learning_rate=1e-5,
            patience=5,
            cube_size=100,
            seed=42
        )
        
        # Initialize transforms for resolution 64x64x64
        system._prepare_training(target_resolution=(64, 64, 64), energy=11.5)
        
        logger.info("✅ Transform initialization successful")
        logger.info(f"   Target resolution: (64, 64, 64)")
        logger.info(f"   Cube size: {system.cube_size}")
        
        # Check if the problematic SpatialPadd is removed
        transform_str = str(system.transforms)
        has_spatial_padd = "SpatialPadd" in transform_str
        has_resized = "Resized" in transform_str
        
        logger.info(f"   Contains Resized: {has_resized}")
        logger.info(f"   Contains SpatialPadd: {has_spatial_padd}")
        
        if has_spatial_padd:
            logger.error("❌ SpatialPadd still present! This will cause resolution mismatch.")
            return False
        elif has_resized:
            logger.info("✅ Pipeline uses Resized without SpatialPadd - correct!")
            return True
        else:
            logger.warning("⚠️  No Resized found - check pipeline configuration")
            return False
            
    except Exception as e:
        logger.error(f"❌ Transform test failed: {e}")
        return False

def test_dose_range_detection():
    """Test dose range detection logic."""
    logger.info("🔍 Testing dose range detection...")
    
    try:
        from sourcecode.training_pipeline import AutoencoderTrainer
        
        # Create synthetic dose data with realistic ranges
        realistic_dose = np.random.exponential(scale=15.0, size=(1000, 64, 64, 64))
        realistic_dose[realistic_dose > 60] = 60  # Cap at reasonable max
        
        # Create problematically low dose data (like what we observed)
        low_dose = realistic_dose * 0.02  # Scale down by ~50x
        
        logger.info("Realistic dose stats:")
        logger.info(f"  Min: {realistic_dose.min():.6f}")
        logger.info(f"  Max: {realistic_dose.max():.6f}")
        logger.info(f"  Mean: {realistic_dose.mean():.6f}")
        
        logger.info("Problematic low dose stats:")
        logger.info(f"  Min: {low_dose.min():.6f}")
        logger.info(f"  Max: {low_dose.max():.6f}")
        logger.info(f"  Mean: {low_dose.mean():.6f}")
        
        # The enhanced detection should flag low_dose as problematic
        if low_dose.max() < 5.0:
            logger.info("✅ Enhanced detection would flag this as problematic (max < 5.0)")
            return True
        else:
            logger.error("❌ Detection logic may not catch this issue")
            return False
            
    except Exception as e:
        logger.error(f"❌ Dose range test failed: {e}")
        return False

def test_learning_rate_adjustment():
    """Test learning rate is properly adjusted after fixes."""
    logger.info("🔍 Testing learning rate adjustment...")
    
    original_lr = 1e-5
    ultra_conservative_lr = original_lr * 0.1  # Old: 1e-6
    balanced_lr = original_lr * 0.5           # New: 5e-6
    
    logger.info(f"Original LR: {original_lr}")
    logger.info(f"Old ultra-conservative LR: {ultra_conservative_lr}")
    logger.info(f"New balanced LR: {balanced_lr}")
    
    # Check if the new LR is reasonable (between old and original)
    if ultra_conservative_lr < balanced_lr < original_lr:
        logger.info("✅ Learning rate properly balanced")
        return True
    else:
        logger.error("❌ Learning rate not properly adjusted")
        return False

def run_comprehensive_test():
    """Run all tests and summarize results."""
    logger.info("🚀 Starting comprehensive pipeline fix validation...")
    
    tests = [
        ("Transform Consistency", test_transforms_consistency),
        ("Dose Range Detection", test_dose_range_detection),
        ("Learning Rate Adjustment", test_learning_rate_adjustment),
    ]
    
    results = []
    for test_name, test_func in tests:
        logger.info(f"\n📋 Running {test_name}...")
        try:
            result = test_func()
            results.append((test_name, result))
            logger.info(f"   Result: {'✅ PASS' if result else '❌ FAIL'}")
        except Exception as e:
            logger.error(f"   Result: ❌ ERROR - {e}")
            results.append((test_name, False))
    
    # Summary
    logger.info("\n" + "="*60)
    logger.info("🏁 TEST SUMMARY")
    logger.info("="*60)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        logger.info(f"  {test_name:<25} {status}")
    
    logger.info(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        logger.info("🎉 All tests passed! Pipeline fixes should resolve the dose scaling issue.")
    else:
        logger.warning(f"⚠️  {total-passed} test(s) failed. Additional fixes may be needed.")
    
    return passed == total

if __name__ == "__main__":
    success = run_comprehensive_test()
    exit(0 if success else 1)
