#!/usr/bin/env python3
"""
Test the corrected inference through SystemManager
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), 'sourcecode')))

# Initialize logging first
import log_config
import logging

# Set up logging to also output to console
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s %(levelname)s:%(name)s: %(message)s')
console_handler.setFormatter(formatter)

# Get root logger and add console handler
root_logger = logging.getLogger()
root_logger.addHandler(console_handler)
root_logger.setLevel(logging.INFO)

logger = logging.getLogger(__name__)

import torch
import numpy as np
from system_manager import SystemManager

def test_system_manager_inference():
    """Test that the SystemManager uses the corrected inference"""
    logger.info("=" * 60)
    logger.info("TESTING SYSTEM MANAGER WITH CORRECTED INFERENCE")
    logger.info("=" * 60)
    
    # Create a simple test CT file
    logger.info("Creating test CT data...")
    test_ct = np.random.rand(100, 100, 100).astype(np.float32)
    test_ct_path = "test_ct.npy"
    np.save(test_ct_path, test_ct)
    logger.info(f"✓ Test CT saved to: {test_ct_path}")
    logger.info(f"Test CT shape: {test_ct.shape}")
    logger.info(f"Test CT range: {test_ct.min():.6f} to {test_ct.max():.6f}")
    
    # Create SystemManager with minimal setup
    logger.info("Creating SystemManager...")
    system_manager = SystemManager(
        root_dir=".",
        transforms=None,
        resolutions=[(64, 64, 64)],
        energies=[11.5],
        quad_energies=[11.5],
        quad_weights=[1.0],
        batch_size=1,
        device=torch.device('cpu'),
        num_epochs=1,
        learning_rate=0.001,
        patience=10,
        cube_size=64
    )
    logger.info("✓ SystemManager created successfully")
    
    # Test inference
    logger.info("Starting inference test...")
    try:
        output_path = system_manager.run_inference(
            test_ct_path, 
            model_checkpoint="unified_energy_conditioned_model_res16.0_energies3.ckpt"
        )
        
        logger.info(f"✓ Inference completed successfully!")
        logger.info(f"Output saved to: {output_path}")
        
        # Check if the output file exists
        if os.path.exists(output_path):
            logger.info("✓ Output file created successfully")
            
            # Load and check the output
            import nibabel as nib
            img = nib.load(output_path)
            data = img.get_fdata()
            
            logger.info(f"Output shape: {data.shape}")
            logger.info(f"Output range: {data.min():.6f} to {data.max():.6f}")
            logger.info("✓ Output file loaded and validated successfully")
            
        else:
            logger.error("✗ Output file not found")
            
    except Exception as e:
        logger.error(f"✗ Error during inference: {e}")
        logger.error("Inference test traceback:", exc_info=True)
        import traceback
        traceback.print_exc()
    
    finally:
        # Clean up
        logger.info("Cleaning up test files...")
        if os.path.exists(test_ct_path):
            os.remove(test_ct_path)
            logger.info(f"✓ Removed test file: {test_ct_path}")
        
        logger.info("=" * 60)
        logger.info("SYSTEM MANAGER INFERENCE TEST COMPLETED")
        logger.info("=" * 60)

if __name__ == "__main__":
    test_system_manager_inference()
