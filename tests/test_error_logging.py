#!/usr/bin/env python3
"""
Test error logging in the corrected inference pipeline
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

def test_error_logging():
    """Test error logging by providing an invalid model checkpoint"""
    logger.info("=" * 60)
    logger.info("TESTING ERROR LOGGING")
    logger.info("=" * 60)
    
    # Create a simple test CT file
    logger.info("Creating test CT data...")
    test_ct = np.random.rand(100, 100, 100).astype(np.float32)
    test_ct_path = "test_ct.npy"
    np.save(test_ct_path, test_ct)
    logger.info(f"✓ Test CT saved to: {test_ct_path}")
    
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
    
    # Test inference with invalid checkpoint (should trigger error logging)
    logger.info("Starting inference test with invalid checkpoint...")
    try:
        output_path = system_manager.run_inference(
            test_ct_path, 
            model_checkpoint="nonexistent_model.ckpt"  # This will cause an error
        )
        
        logger.error("This should not happen - expected an error!")
        
    except Exception as e:
        logger.info(f"✓ Expected error caught: {e}")
        logger.info("✓ Error logging test completed successfully")
    
    finally:
        # Clean up
        logger.info("Cleaning up test files...")
        if os.path.exists(test_ct_path):
            os.remove(test_ct_path)
            logger.info(f"✓ Removed test file: {test_ct_path}")
        
        logger.info("=" * 60)
        logger.info("ERROR LOGGING TEST COMPLETED")
        logger.info("=" * 60)

if __name__ == "__main__":
    test_error_logging()
