#!/usr/bin/env python3
"""
Test GUI Training (nicht das vollständige GUI, nur die Training-Logik)
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), 'sourcecode')))

# Initialize logging first
import log_config
import logging

logger = logging.getLogger(__name__)

import torch
from system_manager import SystemManager
from monai.transforms import Compose

def test_gui_training():
    """Test the GUI training logic without actually running the GUI"""
    
    logger.info("=" * 60)
    logger.info("TESTING GUI TRAINING LOGIC")
    logger.info("=" * 60)
    
    # Create SystemManager similar to GUI
    logger.info("Creating SystemManager for GUI training test...")
    
    system_manager = SystemManager(
        root_dir="traindata",
        transforms=Compose([]),
        resolutions=[16.0],
        energies=[11.5, 15.75, 34.25],  # All energies for corrected training
        quad_energies=[11.5, 15.75, 34.25],
        quad_weights=[0.33, 0.33, 0.34],
        batch_size=1,
        device=torch.device('cpu'),
        num_epochs=2,  # Short test
        learning_rate=0.0001,
        patience=3,
        cube_size=32
    )
    
    logger.info("✓ SystemManager created")
    logger.info(f"Training with energies: {system_manager.energies}")
    
    # Test corrected training (like GUI would do)
    logger.info("Testing energy-conditioned training...")
    try:
        result = system_manager.run_energy_conditioned_training()
        logger.info("✓ Energy-conditioned training completed successfully!")
        logger.info(f"Model saved to: {result['model_path']}")
        
    except Exception as e:
        logger.error(f"Energy-conditioned training failed: {e}")
        logger.error("Training traceback:", exc_info=True)
        return False
    
    logger.info("=" * 60)
    logger.info("GUI TRAINING TEST COMPLETED SUCCESSFULLY")
    logger.info("=" * 60)
    return True

if __name__ == "__main__":
    success = test_gui_training()
    if success:
        print(" GUI training test passed!")
    else:
        print(" GUI training test failed!")
