#!/usr/bin/env python3
"""
Script to run the new energy-conditioned training.

This script trains a SINGLE model with ALL energies, instead of separate models for each energy.
"""

import torch
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'sourcecode'))
import log_config
import logging
from system_manager import SystemManager
from monai.transforms import Compose

# Setup logging
logger = logging.getLogger(__name__)

def main():
    print("🚀 Starting Energy-Conditioned Training")
    print("=" * 60)
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Configuration
    config = {
        'root_dir': '/Users/giannigagliardi/Documents/Git/RadioTherapy/traindata',
        'resolutions': [16.0],  # Start with one resolution for testing
        'energies': [11.5, 15.75, 34.25],  # All available energies
        'quad_energies': [11.5, 15.75, 34.25],  # For inference
        'quad_weights': [1.0, 1.0, 1.0],  # Equal weights for now
        'batch_size': 2,
        'num_epochs': 10,  # Start with fewer epochs for testing
        'learning_rate': 1e-4,
        'patience': 3,
        'cube_size': (64, 64, 64),  # Smaller for faster training
        'seed': 42
    }
    
    print(f"Configuration:")
    print(f"  Energies: {config['energies']}")
    print(f"  Resolutions: {config['resolutions']}")
    print(f"  Batch size: {config['batch_size']}")
    print(f"  Epochs: {config['num_epochs']}")
    print(f"  Cube size: {config['cube_size']}")
    print()
    
    # Create transforms (will be set in SystemManager)
    transforms = Compose([])  # Placeholder, will be overridden
    
    # Initialize SystemManager
    system_manager = SystemManager(
        root_dir=config['root_dir'],
        transforms=transforms,
        resolutions=config['resolutions'],
        energies=config['energies'],
        quad_energies=config['quad_energies'],
        quad_weights=config['quad_weights'],
        batch_size=config['batch_size'],
        device=device,
        num_epochs=config['num_epochs'],
        learning_rate=config['learning_rate'],
        patience=config['patience'],
        cube_size=config['cube_size'],
        seed=config['seed']
    )
    
    try:
        # Run the NEW energy-conditioned training
        print("Starting energy-conditioned training...")
        result = system_manager.run_energy_conditioned_training()
        
        if result:
            print()
            print("✅ Training completed successfully!")
            print(f"Model saved to: {result['model_path']}")
            print(f"Final losses:")
            print(f"  Autoencoder train: {result['losses']['ae_train'][-1]:.4f}")
            print(f"  Autoencoder val: {result['losses']['ae_val'][-1]:.4f}")
            if result['losses']['diff']:
                print(f"  Diffusion: {result['losses']['diff'][-1]:.4f}")
            
            # Test the trained model with a quick inference
            print()
            print("🧪 Testing trained model with quick inference...")
            test_inference(system_manager, result, config)
            
        else:
            print("❌ Training failed or was interrupted")
            
    except Exception as e:
        print(f"❌ Training failed with error: {e}")
        logger.exception("Training failed")
        raise

def test_inference(system_manager, result, config):
    """Quick test of the trained model"""
    try:
        import numpy as np
        from inference_module import InferenceModule
        
        # Load a test CT scan
        test_ct_path = "/Users/giannigagliardi/Documents/Git/RadioTherapy/traindata/11_5/inputcube/235101017859661465075472232303048949736_0.npy"
        ct_data = np.load(test_ct_path)
        ct_tensor = torch.from_numpy(ct_data).float().unsqueeze(0)  # Add channel dim
        
        print(f"Test CT shape: {ct_tensor.shape}")
        
        # Create models_by_energy dict for inference
        models_by_energy = {}
        for energy in config['energies']:
            models_by_energy[energy] = (
                result['autoencoder'],
                result['unet'], 
                result['scheduler']
            )
        
        # Create inference module
        inference_module = InferenceModule(
            models_by_energy=models_by_energy,
            energies=config['energies'],
            energy_weights=config['quad_weights'],
            device=system_manager.device
        )
        
        # Test inference with different energies
        print("Testing inference with different energies:")
        for energy in [11.5, 34.25]:  # Test lowest and highest energy
            dose = inference_module.run_inference_conditioned_on_energy(
                ct_tensor, energy, target_cube_size=(32, 32, 32)  # Smaller for speed
            )
            print(f"  Energy {energy}: output shape={dose.shape}, min={dose.min():.4f}, max={dose.max():.4f}")
        
        print("✅ Inference test completed!")
        
    except Exception as e:
        print(f"⚠️  Inference test failed: {e}")
        logger.warning(f"Inference test failed: {e}")

if __name__ == "__main__":
    main()
