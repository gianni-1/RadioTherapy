#!/usr/bin/env python3
"""
Simple test script for energy-conditioned training
"""

import os
import sys
import torch

# Add paths
sys.path.append('sourcecode')
os.environ['PYTHONPATH'] = '/Users/giannigagliardi/Documents/Git/RadioTherapy'

try:
    from system_manager import SystemManager
    from monai.transforms import Compose
    import logging
    
    print("🚀 Starting Energy-Conditioned Training Test")
    print("=" * 50)
    
    # Setup basic logging
    logging.basicConfig(level=logging.INFO)
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Test configuration - keep it small for testing
    config = {
        'root_dir': 'traindata',
        'resolutions': [16.0],  # Just one resolution
        'energies': [11.5, 15.75],  # Just two energies for faster testing
        'quad_energies': [11.5, 15.75],
        'quad_weights': [1.0, 1.0],
        'batch_size': 1,  # Small batch
        'num_epochs': 2,  # Just 2 epochs for testing
        'learning_rate': 1e-4,
        'patience': 2,
        'cube_size': (32, 32, 32),  # Small cube for speed
        'seed': 42
    }
    
    print(f"Configuration:")
    for key, value in config.items():
        print(f"  {key}: {value}")
    print()
    
    # Initialize SystemManager
    print("Initializing SystemManager...")
    system_manager = SystemManager(
        root_dir=config['root_dir'],
        transforms=Compose([]),  # Will be set in method
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
    
    print("✅ SystemManager initialized successfully")
    
    # Start training
    print("\\nStarting energy-conditioned training...")
    result = system_manager.run_energy_conditioned_training()
    
    if result:
        print("\\n✅ Training completed successfully!")
        print(f"Model saved to: {result['model_path']}")
        print("\\nFinal losses:")
        print(f"  Autoencoder train: {result['losses']['ae_train'][-1]:.4f}")
        print(f"  Autoencoder val: {result['losses']['ae_val'][-1]:.4f}")
        if result['losses']['diff']:
            print(f"  Diffusion: {result['losses']['diff'][-1]:.4f}")
    else:
        print("❌ Training failed")

except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()
