#!/usr/bin/env python3
"""
Corrected Training Script with proper parameters
"""

import os
import sys
import torch
import logging

# Add paths
sys.path.append('sourcecode')
os.environ['PYTHONPATH'] = '/Users/giannigagliardi/Documents/Git/RadioTherapy'

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

print("🚀 CORRECTED TRAINING WITH PROPER PARAMETERS")
print("=" * 60)

try:
    from sourcecode.system_manager import SystemManager
    from monai.transforms import Compose
    
    print("✅ Imports successful")
    
    # CORRECTED PARAMETERS - more training!
    config = {
        'root_dir': 'traindata',
        'resolutions': [16.0],  # Single resolution for now
        'energies': [11.5, 15.75, 34.25],  # All energies
        'quad_energies': [11.5, 15.75, 34.25],
        'quad_weights': [0.33, 0.33, 0.34],  # Equal weights
        'batch_size': 1,
        'num_epochs': 10,  # MORE EPOCHS!
        'learning_rate': 0.0001,  # Smaller learning rate
        'patience': 5,  # More patience
        'cube_size': (32, 32, 32),
        'device': torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        'seed': 42
    }
    
    print(f"🔧 Configuration:")
    for key, value in config.items():
        print(f"  {key}: {value}")
    
    # Initialize SystemManager
    print(f"\n📦 Initializing SystemManager...")
    system_manager = SystemManager(
        root_dir=config['root_dir'],
        transforms=Compose([]),
        resolutions=config['resolutions'],
        energies=config['energies'],
        quad_energies=config['quad_energies'],
        quad_weights=config['quad_weights'],
        batch_size=config['batch_size'],
        device=config['device'],
        num_epochs=config['num_epochs'],
        learning_rate=config['learning_rate'],
        patience=config['patience'],
        cube_size=config['cube_size'],
        seed=config['seed']
    )
    
    print("✅ SystemManager initialized")
    
    # Run corrected training
    print(f"\n🎯 Starting CORRECTED energy-conditioned training...")
    print(f"   This will train for {config['num_epochs']} epochs with proper parameters")
    
    system_manager.run_energy_conditioned_training()
    
    print(f"\n✅ CORRECTED TRAINING COMPLETED!")
    print(f"   Check the new model and run validation again")
    
except Exception as e:
    print(f"❌ Error during corrected training: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "=" * 60)
print("Corrected training script complete!")
