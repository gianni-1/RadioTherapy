#!/usr/bin/env python3
"""
Validation script to test if energy conditioning actually works
This script tests the trained unified model to see if it produces different outputs for different energies
"""

import os
import sys
import torch
import numpy as np
import matplotlib.pyplot as plt

# Add paths
sys.path.append('sourcecode')
os.environ['PYTHONPATH'] = '/Users/giannigagliardi/Documents/Git/RadioTherapy'

try:
    import logging
    
    print("🔍 Energy Conditioning Validation Test")
    print("=" * 50)
    
    # Setup basic logging
    logging.basicConfig(level=logging.INFO)
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Test configuration
    config = {
        'root_dir': 'traindata',
        'resolutions': [16.0],
        'energies': [11.5, 15.75, 34.25],  # Test all three energies
        'batch_size': 1,
        'cube_size': (32, 32, 32),
        'model_path': '/Users/giannigagliardi/Documents/Git/RadioTherapy/unified_energy_conditioned_model_res16.0_energies2.ckpt'
    }
    
    print(f"Model path: {config['model_path']}")
    print(f"Testing energies: {config['energies']}")
    print()
    
    # Check if model exists
    if not os.path.exists(config['model_path']):
        print(f"❌ Model file not found: {config['model_path']}")
        exit(1)
    
    # Load the model checkpoint directly
    print("Loading model checkpoint...")
    checkpoint = torch.load(config['model_path'], map_location=device)
    print("✅ Model checkpoint loaded successfully")
    
    print(f"Checkpoint keys: {list(checkpoint.keys())}")
    
    # Get the actual training configuration from the checkpoint
    if 'training_config' in checkpoint:
        training_config = checkpoint['training_config']
        print(f"Training config: {training_config}")
    
    # Load the autoencoder architecture from our training pipeline
    from generative.networks.nets import AutoencoderKL
    
    # Initialize autoencoder with the ACTUAL parameters from training
    autoencoder = AutoencoderKL(
        spatial_dims=3,
        in_channels=2,  # CT + energy conditioning
        out_channels=1, # Output dose
        num_channels=(32, 32, 32),    # Correct configuration from training
        latent_channels=2,            # Correct configuration from training
        num_res_blocks=1,             # Correct configuration from training
        norm_num_groups=8,            # Correct configuration from training
        attention_levels=(False, False, True),  # Correct configuration from training
    ).to(device)
    
    # Load the trained weights
    if 'autoencoder' in checkpoint:
        autoencoder.load_state_dict(checkpoint['autoencoder'])
        print("✅ Autoencoder state dict loaded from checkpoint")
    else:
        print("❌ Autoencoder not found in checkpoint")
        exit(1)
    
    autoencoder.eval()
    
    print("✅ Autoencoder model loaded and ready for testing")
    
    # Load the same input data but test with different energy values
    print("\\nLoading test data...")
    
    # Load a sample input (we'll use the same physical input but different energy channels)
    sample_input_path = 'traindata/11_5/inputcube/235101017859661465075472232303048949736_0.npy'
    sample_output_path = 'traindata/11_5/outputcube/235101017859661465075472232303048949736_0.npy'
    
    if not os.path.exists(sample_input_path):
        print(f"❌ Sample input not found: {sample_input_path}")
        exit(1)
    
    # Load the physical dose distribution (first channel)
    physical_input = np.load(sample_input_path)
    ground_truth = np.load(sample_output_path)
    
    print(f"Loaded input shape: {physical_input.shape}")
    print(f"Loaded ground truth shape: {ground_truth.shape}")
    
    # Test energy conditioning
    print("\\n🧪 Testing Energy Conditioning...")
    print("-" * 30)
    
    results = {}
    predictions = {}
    
    for energy in config['energies']:
        print(f"\\nTesting energy: {energy} MeV")
        
        # Create energy-conditioned input
        # Channel 0: physical dose distribution
        # Channel 1: energy value (normalized)
        energy_normalized = energy / 40.0  # Normalize energy to [0,1] range approximately
        
        # Create 2-channel input
        input_2channel = np.zeros((2, *physical_input.shape))
        input_2channel[0] = physical_input  # Physical dose
        input_2channel[1] = energy_normalized  # Energy channel
        
        # Convert to tensor and add batch dimension
        input_tensor = torch.tensor(input_2channel, dtype=torch.float32).unsqueeze(0).to(device)
        
        print(f"  Input tensor shape: {input_tensor.shape}")
        print(f"  Energy channel value: {energy_normalized:.4f}")
        
        # Run inference
        with torch.no_grad():
            prediction = autoencoder(input_tensor)
            # AutoencoderKL returns a tuple (reconstruction, latent), we want the reconstruction
            if isinstance(prediction, tuple):
                prediction = prediction[0]  # Get the reconstruction
            prediction_np = prediction.cpu().numpy().squeeze()
        
        predictions[energy] = prediction_np
        
        # Calculate some statistics
        pred_mean = np.mean(prediction_np)
        pred_std = np.std(prediction_np)
        pred_max = np.max(prediction_np)
        pred_min = np.min(prediction_np)
        
        results[energy] = {
            'mean': pred_mean,
            'std': pred_std,
            'max': pred_max,
            'min': pred_min
        }
        
        print(f"  Prediction statistics:")
        print(f"    Mean: {pred_mean:.6f}")
        print(f"    Std:  {pred_std:.6f}")
        print(f"    Max:  {pred_max:.6f}")
        print(f"    Min:  {pred_min:.6f}")
    
    # Compare predictions between energies
    print("\\n📊 Energy Conditioning Analysis")
    print("-" * 30)
    
    energy_list = list(config['energies'])
    
    # Calculate differences between energy predictions
    for i in range(len(energy_list)):
        for j in range(i + 1, len(energy_list)):
            energy1, energy2 = energy_list[i], energy_list[j]
            pred1, pred2 = predictions[energy1], predictions[energy2]
            
            # Calculate various difference metrics
            mse = np.mean((pred1 - pred2) ** 2)
            mae = np.mean(np.abs(pred1 - pred2))
            max_diff = np.max(np.abs(pred1 - pred2))
            correlation = np.corrcoef(pred1.flatten(), pred2.flatten())[0, 1]
            
            print(f"\\n{energy1} MeV vs {energy2} MeV:")
            print(f"  MSE:         {mse:.8f}")
            print(f"  MAE:         {mae:.8f}")
            print(f"  Max Diff:    {max_diff:.8f}")
            print(f"  Correlation: {correlation:.6f}")
            
            # Energy conditioning is working if:
            # 1. MSE > 0 (predictions are different)
            # 2. Correlation < 1 (not identical)
            # 3. Differences are meaningful relative to the prediction scale
            
            if mse > 1e-8 and correlation < 0.99:
                print(f"  ✅ ENERGY CONDITIONING DETECTED - Model responds to energy!")
            elif mse < 1e-10:
                print(f"  ❌ NO ENERGY CONDITIONING - Predictions are identical")
            else:
                print(f"  ⚠️  WEAK ENERGY CONDITIONING - Small differences detected")
    
    # Statistical summary
    print("\\n📈 Summary Statistics")
    print("-" * 20)
    
    means = [results[e]['mean'] for e in energy_list]
    stds = [results[e]['std'] for e in energy_list]
    
    mean_variation = np.std(means) / np.mean(means) if np.mean(means) != 0 else 0
    std_variation = np.std(stds) / np.mean(stds) if np.mean(stds) != 0 else 0
    
    print(f"Mean variation across energies: {mean_variation:.6f}")
    print(f"Std variation across energies:  {std_variation:.6f}")
    
    # Overall assessment
    print("\\n🎯 Overall Assessment")
    print("-" * 20)
    
    # Check if any meaningful differences exist
    max_mse = 0
    min_correlation = 1.0
    
    for i in range(len(energy_list)):
        for j in range(i + 1, len(energy_list)):
            pred1, pred2 = predictions[energy_list[i]], predictions[energy_list[j]]
            mse = np.mean((pred1 - pred2) ** 2)
            correlation = np.corrcoef(pred1.flatten(), pred2.flatten())[0, 1]
            max_mse = max(max_mse, mse)
            min_correlation = min(min_correlation, correlation)
    
    if max_mse > 1e-6 and min_correlation < 0.95:
        print("✅ SUCCESS: Energy conditioning is working!")
        print("   The model produces meaningfully different outputs for different energies.")
        print("   The original problem has been SOLVED! 🎉")
    elif max_mse > 1e-8:
        print("⚠️  PARTIAL SUCCESS: Weak energy conditioning detected.")
        print("   The model shows some response to energy, but may need more training.")
    else:
        print("❌ FAILURE: No energy conditioning detected.")
        print("   The model ignores the energy input. Problem NOT solved.")
    
    print(f"\\nMax MSE between energies: {max_mse:.2e}")
    print(f"Min correlation between energies: {min_correlation:.6f}")

except Exception as e:
    print(f"❌ Error during validation: {e}")
    import traceback
    traceback.print_exc()

print("\\n" + "=" * 50)
print("Energy conditioning validation complete!")
