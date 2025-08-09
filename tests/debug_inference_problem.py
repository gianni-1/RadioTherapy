#!/usr/bin/env python3
"""
Debug-Script um das Inference-Problem zu identifizieren
"""

import os
import sys
import torch
import numpy as np
import logging

# Add paths
sys.path.append('sourcecode')
os.environ['PYTHONPATH'] = '/Users/giannigagliardi/Documents/Git/RadioTherapy'

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

print(" DEBUG: Inference Problem Analysis")
print("=" * 60)

# Test: Lade den gleichen Input/Output wie in visualizationTEST
input_file = 'traindata/11_5/inputcube/235101017859661465075472232303048949736_0.npy'
output_file = 'traindata/11_5/outputcube/235101017859661465075472232303048949736_0.npy'

print(f"Loading test files:")
print(f"  Input: {input_file}")
print(f"  Output: {output_file}")

# Lade die Daten
if os.path.exists(input_file) and os.path.exists(output_file):
    input_data = np.load(input_file)
    ground_truth = np.load(output_file)
    
    print(f"\n Ground Truth Analysis:")
    print(f"  Input shape: {input_data.shape}")
    print(f"  Output shape: {ground_truth.shape}")
    print(f"  Output min/max: {ground_truth.min():.6f} / {ground_truth.max():.6f}")
    print(f"  Output mean/std: {ground_truth.mean():.6f} / {ground_truth.std():.6f}")
    print(f"  Output non-zero elements: {np.count_nonzero(ground_truth)}")
    
    # Test 1: Direkte Validation mit unserem Modell
    print(f"\n🧪 Test 1: Direct Model Validation")
    
    # Lade das trainierte Modell
    model_path = '/Users/giannigagliardi/Documents/Git/RadioTherapy/unified_energy_conditioned_model_res16.0_energies2.ckpt'
    
    if os.path.exists(model_path):
        print(f"Loading model: {model_path}")
        
        device = torch.device("cpu")
        checkpoint = torch.load(model_path, map_location=device)
        
        # Lade Autoencoder
        from generative.networks.nets import AutoencoderKL
        autoencoder = AutoencoderKL(
            spatial_dims=3,
            in_channels=2,
            out_channels=1,
            num_channels=(32, 32, 32),
            latent_channels=2,
            num_res_blocks=1,
            norm_num_groups=8,
            attention_levels=(False, False, True),
        ).to(device)
        
        autoencoder.load_state_dict(checkpoint['autoencoder'])
        autoencoder.eval()
        
        print(" Model loaded successfully")
        
        # Test: Verwende die gleiche Energie-Konditionierung wie im Training
        print(f"\n Test 2: Energy Conditioning")
        
        # Erstelle Input mit Energy-Kanal (wie im Training)
        energy_value = 11.5
        energy_normalized = energy_value / 40.0  # Normalisierung wie in validate_energy_conditioning.py
        
        # Create 2-channel input
        input_2channel = np.zeros((2, *input_data.shape))
        input_2channel[0] = input_data  # Physical dose
        input_2channel[1] = energy_normalized  # Energy channel
        
        # Convert to tensor
        input_tensor = torch.tensor(input_2channel, dtype=torch.float32).unsqueeze(0).to(device)
        
        print(f"  Input tensor shape: {input_tensor.shape}")
        print(f"  Energy channel value: {energy_normalized:.4f}")
        print(f"  Input channel 0 (CT) min/max: {input_tensor[0,0].min():.6f} / {input_tensor[0,0].max():.6f}")
        print(f"  Input channel 1 (Energy) min/max: {input_tensor[0,1].min():.6f} / {input_tensor[0,1].max():.6f}")
        
        # Test direkte Autoencoder-Rekonstruktion
        print(f"\n Test 3: Direct Autoencoder Reconstruction")
        with torch.no_grad():
            # Encode
            encoded = autoencoder.encode(input_tensor)
            if isinstance(encoded, tuple):
                latent = encoded[0]
            else:
                latent = encoded.latent_dist.sample()
            
            print(f"  Latent shape: {latent.shape}")
            print(f"  Latent min/max: {latent.min():.6f} / {latent.max():.6f}")
            
            # Decode
            decoded = autoencoder.decode(latent)
            if isinstance(decoded, tuple):
                reconstruction = decoded[0]
            else:
                reconstruction = decoded
            
            reconstruction_np = reconstruction.cpu().numpy().squeeze()
            
            print(f"  Reconstruction shape: {reconstruction_np.shape}")
            print(f"  Reconstruction min/max: {reconstruction_np.min():.6f} / {reconstruction_np.max():.6f}")
            print(f"  Reconstruction mean/std: {reconstruction_np.mean():.6f} / {reconstruction_np.std():.6f}")
            
            # Vergleiche mit Ground Truth
            print(f"\n Comparison with Ground Truth:")
            mse = np.mean((reconstruction_np - ground_truth) ** 2)
            mae = np.mean(np.abs(reconstruction_np - ground_truth))
            correlation = np.corrcoef(reconstruction_np.flatten(), ground_truth.flatten())[0, 1]
            
            print(f"  MSE: {mse:.8f}")
            print(f"  MAE: {mae:.8f}")
            print(f"  Correlation: {correlation:.6f}")
            
            if correlation > 0.5:
                print("   GOOD: Reconstruction correlates with ground truth")
            else:
                print("   BAD: Poor correlation with ground truth")
                
            # Speichere zur Visualisierung
            np.save('debug_reconstruction.npy', reconstruction_np)
            np.save('debug_ground_truth.npy', ground_truth)
            np.save('debug_input.npy', input_data)
            
            print(f"\n Saved debug files:")
            print(f"  debug_reconstruction.npy - Model reconstruction")
            print(f"  debug_ground_truth.npy - Ground truth")
            print(f"  debug_input.npy - Input data")
            
    else:
        print(f" Model not found: {model_path}")
        
else:
    print(f" Test files not found!")
    print(f"  Looking for: {input_file}")
    print(f"  Looking for: {output_file}")

print("\n" + "=" * 60)
print("Debug analysis complete!")
