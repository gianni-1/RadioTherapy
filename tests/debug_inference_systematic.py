#!/usr/bin/env python3
"""
Systematische Analyse der Inference-Pipeline Probleme
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

print("🔍 SYSTEMATIC INFERENCE ANALYSIS")
print("=" * 60)

# Test verschiedene Cube-Größen und Normalisierungen
input_file = 'traindata/11_5/inputcube/235101017859661465075472232303048949736_0.npy'
output_file = 'traindata/11_5/outputcube/235101017859661465075472232303048949736_0.npy'

if os.path.exists(input_file) and os.path.exists(output_file):
    input_data = np.load(input_file)
    ground_truth = np.load(output_file)
    
    print(f"Original data shapes: {input_data.shape} -> {ground_truth.shape}")
    
    # Lade das trainierte Modell
    model_path = '/Users/giannigagliardi/Documents/Git/RadioTherapy/unified_energy_conditioned_model_res16.0_energies2.ckpt'
    device = torch.device("cpu")
    checkpoint = torch.load(model_path, map_location=device)
    
    # Schaue dir die Training-Konfiguration an
    print(f"\n📋 Training Configuration:")
    if 'training_config' in checkpoint:
        config = checkpoint['training_config']
        print(f"  Cube size: {config.get('cube_size', 'Not found')}")
        print(f"  Batch size: {config.get('batch_size', 'Not found')}")
        print(f"  Epochs: {config.get('num_epochs', 'Not found')}")
        print(f"  Learning rate: {config.get('learning_rate', 'Not found')}")
        
        # Verwende die TRAINING cube_size!
        if 'cube_size' in config:
            training_cube_size = config['cube_size']
            print(f"   Using training cube size: {training_cube_size}")
        else:
            training_cube_size = (32, 32, 32)  # Default
            print(f"    Using default cube size: {training_cube_size}")
    else:
        training_cube_size = (32, 32, 32)
        print(f"    No training config found, using default: {training_cube_size}")
    
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
    
    print(f"\n🔄 CORRECTED TEST: Using training cube size")
    
    # Resize input auf training cube size
    from torch.nn.functional import interpolate
    
    # Konvertiere zu Tensor und füge Batch-Dimension hinzu
    input_tensor = torch.tensor(input_data, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)
    ground_truth_tensor = torch.tensor(ground_truth, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)
    
    # Resize auf Training-Größe
    input_resized = interpolate(input_tensor, size=training_cube_size, mode='trilinear', align_corners=True)
    ground_truth_resized = interpolate(ground_truth_tensor, size=training_cube_size, mode='trilinear', align_corners=True)
    
    print(f"  Resized input: {input_tensor.shape} -> {input_resized.shape}")
    print(f"  Resized ground truth: {ground_truth_tensor.shape} -> {ground_truth_resized.shape}")
    
    # Teste verschiedene Energie-Normalisierungen
    energy_value = 11.5
    energy_normalizations = [
        ("div_40", energy_value / 40.0),
        ("div_100", energy_value / 100.0),
        ("div_50", energy_value / 50.0),
        ("raw", energy_value),
        ("normalized_0_1", (energy_value - 11.5) / (34.25 - 11.5))  # Min-Max normalization
    ]
    
    best_correlation = -1
    best_method = None
    best_reconstruction = None
    
    for method_name, energy_normalized in energy_normalizations:
        print(f"\n🧪 Testing energy normalization: {method_name} (value: {energy_normalized:.6f})")
        
        # Erstelle 2-channel input
        input_2channel = torch.zeros((1, 2, *training_cube_size), device=device)
        input_2channel[0, 0] = input_resized.squeeze()  # CT data
        input_2channel[0, 1] = energy_normalized  # Energy channel
        
        with torch.no_grad():
            # Encode
            encoded = autoencoder.encode(input_2channel)
            if isinstance(encoded, tuple):
                latent = encoded[0]
            else:
                latent = encoded.latent_dist.sample()
            
            # Decode
            decoded = autoencoder.decode(latent)
            if isinstance(decoded, tuple):
                reconstruction = decoded[0]
            else:
                reconstruction = decoded
            
            # Resize zurück auf Original-Größe für Vergleich
            reconstruction_full = interpolate(reconstruction, size=(100, 100, 100), mode='trilinear', align_corners=True)
            
            reconstruction_np = reconstruction_full.cpu().numpy().squeeze()
            
            # Vergleiche mit Ground Truth
            mse = np.mean((reconstruction_np - ground_truth) ** 2)
            mae = np.mean(np.abs(reconstruction_np - ground_truth))
            correlation = np.corrcoef(reconstruction_np.flatten(), ground_truth.flatten())[0, 1]
            
            print(f"    MSE: {mse:.8f}")
            print(f"    MAE: {mae:.8f}")
            print(f"    Correlation: {correlation:.6f}")
            print(f"    Reconstruction min/max: {reconstruction_np.min():.6f} / {reconstruction_np.max():.6f}")
            
            if correlation > best_correlation:
                best_correlation = correlation
                best_method = method_name
                best_reconstruction = reconstruction_np.copy()
                
                print(f"    ✅ NEW BEST METHOD!")
    
    print(f"\n🎯 BEST RESULT:")
    print(f"  Method: {best_method}")
    print(f"  Correlation: {best_correlation:.6f}")
    
    if best_correlation > 0.3:
        print(f"   GOOD: Reasonable correlation found!")
    else:
        print(f"   STILL BAD: Poor correlation even with corrections")
    
    # Save best reconstruction
    if best_reconstruction is not None:
        np.save('debug_best_reconstruction.npy', best_reconstruction)
        print(f"  💾 Saved best reconstruction to: debug_best_reconstruction.npy")
    
    print(f"\n🔧 RECOMMENDATION:")
    if best_correlation > 0.3:
        print(f"  Use energy normalization method: {best_method}")
        print(f"  Ensure cube size matches training: {training_cube_size}")
        print(f"  Fix inference pipeline to use these parameters!")
    else:
        print(f"  The model may not be properly trained for energy conditioning")
        print(f"  Consider retraining with more epochs or different parameters")

else:
    print(f" Test files not found!")

print("\n" + "=" * 60)
print("Systematic analysis complete!")
