#!/usr/bin/env python3
"""
Überprüfe die Trainings-Daten auf Konsistenz
"""

import os
import sys
import torch
import numpy as np
import matplotlib.pyplot as plt

# Add paths
sys.path.append('sourcecode')
os.environ['PYTHONPATH'] = '/Users/giannigagliardi/Documents/Git/RadioTherapy'

print("🔍 TRAINING DATA VALIDATION")
print("=" * 50)

# Teste mehrere Input-Output Paare
test_files = [
    'traindata/11_5/inputcube/235101017859661465075472232303048949736_0.npy',
    'traindata/15_75/inputcube/235101017859661465075472232303048949736_0.npy',
    'traindata/34_25/inputcube/235101017859661465075472232303048949736_0.npy'
]

output_files = [
    'traindata/11_5/outputcube/235101017859661465075472232303048949736_0.npy',
    'traindata/15_75/outputcube/235101017859661465075472232303048949736_0.npy',
    'traindata/34_25/outputcube/235101017859661465075472232303048949736_0.npy'
]

energies = [11.5, 15.75, 34.25]

for i, (input_file, output_file, energy) in enumerate(zip(test_files, output_files, energies)):
    print(f"\n📊 Testing Energy {energy} MeV:")
    print(f"  Input:  {input_file}")
    print(f"  Output: {output_file}")
    
    if os.path.exists(input_file) and os.path.exists(output_file):
        input_data = np.load(input_file)
        output_data = np.load(output_file)
        
        print(f"  ✅ Files exist")
        print(f"  Input shape: {input_data.shape}")
        print(f"  Output shape: {output_data.shape}")
        print(f"  Input min/max: {input_data.min():.6f} / {input_data.max():.6f}")
        print(f"  Output min/max: {output_data.min():.6f} / {output_data.max():.6f}")
        print(f"  Output mean/std: {output_data.mean():.6f} / {output_data.std():.6f}")
        print(f"  Output non-zero: {np.count_nonzero(output_data)}")
        
        # Check if output is all zeros or constant
        if np.all(output_data == 0):
            print(f"  ❌ OUTPUT IS ALL ZEROS!")
        elif np.all(output_data == output_data.flat[0]):
            print(f"  ❌ OUTPUT IS CONSTANT!")
        else:
            print(f"  ✅ Output has variation")
            
        # Quick correlation check between input and output
        correlation = np.corrcoef(input_data.flatten(), output_data.flatten())[0, 1]
        print(f"  Input-Output correlation: {correlation:.6f}")
        
    else:
        print(f"  ❌ Files missing")

# Teste das Data Loading System
print(f"\n🔄 Testing Data Loading System:")

try:
    from sourcecode.data_management import DataLoaderModule
    from monai.transforms import Compose
    
    # Test data loading
    data_loader = DataLoaderModule(
        root_dir='traindata',
        transforms=Compose([]),
        resolutions=[16.0],
        energies=[11.5, 15.75, 34.25],
        batch_size=1,
        cube_size=(32, 32, 32),
        seed=42
    )
    
    print(f"  ✅ DataLoaderModule imported successfully")
    
    # Get a sample
    train_loader, val_loader, _ = data_loader.get_data_loaders()
    print(f"  Train batches: {len(train_loader)}")
    print(f"  Val batches: {len(val_loader)}")
    
    # Test one batch
    for batch in train_loader:
        input_batch = batch['inputcube']
        output_batch = batch['outputcube']
        energy_batch = batch['energy']
        
        print(f"  Sample batch:")
        print(f"    Input shape: {input_batch.shape}")
        print(f"    Output shape: {output_batch.shape}")
        print(f"    Energy values: {energy_batch}")
        print(f"    Input min/max: {input_batch.min():.6f} / {input_batch.max():.6f}")
        print(f"    Output min/max: {output_batch.min():.6f} / {output_batch.max():.6f}")
        print(f"    Output mean/std: {output_batch.mean():.6f} / {output_batch.std():.6f}")
        
        # Check for problems
        if torch.all(output_batch == 0):
            print(f"    ❌ BATCH OUTPUT IS ALL ZEROS!")
        elif torch.all(output_batch == output_batch.flat[0]):
            print(f"    ❌ BATCH OUTPUT IS CONSTANT!")
        else:
            print(f"    ✅ Batch output has variation")
            
        break  # Only test first batch
        
except Exception as e:
    print(f"  ❌ Error testing data loading: {e}")

# Test if the model can even learn identity mapping
print(f"\n🧪 Testing if model can learn identity mapping:")

# Create simple test: input = output
device = torch.device("cpu")
test_input = torch.randn(1, 2, 32, 32, 32)  # 2 channels for energy conditioning
test_output = test_input[:, 0:1, :, :, :]  # Take only first channel as output

print(f"  Test input shape: {test_input.shape}")
print(f"  Test output shape: {test_output.shape}")
print(f"  Input min/max: {test_input.min():.6f} / {test_input.max():.6f}")
print(f"  Output min/max: {test_output.min():.6f} / {test_output.max():.6f}")

# Load model and test
model_path = '/Users/giannigagliardi/Documents/Git/RadioTherapy/unified_energy_conditioned_model_res16.0_energies2.ckpt'
if os.path.exists(model_path):
    checkpoint = torch.load(model_path, map_location=device)
    
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
    
    with torch.no_grad():
        encoded = autoencoder.encode(test_input)
        if isinstance(encoded, tuple):
            latent = encoded[0]
        else:
            latent = encoded.latent_dist.sample()
        
        decoded = autoencoder.decode(latent)
        if isinstance(decoded, tuple):
            reconstruction = decoded[0]
        else:
            reconstruction = decoded
    
    print(f"  Model reconstruction shape: {reconstruction.shape}")
    print(f"  Model reconstruction min/max: {reconstruction.min():.6f} / {reconstruction.max():.6f}")
    
    # Test correlation
    correlation = np.corrcoef(test_input[:, 0].flatten(), reconstruction.flatten())[0, 1]
    print(f"  Correlation with first input channel: {correlation:.6f}")
    
    if correlation > 0.5:
        print(f"  ✅ Model can reconstruct reasonably well")
    else:
        print(f"  ❌ Model cannot even reconstruct simple inputs")

print("\n" + "=" * 50)
print("Training data validation complete!")
