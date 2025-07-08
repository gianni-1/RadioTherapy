#!/usr/bin/env python3
"""
Simple test to compare inference output with ground truth data.
"""

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from pathlib import Path
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    """
    Simple comparison between inference and ground truth.
    """
    logger.info("=== SIMPLE INFERENCE TEST ===")
    
    # 1. Load ground truth data
    logger.info("Loading ground truth data...")
    
    # Load input and output from 11.5 keV energy
    input_file = Path("traindata/11_5/inputcube/235101017859661465075472232303048949736_0.npy")
    output_file = Path("traindata/11_5/outputcube/235101017859661465075472232303048949736_0.npy")
    
    if not input_file.exists() or not output_file.exists():
        logger.error(f"Files not found: {input_file}, {output_file}")
        return
    
    input_data = np.load(input_file)
    output_data = np.load(output_file)
    
    logger.info(f"Input shape: {input_data.shape}, range: {input_data.min():.3f} to {input_data.max():.3f}")
    logger.info(f"Output shape: {output_data.shape}, range: {output_data.min():.6f} to {output_data.max():.6f}")
    
    # 2. Load the trained model
    logger.info("Loading model...")
    
    model_path = "unified_energy_conditioned_model_res16.0_energies3.ckpt"
    if not Path(model_path).exists():
        logger.error(f"Model not found: {model_path}")
        return
    
    try:
        checkpoint = torch.load(model_path, map_location='cpu')
        logger.info(f"Checkpoint keys: {list(checkpoint.keys())}")
        
        # Try to understand the model structure
        if 'unet' in checkpoint:
            unet_state = checkpoint['unet']
            autoencoder_state = checkpoint.get('autoencoder', {})
            scheduler_state = checkpoint.get('scheduler', {})
            
            logger.info(f"Found separate components: unet, autoencoder, scheduler")
            logger.info(f"UNet state keys (first 10): {list(unet_state.keys())[:10]}")
            
            # Check UNet architecture
            input_layer_found = False
            for key in unet_state.keys():
                if 'conv' in key.lower() and 'weight' in key:
                    weight_shape = unet_state[key].shape
                    logger.info(f"UNet layer {key}: weight shape {weight_shape}")
                    if not input_layer_found:
                        logger.info(f"First conv layer appears to expect {weight_shape[1]} input channels")
                        input_layer_found = True
                    break
                    
            model_state = unet_state
            
        elif 'model_state_dict' in checkpoint:
            model_state = checkpoint['model_state_dict']
        else:
            model_state = checkpoint
            
        # Print some keys to understand the model
        state_keys = list(model_state.keys())[:10]  # First 10 keys
        logger.info(f"Model state keys (first 10): {state_keys}")
        
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        return
    
    # 3. Simple direct comparison - can we even load the model?
    logger.info("Attempting to create and load model...")
    
    try:
        # Try to create a simple model that matches the architecture
        from generative.networks.nets import DiffusionModelUNet
        
        # Use the EXACT configuration from the checkpoint
        configs = [
            {
                "in_channels": 2,
                "out_channels": 2,
                "cross_attention_dim": 2,
                "num_channels": (32, 64, 64),
                "name": "EXACT match from checkpoint"
            },
        ]
        
        for config in configs:
            try:
                logger.info(f"Testing {config['name']} configuration...")
                
                model = DiffusionModelUNet(
                    spatial_dims=3,
                    in_channels=config["in_channels"],
                    out_channels=config["out_channels"],
                    num_res_blocks=1,  # From SystemManager
                    num_channels=config["num_channels"],
                    attention_levels=(False, True, True),
                    num_head_channels=(0, 32, 32),  # From SystemManager
                    with_conditioning=True,
                    transformer_num_layers=1,
                    cross_attention_dim=config["cross_attention_dim"],
                    upcast_attention=True,
                )
                
                # Try to load state dict
                missing_keys, unexpected_keys = model.load_state_dict(model_state, strict=False)
                
                logger.info(f"  Missing keys: {len(missing_keys)}")
                logger.info(f"  Unexpected keys: {len(unexpected_keys)}")
                
                if len(missing_keys) == 0:
                    logger.info(f"  ✓ {config['name']} configuration loaded successfully!")
                    
                    # Test inference
                    model.eval()
                    
                    # Prepare input
                    input_tensor = torch.from_numpy(input_data).float().unsqueeze(0).unsqueeze(0)  # [1, 1, D, H, W]
                    
                    # Resize to model expected size (64x64x64)
                    if input_tensor.shape[2:] != (64, 64, 64):
                        input_tensor = F.interpolate(input_tensor, size=(64, 64, 64), mode='trilinear', align_corners=False)
                    
                    if config["in_channels"] == 2:
                        # Add energy channel
                        energy_value = 11.5
                        energy_normalized = energy_value / 100.0  # Normalize
                        energy_channel = torch.full_like(input_tensor, energy_normalized)
                        input_tensor = torch.cat([input_tensor, energy_channel], dim=1)
                    
                    logger.info(f"  Input tensor shape: {input_tensor.shape}")
                    
                    # Create context for cross-attention (2D context)
                    energy_tensor = torch.tensor([[energy_value / 100.0, 1.0]], dtype=torch.float32)  # [1, 2]
                    context = energy_tensor.unsqueeze(1)  # [1, 1, 2] - Add sequence dimension
                    
                    # Create timestep
                    timestep = torch.tensor([0], dtype=torch.long)
                    
                    # Run inference
                    with torch.no_grad():
                        try:
                            output = model(
                                x=input_tensor,
                                timesteps=timestep,
                                context=context
                            )
                            
                            logger.info(f"  ✓ Inference successful! Output shape: {output.shape}")
                            logger.info(f"  Output range: {output.min():.6f} to {output.max():.6f}")
                            
                            # If model outputs 2 channels, take the first one
                            if output.shape[1] == 2:
                                output = output[:, :1]  # Take first channel
                                logger.info(f"  Took first channel, new shape: {output.shape}")
                            
                            # Resize back to original size
                            if output.shape[2:] != output_data.shape:
                                output = F.interpolate(output, size=output_data.shape, mode='trilinear', align_corners=False)
                            
                            # Compare with ground truth
                            output_np = output[0, 0].cpu().numpy()
                            
                            # Statistical comparison
                            logger.info("  === COMPARISON WITH GROUND TRUTH ===")
                            logger.info(f"  Ground truth range: {output_data.min():.6f} to {output_data.max():.6f}")
                            logger.info(f"  Inference range: {output_np.min():.6f} to {output_np.max():.6f}")
                            
                            # Check if ranges are reasonable
                            gt_max = output_data.max()
                            inf_max = output_np.max()
                            
                            if gt_max > 0:
                                scale_ratio = inf_max / gt_max
                                logger.info(f"  Scale ratio (inference/ground_truth): {scale_ratio:.6f}")
                            
                            # Correlation
                            try:
                                correlation = np.corrcoef(output_data.flatten(), output_np.flatten())[0, 1]
                                logger.info(f"  Correlation: {correlation:.6f}")
                                
                                if abs(correlation) < 0.1:
                                    logger.info("  🚨 VERY LOW CORRELATION!")
                                elif abs(correlation) < 0.3:
                                    logger.info("  ⚠️  LOW CORRELATION")
                                else:
                                    logger.info("  ✓ Reasonable correlation")
                            except:
                                logger.info("  Could not compute correlation")
                            
                            # Save visualization
                            fig, axes = plt.subplots(2, 3, figsize=(15, 10))
                            
                            # Take middle slices
                            slice_idx = output_data.shape[2] // 2
                            
                            # Ground truth
                            im1 = axes[0, 0].imshow(output_data[:, :, slice_idx], cmap='hot')
                            axes[0, 0].set_title(f'Ground Truth\nMax: {output_data.max():.6f}')
                            axes[0, 0].axis('off')
                            plt.colorbar(im1, ax=axes[0, 0])
                            
                            # Inference
                            im2 = axes[0, 1].imshow(output_np[:, :, slice_idx], cmap='hot')
                            axes[0, 1].set_title(f'Inference\nMax: {output_np.max():.6f}')
                            axes[0, 1].axis('off')
                            plt.colorbar(im2, ax=axes[0, 1])
                            
                            # Difference
                            diff = output_np - output_data
                            im3 = axes[0, 2].imshow(diff[:, :, slice_idx], cmap='RdBu_r')
                            axes[0, 2].set_title(f'Difference\nMax: {abs(diff).max():.6f}')
                            axes[0, 2].axis('off')
                            plt.colorbar(im3, ax=axes[0, 2])
                            
                            # Show different slice (axial)
                            slice_idx_y = output_data.shape[1] // 2
                            
                            # Ground truth (axial)
                            im4 = axes[1, 0].imshow(output_data[:, slice_idx_y, :], cmap='hot')
                            axes[1, 0].set_title('Ground Truth (Axial)')
                            axes[1, 0].axis('off')
                            plt.colorbar(im4, ax=axes[1, 0])
                            
                            # Inference (axial)
                            im5 = axes[1, 1].imshow(output_np[:, slice_idx_y, :], cmap='hot')
                            axes[1, 1].set_title('Inference (Axial)')
                            axes[1, 1].axis('off')
                            plt.colorbar(im5, ax=axes[1, 1])
                            
                            # Difference (axial)
                            im6 = axes[1, 2].imshow(diff[:, slice_idx_y, :], cmap='RdBu_r')
                            axes[1, 2].set_title('Difference (Axial)')
                            axes[1, 2].axis('off')
                            plt.colorbar(im6, ax=axes[1, 2])
                            
                            plt.tight_layout()
                            plt.savefig(f'comparison_correct_model.png', dpi=150, bbox_inches='tight')
                            logger.info(f"  Saved comparison as comparison_correct_model.png")
                            
                            return True  # Success
                            
                        except Exception as e:
                            logger.info(f"  ✗ Inference failed: {e}")
                            import traceback
                            traceback.print_exc()
                            
                else:
                    logger.info(f"  ✗ {config['name']} configuration has missing keys")
                    
            except Exception as e:
                logger.info(f"  ✗ Error with {config['name']} configuration: {e}")
                
    except Exception as e:
        logger.error(f"Error creating model: {e}")
        return
    
    logger.info("=== TEST COMPLETED ===")

if __name__ == "__main__":
    main()
