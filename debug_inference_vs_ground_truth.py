#!/usr/bin/env python3
"""
Debug script to compare inference results with ground truth data.
This script will help identify why inference results differ so dramatically from expected outputs.
"""

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from pathlib import Path
import logging
import sys
import os

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Add the sourcecode directory to the path
sourcecode_dir = Path(__file__).parent / "sourcecode"
sys.path.insert(0, str(sourcecode_dir))

def load_training_data_sample(energy_value=11.5):
    """
    Load a sample from the training data to compare with inference results.
    """
    logger.info(f"Loading training data sample for energy {energy_value}")
    
    # Look for training data directories
    traindata_dir = Path("traindata")
    if not traindata_dir.exists():
        logger.error("traindata directory not found!")
        return None, None
    
    # Find files for the specific energy
    input_files = []
    output_files = []
    
    # Look for energy-specific subdirectories
    energy_dir_name = str(energy_value).replace(".", "_")
    energy_dir = traindata_dir / energy_dir_name
    
    if energy_dir.exists():
        logger.info(f"Found energy directory: {energy_dir}")
        
        # Look for inputcube and outputcube subdirectories
        input_dir = energy_dir / "inputcube"
        output_dir = energy_dir / "outputcube"
        
        if input_dir.exists():
            input_files = list(input_dir.glob("*.npy"))
            logger.info(f"Found {len(input_files)} input files in {input_dir}")
        
        if output_dir.exists():
            output_files = list(output_dir.glob("*.npy"))
            logger.info(f"Found {len(output_files)} output files in {output_dir}")
            
    else:
        logger.warning(f"No energy directory found for {energy_value} (looking for {energy_dir})")
        # Fallback: search all files
        for file in traindata_dir.glob("**/*"):
            if file.is_file():
                filename = file.name.lower()
                if str(energy_value).replace(".", "_") in filename or f"{energy_value:.0f}" in filename:
                    if "input" in filename or "ct" in filename:
                        input_files.append(file)
                    elif "output" in filename or "dose" in filename:
                        output_files.append(file)
    
    logger.info(f"Found {len(input_files)} input files and {len(output_files)} output files for energy {energy_value}")
    
    if not input_files or not output_files:
        logger.warning(f"No matching files found for energy {energy_value}")
        return None, None
    
    # Load the first matching pair
    input_file = input_files[0]
    output_file = output_files[0]
    
    logger.info(f"Loading input: {input_file}")
    logger.info(f"Loading output: {output_file}")
    
    try:
        # Try different loading methods
        if input_file.suffix == '.npy':
            input_data = np.load(input_file)
        elif input_file.suffix == '.npz':
            input_data = np.load(input_file)
            # If it's a .npz file, get the first array
            if hasattr(input_data, 'files'):
                input_data = input_data[input_data.files[0]]
        else:
            logger.error(f"Unsupported file format: {input_file.suffix}")
            return None, None
            
        if output_file.suffix == '.npy':
            output_data = np.load(output_file)
        elif output_file.suffix == '.npz':
            output_data = np.load(output_file)
            if hasattr(output_data, 'files'):
                output_data = output_data[output_data.files[0]]
        else:
            logger.error(f"Unsupported file format: {output_file.suffix}")
            return None, None
            
        logger.info(f"Loaded input: shape={input_data.shape}, min={input_data.min():.4f}, max={input_data.max():.4f}")
        logger.info(f"Loaded output: shape={output_data.shape}, min={output_data.min():.4f}, max={output_data.max():.4f}")
        
        return input_data, output_data
        
    except Exception as e:
        logger.error(f"Error loading data: {e}")
        return None, None

def compare_inference_with_ground_truth(energy_value=11.5, model_path=None):
    """
    Compare inference results with ground truth data.
    """
    logger.info("=== COMPARING INFERENCE WITH GROUND TRUTH ===")
    
    # Load training data sample
    input_data, ground_truth = load_training_data_sample(energy_value)
    if input_data is None or ground_truth is None:
        logger.error("Could not load training data sample")
        return
    
    # Convert to torch tensors
    if isinstance(input_data, np.ndarray):
        input_tensor = torch.from_numpy(input_data).float()
    else:
        input_tensor = input_data.float()
    
    if isinstance(ground_truth, np.ndarray):
        ground_truth_tensor = torch.from_numpy(ground_truth).float()
    else:
        ground_truth_tensor = ground_truth.float()
    
    # Ensure proper shape (add batch and channel dimensions if needed)
    if input_tensor.dim() == 3:
        input_tensor = input_tensor.unsqueeze(0).unsqueeze(0)  # [1, 1, D, H, W]
    elif input_tensor.dim() == 4:
        input_tensor = input_tensor.unsqueeze(0)  # [1, C, D, H, W]
    
    if ground_truth_tensor.dim() == 3:
        ground_truth_tensor = ground_truth_tensor.unsqueeze(0).unsqueeze(0)  # [1, 1, D, H, W]
    elif ground_truth_tensor.dim() == 4:
        ground_truth_tensor = ground_truth_tensor.unsqueeze(0)  # [1, C, D, H, W]
    
    logger.info(f"Input tensor shape: {input_tensor.shape}")
    logger.info(f"Ground truth tensor shape: {ground_truth_tensor.shape}")        # Load and run inference
        try:
            from inference_module import InferenceModule
            
            # Load the model checkpoint
            if model_path:
                logger.info(f"Loading model from: {model_path}")
                checkpoint = torch.load(model_path, map_location='cpu')
                
                # Extract model components from checkpoint
                if 'model_state_dict' in checkpoint:
                    model_state = checkpoint['model_state_dict']
                else:
                    model_state = checkpoint
                
                # Create a simple models_by_energy dictionary
                # For now, we'll use a dummy structure since we only have one energy
                models_by_energy = {
                    energy_value: (None, None, None)  # autoencoder, unet, scheduler
                }
                
                # Initialize inference module
                inference_module = InferenceModule(
                    models_by_energy=models_by_energy,
                    energies=[energy_value],
                    energy_weights=[1.0],
                    device='cpu'
                )
                
                # Since we can't easily reconstruct the full model from checkpoint,
                # let's do a simpler approach: load the model directly
                logger.info("Attempting to load model directly from checkpoint...")
                
                # Actually, let's use a different approach - just load and run the model directly
                # without going through the InferenceModule
                
                from generative.networks.nets import DiffusionModelUNet
                
                # Create model
                model = DiffusionModelUNet(
                    spatial_dims=3,
                    in_channels=1,
                    out_channels=1,
                    num_res_blocks=2,
                    num_channels=(32, 64, 64),
                    attention_levels=(False, True, True),
                    num_head_channels=64,
                    with_conditioning=True,
                    transformer_num_layers=1,
                    cross_attention_dim=512,
                    upcast_attention=True,
                )
                
                # Load state dict
                model.load_state_dict(model_state, strict=False)
                model.eval()
                
                logger.info("Model loaded successfully")
                
                # Create energy conditioning
                energy_tensor = torch.tensor([energy_value], dtype=torch.float32).unsqueeze(0)
                position_tensor = torch.tensor([0.5], dtype=torch.float32).unsqueeze(0)  # Middle position
                context = torch.cat([energy_tensor, position_tensor], dim=1)  # Shape: [1, 2]
                
                # Run inference
                logger.info("Running model inference...")
                with torch.no_grad():
                    # Resize input to match model expectations
                    if input_tensor.shape[2:] != (64, 64, 64):
                        input_resized = F.interpolate(
                            input_tensor, 
                            size=(64, 64, 64), 
                            mode='trilinear', 
                            align_corners=False
                        )
                    else:
                        input_resized = input_tensor
                    
                    logger.info(f"Input tensor for inference: {input_resized.shape}")
                    
                    # Add noise (as in diffusion process)
                    noise = torch.randn_like(input_resized)
                    
                    # Create timestep
                    timestep = torch.tensor([0], dtype=torch.long)
                    
                    # Run model
                    inference_output = model(
                        x=noise,  # Start with noise
                        timesteps=timestep,
                        context=context.unsqueeze(0).repeat(input_resized.shape[0], 1, 1)
                    )
                    
                    logger.info(f"Raw model output shape: {inference_output.shape}")
                    
                    # Resize back to original size if needed
                    if ground_truth_tensor.shape[2:] != inference_output.shape[2:]:
                        inference_output = F.interpolate(
                            inference_output,
                            size=ground_truth_tensor.shape[2:],
                            mode='trilinear',
                            align_corners=False
                        )
                
            else:
                logger.error("No model path provided")
                return
        
        logger.info(f"Inference output shape: {inference_output.shape}")
        logger.info(f"Inference output min/max: {inference_output.min():.4f}/{inference_output.max():.4f}")
        
        # Ensure same shape for comparison
        if inference_output.shape != ground_truth_tensor.shape:
            logger.warning(f"Shape mismatch: inference={inference_output.shape}, ground_truth={ground_truth_tensor.shape}")
            
            # Try to match shapes
            if inference_output.dim() == 5 and ground_truth_tensor.dim() == 5:
                # Both have batch and channel dimensions
                if inference_output.shape[0] == ground_truth_tensor.shape[0]:
                    # Same batch size, check channels
                    if inference_output.shape[1] != ground_truth_tensor.shape[1]:
                        logger.info("Adjusting channel dimension...")
                        if inference_output.shape[1] == 1:
                            pass  # Keep as is
                        else:
                            inference_output = inference_output[:, :1]  # Take first channel
                
                # Check spatial dimensions
                if inference_output.shape[2:] != ground_truth_tensor.shape[2:]:
                    logger.info("Spatial dimensions don't match - this might be the issue!")
                    logger.info(f"Inference spatial: {inference_output.shape[2:]}")
                    logger.info(f"Ground truth spatial: {ground_truth_tensor.shape[2:]}")
        
        # Statistical comparison
        logger.info("=== STATISTICAL COMPARISON ===")
        
        # Ground truth stats
        gt_min, gt_max = ground_truth_tensor.min().item(), ground_truth_tensor.max().item()
        gt_mean, gt_std = ground_truth_tensor.mean().item(), ground_truth_tensor.std().item()
        
        # Inference stats
        inf_min, inf_max = inference_output.min().item(), inference_output.max().item()
        inf_mean, inf_std = inference_output.mean().item(), inference_output.std().item()
        
        logger.info(f"Ground Truth - Min: {gt_min:.6f}, Max: {gt_max:.6f}, Mean: {gt_mean:.6f}, Std: {gt_std:.6f}")
        logger.info(f"Inference     - Min: {inf_min:.6f}, Max: {inf_max:.6f}, Mean: {inf_mean:.6f}, Std: {inf_std:.6f}")
        
        # Range comparison
        gt_range = gt_max - gt_min
        inf_range = inf_max - inf_min
        logger.info(f"Ground Truth Range: {gt_range:.6f}")
        logger.info(f"Inference Range: {inf_range:.6f}")
        logger.info(f"Range Ratio (inf/gt): {inf_range/gt_range:.6f}" if gt_range > 0 else "Ground truth range is zero!")
        
        # Mean comparison
        mean_ratio = inf_mean / gt_mean if gt_mean != 0 else float('inf')
        logger.info(f"Mean Ratio (inf/gt): {mean_ratio:.6f}")
        
        # Try correlation if shapes match
        if inference_output.shape == ground_truth_tensor.shape:
            try:
                # Flatten tensors for correlation
                inf_flat = inference_output.flatten()
                gt_flat = ground_truth_tensor.flatten()
                
                # Compute correlation
                correlation = torch.corrcoef(torch.stack([inf_flat, gt_flat]))[0, 1].item()
                logger.info(f"Correlation coefficient: {correlation:.6f}")
                
                if abs(correlation) < 0.1:
                    logger.warning("🚨 VERY LOW CORRELATION - inference and ground truth are almost uncorrelated!")
                elif abs(correlation) < 0.3:
                    logger.warning("⚠️  LOW CORRELATION - significant differences detected")
                else:
                    logger.info("✓ Reasonable correlation detected")
                    
            except Exception as e:
                logger.error(f"Could not compute correlation: {e}")
        
        # Check if values are in completely different ranges
        if gt_range > 0 and inf_range > 0:
            if inf_max < gt_min or inf_min > gt_max:
                logger.warning("🚨 NO OVERLAP - inference and ground truth value ranges don't overlap at all!")
            elif (inf_max - inf_min) / (gt_max - gt_min) > 100 or (gt_max - gt_min) / (inf_max - inf_min) > 100:
                logger.warning("⚠️  EXTREME SCALE DIFFERENCE - ranges differ by more than 100x")
        
        # Visual comparison (save plots)
        logger.info("Creating visualization comparison...")
        
        # Take middle slice for visualization
        if ground_truth_tensor.dim() == 5:
            gt_slice = ground_truth_tensor[0, 0, ground_truth_tensor.shape[2]//2, :, :].cpu().numpy()
        else:
            gt_slice = ground_truth_tensor[ground_truth_tensor.shape[0]//2, :, :].cpu().numpy()
        
        if inference_output.dim() == 5:
            inf_slice = inference_output[0, 0, inference_output.shape[2]//2, :, :].cpu().numpy()
        else:
            inf_slice = inference_output[inference_output.shape[0]//2, :, :].cpu().numpy()
        
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # Ground truth
        im1 = axes[0].imshow(gt_slice, cmap='hot')
        axes[0].set_title(f'Ground Truth\nMin: {gt_min:.3f}, Max: {gt_max:.3f}')
        axes[0].axis('off')
        plt.colorbar(im1, ax=axes[0])
        
        # Inference result
        im2 = axes[1].imshow(inf_slice, cmap='hot')
        axes[1].set_title(f'Inference Result\nMin: {inf_min:.3f}, Max: {inf_max:.3f}')
        axes[1].axis('off')
        plt.colorbar(im2, ax=axes[1])
        
        # Difference (if shapes match)
        if gt_slice.shape == inf_slice.shape:
            diff = inf_slice - gt_slice
            im3 = axes[2].imshow(diff, cmap='RdBu_r')
            axes[2].set_title(f'Difference (Inf - GT)\nMin: {diff.min():.3f}, Max: {diff.max():.3f}')
            axes[2].axis('off')
            plt.colorbar(im3, ax=axes[2])
        else:
            axes[2].text(0.5, 0.5, f'Shape mismatch:\nGT: {gt_slice.shape}\nInf: {inf_slice.shape}', 
                        ha='center', va='center', transform=axes[2].transAxes)
            axes[2].set_title('Cannot compute difference')
        
        plt.tight_layout()
        plt.savefig('inference_vs_ground_truth_comparison.png', dpi=150, bbox_inches='tight')
        logger.info("Saved comparison plot as 'inference_vs_ground_truth_comparison.png'")
        
        # Summary
        logger.info("=== SUMMARY ===")
        if abs(correlation) < 0.1 if 'correlation' in locals() else True:
            logger.info("🚨 CRITICAL ISSUE: Inference results don't match ground truth at all!")
            logger.info("Possible causes:")
            logger.info("1. Wrong model loaded (not trained on this data)")
            logger.info("2. Incorrect preprocessing/normalization")
            logger.info("3. Model not properly trained")
            logger.info("4. Wrong energy conditioning")
            logger.info("5. Data loading issue")
        else:
            logger.info("✓ Inference results show some correlation with ground truth")
        
        return {
            'ground_truth_stats': (gt_min, gt_max, gt_mean, gt_std),
            'inference_stats': (inf_min, inf_max, inf_mean, inf_std),
            'correlation': correlation if 'correlation' in locals() else None,
            'shapes_match': inference_output.shape == ground_truth_tensor.shape
        }
        
    except Exception as e:
        logger.error(f"Error during inference comparison: {e}")
        import traceback
        traceback.print_exc()
        return None

def main():
    """
    Main function to run the comparison.
    """
    logger.info("Starting inference vs ground truth comparison...")
    
    # Check if we have the new model
    newest_model = None
    model_files = list(Path(".").glob("*.ckpt"))
    if model_files:
        newest_model = max(model_files, key=lambda x: x.stat().st_mtime)
        logger.info(f"Found newest model: {newest_model}")
    
    # Run comparison
    result = compare_inference_with_ground_truth(
        energy_value=11.5,
        model_path=str(newest_model) if newest_model else None
    )
    
    if result:
        logger.info("Comparison completed successfully!")
    else:
        logger.error("Comparison failed!")

if __name__ == "__main__":
    main()
