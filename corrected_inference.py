#!/usr/bin/env python3
"""
Corrected inference pipeline that properly uses the energy-conditioned model.
"""

import torch
import torch.nn.functional as F
import numpy as np
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

class CorrectedInferenceModule:
    """
    Fixed inference module that properly handles the energy-conditioned model.
    """
    
    def __init__(self, model_path, device='cpu'):
        """
        Initialize with the corrected energy-conditioned model.
        
        Args:
            model_path: Path to the unified model checkpoint
            device: Device to run inference on
        """
        logger.info("=" * 60)
        logger.info("CORRECTED INFERENCE MODULE INITIALIZATION")
        logger.info("=" * 60)
        
        self.device = device
        self.model_path = model_path
        
        # Load the checkpoint
        logger.info(f"Loading model from: {model_path}")
        logger.info(f"Device: {device}")
        
        try:
            # Try loading with weights_only=False to avoid the warning and ensure compatibility
            logger.info("Attempting to load checkpoint...")
            self.checkpoint = torch.load(model_path, map_location=device, weights_only=False)
            logger.info("✓ Checkpoint loaded successfully via direct path")
        except Exception as e:
            # If that fails, try opening as file explicitly
            logger.warning(f"Direct load failed: {e}")
            logger.info("Attempting to load checkpoint via file handle...")
            try:
                with open(model_path, 'rb') as f:
                    self.checkpoint = torch.load(f, map_location=device, weights_only=False)
                logger.info("✓ Checkpoint loaded successfully via file handle")
            except Exception as e2:
                logger.error(f"Both loading methods failed: {e2}")
                raise e2
        
        # Extract energies and scale factor from checkpoint
        self.energies = self.checkpoint.get('energies', [11.5, 15.75, 34.25])
        self.scale_factor = self.checkpoint.get('scale_factor', 0.18215)
        
        logger.info(f"✓ Model energies: {self.energies}")
        logger.info(f"✓ Scale factor: {self.scale_factor}")
        logger.info(f"✓ Checkpoint keys: {list(self.checkpoint.keys())}")
        
        # Load models
        logger.info("Loading model components...")
        self._load_models()
        logger.info("✓ Corrected inference module initialized successfully")
    
    def _load_models(self):
        """Load autoencoder, unet, and scheduler from checkpoint."""
        try:
            logger.info("  Loading required modules...")
            # Import required modules
            from generative.networks.nets import AutoencoderKL, DiffusionModelUNet
            from generative.inferers import LatentDiffusionInferer
            from monai.utils import set_determinism
            logger.info("  ✓ Modules imported successfully")
            
            # Load autoencoder
            logger.info("  Creating autoencoder...")
            self.autoencoder = AutoencoderKL(
                spatial_dims=3,
                in_channels=2,  # CT + Energy channel
                out_channels=1,
                num_channels=(32, 32, 32),
                latent_channels=2,
                num_res_blocks=1,
                norm_num_groups=8,
                attention_levels=(False, False, True),
            ).to(self.device)
            logger.info(f"  ✓ Autoencoder created (in_channels=2, out_channels=1)")
            
            # Load UNet
            logger.info("  Creating UNet...")
            self.unet = DiffusionModelUNet(
                spatial_dims=3,
                in_channels=2,  # Latent space is 2 channels
                out_channels=2,
                with_conditioning=True,
                cross_attention_dim=2,
                num_res_blocks=1,
                num_channels=(32, 64, 64),
                attention_levels=(False, True, True),
                num_head_channels=(0, 32, 32),
            ).to(self.device)
            logger.info(f"  ✓ UNet created (in_channels=2, out_channels=2, cross_attention_dim=2)")
            
            # Load scheduler
            logger.info("  Creating scheduler...")
            from generative.networks.schedulers.ddpm import DDPMScheduler
            
            self.scheduler = DDPMScheduler(
                num_train_timesteps=1000, 
                beta_start=0.0015, 
                beta_end=0.0195
            )
            logger.info(f"  ✓ DDPM Scheduler created (timesteps=1000)")
            
            # Load state dicts
            logger.info("  Loading model weights...")
            
            # Check autoencoder weights
            ae_state_dict = self.checkpoint['autoencoder']
            logger.info(f"  Autoencoder checkpoint keys: {len(ae_state_dict)} parameters")
            self.autoencoder.load_state_dict(ae_state_dict)
            logger.info("  ✓ Autoencoder weights loaded")
            
            # Check UNet weights
            unet_state_dict = self.checkpoint['unet']
            logger.info(f"  UNet checkpoint keys: {len(unet_state_dict)} parameters")
            self.unet.load_state_dict(unet_state_dict)
            logger.info("  ✓ UNet weights loaded")
            
            # Set to eval mode
            self.autoencoder.eval()
            self.unet.eval()
            logger.info("  ✓ Models set to evaluation mode")
            
            # Create inferer
            logger.info("  Creating inferer...")
            self.inferer = LatentDiffusionInferer(
                scheduler=self.scheduler, 
                scale_factor=self.scale_factor
            )
            logger.info(f"  ✓ LatentDiffusionInferer created (scale_factor={self.scale_factor})")
            
        except Exception as e:
            logger.error(f"Failed to load models: {e}")
            logger.error("Model loading traceback:", exc_info=True)
            raise e
        
        logger.info("  ✓ All models loaded successfully")
    
    
    def preprocess_ct(self, ct_array, target_cube_size=(64, 64, 64)):
        """
        Preprocess CT data for inference.
        
        Args:
            ct_array: CT data as numpy array
            target_cube_size: Target size for processing
            
        Returns:
            Preprocessed CT tensor
        """
        # Convert to tensor
        if isinstance(ct_array, np.ndarray):
            ct_tensor = torch.from_numpy(ct_array).float()
        else:
            ct_tensor = ct_array.float()
        
        # Add batch and channel dimensions if needed
        if ct_tensor.dim() == 3:
            ct_tensor = ct_tensor.unsqueeze(0).unsqueeze(0)  # [1, 1, D, H, W]
        elif ct_tensor.dim() == 4:
            ct_tensor = ct_tensor.unsqueeze(0)  # [1, C, D, H, W]
        
        # Resize to target size
        if ct_tensor.shape[2:] != target_cube_size:
            ct_tensor = F.interpolate(
                ct_tensor,
                size=target_cube_size,
                mode='trilinear',
                align_corners=False
            )
        
        # Normalize CT values (similar to training preprocessing)
        # Clamp extreme values and normalize to 0-1 range
        ct_tensor = torch.clamp(ct_tensor, -1000, 3000)  # Typical CT range
        ct_tensor = (ct_tensor + 1000) / 4000  # Normalize to 0-1
        
        return ct_tensor.to(self.device)
    
    def run_inference(self, ct_tensor, quad_energies=None, quad_weights=None, target_cube_size=(64, 64, 64)):
        """
        Run inference on CT data with quadrature integration over energies.
        
        Args:
            ct_tensor: CT data tensor or numpy array
            quad_energies: List of quadrature energies (if None, uses single energy)
            quad_weights: List of quadrature weights (if None, uses single energy)
            target_cube_size: Target cube size for processing
            
        Returns:
            Dose distribution tensor
        """
        if quad_energies is None or quad_weights is None:
            # Single energy inference
            logger.info("=== RUNNING SINGLE ENERGY INFERENCE ===")
            return self.run_inference_single_energy(ct_tensor, target_cube_size=target_cube_size)
        
        logger.info("=" * 60)
        logger.info("RUNNING CORRECTED INFERENCE WITH QUADRATURE")
        logger.info("=" * 60)
        logger.info(f"Quadrature energies: {quad_energies}")
        logger.info(f"Quadrature weights: {quad_weights}")
        logger.info(f"Number of quadrature points: {len(quad_energies)}")
        
        try:
            # Run inference for each energy and aggregate
            total_dose = None
            
            for i, (energy, weight) in enumerate(zip(quad_energies, quad_weights)):
                logger.info(f"Processing quadrature point {i+1}/{len(quad_energies)}: energy={energy:.3f} keV, weight={weight:.6f}")
                
                # Run inference for this energy
                dose = self.run_inference_single_energy(ct_tensor, target_energy=energy, target_cube_size=target_cube_size)
                logger.info(f"  ✓ Dose computed: shape={dose.shape}, range=[{dose.min():.6f}, {dose.max():.6f}]")
                
                # Weight and accumulate
                if total_dose is None:
                    total_dose = dose * weight
                    logger.info(f"  ✓ Initialized total dose (weighted by {weight:.6f})")
                else:
                    total_dose += dose * weight
                    logger.info(f"  ✓ Added to total dose (weighted by {weight:.6f})")
            
            logger.info("=" * 60)
            logger.info("QUADRATURE INTEGRATION COMPLETED SUCCESSFULLY")
            logger.info(f"Final dose: shape={total_dose.shape}, range=[{total_dose.min():.6f}, {total_dose.max():.6f}]")
            logger.info("=" * 60)
            return total_dose
            
        except Exception as e:
            logger.error("QUADRATURE INFERENCE FAILED")
            logger.error(f"Error: {e}")
            logger.error("Quadrature inference traceback:", exc_info=True)
            raise e
    
    def run_inference_single_energy(self, ct_tensor, target_energy=None, target_cube_size=(64, 64, 64)):
        """
        Run inference on CT data for a single energy.
        
        Args:
            ct_tensor: CT data tensor or numpy array
            target_energy: Target energy for inference (if None, uses first available)
            target_cube_size: Target cube size for processing
            
        Returns:
            Dose distribution tensor
        """
        logger.info("-" * 40)
        logger.info("SINGLE ENERGY INFERENCE STARTED")
        logger.info("-" * 40)
        
        try:
            # Preprocess CT
            original_shape = ct_tensor.shape if hasattr(ct_tensor, 'shape') else ct_tensor.shape
            logger.info(f"Original CT shape: {original_shape}")
            
            input_tensor = self.preprocess_ct(ct_tensor, target_cube_size)
            logger.info(f"✓ Input preprocessed: {input_tensor.shape}")
            
            # Choose energy
            if target_energy is None:
                target_energy = self.energies[0]
                logger.info(f"No target energy specified, using default: {target_energy} keV")
            else:
                logger.info(f"Using specified energy: {target_energy} keV")
            
            # Add energy conditioning
            B, C, D, H, W = input_tensor.shape
            normalized_energy = target_energy / 100.0  # Normalize like in training
            energy_tensor = torch.full((B, 1, D, H, W), normalized_energy, device=self.device)
            conditioned_input = torch.cat([input_tensor, energy_tensor], dim=1)
            
            logger.info(f"✓ Energy conditioning added:")
            logger.info(f"  - Input shape: {input_tensor.shape}")
            logger.info(f"  - Energy tensor shape: {energy_tensor.shape}")
            logger.info(f"  - Conditioned input shape: {conditioned_input.shape}")
            logger.info(f"  - Raw energy: {target_energy}, normalized: {normalized_energy:.6f}")
            
            with torch.no_grad():
                # 1. Encode with autoencoder
                logger.info("Step 1: Encoding with autoencoder...")
                encoded = self.autoencoder.encode(conditioned_input)
                if isinstance(encoded, tuple):
                    latent = encoded[0]
                else:
                    latent = encoded.latent_dist.sample()
            
            logger.info(f"Latent shape: {latent.shape}")
            logger.info(f"Latent range: {latent.min():.4f} to {latent.max():.4f}")
            
            # 2. Create context for cross-attention
            energy_weight = 1.0  # Could be adjusted based on energy
            context = torch.tensor(
                [[normalized_energy, energy_weight]], 
                dtype=torch.float32, 
                device=self.device
            ).unsqueeze(1)  # [1, 1, 2] - Add sequence dimension
            
            logger.info(f"Context shape: {context.shape}")
            logger.info(f"Context values: {context}")
            
            # 3. Run diffusion sampling
            logger.info("Running diffusion sampling...")
            
            # Create noise for sampling
            noise_shape = latent.shape
            noise = torch.randn(noise_shape, device=self.device)
            
            logger.info(f"Noise shape: {noise_shape}")
            logger.info(f"Using context: {context}")
            
            # Use the inferer for proper diffusion sampling
            # The LatentDiffusionInferer handles the full pipeline internally
            sampled_output = self.inferer.sample(
                input_noise=noise,
                autoencoder_model=self.autoencoder,
                diffusion_model=self.unet,
                scheduler=self.scheduler,
                conditioning=context,
                mode="crossattn"
            )
            
            logger.info(f"Sampled output shape: {sampled_output.shape}")
            logger.info(f"Sampled output range: {sampled_output.min():.4f} to {sampled_output.max():.4f}")
            
            # The inferer should return the decoded output directly
            final_output = sampled_output
            
            # 5. Take only the dose channel (first channel) if multi-channel
            if final_output.shape[1] > 1:
                dose_output = final_output[:, :1]  # Take first channel
                logger.info(f"Took first channel for dose, shape: {dose_output.shape}")
            else:
                dose_output = final_output
            
            logger.info(f"Final dose output shape: {dose_output.shape}")
            logger.info(f"Final dose range: {dose_output.min():.6f} to {dose_output.max():.6f}")
            
            # 6. Resize back to original size if needed
            if hasattr(ct_tensor, 'shape'):
                original_spatial = ct_tensor.shape if len(ct_tensor.shape) == 3 else ct_tensor.shape[-3:]
            else:
                original_spatial = original_shape if len(original_shape) == 3 else original_shape[-3:]
            
            if dose_output.shape[2:] != original_spatial:
                logger.info(f"Resizing from {dose_output.shape[2:]} to {original_spatial}")
                dose_output = F.interpolate(
                    dose_output,
                    size=original_spatial,
                    mode='trilinear',
                    align_corners=False
                )
            
            # 7. Remove batch dimension
            final_dose = dose_output[0, 0]  # Remove batch and channel dims
            
            logger.info(f"Final output shape: {final_dose.shape}")
            logger.info(f"Final output range: {final_dose.min():.6f} to {final_dose.max():.6f}")
            
            logger.info("✓ SINGLE ENERGY INFERENCE COMPLETED SUCCESSFULLY")
            logger.info("-" * 40)
            return final_dose
            
        except Exception as e:
            logger.error("SINGLE ENERGY INFERENCE FAILED")
            logger.error(f"Error: {e}")
            logger.error("Single energy inference traceback:", exc_info=True)
            raise e


def test_corrected_inference():
    """Test the corrected inference module."""
    import matplotlib.pyplot as plt
    
    # Load test data
    input_file = Path("traindata/11_5/inputcube/235101017859661465075472232303048949736_0.npy")
    output_file = Path("traindata/11_5/outputcube/235101017859661465075472232303048949736_0.npy")
    
    if not input_file.exists():
        print(f"Test input file not found: {input_file}")
        return
    
    input_data = np.load(input_file)
    ground_truth = np.load(output_file)
    
    print(f"Input shape: {input_data.shape}")
    print(f"Ground truth shape: {ground_truth.shape}")
    print(f"Ground truth range: {ground_truth.min():.6f} to {ground_truth.max():.6f}")
    
    # Run corrected inference
    model_path = "unified_energy_conditioned_model_res16.0_energies3.ckpt"
    
    try:
        inference_module = CorrectedInferenceModule(model_path, device='cpu')
        
        # Run inference
        result = inference_module.run_inference_single_energy(
            ct_tensor=input_data,
            target_energy=11.5,
            target_cube_size=(64, 64, 64)
        )
        
        result_np = result.cpu().numpy()
        print(f"Inference result shape: {result_np.shape}")
        print(f"Inference result range: {result_np.min():.6f} to {result_np.max():.6f}")
        
        # Compare with ground truth
        if result_np.shape == ground_truth.shape:
            correlation = np.corrcoef(result_np.flatten(), ground_truth.flatten())[0, 1]
            print(f"Correlation with ground truth: {correlation:.6f}")
            
            # Visualize comparison
            fig, axes = plt.subplots(2, 3, figsize=(15, 10))
            
            slice_idx = ground_truth.shape[2] // 2
            
            # Ground truth
            im1 = axes[0, 0].imshow(ground_truth[:, :, slice_idx], cmap='hot')
            axes[0, 0].set_title(f'Ground Truth\nMax: {ground_truth.max():.6f}')
            axes[0, 0].axis('off')
            plt.colorbar(im1, ax=axes[0, 0])
            
            # Inference result
            im2 = axes[0, 1].imshow(result_np[:, :, slice_idx], cmap='hot')
            axes[0, 1].set_title(f'Corrected Inference\nMax: {result_np.max():.6f}')
            axes[0, 1].axis('off')
            plt.colorbar(im2, ax=axes[0, 1])
            
            # Difference
            diff = result_np - ground_truth
            im3 = axes[0, 2].imshow(diff[:, :, slice_idx], cmap='RdBu_r')
            axes[0, 2].set_title(f'Difference\nMax: {abs(diff).max():.6f}')
            axes[0, 2].axis('off')
            plt.colorbar(im3, ax=axes[0, 2])
            
            # Axial view
            slice_idx_y = ground_truth.shape[1] // 2
            
            im4 = axes[1, 0].imshow(ground_truth[:, slice_idx_y, :], cmap='hot')
            axes[1, 0].set_title('Ground Truth (Axial)')
            axes[1, 0].axis('off')
            plt.colorbar(im4, ax=axes[1, 0])
            
            im5 = axes[1, 1].imshow(result_np[:, slice_idx_y, :], cmap='hot')
            axes[1, 1].set_title('Corrected Inference (Axial)')
            axes[1, 1].axis('off')
            plt.colorbar(im5, ax=axes[1, 1])
            
            im6 = axes[1, 2].imshow(diff[:, slice_idx_y, :], cmap='RdBu_r')
            axes[1, 2].set_title('Difference (Axial)')
            axes[1, 2].axis('off')
            plt.colorbar(im6, ax=axes[1, 2])
            
            plt.tight_layout()
            plt.savefig('corrected_inference_comparison.png', dpi=150, bbox_inches='tight')
            print("Saved comparison as 'corrected_inference_comparison.png'")
            
        else:
            print(f"Shape mismatch: result {result_np.shape} vs ground truth {ground_truth.shape}")
        
    except Exception as e:
        print(f"Error during corrected inference: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    test_corrected_inference()
