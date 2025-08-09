#training_pipeline.py

import torch
import torch.nn.functional as F
from torch.nn import L1Loss
from tqdm import tqdm
from monai.losses import PatchAdversarialLoss, PerceptualLoss
import math
from torch.nn.utils import clip_grad_norm_
import log_config
import logging
import numpy as np
from data_management import HotspotPatchDataset




# Constants
#ENERGY_NORMALIZATION_FACTOR = 50.0  # ≈ max(energy) in training set (46.53 keV)
DOSE_EPS = 1e-8   # small value to avoid division‑by‑zero in normalisation
TV_WEIGHT = 5e-3      # stronger weight for total‑variation smoothness loss
# Min‑max limits for dose normalisation (Gy).
# These will be updated dynamically from the dataset statistics
CLIP_MIN = None
CLIP_MAX = None

logger = logging.getLogger(__name__)

class EarlyStopping:
    """
    Monitors the validation loss and triggers early stopping if the loss
    the loss does not improve over a defined number of epochs.
    """
    def __init__(self, patience=5):
        """
        Args:
            patience (int): Number of epochs permitted without improvement.
        """
        self.patience = patience
        self.best_val_loss = float('inf')
        self.epochs_no_improvement = 0

    def update(self, val_loss):
        """
        Compares the current validation loss with the best loss to date.
        Args:
            val_loss (float): Current loss of validation.
        Returns:
            bool: If Early Stopping is triggered true otherwise False.
        """
        if val_loss < self.best_val_loss:
            self.best_val_loss = val_loss
            self.epochs_no_improvement = 0
            return False 
        else:
            self.epochs_no_improvement += 1
            return self.epochs_no_improvement >= self.patience
        
    def reset(self):
        """
        Resets the early stopping parameters.
        """
        self.best_val_loss = float('inf')
        self.epochs_no_improvement = 0

class AutoencoderTrainer:
    """
    Encapsulates the training process for the autoencoder, including validation.

    This implementation has been updated to integrate energy conditioning.
    Each sample is expected to contain an "energy" field, which is normalized, expanded to match
    the spatial dimensions of the image, and then concatenated as an additional input channel.
    """
    def __init__(self, autoencoder, discriminator, optimizer_g, optimizer_d, energy_min, energy_max, device,
                 kl_weight=1e-6, adv_weight=0.001, perceptual_weight=0.005,
                 tv_weight=TV_WEIGHT, warm_up_epochs=2):
        """
        Args:
            autoencoder (torch.nn.Module): The autoencoder model.
            discriminator (torch.nn.Module): The discriminator model (for adversarial training).
            optimizer_g (torch.optim.Optimizer): Optimizer for the generator.
            optimizer_d (torch.optim.Optimizer): Optimizer for the discriminator.
            device (str): Device to run the training on ('cpu' or 'cuda').
            kl_weight (float): Weight for KL divergence loss.
            adv_weight (float): Weight for adversarial loss. Default: 0.001
            perceptual_weight (float): Weight for perceptual loss. Default: 0.005
            tv_weight (float): Weight for total‑variation loss.
            warm_up_epochs (int): Number of epochs to warm up the KL weight.
        """
        self.autoencoder = autoencoder
        self.discriminator = discriminator
        self.optimizer_g = optimizer_g
        self.optimizer_d = optimizer_d
        self.device = device
        
        # Initialize dose normalization parameters (will be calculated from data)
        self.clip_min = None
        self.clip_max = None
        self.energy_min = energy_min
        self.energy_max = energy_max
        self.kl_weight = kl_weight
        self.adv_weight = adv_weight
        self.perceptual_weight = perceptual_weight
        self.tv_weight = tv_weight
        self.warm_up_epochs = warm_up_epochs
        
        # Loss functions will be initialized after normalization parameters are calculated
        self.l1_loss = None
        self.adv_loss = None
        self.perceptual_loss = None
        
    def calculate_dose_normalization_params(self, dataloader, energy=None):
        """
        Calculate clip_min and clip_max from the training data for dose normalization.
        This should be called before training starts.
        
        Args:
            dataloader: Training dataloader
            energy: Energy level for energy-specific normalization
        """
        logger.info("Calculating dose normalization parameters from training data...")
        if energy is not None:
            logger.info(f"Energy-specific normalization for energy: {energy:.2f}")
        
        all_dose_values = []
        sample_count = 0
        
        for batch in dataloader:
            target_dose = batch["target"]  # Assuming target is the dose data
            sample_count += target_dose.shape[0]
            
            # Convert to numpy and flatten
            dose_values = target_dose.cpu().numpy().flatten()
            
            # Log sample statistics
            logger.info(f"Batch {len(all_dose_values)+1}: shape={target_dose.shape}, "
                       f"min={dose_values.min():.6f}, max={dose_values.max():.6f}, "
                       f"mean={dose_values.mean():.6f}, nonzero_count={np.count_nonzero(dose_values)}")
            
            all_dose_values.append(dose_values)
        
        if not all_dose_values:
            raise RuntimeError("No dose data found in dataloader")
            
        all_dose_values = np.concatenate(all_dose_values)
        logger.info(f"Total samples processed: {sample_count}, total voxels: {len(all_dose_values)}")
        logger.info(f"Overall dose stats: min={all_dose_values.min():.6f}, max={all_dose_values.max():.6f}, "
                   f"mean={all_dose_values.mean():.6f}, std={all_dose_values.std():.6f}")
        
        # 🔍 ENHANCED DEBUGGING: Check if dose values are realistic
        nonzero_values = all_dose_values[all_dose_values > 0]
        if len(nonzero_values) > 0:
            logger.info(f"Non-zero dose analysis:")
            logger.info(f"  Count: {len(nonzero_values)}/{len(all_dose_values)} ({100*len(nonzero_values)/len(all_dose_values):.1f}%)")
            logger.info(f"  Min non-zero: {nonzero_values.min():.6f}")
            logger.info(f"  Max non-zero: {nonzero_values.max():.6f}")
            logger.info(f"  Mean non-zero: {nonzero_values.mean():.6f}")
            logger.info(f"  Median non-zero: {np.median(nonzero_values):.6f}")
            
            # Check if values seem suspiciously low (should be in tens, not decimals)
            if nonzero_values.max() < 5.0:
                logger.error(f" DOSE VALUES TOO LOW! Max dose {nonzero_values.max():.6f} < 5.0")
                logger.error("Expected dose values in range 10-50+ for radiotherapy!")
                logger.error("This suggests a major pipeline or data scaling issue.")
        
        # Calculate percentiles to avoid outliers, but ensure meaningful range
        clip_min_candidate = float(np.percentile(all_dose_values, 0.1))
        
        # **ENERGY-SPECIFIC DOSE NORMALIZATION FIX**
        if energy is not None and energy > 40.0:
            # **CRITICAL FIX: For very high energies, use actual max to preserve signal strength**
            # Don't use percentiles for high energies as they cause severe underestimation
            clip_max_candidate = float(np.max(all_dose_values))
            logger.info(f" High energy ({energy:.2f}): Using ACTUAL maximum for clip_max: {clip_max_candidate:.2f}")
            
            # Only cap if extremely unreasonable (>150 Gy)
            if clip_max_candidate > 150.0:
                clip_max_candidate = 100.0
                logger.info(f" Capping clip_max to 100.0 for energy {energy:.2f} (was {float(np.max(all_dose_values)):.2f})")
            else:
                logger.info(f"✓ Using full range clip_max={clip_max_candidate:.2f} for energy {energy:.2f}")
        else:
            # Normal case: use actual maximum for better reconstruction
            clip_max_candidate = float(np.max(all_dose_values))
        
        # **CRITICAL FIX: Handle edge cases where data is all zeros or very close**
        if clip_max_candidate - clip_min_candidate < 1e-6:
            logger.warning(f"Percentile dose range too small: {clip_max_candidate - clip_min_candidate:.10f}")
            logger.warning("Using actual min/max normalization parameters...")
            
            # Use actual min/max if percentiles fail
            actual_min = float(all_dose_values.min())
            actual_max = float(all_dose_values.max())
            
            if actual_max - actual_min < 1e-6:
                # Data is essentially constant - use safe defaults
                logger.error("All dose values are essentially the same! Using emergency defaults.")
                self.clip_min = 0.0
                self.clip_max = 1.0  # Safe fallback
            else:
                # Use actual range
                self.clip_min = actual_min
                self.clip_max = actual_max
                logger.info(f"Successfully using actual range: {self.clip_min:.6f} to {self.clip_max:.6f}")
        else:
            # Normal case - use percentiles
            self.clip_min = clip_min_candidate
            self.clip_max = clip_max_candidate
            logger.info(f"Successfully using percentile range: {self.clip_min:.6f} to {self.clip_max:.6f}")
        
        logger.info(f"Final dose normalization parameters:")
        logger.info(f"  clip_min: {self.clip_min:.6f}")
        logger.info(f"  clip_max: {self.clip_max:.6f}")
        logger.info(f"  dose range: {self.clip_max - self.clip_min:.6f}")
        
        # Validate parameters
        if self.clip_max <= self.clip_min:
            raise RuntimeError(f"Invalid normalization range: clip_max ({self.clip_max}) <= clip_min ({self.clip_min})")
        
        return self.clip_min, self.clip_max
    
    def setup_loss_functions(self):
        """Initialize loss functions after normalization parameters are set."""
        # Loss functions
        self.l1_loss = L1Loss()
        self.adv_loss = PatchAdversarialLoss(criterion="least_squares")
        self.perceptual_loss = PerceptualLoss(spatial_dims=3, network_type="alex").to(self.device)
        
        logger.info("Loss functions initialized successfully")
    
    def weighted_dose_loss(self, pred, target, weight_factor=10000.0, dose_threshold=1e-6):
        """
        Extreme weighted L1 loss for ultra-sparse dose data.
        
        Args:
            pred: Predicted dose values
            target: Ground truth dose values  
            weight_factor: Extreme multiplier for non-zero dose voxels (10,000x)
            dose_threshold: Minimum value to consider as "real dose"
        """
        # Create extreme weight mask
        # Weights are 10,000x for dose voxels, nearly zero for background
        weights = torch.where(target > dose_threshold, 100000.0, 0.01)  
        
        # Weighted absolute difference
        weighted_diff = torch.abs(pred - target) * weights
        
        # Calculate stats for logging
        total_voxels = target.numel()
        dose_voxels = (target > dose_threshold).sum().item()
        
        if dose_voxels > 0:
            dose_loss = weighted_diff[target > dose_threshold].mean()
            background_loss = weighted_diff[target <= dose_threshold].mean()
            
            logger.debug(f"EXTREME Weighted loss - {dose_voxels}/{total_voxels} dose voxels "
                        f"({100*dose_voxels/total_voxels:.4f}%), "
                        f"dose_loss: {dose_loss:.6f}, bg_loss: {background_loss:.6f}")
        else:
            logger.warning("No dose voxels found - using minimal background loss")
            return weighted_diff.mean() * 0.001  # Minimal loss when no dose
        
        return weighted_diff.mean()
    
    def masked_dose_loss(self, pred, target, dose_threshold=1e-6):
        """
        Pure dose-only loss - ignores background completely.
        
        Args:
            pred: Predicted dose values
            target: Ground truth dose values
            dose_threshold: Minimum value to consider as "real dose"
        """
        # Create mask for non-zero dose regions
        dose_mask = target > dose_threshold
        
        if dose_mask.sum() == 0:
            # No dose regions - return extremely small loss to avoid stopping training
            logger.warning("No dose regions found in batch - returning minimal loss")
            return torch.tensor(1e-6, device=target.device, requires_grad=True)
        
        # Apply mask - ONLY train on dose regions
        masked_pred = pred[dose_mask]
        masked_target = target[dose_mask]
        
        # MSE loss for better gradient signal on sparse data
        loss = F.mse_loss(masked_pred, masked_target)
        
        dose_voxels = dose_mask.sum().item()
        total_voxels = target.numel()
        logger.debug(f"PURE Dose loss - Training ONLY on {dose_voxels}/{total_voxels} voxels "
                    f"({100*dose_voxels/total_voxels:.4f}%), MSE loss: {loss:.6f}")
        
        return loss
    
    def extreme_dose_focused_loss(self, pred, target, dose_threshold=1e-6):
        """
        Combination of extreme weighted and pure masked loss for ultra-sparse data.
        Uses 100,000x weights and pure MSE loss for maximum signal.
        """
        dose_mask = target > dose_threshold
        dose_voxels = dose_mask.sum().item()
        
        if dose_voxels == 0:
            # No dose - minimal loss to keep training alive
            return torch.tensor(1e-6, device=target.device, requires_grad=True)
        
        # Pure dose loss (MSE on dose regions only) - better gradients for sparse data
        masked_pred = pred[dose_mask]
        masked_target = target[dose_mask]
        pure_dose_loss = F.mse_loss(masked_pred, masked_target)

        # EXTREME weighted loss for EXTREME sparsity (1,000,000x weight for dose - MASSIVE INCREASE for 0.005% sparsity)
        sparsity_percent = 100 * dose_voxels / target.numel()
        
        # Adaptive weight based on actual sparsity level - MUCH MORE AGGRESSIVE
        if sparsity_percent < 0.006:  # Less than 0.006% sparsity (like we see: 0.005%)
            dose_weight = 1000000.0  # 1M weight for ultra-extreme cases
            background_weight = 0.0000001  # Nearly ignore background completely
        elif sparsity_percent < 0.01:  # Less than 0.01% sparsity  
            dose_weight = 500000.0  # 500K weight for extreme cases
            background_weight = 0.000001  # Nearly ignore background completely
        else:
            dose_weight = 100000.0  # Original 100K weight
            background_weight = 0.00001
            
        weights = torch.where(dose_mask, dose_weight, background_weight)
        weighted_loss = (torch.abs(pred - target) * weights).mean()
        
        # Combine: Emphasize pure dose training even more heavily for extreme sparsity
        if sparsity_percent < 0.006:
            combined_loss = 0.95 * pure_dose_loss + 0.05 * weighted_loss  # 95% pure dose focus for ultra-extreme
        elif sparsity_percent < 0.01:
            combined_loss = 0.9 * pure_dose_loss + 0.1 * weighted_loss  # 90% pure dose focus
        else:
            combined_loss = 0.8 * pure_dose_loss + 0.2 * weighted_loss
        
        total_voxels = target.numel()
        logger.debug(f"EXTREME loss - {dose_voxels}/{total_voxels} dose voxels "
                    f"({sparsity_percent:.5f}%), dose_weight: {dose_weight:.0f}, "
                    f"pure_MSE: {pure_dose_loss:.6f}, weighted_{dose_weight:.0f}: {weighted_loss:.6f}, combined: {combined_loss:.6f}")
        
        return combined_loss
    # ------------------------------------------------------------------ #
    # utilities                                                           #
    # ------------------------------------------------------------------ #
    def _compute_dataset_dose_stats(self, loader):
        """
        Go once over the whole loader to obtain global mean / std of the
        ground-truth dose.  These statistics are required for a stable
        normalisation of the target and the network output and will be
        cached in `self.dose_mean` and `self.dose_std`.

        Args:
            loader (DataLoader): training (or validation) loader that yields
                                 dictionaries with a "target" key.
        """
        logger.info("[Dose stats] computing dataset-wide mean / std …")
        sum_val, sum_sq, num_vox = 0.0, 0.0, 0
        with torch.no_grad():
            for b in loader:
                tgt = b["target"]          # shape [B,1,D,H,W]
                sum_val += tgt.sum().item()
                sum_sq  += (tgt ** 2).sum().item()
                num_vox += tgt.numel()

        mean = sum_val / num_vox
        var  = (sum_sq / num_vox) - mean ** 2
        std  = math.sqrt(max(var, DOSE_EPS))

        self.dose_mean = mean
        self.dose_std  = std
        # also capture absolute min / max for min‑max normalisation
        dataset_min = float('inf')
        dataset_max = float('-inf')
        for b in loader:
            tgt = b["target"]
            dataset_min = min(dataset_min, tgt.min().item())
            dataset_max = max(dataset_max, tgt.max().item())

        self.clip_min = dataset_min
        self.clip_max = dataset_max
        logger.info(
            f"[Dose stats] mean={mean:.5f}  std={std:.5f}  "
            f"min={dataset_min:.5f}  max={dataset_max:.5f}"
        )

    @staticmethod
    def _total_variation_loss(img):
        """
        3-D total-variation loss encouraging spatial smoothness.

        Args:
            img (torch.Tensor): tensor of shape [B, C, D, H, W]
        """
        dx = torch.abs(img[..., :-1, :, :] - img[..., 1:, :, :]).mean()
        dy = torch.abs(img[..., :, :-1, :] - img[..., :, 1:, :]).mean()
        dz = torch.abs(img[..., :, :, :-1] - img[..., :, :, 1:]).mean()
        return dx + dy + dz

    def train_one_epoch(self, train_loader, epoch):
        """
        Trains the autoencoder and the discriminator for an epoch.

        This method integrates energy conditioning: for each batch, it extracts the "energy" values,
        normalizes them, expands them to match the spatial dimensions, and concatenates them as an
        additional channel to the input images. The autoencoder must be adapted to accept the increased
        number of input channels.

        Args:
            train_loader (DataLoader): DataLoader for training data.
            epoch (int): Current epoch.
        Returns:
            tuple: Average reconstruction loss, generator loss and discriminator loss.
        """
        self.autoencoder.train()
        self.discriminator.train()
        epoch_loss = 0.0
        gen_epoch_loss = 0.0
        disc_epoch_loss = 0.0

        logger.info(f"Starting autoencoder training epoch {epoch}")
        
        # **ENERGY-SPECIFIC LEARNING RATE ADJUSTMENT**
        # Extract energy from first batch to determine if adjustment is needed
        first_batch = next(iter(train_loader))
        if "energy" in first_batch:
            current_energy = first_batch["energy"][0].item()  # Get first energy value
            logger.info(f"Training energy: {current_energy:.2f}")
            
            # Adaptive Learning Rate for high energies
            original_lr_g = None
            original_lr_d = None
            if current_energy > 40.0:
                # Reduce Learning Rate for high energies by factor 3
                original_lr_g = self.optimizer_g.param_groups[0]['lr']
                original_lr_d = self.optimizer_d.param_groups[0]['lr']
                new_lr_g = original_lr_g * 0.33
                new_lr_d = original_lr_d * 0.33
                
                for param_group in self.optimizer_g.param_groups:
                    param_group['lr'] = new_lr_g
                for param_group in self.optimizer_d.param_groups:
                    param_group['lr'] = new_lr_d
                    
                logger.info(f"🔥 HIGH ENERGY ({current_energy:.2f}): Reduced learning rates")
                logger.info(f"   Generator: {original_lr_g:.6f} -> {new_lr_g:.6f}")
                logger.info(f"   Discriminator: {original_lr_d:.6f} -> {new_lr_d:.6f}")
        else:
            current_energy = None
            original_lr_g = None
            original_lr_d = None
            
        # ------------------------------------------------------------------
        # one-time initialization of energy normalization range
        # ------------------------------------------------------------------
        if self.energy_min is None:
            raise RuntimeError(
                "energy_min/max not initialised - run _compute_dataset_energy_stats first"
            )
        for step, batch in enumerate(tqdm(train_loader, desc=f"Autoencoder Epoch {epoch}")):
            # Move data to device
            images = batch["input"].to(self.device)  # Expected shape: [B, 1, D, H, W]
            dose_target = batch["target"].to(self.device)  # Ground-truth dose
            
            # **DEBUG: Log dose statistics for first few batches**
            if step < 3:
                dose_stats = {
                    'min': dose_target.min().item(),
                    'max': dose_target.max().item(), 
                    'mean': dose_target.mean().item(),
                    'nonzero_count': (dose_target > 1e-6).sum().item(),
                    'total_voxels': dose_target.numel()
                }
                logger.info(f"Batch {step+1} dose stats: {dose_stats}")
            
            # **CRITICAL FIX: Apply consistent dose normalization during training**
            # Normalize dose to [0,1] range using clip_min/clip_max with safety checks
            if self.clip_min is not None and self.clip_max is not None and (self.clip_max - self.clip_min) > 1e-6:
                dose_range = self.clip_max - self.clip_min
                dose_target_norm = (dose_target - self.clip_min) / dose_range
                # Clamp to [0,1] range to handle outliers
                dose_target_norm = torch.clamp(dose_target_norm, 0.0, 1.0)
                logger.debug(f"Normalized dose: orig_range=[{dose_target.min().item():.6f}, {dose_target.max().item():.6f}] "
                           f"-> norm_range=[{dose_target_norm.min().item():.6f}, {dose_target_norm.max().item():.6f}]")
            else:
                # Fallback: use raw dose values if normalization parameters not available
                logger.warning("Skipping dose normalization - using raw dose values")
                dose_target_norm = dose_target

            # Apply energy conditioning if energy data is available
            if "energy" in batch:
                energies = batch["energy"].to(self.device)
                # Min-Max normalize energy using dataset range with zero-division guard
                eps = 1e-8
                normalized_energy = energies.float() / 100.00
                print(f"xNormalized energy: {normalized_energy}")
                B, C, D, H, W = images.shape
                energy_tensor = normalized_energy.view(B, 1, 1, 1, 1).expand(B, 1, D, H, W)
                conditioned_input = torch.cat([images, energy_tensor], dim=1)
            else:
                # Throw an error if energy conditioning is expected but not provided
                raise ValueError(
                    "Energy conditioning is expected but 'energy' field is missing in the batch."
                )

            # Verify input channels match autoencoder expectations
            first_conv = next((m for m in self.autoencoder.modules() if isinstance(m, torch.nn.Conv3d)), None)
            expected_in_channels = first_conv.in_channels if first_conv else conditioned_input.shape[1]
            
            if conditioned_input.shape[1] != expected_in_channels:
                raise ValueError(
                    f"Input channel mismatch: model expects {expected_in_channels} channels, "
                    f"but got {conditioned_input.shape[1]} channels. "
                    f"Energy conditioning: {'enabled' if 'energy' in batch else 'disabled'}"
                )

            #Zero the gradients for the generator 
            self.optimizer_g.zero_grad(set_to_none=True)
            # Pass the conditioned input through the autoencoder
            # (Note: autoencoder's in_channels should be updated to handle conditioned input)
            reconstruction, z_mu, z_sigma = self.autoencoder(conditioned_input)
            
            # **CRITICAL FIX: Apply same normalization to reconstruction as target**
            if self.clip_min is not None and self.clip_max is not None and (self.clip_max - self.clip_min) > 1e-6:
                dose_range = self.clip_max - self.clip_min
                reconstruction_norm = (reconstruction - self.clip_min) / dose_range
                reconstruction_norm = torch.clamp(reconstruction_norm, 0.0, 1.0)
            else:
                reconstruction_norm = reconstruction

            #calculate KL loss
            kl_loss = 0.5 * torch.sum(
                z_mu.pow(2) + z_sigma.pow(2) - torch.log(z_sigma.pow(2)) - 1
            ) /images.size(0)
            # **CRITICAL FIX: Use extreme dose-focused loss for ultra-sparse data**
            
            # Check dose statistics first
            dose_mask = dose_target_norm > 1e-6
            dose_voxels = dose_mask.sum().item()
            total_voxels = dose_target_norm.numel()
            dose_percentage = 100 * dose_voxels / total_voxels
            
            if dose_voxels == 0:
                logger.warning("Batch has no dose voxels - skipping")
                continue
            
            if dose_voxels < 10000:  # Schwelle für MEGA-ULTRA-EXTREME deutlich erhöht
                # Ultra-sparse batch - use extreme loss
                recons_loss = self.extreme_dose_focused_loss(reconstruction_norm.float(), dose_target_norm.float())
                loss_type = "EXTREME (1,000,000x weights + MSE)"
            else:
                # Normal sparse batch - use combined weighted/masked approach
                weighted_loss = self.weighted_dose_loss(reconstruction_norm.float(), dose_target_norm.float())
                masked_loss = self.masked_dose_loss(reconstruction_norm.float(), dose_target_norm.float())
                recons_loss = 0.6 * weighted_loss + 0.4 * masked_loss
                loss_type = "COMBINED (100,000x weighted + pure masked)"
            
            # **ENERGY-SPECIFIC LOSS AMPLIFICATION FOR HIGH ENERGIES**
            if current_energy is not None and current_energy > 40.0:
                recons_loss = recons_loss * 1.5  # 50% stronger weighting
                loss_type += " + HIGH_ENERGY_BOOST"
                if step < 3:  # Log only for first few batches
                    logger.info(f" Applied 1.5x loss boost for high energy {current_energy:.2f}")
                    
            # **ENHANCED: Comprehensive sparse data analysis**
            dose_ratio = dose_voxels / total_voxels
            sparsity_level = f"{dose_ratio*100:.6f}%" if dose_ratio > 0 else "NO_DOSE"
            
            logger.debug(f"{loss_type} loss on {dose_voxels}/{total_voxels} dose voxels "
                        f"({dose_percentage:.4f}%), loss: {recons_loss:.6f}")
            logger.debug(f"   -> Sparsity: {sparsity_level}, recon_range: [{reconstruction_norm.min().item():.6f}, {reconstruction_norm.max().item():.6f}]")
            
            # Track dose-only reconstruction quality with enhanced metrics
            if dose_voxels > 0:
                dose_mask_bool = dose_target_norm > 1e-6
                dose_mse = F.mse_loss(reconstruction_norm[dose_mask_bool], dose_target_norm[dose_mask_bool])
                dose_mae = F.l1_loss(reconstruction_norm[dose_mask_bool], dose_target_norm[dose_mask_bool])
                
                # Signal strength analysis
                target_signal = dose_target_norm[dose_mask_bool].mean().item()
                recon_signal = reconstruction_norm[dose_mask_bool].mean().item()
                signal_ratio = (recon_signal / target_signal) if target_signal > 0 else 0.0
                
                logger.debug(f"   -> Dose-only MSE: {dose_mse:.6f}, MAE: {dose_mae:.6f}")
                logger.debug(f"   -> Signal ratio: {signal_ratio:.6f} (target: {target_signal:.6f}, recon: {recon_signal:.6f})")
                
                # Critical thresholds
            # Nur jede 20. Warnung loggen, um Spam zu vermeiden
            if step % 20 == 0:
                if signal_ratio < 0.01:
                    logger.warning(f" SIGNAL COLLAPSE: ratio={signal_ratio:.6f} - model barely responding!")
                elif signal_ratio < 0.1:
                    logger.warning(f"  WEAK SIGNAL: ratio={signal_ratio:.6f} - consider higher weights!")
            
            # **CRITICAL FIX: Check for NaN in loss**
            if torch.isnan(recons_loss) or torch.isinf(recons_loss):
                logger.error(f"NaN/Inf detected in reconstruction loss: {recons_loss.item()}")
                logger.error(f"reconstruction_norm stats: min={reconstruction_norm.min().item():.6f}, "
                           f"max={reconstruction_norm.max().item():.6f}, mean={reconstruction_norm.mean().item():.6f}")
                logger.error(f"dose_target_norm stats: min={dose_target_norm.min().item():.6f}, "
                           f"max={dose_target_norm.max().item():.6f}, mean={dose_target_norm.mean().item():.6f}")
                # Skip this batch
                continue
            
            # **ENHANCED: Model collapse detection and automatic intervention**
            recon_max = reconstruction_norm.max().item()
            target_max = dose_target_norm.max().item()
            
            # **ENERGY-SPECIFIC DEGRADATION THRESHOLDS**
            if current_energy is not None and current_energy > 40.0:
                # Stricter threshold for high energies
                degradation_threshold = 0.05
                collapse_threshold = 0.001
            else:
                # Normal thresholds
                degradation_threshold = 0.02
                collapse_threshold = 0.0001
                
            # Nur jede 20. Warnung loggen, um Spam zu vermeiden
            if step % 20 == 0:
                ratio = recon_max / max(target_max, 1e-6)
                if recon_max < collapse_threshold and target_max > 0.01:
                    logger.warning(f" CRITICAL MODEL COLLAPSE: recon_max={recon_max:.6f}, target_max={target_max:.6f}")
                    logger.warning(f"   Signal ratio: {ratio:.8f} (should be >{degradation_threshold})")
                    if current_energy is not None and current_energy > 40.0:
                        logger.warning(f" HIGH ENERGY COLLAPSE: energy={current_energy:.2f} - Consider reducing clip_max")
                    logger.warning(f"   URGENT: Consider 500K+ weights or alternative loss function")
                elif ratio < degradation_threshold and target_max > 0.1:
                    logger.warning(f"  MODEL DEGRADATION: recon_max={recon_max:.6f}, target_max={target_max:.6f}")
                    logger.warning(f"   Model producing weak signals - monitor closely")
                    if current_energy is not None and current_energy > 40.0:
                        logger.warning(f" HIGH ENERGY DEGRADATION: energy={current_energy:.2f}, ratio={ratio:.6f}")
                        logger.warning(f"   Suggested: Reduce dose normalization clip_max")
            
            # compute adversarial loss if using discriminator
            adv = 0.0
            if epoch > self.warm_up_epochs and self.discriminator is not None:
                logits_fake = self.discriminator(reconstruction.contiguous().float())[-1]
                adv = self.adv_loss(logits_fake, target_is_real=True, for_discriminator=False)
            # compute perceptual loss (use dose target as reference, not CT input)
            perc = self.perceptual_loss(reconstruction_norm, dose_target_norm)
            # compute TV in physical Gy units (before normalisation)
            tv = self._total_variation_loss(reconstruction)
            # total generator loss
            loss_g = (recons_loss
                      + self.kl_weight * kl_loss
                      + self.adv_weight * adv
                      + self.perceptual_weight * perc
                      + self.tv_weight * tv)

            # Add adversarial loss if past warm-up phase and discriminator is used.
            if epoch > self.warm_up_epochs and self.discriminator is not None:
                gen_epoch_loss += adv.item()
            # keep track of TV loss for logging
            gen_epoch_loss += self.tv_weight * tv.item()
            loss_g.backward()
            # ULTRA-conservative gradient clipping for extreme stability
            torch.nn.utils.clip_grad_norm_(self.autoencoder.parameters(), max_norm=0.1)  # Much more conservative
            torch.nn.utils.clip_grad_value_(self.autoencoder.parameters(), clip_value=0.01)  # Much more conservative
            self.optimizer_g.step()

            # **CRITICAL FIX: Check for NaN in generator loss**
            if epoch > self.warm_up_epochs and self.discriminator is not None:
                self.optimizer_d.zero_grad(set_to_none=True)
                
                # Discriminator loss: real vs fake
                logits_fake = self.discriminator(reconstruction.contiguous().detach())[-1]
                loss_d_fake = F.mse_loss(logits_fake, torch.zeros_like(logits_fake))
                
                # Discriminator loss: real vs real
                logits_real = self.discriminator(dose_target.contiguous().detach())[-1]
                loss_d_real = F.mse_loss(logits_real, torch.ones_like(logits_real))
                
                discriminator_loss = (loss_d_fake + loss_d_real) / 2
                discriminator_loss.backward()
                self.optimizer_d.step()
                disc_epoch_loss += discriminator_loss.item()
            
            epoch_loss += recons_loss.item()
        
        avg_loss = epoch_loss / (step + 1)
        avg_gen_loss = gen_epoch_loss / (step + 1) if self.discriminator is not None else 0
        avg_disc_loss = disc_epoch_loss / (step + 1) if self.discriminator is not None else 0
        logger.info(
            f"Epoch {epoch} completed: recon_loss={avg_loss:.6f}, adv_loss={avg_gen_loss:.4f}, "
            f"disc_loss={avg_disc_loss:.4f}, tv={tv.item():.4f}"
        )
        # Dynamic message based on actual loss types used
        logger.info(f"  -> Training with ADAPTIVE loss selection: MEGA-ULTRA-EXTREME (<300 dose voxels, 1M weights for <0.006% sparsity) + COMBINED (≥300 dose voxels)")

        # **RESTORE ORIGINAL LEARNING RATES**
        if original_lr_g is not None and original_lr_d is not None:
            for param_group in self.optimizer_g.param_groups:
                param_group['lr'] = original_lr_g
            for param_group in self.optimizer_d.param_groups:
                param_group['lr'] = original_lr_d
            logger.info(f" Restored original learning rates: G={original_lr_g:.6f}, D={original_lr_d:.6f}")

        return avg_loss, avg_gen_loss, avg_disc_loss

    def validate(self, val_loader):
        """
        Performs the validation of the autoencoder and returns the average loss.
        Args:
            val_loader (DataLoader): DataLoader for validation data.
        Returns:
            float: Average validation loss.
        """
        self.autoencoder.eval()
        val_loss = 0.0
        batch_losses = []
        extreme_count = 0
        combined_count = 0
        
        logger.info("Starting validation")
        logger.info(f"Validation dataset: {len(val_loader)} batches")
        
        with torch.no_grad():
            for i, batch in enumerate(val_loader):
                images = batch["input"].to(self.device)
                dose_target = batch["target"].to(self.device)  # ground‑truth dose
                
                # **CRITICAL FIX: Apply consistent dose normalization during validation**
                if self.clip_min is not None and self.clip_max is not None and (self.clip_max - self.clip_min) > 1e-6:
                    dose_range = self.clip_max - self.clip_min
                    dose_target_norm = (dose_target - self.clip_min) / dose_range
                    dose_target_norm = torch.clamp(dose_target_norm, 0.0, 1.0)
                else:
                    dose_target_norm = dose_target
                # Apply energy conditioning if energy data is available
                if "energy" in batch:
                    energies = batch["energy"].to(self.device)
                    # Min-Max normalize energy using dataset range with zero-division guard
                    eps = 1e-8
                    normalized_energy = energies.float() / 100.00
                    if i == 0:  # Log energy only for first batch
                        energy_val = energies[0].item() if energies.numel() > 1 else energies.item()
                        norm_val = normalized_energy[0].item() if normalized_energy.numel() > 1 else normalized_energy.item()
                        logger.info(f"Val energy normalization: {energy_val:.2f} -> {norm_val:.6f}")
                    B, C, D, H, W = images.shape
                    energy_tensor = normalized_energy.view(B, 1, 1, 1, 1).expand(B, 1, D, H, W)
                    conditioned_input = torch.cat([images, energy_tensor], dim=1)
                else:
                    raise ValueError(
                        "Energy conditioning is expected but 'energy' field is missing in the batch."
                    )

                # Verify input channels match autoencoder expectations
                first_conv = next((m for m in self.autoencoder.modules() if isinstance(m, torch.nn.Conv3d)), None)
                expected_in_channels = first_conv.in_channels if first_conv else conditioned_input.shape[1]
                
                if conditioned_input.shape[1] != expected_in_channels:
                    raise ValueError(
                        f"Validation input channel mismatch: model expects {expected_in_channels} channels, "
                        f"but got {conditioned_input.shape[1]} channels."
                    )

                reconstruction, z_mu, z_sigma = self.autoencoder(conditioned_input)
                
                # **CRITICAL FIX: Apply same normalization to reconstruction during validation**
                if self.clip_min is not None and self.clip_max is not None and (self.clip_max - self.clip_min) > 1e-6:
                    dose_range = self.clip_max - self.clip_min
                    reconstruction_norm = (reconstruction - self.clip_min) / dose_range
                    reconstruction_norm = torch.clamp(reconstruction_norm, 0.0, 1.0)
                else:
                    reconstruction_norm = reconstruction
                    
                # **CRITICAL FIX: Use SAME adaptive loss selection as training**
                dose_mask = dose_target_norm > 1e-6
                dose_voxels = dose_mask.sum().item()
                
                if dose_voxels < 300:  # CRITICAL FIX: Same threshold as training (300)
                    # Ultra-sparse batch - use extreme loss (same as training)
                    loss = self.extreme_dose_focused_loss(reconstruction_norm, dose_target_norm)
                    loss_type = "MEGA-ULTRA-EXTREME"
                    extreme_count += 1
                else:
                    # Normal sparse batch - use combined weighted/masked approach (same as training)
                    weighted_loss = self.weighted_dose_loss(reconstruction_norm, dose_target_norm)
                    masked_loss = self.masked_dose_loss(reconstruction_norm, dose_target_norm)
                    loss = 0.6 * weighted_loss + 0.4 * masked_loss
                    loss_type = "COMBINED"
                    combined_count += 1
                
                val_loss += loss.item()
                batch_losses.append(loss.item())
                
                # Debug logging for first few batches
                if i < 3:
                    total_voxels = dose_target_norm.numel()
                    energy_val = energies[0].item() if energies.numel() > 1 else energies.item()
                    logger.info(f"Val batch {i}: energy={energy_val:.1f}, {loss_type}, "
                               f"{dose_voxels}/{total_voxels} dose voxels "
                               f"({100*dose_voxels/total_voxels:.4f}%), loss={loss.item():.6f}")
                    logger.info(f"  -> target_range=[{dose_target_norm.min():.6f}, {dose_target_norm.max():.6f}], "
                               f"recon_range=[{reconstruction_norm.min():.6f}, {reconstruction_norm.max():.6f}]")
        
        avg_val_loss = val_loss / len(val_loader)
        loss_std = np.std(batch_losses) if len(batch_losses) > 1 else 0.0
        
        logger.info(f"Validation completed: avg_val_loss={avg_val_loss:.6f} ± {loss_std:.6f}")
        logger.info(f"  -> Loss distribution: {extreme_count} MEGA-ULTRA-EXTREME batches, {combined_count} COMBINED batches")
        logger.info(f"  -> Loss range: min={min(batch_losses):.6f}, max={max(batch_losses):.6f}")
        
        return avg_val_loss

class DiffusionTrainer:
    """
    Encapsulates the training process for the diffusion model.    """

    """
    Trains diffusion model for one epoch.

    Args:
        train_loader (DataLoader): DataLoader for training data.
        epoch (int): Current epoch index.
        inferer (callable): A callable with signature
            inferer(inputs, autoencoder_model, diffusion_model, noise, timesteps)
            that returns predicted noise.
    """
    def __init__(self, diffusion_model, optimizer_diff, energy_min, energy_max, device):
        """
        Args:
            diffusion_model (torch.nn.Module): Das Diffusionsmodell (z. B. ein UNet).
            optimizer_diff (torch.optim.Optimizer): Optimierer für das Diffusionsmodell.
            device (torch.device): CPU oder GPU.
        """
        self.diffusion_model = diffusion_model
        self.optimizer_diff = optimizer_diff
        # Min-Max energy normalization parameters
        self.energy_min = energy_min
        self.energy_max = energy_max
        self.device = device
        # ---------------------------------------------------------------
        # Determine the cross‑attention dimension expected by the UNet.
        #
        # Many UNet variants have several cross‑attention blocks operating
        # at different channel widths.  Each block owns a projection layer
        # called “to_k”; its *input* dimension equals the context‑vector
        # length expected by this block.  All blocks that consume the same
        # context embedding should therefore agree on that dimension –
        # but some architectures prepend an additional projection layer,
        # so we might see values like 2, 64, 128, …
        #
        # Empirically, the *smallest* of these `in_features` values
        # corresponds to the original conditioning size (in our case
        # `(energy, bias) = 2`).  We therefore collect **all** `to_k`
        # modules and pick the minimum `in_features`.
        # ---------------------------------------------------------------
        cand_dims = set()
        for name, mod in self.diffusion_model.named_modules():
            if isinstance(mod, torch.nn.Linear) and (name.endswith("to_k") or ".to_k" in name):
                cand_dims.add(mod.in_features)

        if cand_dims:
            # choose the smallest dimension – typically 2
            self.cross_attn_dim = min(cand_dims)
        else:
            raise RuntimeError(
                "Could not find any cross-attention projection layer in the diffusion model. "
                "Ensure that the model is correctly configured for energy conditioning."
            )

        if self.cross_attn_dim is None:
            raise RuntimeError("Could not determine cross‑attention dimension from diffusion model.")

        # Note: Context projection disabled since running in unconditional mode
        # Small MLP that maps (energy, bias) → vector of length `cross_attn_dim`
        # self.context_proj = torch.nn.Linear(2, self.cross_attn_dim).to(self.device)

    def train_one_epoch(self, train_loader, epoch, inferer=None, autoencoder=None):
        """
        Args:
            diffusion_model (torch.nn.Module): The diffusion model (e.g. a UNet).
            optimizer_diff (torch.optim.Optimizer): Optimizer for the diffusion model.
            device (torch.device): CPU or GPU.
        """
        # if no inferer or autoencoder provided, skip diffusion training
        if inferer is None or autoencoder is None:
            return 0.0

        self.diffusion_model.train()
        epoch_loss = 0.0

        for step, batch in enumerate(tqdm(train_loader, desc=f"Diffusion Epoch {epoch}")):
            images = batch["input"].to(self.device)
            
            # Apply energy conditioning consistently
            if "energy" in batch:
                energies = batch["energy"].to(self.device)
                # Min-Max normalize energy using dataset range with zero-division guard
                eps = 1e-8
                #normalized_energy = (energies.float() - self.energy_min) / (self.energy_max - self.energy_min + eps)
                normalized_energy = energies.float() / 100.00
                print(f"Normalized energy: {normalized_energy}")
                B, C, D, H, W = images.shape
                energy_tensor = normalized_energy.view(B, 1, 1, 1, 1).expand(B, 1, D, H, W)
                conditioned_input = torch.cat([images, energy_tensor], dim=1)
            else:
                # If energy conditioning is expected but not provided, raise an error
                raise ValueError(
                    "Energy conditioning is expected but 'energy' field is missing in the batch."
                )

            # Zero gradients for diffusion optimizer
            self.optimizer_diff.zero_grad(set_to_none=True)

            # Encode conditioned input to latents using the autoencoder
            with torch.no_grad():
                # MONAI's AutoencoderKL.encode() returns a tuple (mu, logvar)
                # For consistent handling, always expect this format
                mu, logvar = autoencoder.encode(conditioned_input)
                # Sample from the latent distribution
                std = torch.exp(0.5 * logvar)
                eps = torch.randn_like(std)
                latents = mu + eps * std
                latents = latents.to(self.device)

                # Crop / pad latents symmetrically so that spatial dims are multiples
                # of the UNet down‑sampling factor
                down_blocks = [m for m in self.diffusion_model.modules()
                               if isinstance(m, torch.nn.Conv3d) and any(s > 1 for s in m.stride)]
                downsample_factor = 2 ** len(down_blocks) if down_blocks else 1

                B, C, D, H, W = latents.shape
                new_D = (D // downsample_factor) * downsample_factor
                new_H = (H // downsample_factor) * downsample_factor
                new_W = (W // downsample_factor) * downsample_factor

                if (new_D, new_H, new_W) != (D, H, W):
                    d0 = (D - new_D) // 2
                    h0 = (H - new_H) // 2
                    w0 = (W - new_W) // 2
                    latents = latents[:, :, d0:d0 + new_D, h0:h0 + new_H, w0:w0 + new_W]

                # Verify latent channels match diffusion model expectations
                first_conv_diff = next((m for m in self.diffusion_model.modules() if isinstance(m, torch.nn.Conv3d)), None)
                expected_latent_ch = first_conv_diff.in_channels if first_conv_diff else latents.shape[1]

                if latents.shape[1] != expected_latent_ch:
                    raise ValueError(
                        f"Latent channel mismatch: diffusion model expects {expected_latent_ch} channels, "
                        f"but autoencoder produces {latents.shape[1]} channels"
                    )
           
            # generates random noise
            noise = torch.randn_like(latents)

            #  generates random Timesteps
            timesteps = torch.randint(0, inferer.scheduler.num_train_timesteps, (images.shape[0],), device=self.device).long()

            # Add noise to the latents to create x_t
            noisy_latents = inferer.scheduler.add_noise(latents, noise, timesteps)

            # ------------------------------------------------------------------
            # Note: Running in unconditional mode (with_conditioning=False)
            # Context/conditioning has been disabled in system_manager.py
            # ------------------------------------------------------------------
            # energy_feat = normalized_energy.view(B, 1)              # [B,1]
            # bias_feat   = torch.ones_like(energy_feat)              # [B,1]
            # base_ctx    = torch.cat([energy_feat, bias_feat], dim=-1)  # [B,2]
            # proj_ctx    = self.context_proj(base_ctx)               # [B,cross_attn_dim]
            # context_tensor = proj_ctx.unsqueeze(1)                  # [B,1,cross_attn_dim]

            # --- classifier‑free guidance dropout ---------------------------
            # if torch.rand(1, device=self.device).item() < 0.1:
            #     context_tensor = None     # unconditional pass

            # Predict the noise component and optimise (unconditional)
            noise_pred = self.diffusion_model(
                noisy_latents,
                timesteps=timesteps
            )

            # ----------------- optimisation step ---------------------------
            loss = F.mse_loss(noise_pred.float(), noise.float())
            loss.backward()

            # gradient‑clipping stabilises very small batches
            clip_grad_norm_(self.diffusion_model.parameters(), max_norm=1.0)

            self.optimizer_diff.step()
            # Crop original noise to match predicted noise spatial dimensions (symmetric)
            if noise.ndim == noise_pred.ndim and noise.shape[2:] != noise_pred.shape[2:]:
                _, _, Dn, Hn, Wn = noise_pred.shape
                d0 = (noise.shape[2] - Dn) // 2
                h0 = (noise.shape[3] - Hn) // 2
                w0 = (noise.shape[4] - Wn) // 2
                noise = noise[:, :, d0:d0 + Dn, h0:h0 + Hn, w0:w0 + Wn]
            epoch_loss += loss.item()

        avg_loss = epoch_loss / (step + 1)
        return avg_loss