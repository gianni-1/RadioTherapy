# system_manager.py

import shutil
import os
import torch
import log_config
import logging
import glob
import json
import numpy as np
logger = logging.getLogger(__name__)

from data_management import DataLoaderModule, HotspotPatchDataset
from training_pipeline import AutoencoderTrainer, DiffusionTrainer, EarlyStopping
from inference_module import InferenceModule
from monai.transforms import Compose, LoadImaged, EnsureChannelFirstd, ToTensord
from visualization import Visualization
from monai.utils import first, set_determinism
from monai.data import NibabelReader
from generative.networks.nets import AutoencoderKL, PatchDiscriminator, DiffusionModelUNet
from generative.inferers import LatentDiffusionInferer
from torch.optim import Adam
from torch.amp import autocast
from generative.networks.schedulers.ddpm import DDPMScheduler
from monai.transforms import (
    Compose, LoadImaged, EnsureChannelFirstd, Lambdad,
    EnsureTyped, Orientationd, Resized, SpatialPadd,
    CenterSpatialCropd, ScaleIntensityRangePercentilesd, ToTensord,
)
from monai.data import NumpyReader

class SystemManager:
    """
    The SystemManager class orchestrates the complete training and inference workflow.
    
    It iterates over each combination of spatial resolution and energy level:
      - Loads the dataset using the DataLoaderModule.
      - Splits the data into training and validation sets.
      - Initializes and trains the autoencoder (and optionally diffusion model) 
        using early stopping based on validation loss.
      - Sets a training-completion flag for subsequent inference.
    """
    def __init__(self, root_dir, transforms, resolutions, energies, energy_min, energy_max, quad_energies, quad_weights, batch_size, device, num_epochs, learning_rate, patience, cube_size, seed=42):
        """
        Initializes the SystemManager with configuration parameters.

        Args:
            root_dir (str): Root directory of the dataset.
            transforms (monai.transforms.Compose): Preprocessing transforms to apply.
            resolutions (list of tuple): List of resolutions (e.g., [(64, 64, 64), (32, 32, 32)]).
            energies (list of int): List of energy levels (e.g., [62, 75, 90]).
            batch_size (int): Batch size for DataLoader.
            device (torch.device): Device (CPU or GPU) for computations.
            seed (int): Random seed for reproducibility.
        """
        self.root_dir = root_dir
        self.transforms = transforms
        self.resolutions = resolutions
        self.energies = energies
        self.energy_min = energy_min
        self.energy_max = energy_max
        self.batch_size = batch_size
        self.device = device
        self.seed = seed
        self.num_epochs = num_epochs
        self.learning_rate = learning_rate
        self.patience = patience
        self.training_complete = False
        self.stop_training = False
        # placeholders for trained models
        self.autoencoder = None
        self.unet = None
        self.scheduler = None
        self.cube_size = cube_size
        self.quad_energies = quad_energies
        self.quad_weights = quad_weights
        #keep models per energy for quadrature inference
        self.models_by_energy = {}
        # track individual checkpoints in memory
        self.saved_ckpts = {}

    def save_models(self, autoencoder, unet, optimizer_diff, optimizer_g,
                    optimizer_d, epoch, res, energy,
                    dose_mean=None, dose_std=None,
                    scale_factor=None, clip_min=0.0, clip_max=None,
                    smoothing_kernel=0):
        # build checkpoint dict in memory (no file I/O)
        ckpt = {
            "autoencoder": autoencoder.state_dict(),
            "unet": unet.state_dict(),
            "optimizer_diff": optimizer_diff.state_dict(),
            "optimizer_g": optimizer_g.state_dict(),
            "optimizer_d": optimizer_d.state_dict(),
            "epoch": epoch,
            "scale_factor": scale_factor,
            "clip_min": clip_min,
            "clip_max": clip_max,
            "smoothing_kernel": smoothing_kernel,
            "dose_mean": dose_mean,
            "dose_std": dose_std,
            "resolution": res,
            "energy": energy,
            "learning_rate": self.learning_rate,
            "batch_size": self.batch_size,
            "cube_size": self.cube_size,
        }
        # build a resolution string for the key
        if isinstance(res, (tuple, list)):
            res_str = "x".join(str(r) for r in res)
        else:
            res_str = str(res)
        key = f"res{res_str}_e{energy:.2f}"
        self.saved_ckpts[key] = ckpt

    def save_combined_checkpoint(self, output_filename="combined_models.ckpt"):
        """
        Save all in-memory checkpoints into a single combined file.
        """
        # Extract dose normalization parameters for each energy/resolution combination
        dose_normalization_params = {}
        
        for key, ckpt in self.saved_ckpts.items():
            # Extract energy and resolution from key (e.g., "res25_e46.53")
            if 'clip_min' in ckpt and 'clip_max' in ckpt and 'energy' in ckpt:
                dose_normalization_params[key] = {
                    'clip_min': ckpt['clip_min'],
                    'clip_max': ckpt['clip_max'],
                    'energy': ckpt['energy'],
                    'resolution': ckpt.get('resolution', None),
                    'scale_factor': ckpt.get('scale_factor', 1.0),
                    'dose_mean': ckpt.get('dose_mean', None),
                    'dose_std': ckpt.get('dose_std', None)
                }
                logger.info(f"✓ Dose params for {key}: clip_min={ckpt['clip_min']:.6f}, clip_max={ckpt['clip_max']:.6f}")
        
        combined = {
            "models_by_energy": self.saved_ckpts,
            "dose_normalization_params": dose_normalization_params
        }
        output_path = os.path.join(self.root_dir, output_filename)
        torch.save(combined, output_path)
        logger.info(f"✓ Combined checkpoint saved to {output_path}")
        logger.info(f"✓ Saved dose normalization parameters for {len(dose_normalization_params)} energy/resolution combinations")
        return output_path

    def run_training(self):
        """
        Executes the full training pipeline over all resolution and energy combinations.
        
        For each combination, the following steps are executed:
          1. Load and split the dataset filtered by the current energy.
          2. Create DataLoaders for training and validation.
          3. Initialize the autoencoder, discriminator, and diffusion models.
          4. Run the autoencoder training loop with early stopping based on validation loss.
          5. Run the diffusion model training loop.
          6. Save the models for the current configuration.
        
        After all configurations have been processed, a flag is set to indicate that training is complete.
        """
        # Print hyperparameters for verification
        logger.info("Starting CORRECTED training with hyperparameters:")
        logger.info(f"  batch_size={self.batch_size}, num_epochs={self.num_epochs}, learning_rate={self.learning_rate}, patience={self.patience}, cube_size={self.cube_size}")
        logger.info(f"  resolutions={self.resolutions}, energies={self.energies}")
        
        # CORRECTED TRAINING STRATEGY: Use single resolution (64x64x64) for all energies
        target_resolution = (64, 64, 64)  # Fixed single resolution for stability
        logger.info(f"🔧 CORRECTED: Using SINGLE resolution {target_resolution} for ALL energies (instead of multi-resolution)")
        logger.info(f"🔧 CORRECTED: Energy Loop -> Resolution Loop (proper architecture)")
        
        # instantiate models - before the loop to avoid re-instantiation
        autoencoder = AutoencoderKL(
            spatial_dims=3,
            in_channels=2,
            out_channels=1,
            num_channels=(32, 32, 32),
            latent_channels=2,
            num_res_blocks=1,
            norm_num_groups=8,
            attention_levels=(False, False, True),
        ).to(self.device)
        discriminator = PatchDiscriminator(
            spatial_dims=3,
            num_layers_d=3, 
            num_channels=32, 
            in_channels=1, 
            out_channels=1
        ).to(self.device)
        unet = DiffusionModelUNet(
            spatial_dims=3,
            in_channels=2,
            out_channels=2,
            with_conditioning=False,  # Disabled cross-attention conditioning
            num_res_blocks=1,
            num_channels=(32, 64, 64),
            attention_levels=(False, True, True),
            num_head_channels=(0, 64, 64),
        ).to(self.device)

        # initialize history lists for plotting
        ae_train_losses = []
        ae_val_losses   = []
        gen_losses      = []
        disc_losses     = []
        diff_losses     = []
        
        # CORRECTED TRAINING LOOP: Energy as Outer Loop, Single Resolution
        for energy in self.energies:
            if self.stop_training:
                logger.info("Training aborted by user.")
                return
            logger.info(f"🔧 --- CORRECTED: Training energy={energy} eV at resolution={target_resolution} ---")
            logger.info(f"🔧 CORRECTED: Pipeline uses consistent resolution: {target_resolution}")
            # Separate transforms for patch-based training and validation
            patch_transforms = Compose([
                EnsureTyped(keys=["input", "target"]),
                Orientationd(keys=["input", "target"], axcodes="RAS"),
                Resized(keys=["input", "target"], spatial_size=target_resolution, mode=("bilinear", "nearest")),
                ScaleIntensityRangePercentilesd(keys="input", lower=0, upper=99.5, b_min=0, b_max=1),
                ToTensord(keys=["input", "target"]),
                EnsureTyped(keys=["energy"]),
                ToTensord(keys=["energy"])
            ])
            val_transforms = Compose([
                LoadImaged(keys=["input", "target"], reader=NumpyReader),
                EnsureChannelFirstd(keys=["input", "target"]),
                EnsureTyped(keys=["input", "target"]),
                Orientationd(keys=["input", "target"], axcodes="RAS"),
                Resized(keys=["input", "target"], spatial_size=target_resolution, mode=("bilinear", "nearest")),
                ScaleIntensityRangePercentilesd(keys="input", lower=0, upper=99.5, b_min=0, b_max=1),
                ToTensord(keys=["input", "target"]),
                EnsureTyped(keys=["energy"]),
                ToTensord(keys=["energy"])
            ])
                

            # Use HotspotPatchDataset for patch-based training
            logger.info("Using HotspotPatchDataset for patch-based training.")
            train_ds = HotspotPatchDataset(
                root_dir=self.root_dir,
                section=None,
                patch_size=(32, 32, 32),
                min_hotspot_voxels=1000,
                dose_threshold=0.5,
                max_patches=6,
                random_patches=0,
                transforms=patch_transforms,
                energy=energy,
                max_patches_per_energy=80 if energy <= 40.0 else 60  # Less patches for high energy
            )
            # Für Validation jetzt auch HotspotPatchDataset verwenden (ohne random_patches)
            val_ds = HotspotPatchDataset(
                root_dir=self.root_dir,
                section=None,
                patch_size=(32, 32, 32),
                min_hotspot_voxels=1000,
                dose_threshold=0.5,
                max_patches=2,
                random_patches=0,
                transforms=patch_transforms,
                energy=energy,
                max_patches_per_energy=20 if energy <= 40.0 else 15  # Less patches for high energy validation
            )
            # DataLoader wie gehabt
            from data_management import DataLoaderModule
            data_module = DataLoaderModule(
                root_dir=self.root_dir,
                transforms=patch_transforms
            )
            train_loader = data_module.create_data_loader(train_ds, self.batch_size, shuffle=True)
            val_loader   = data_module.create_data_loader(val_ds, self.batch_size, shuffle=False)
            logger.info(f"Found {len(val_ds)} validation patches for energy={energy}, resolution={target_resolution}")

            scheduler = DDPMScheduler(
                num_train_timesteps=1000,
                schedule="scaled_linear_beta",
                beta_start=0.0015,
                beta_end=0.0195,
            )
        
            # ### Scaling factor
            #
            # As mentioned in Rombach et al. [1] Section 4.3.2 and D.1, the signal-to-noise ratio (induced by the scale of the latent space) can affect the results obtained with the LDM, if the standard deviation of the latent space distribution drifts too much from that of a Gaussian. 
            # For this reason, it is best practice to use a scaling factor to adapt this standard deviation.
            #
            # _Note: In case where the latent space is close to a Gaussian distribution, the scaling factor will be close to one, and the results will not differ from those obtained when it is not used._
            #

            # +
            with torch.no_grad():
                with autocast('cuda', enabled=True):
                    first_batch = first(train_loader)
                    # Build conditioned input for autoencoder with energy channel if available
                    images = first_batch["input"].to(self.device)
                    energies = first_batch.get("energy", None)
                    if energies is not None:
                        energies = energies.to(self.device)
                        normalized_energy = energies.float() / 100.0  # match training normalization
                        B, C, D, H, W = images.shape
                        energy_tensor = normalized_energy.view(B, 1, 1, 1, 1).expand(B, 1, D, H, W)
                        conditioned = torch.cat([images, energy_tensor], dim=1)
                    else:
                        conditioned = images
                    # Encode to latents using conditioned input
                    z = autoencoder.encode_stage_2_inputs(conditioned)

            # **CRITICAL FIX: Handle NaN in scale factor calculation**
            z_std = torch.std(z)
            logger.info(f"Latent std: {z_std.item():.6f}")
            
            if torch.isnan(z_std) or torch.isinf(z_std) or z_std.item() < 1e-8:
                logger.warning(f"Invalid latent std ({z_std.item()}), using fallback scale_factor=1.0")
                scale_factor = 1.0
            else:
                scale_factor = 1 / z_std.item()
                # Clamp scale factor to reasonable range
                scale_factor = max(min(scale_factor, 10.0), 0.1)
            
            logger.info(f"Scaling factor set to {scale_factor}")
            
            # **Additional validation**
            if not np.isfinite(scale_factor):
                logger.error(f"Scale factor is not finite: {scale_factor}")
                scale_factor = 1.0

            inferer = LatentDiffusionInferer(scheduler, scale_factor=scale_factor)

            # 🔧 AGGRESSIVE learning rate for faster convergence (problems identified)
            # Current signal ratios 0.03-0.08 too low, need stronger learning
            aggressive_lr = self.learning_rate * 1.0  # Use full original LR (1e-5)
            logger.info(f"Using AGGRESSIVE learning rate: {aggressive_lr} (original: {self.learning_rate}) for faster convergence")
            
            opt_g = Adam(autoencoder.parameters(), lr=aggressive_lr, weight_decay=1e-5)  # Reduced weight decay
            opt_d = Adam(discriminator.parameters(), lr=aggressive_lr, weight_decay=1e-5)
            opt_diff = Adam(unet.parameters(), lr=aggressive_lr, weight_decay=1e-5)
            
            # trainers and early stopping
            ae_trainer = AutoencoderTrainer(autoencoder, discriminator, opt_g, opt_d, self.energy_min, self.energy_max, self.device)
            
            # **CRITICAL FIX: Calculate dose normalization parameters before training**
            logger.info("Calculating dose normalization parameters from training data...")
            try:
                ae_trainer.calculate_dose_normalization_params(train_loader, energy=energy)
            except Exception as e:
                logger.error(f"Failed to calculate dose normalization parameters: {e}")
                logger.error("This usually indicates problems with the training data.")
                raise
            
            # Validate that normalization parameters are reasonable
            if ae_trainer.clip_min is None or ae_trainer.clip_max is None:
                raise RuntimeError("clip_min or clip_max is None after calculation")
            
            if ae_trainer.clip_max <= ae_trainer.clip_min:
                raise RuntimeError(f"Invalid dose range: clip_max ({ae_trainer.clip_max}) <= clip_min ({ae_trainer.clip_min})")
            
            # Initialize loss functions after normalization parameters are set
            ae_trainer.setup_loss_functions()
            
            logger.info(f"Starting autoencoder training for resolution={target_resolution}, energy={energy}")
            stopper = EarlyStopping(patience=self.patience)
            # autoencoder training loop
            for epoch in range(self.num_epochs):
                if self.stop_training:
                    logger.info(f"Autoencoder training aborted by user at epoch {epoch} for resolution={target_resolution}, energy={energy}")
                    break
                train_loss, gen_loss, disc_loss = ae_trainer.train_one_epoch(train_loader, epoch)
                val_loss = ae_trainer.validate(val_loader)
                # record losses
                ae_train_losses.append(train_loss)
                ae_val_losses.append(val_loss)
                gen_losses.append(gen_loss)
                disc_losses.append(disc_loss)
                if stopper.update(val_loss):
                    logger.info(f"Early stopping autoencoder at epoch {epoch+1} for resolution={target_resolution}, energy={energy}")
                    break
            
            # now diffusion training
            logger.info(f"Starting diffusion training for resolution={target_resolution}, energy={energy}")
            diff_trainer = DiffusionTrainer(unet, opt_diff, self.energy_min, self.energy_max, self.device)
            
            # diffusion (UNet) training loop
            for epoch in range(self.num_epochs):
                if self.stop_training:
                    logger.info(f"Diffusion training aborted by user at epoch {epoch} for resolution={target_resolution}, energy={energy}")
                    break
                diff_loss = diff_trainer.train_one_epoch(train_loader, epoch, inferer, autoencoder)
                # record diffusion loss
                diff_losses.append(diff_loss)

            # plot loss curves for this config
            Visualization.plot_loss_curves(
                ae_train_losses, ae_val_losses,
                gen_losses, disc_losses,
                diff_losses,
                resolution=target_resolution, energy=energy
            )
            
            # **THIS IS THE CRUCIAL ADDITION:**
            # Save individual model checkpoint to memory after training each configuration
            self.save_models(
                autoencoder=autoencoder,
                unet=unet,
                optimizer_diff=opt_diff,
                optimizer_g=opt_g,
                optimizer_d=opt_d,
                epoch=epoch,  # last epoch
                res=target_resolution,
                energy=energy,
                dose_mean=getattr(ae_trainer, 'dose_mean', None),
                dose_std=getattr(ae_trainer, 'dose_std', None),
                scale_factor=scale_factor,
                clip_min=getattr(ae_trainer, 'clip_min', 0.0),  # FIX: Get from trainer
                clip_max=getattr(ae_trainer, 'clip_max', None),  # FIX: Get from trainer
                smoothing_kernel=0,
            )
            logger.info(f"✓ Model checkpoint saved to memory for resolution={target_resolution}, energy={energy}")

        # after all individual trainings, save a single combined checkpoint
        self.save_combined_checkpoint()
        self.training_complete = True
        logger.info("All training finished for all configurations.")

    def run_inference(self, ct_file_path, model_checkpoint=None):
        """
        Load a CT scan and run inference to compute dose distribution.
        """
        logger.info("=" * 60)
        logger.info("STARTING INFERENCE WORKFLOW")
        logger.info("=" * 60)
        logger.info(f"CT file path: {ct_file_path}")
        logger.info(f"Model checkpoint: {model_checkpoint}")
        logger.info(f"Device: {self.device}")
        
        # determine model sources and load checkpoint dict if needed
        ckpt = None
        if model_checkpoint is None:
            if not self.models_by_energy:
                logger.error("No trained models found and no checkpoint provided")
                raise RuntimeError("No trained models found. Train the models first.")
            logger.info("Using pre-trained models from training session")
        elif isinstance(model_checkpoint, str):
            logger.info(f"Loading checkpoint from file: {model_checkpoint}")
            import torch as _torch
            try:
                ckpt = _torch.load(model_checkpoint, map_location=self.device, weights_only=False)
                logger.info("✓ Checkpoint loaded successfully")
            except Exception as e:
                logger.warning(f"Direct load failed: {e}")
                logger.info("Attempting to load via file handle...")
                with open(model_checkpoint, 'rb') as f:
                    ckpt = _torch.load(f, map_location=self.device, weights_only=False)
                logger.info("✓ Checkpoint loaded via file handle")
        elif isinstance(model_checkpoint, dict) and 'autoencoder' in model_checkpoint and 'unet' in model_checkpoint:
            logger.info("Using provided checkpoint dictionary")
            ckpt = model_checkpoint
        else:
            logger.error("Invalid model checkpoint provided")
            raise ValueError("Invalid model checkpoint. Provide a path or a dict with 'autoencoder' and 'unet' keys.")
        # if we have a checkpoint dict, rebuild models from state_dict
        if ckpt is not None:
            # --- Combined checkpoint support ---
            models_by_energy_ckpt = ckpt.get("models_by_energy", None)
            if models_by_energy_ckpt is not None:
                logger.info("Detected combined checkpoint with per-energy models, loading all energies")
                self.models_by_energy.clear()
                for energy_str, model_dict in models_by_energy_ckpt.items():
                    # Extract energy from key format like "res16x16x16_e46.53" or "res16_e46.53"
                    if "_e" in energy_str:
                        energy_val = float(energy_str.split("_e")[1])
                    else:
                        # Fallback: try to convert the whole string (for backward compatibility)
                        energy_val = float(energy_str)
                        logger.warning(f"Using fallback energy extraction for key: {energy_str}")
                    # instantiate autoencoder
                    ae = AutoencoderKL(
                        spatial_dims=3,
                        in_channels=2,
                        out_channels=1,
                        num_channels=(32, 32, 32),
                        latent_channels=2,
                        num_res_blocks=1,
                        norm_num_groups=8,
                        attention_levels=(False, False, True),
                    ).to(self.device)
                    ae.load_state_dict(model_dict["autoencoder"])
                
                    # instantiate unet
                    un = DiffusionModelUNet(
                        spatial_dims=3,
                        in_channels=2,
                        out_channels=2,
                        with_conditioning=False,  # Disabled cross-attention conditioning
                        num_res_blocks=1,
                        num_channels=(32, 64, 64),
                        attention_levels=(False, True, True),
                        num_head_channels=(0, 64, 64),
                    ).to(self.device)
                    un.load_state_dict(model_dict["unet"], strict=False)
                    # scheduler
                    sched = DDPMScheduler(
                        num_train_timesteps=1000,
                        schedule="scaled_linear_beta",
                        beta_start=0.0015,
                        beta_end=0.0195,
                    )
                    self.models_by_energy[energy_val] = (ae, un, sched)
                logger.info("✓ Loaded combined models for all energies")
                
                # Update quad_energies and quad_weights based on available models
                available_energies = sorted(self.models_by_energy.keys())
                logger.info(f"Available energies in checkpoint: {available_energies}")
                
                # Validate that all required quad_energies are available in the checkpoint
                missing_energies = []
                for energy in self.quad_energies:
                    if energy not in available_energies:
                        missing_energies.append(energy)
                
                if missing_energies:
                    logger.warning(f"Missing energies in checkpoint: {missing_energies}")
                    logger.warning(f"Available energies: {available_energies}")
                    logger.warning(f"Requested energies: {self.quad_energies}")
                    # Filter out missing energies and adjust weights
                    original_energies = self.quad_energies.copy()
                    original_weights = self.quad_weights.copy()
                    
                    filtered_energies = []
                    filtered_weights = []
                    for i, energy in enumerate(original_energies):
                        if energy in available_energies:
                            filtered_energies.append(energy)
                            filtered_weights.append(original_weights[i])
                    
                    if filtered_energies:
                        # Renormalize weights
                        total_weight = sum(filtered_weights)
                        self.quad_energies = filtered_energies
                        self.quad_weights = [w/total_weight for w in filtered_weights]
                        logger.info(f"Filtered quad_energies: {self.quad_energies}")
                        logger.info(f"Renormalized quad_weights: {self.quad_weights}")
                    else:
                        raise ValueError(f"No quadrature energies available in checkpoint! Available: {available_energies}, Requested: {original_energies}")
                else:
                    logger.info("✓ All quadrature energies are available in checkpoint")
                    logger.info(f"Using quad_energies: {self.quad_energies}")
                    logger.info(f"Using quad_weights: {self.quad_weights}")
                
                
                # Load common parameters by averaging over all energy checkpoints
                if models_by_energy_ckpt:
                    scale_factors = []
                    dose_means = []
                    dose_stds = []
                    clip_mins = []
                    clip_maxs = []
                    smoothing_kernels = []
                    
                    for energy_key, model_dict in models_by_energy_ckpt.items():
                        # Only exclude None values, but include 0.0 and other numeric values
                        sf = model_dict.get("scale_factor")
                        if sf is not None:
                            scale_factors.append(sf)
                        
                        dm = model_dict.get("dose_mean")
                        if dm is not None:
                            dose_means.append(dm)
                        
                        ds = model_dict.get("dose_std")
                        if ds is not None:
                            dose_stds.append(ds)
                        
                        cm = model_dict.get("clip_min")
                        if cm is not None:
                            clip_mins.append(cm)
                        
                        # clip_max can legitimately be None
                        clip_maxs.append(model_dict.get("clip_max"))
                        
                        sk = model_dict.get("smoothing_kernel")
                        if sk is not None:
                            smoothing_kernels.append(sk)
                    
                    # Calculate averages with fallback defaults if no valid values found
                    self.scale_factor = sum(scale_factors) / len(scale_factors) if scale_factors else 1.0
                    self.dose_mean = sum(dose_means) / len(dose_means) if dose_means else 0.0
                    self.dose_std = sum(dose_stds) / len(dose_stds) if dose_stds else 1.0
                    self.clip_min = sum(clip_mins) / len(clip_mins) if clip_mins else 0.0
                    self.smoothing_kernel = sum(smoothing_kernels) / len(smoothing_kernels) if smoothing_kernels else 0
                    
                    # For clip_max, use the average of non-None values, or None if all are None
                    valid_clip_maxs = [x for x in clip_maxs if x is not None]
                    self.clip_max = sum(valid_clip_maxs) / len(valid_clip_maxs) if valid_clip_maxs else None
                    
                    logger.info(f"Loaded averaged parameters from {len(models_by_energy_ckpt)} energy models:")
                    logger.info(f"  scale_factor={self.scale_factor:.6f} (from {len(scale_factors)} values)")
                    logger.info(f"  dose_mean={self.dose_mean:.6f} (from {len(dose_means)} values)")
                    logger.info(f"  dose_std={self.dose_std:.6f} (from {len(dose_stds)} values)")
                    logger.info(f"  clip_min={self.clip_min:.6f} (from {len(clip_mins)} values)")
                    logger.info(f"  clip_max={self.clip_max} (from {len(valid_clip_maxs)} values)")
                    logger.info(f"  smoothing_kernel={self.smoothing_kernel:.1f} (from {len(smoothing_kernels)} values)")
                
                # switch to in-memory inference path
                model_checkpoint = None
            else:
                # --- End combined checkpoint support ---

                logger.info("Rebuilding models from checkpoint state_dict...")
                ae = AutoencoderKL(spatial_dims=3, in_channels=2, out_channels=1,
                                    num_channels=(32, 32, 32), latent_channels=2,
                                    num_res_blocks=1, norm_num_groups=8,
                                    attention_levels=(False, False, True)).to(self.device)
                ae.load_state_dict(ckpt['autoencoder'])
                logger.info("✓ Autoencoder loaded from checkpoint")

                unet_state = ckpt['unet']

                un = DiffusionModelUNet(
                    spatial_dims=3, in_channels=2, out_channels=2,
                    with_conditioning=False,  # Disabled cross-attention conditioning
                    num_res_blocks=1, num_channels=(32, 64, 64),
                    attention_levels=(False, True, True),
                    num_head_channels=(0, 64, 64)
                ).to(self.device)

                # Filter checkpoint to only matching shapes before loading
                pretrained_dict = ckpt['unet']
                model_dict = un.state_dict()
                filtered_dict = {
                    k: v for k, v in pretrained_dict.items()
                    if k in model_dict and model_dict[k].shape == v.shape
                }
                missing = set(model_dict.keys()) - set(filtered_dict.keys())
                unexpected = set(pretrained_dict.keys()) - set(filtered_dict.keys())
                # Load only matching parameters
                un.load_state_dict(filtered_dict, strict=False)
                if missing or unexpected:
                    logger.warning(f"UNet checkpoint loaded with missing keys: {sorted(missing)} and unexpected keys: {sorted(unexpected)}. Mismatched shapes filtered out.")
                else:
                    logger.info("✓ UNet loaded from checkpoint (all keys matched)")

                sched = DDPMScheduler(num_train_timesteps=1000,
                                      schedule="scaled_linear_beta",
                                      beta_start=0.0015, beta_end=0.0195)
                self.models_by_energy = {energy: (ae, un, sched) for energy in self.quad_energies}
                self.autoencoder, self.unet, self.scheduler = ae, un, sched
                logger.info("✓ Scheduler created and models assigned")

                # Retrieve dataset‑wide dose statistics for de‑normalisation
                self.dose_mean = ckpt.get("dose_mean", 0.0)
                self.dose_std  = ckpt.get("dose_std",  1.0)
                logger.info(f"Dose scaling loaded from checkpoint: mean={self.dose_mean}, std={self.dose_std}")
                self.scale_factor     = ckpt.get("scale_factor", 1.0)
                self.clip_min         = ckpt.get("clip_min", 0.0)
                self.clip_max         = ckpt.get("clip_max", None)
                self.smoothing_kernel = ckpt.get("smoothing_kernel", 0)
        
        # lazy import to avoid circular
        import nibabel as nib
        import numpy as np
        from corrected_inference import CorrectedInferenceModule

        # build tensor from CT file: support both NIfTI (.nii, .nii.gz) and NumPy (.npy)
        logger.info("Loading CT data...")
        logger.info(f"CT file path: {ct_file_path}")
        
        path_lower = ct_file_path.lower()
        if path_lower.endswith('.nii') or path_lower.endswith('.nii.gz'):
            logger.info("Loading NIfTI file...")
            nifti_img = nib.load(ct_file_path)
            arr = np.asarray(nifti_img.dataobj)
            logger.info(f"✓ NIfTI loaded: shape={arr.shape}, dtype={arr.dtype}")
        elif path_lower.endswith('.npy'):
            logger.info("Loading NumPy file...")
            arr = np.load(ct_file_path)
            logger.info(f"✓ NumPy loaded: shape={arr.shape}, dtype={arr.dtype}")
        else:
            logger.error(f"Unsupported CT file format: {ct_file_path}")
            raise ValueError(f"Unsupported CT file format: {ct_file_path}")
        
        logger.info(f"CT data range: {arr.min():.6f} to {arr.max():.6f}")
        ct_tensor = torch.from_numpy(arr).unsqueeze(0).to(self.device)  # shape [1, D, H, W]
        logger.info(f"CT tensor shape: {ct_tensor.shape}")
        
        # Use corrected inference module
        logger.info("Setting up corrected inference module...")
        try:
            if ckpt is not None and "models_by_energy" in ckpt:
                # Use already loaded models from combined checkpoint
                logger.info("Using already loaded models from combined checkpoint")
                inf_mod = CorrectedInferenceModule(
                    models_by_energy=self.models_by_energy,
                    device=self.device,
                    energies=self.quad_energies,
                    scale_factor=self.scale_factor,
                    dose_mean=self.dose_mean,
                    dose_std=self.dose_std,
                    clip_min=self.clip_min,
                    clip_max=self.clip_max,
                    smoothing_kernel=self.smoothing_kernel
                )
            elif model_checkpoint is not None:
                # Load from checkpoint path
                logger.info("Loading models from checkpoint path")
                inf_mod = CorrectedInferenceModule(
                    checkpoint_path=model_checkpoint,
                    device=self.device,
                    energies=self.quad_energies
                )
            else:
                # Use pre-trained models from training session
                logger.info("Using pre-trained models from training session")
                inf_mod = CorrectedInferenceModule(
                    models_by_energy=self.models_by_energy,
                    device=self.device,
                    energies=self.quad_energies,
                    scale_factor=getattr(self, 'scale_factor', 1.0),
                    dose_mean=getattr(self, 'dose_mean', 0.0),
                    dose_std=getattr(self, 'dose_std', 1.0),
                    clip_min=getattr(self, 'clip_min', 0.0),
                    clip_max=getattr(self, 'clip_max', None),
                    smoothing_kernel=getattr(self, 'smoothing_kernel', 0)
                )
            logger.info("✓ CorrectedInferenceModule created successfully")
        except Exception as e:
            logger.error(f"Failed to create CorrectedInferenceModule: {e}")
            logger.error("CorrectedInferenceModule creation traceback:", exc_info=True)
            raise e
        
        # run quadrature-based inference over all energies
        logger.info("Running corrected inference...")
        logger.info(f"Quadrature energies: {self.quad_energies}")
        logger.info(f"Quadrature weights: {self.quad_weights}")
        
        try:
            dose = inf_mod.run_inference(ct_tensor, self.quad_energies, self.quad_weights)
            logger.info("✓ Corrected inference completed successfully")
            logger.info(f"Dose tensor shape: {dose.shape}")
            logger.info(f"Dose range: {dose.min():.6f} to {dose.max():.6f}")
        except Exception as e:
            logger.error(f"Corrected inference failed: {e}")
            logger.error("Corrected inference traceback:", exc_info=True)
            raise e
        # convert to numpy and remove batch dim
        logger.info("Converting output to NumPy and preparing NIfTI...")
        dose_np = dose.detach().cpu().numpy()
        if dose_np.ndim == 4 and dose_np.shape[0] == 1:
            dose_np = dose_np[0]
        logger.info(f"Final dose shape: {dose_np.shape}")
        logger.info(f"Final dose range: {dose_np.min():.6f} to {dose_np.max():.6f}")
        
        # create affine
        import numpy as _np, nibabel as _nib, json as _json, os as _os
        from nibabel.nifti1 import Nifti1Extension
        affine = _np.eye(4)
        img = _nib.Nifti1Image(dose_np, affine)
        logger.info("✓ NIfTI image created")
        
        # attach cubes.json manifest if available
        manifest_path = _os.path.join(self.root_dir, 'cubes.json')
        if _os.path.exists(manifest_path):
            logger.info(f"Loading manifest from: {manifest_path}")
            with open(manifest_path, 'r') as mf:
                manifest = _json.load(mf)
            # Encode manifest JSON to bytes for NIfTI extension
            ext = Nifti1Extension('comment', _json.dumps(manifest).encode('utf-8'))
            img.header.extensions.append(ext)
            logger.info("✓ Manifest attached to NIfTI header")
        else:
            logger.info("No cubes.json manifest found, skipping manifest attachment")
        
        # save NIfTI file
        out_path = _os.path.join(self.root_dir, 'inference_with_manifest.nii.gz')
        logger.info(f"Saving inference result to: {out_path}")
        _nib.save(img, out_path)
        logger.info("✓ NIfTI file saved successfully")
        
        logger.info("=" * 60)
        logger.info("INFERENCE WORKFLOW COMPLETED SUCCESSFULLY")
        logger.info(f"Output file: {out_path}")
        logger.info("=" * 60)
        # set result path
        self.dose_result_path = out_path
        return out_path

    def run_energy_conditioned_training(self):
        """
        Executes energy-conditioned training pipeline - trains a SINGLE model with ALL energies.
        
        This is the new approach that trains one model with energy conditioning,
        instead of separate models for each energy.
        
        Steps:
          1. Load dataset with ALL energies (no filtering by energy)
          2. Create DataLoaders for training and validation
          3. Initialize models with 2-channel input (CT + Energy)
          4. Train autoencoder with energy conditioning
          5. Train diffusion model with energy conditioning
          6. Save the unified model
        """
        logger.info("=== STARTING ENERGY-CONDITIONED TRAINING ===")
        logger.info("Training a SINGLE model with ALL energies instead of separate models")
        logger.info("Starting training with hyperparameters:")
        logger.info(f"  batch_size={self.batch_size}, num_epochs={self.num_epochs}, learning_rate={self.learning_rate}, patience={self.patience}, cube_size={self.cube_size}")
        logger.info(f"  resolutions={self.resolutions}, energies={self.energies}")
        
        # For simplicity, use first resolution only (can be extended later)
        res = self.resolutions[0]
        logger.info(f"--- Training unified model at resolution={res} with energies={self.energies} ---")
        
        # Setup transforms
        logger.info(f"🔧 CORRECTED: Pipeline uses consistent resolution: {res}")
        self.transforms = Compose([
            LoadImaged(keys=["input", "target"], reader=NumpyReader),
            EnsureChannelFirstd(keys=["input", "target"]),
            EnsureTyped(keys=["input", "target"]),
            Orientationd(keys=["input", "target"], axcodes="RAS"),
            # 🔥 CRITICAL FIX: Use Resized instead of Spacingd for array dimensions
            Resized(keys=["input", "target"], spatial_size=res, mode=("bilinear", "nearest")),
            # 🔧 REMOVED: SpatialPadd and CenterSpatialCropd that destroyed the target resolution
            # SpatialPadd(keys=["input", "target"], spatial_size=self.cube_size, method="symmetric"),
            # CenterSpatialCropd(keys=["input", "target"], roi_size=self.cube_size),
            ScaleIntensityRangePercentilesd(
                keys="input", lower=0, upper=99.5, b_min=0, b_max=1
            ),
            ToTensord(keys=["input", "target"]),
            EnsureTyped(keys=["energy"]),
            ToTensord(keys=["energy"])
        ])
        
        # Initialize history lists for plotting
        ae_train_losses = []
        ae_val_losses = []
        gen_losses = []
        disc_losses = []
        diff_losses = []
        
        # Load dataset WITHOUT filtering by energy (this is the key change!)
        data_module = DataLoaderModule(
            root_dir=self.root_dir,
            transforms=self.transforms
        )
        # Load complete dataset with ALL energies
        ds_full = data_module.load_dataset(section=None)
        # NO FILTERING BY ENERGY! - This is the crucial difference
        logger.info(f"Loaded {len(ds_full)} samples with ALL energies: {set([float(s['energy'].item()) for s in ds_full])}")
        
        train_ds, val_ds = data_module.split_dataset(ds_full)
        train_loader = data_module.create_data_loader(train_ds, self.batch_size, shuffle=True)
        val_loader = data_module.create_data_loader(val_ds, self.batch_size, shuffle=False)
        
        logger.info(f"Training set: {len(train_ds)} samples")
        logger.info(f"Validation set: {len(val_ds)} samples")
        
        # Instantiate models with 2-channel input for energy conditioning
        autoencoder = AutoencoderKL(
            spatial_dims=3,
            in_channels=2,  # CT + Energy channel
            out_channels=1,
            num_channels=(32, 32, 32),
            latent_channels=2,
            num_res_blocks=1,
            norm_num_groups=8,
            attention_levels=(False, False, True),
        ).to(self.device)
        
        discriminator = PatchDiscriminator(
            spatial_dims=3,
            num_layers_d=3,
            num_channels=32,
            in_channels=1,  # Output is still 1 channel
            out_channels=1
        ).to(self.device)
        
        unet = DiffusionModelUNet(
            spatial_dims=3,
            in_channels=2,   # Latent space is 2 channels
            out_channels=2,
            with_conditioning=False,  # Disabled cross-attention conditioning
            num_res_blocks=1,
            num_channels=(32, 64, 64),
            attention_levels=(False, True, True),
            num_head_channels=(0, 64, 64),
        ).to(self.device)
        
        scheduler = DDPMScheduler(num_train_timesteps=1000, beta_start=0.0015, beta_end=0.0195)
        
        # Calculate latent scaling factor
        with torch.no_grad():
            # Take a sample from training data to compute scaling factor
            sample_batch = first(train_loader)
            sample_input = sample_batch["input"].to(self.device)
            sample_energy = sample_batch["energy"].to(self.device)
            
            # Apply energy conditioning for scaling factor calculation
            B, C, D, H, W = sample_input.shape
            normalized_energy = sample_energy.float() / 100.0
            energy_tensor = normalized_energy.view(B, 1, 1, 1, 1).expand(B, 1, D, H, W)
            conditioned_input = torch.cat([sample_input, energy_tensor], dim=1)
            
            encoded = autoencoder.encode(conditioned_input)
            if isinstance(encoded, tuple):
                z = encoded[0]
            else:
                z = encoded.latent_dist.sample()
            scale_factor = 1 / torch.std(z)
            
        logger.info(f"Scaling factor set to {scale_factor}")
        
        # Train Autoencoder with energy conditioning
        logger.info(f"Starting autoencoder training for unified model")
        
        # Create optimizers with weight decay for stability
        optimizer_g = torch.optim.Adam(autoencoder.parameters(), lr=self.learning_rate, weight_decay=1e-5)
        optimizer_d = torch.optim.Adam(discriminator.parameters(), lr=self.learning_rate, weight_decay=1e-5)
        
        ae_trainer = AutoencoderTrainer(
            autoencoder=autoencoder,
            discriminator=discriminator,
            optimizer_g=optimizer_g,
            optimizer_d=optimizer_d,
            device=self.device
        )
        
        early_stopping = EarlyStopping(patience=self.patience)
        
        for epoch in range(self.num_epochs):
            if self.stop_training:
                logger.info("Training aborted by user.")
                return
                
            recon_loss, adv_loss, disc_loss = ae_trainer.train_one_epoch(train_loader, epoch)
            ae_train_losses.append(recon_loss)
            gen_losses.append(adv_loss)
            disc_losses.append(disc_loss)
            
            val_loss = ae_trainer.validate(val_loader)
            ae_val_losses.append(val_loss)
            
            early_stop = early_stopping.update(val_loss)
            if early_stop:
                logger.info(f"Early stopping triggered at epoch {epoch}")
                break
        
        # Train Diffusion Model with energy conditioning
        logger.info(f"Starting diffusion training for unified model")
        
        # Create optimizer for diffusion model
        optimizer_diff = torch.optim.Adam(unet.parameters(), lr=self.learning_rate)
        
        diff_trainer = DiffusionTrainer(
            diffusion_model=unet,
            optimizer_diff=optimizer_diff,
            device=self.device
        )
        
        # Train diffusion for fewer epochs (typically 10-20)
        from generative.inferers import LatentDiffusionInferer
        inferer = LatentDiffusionInferer(scheduler=scheduler, scale_factor=scale_factor)
        
        diff_epochs = max(10, self.num_epochs // 2)
        for epoch in range(diff_epochs):
            if self.stop_training:
                logger.info("Training aborted by user.")
                return
                
            diff_loss = diff_trainer.train_one_epoch(train_loader, epoch, inferer=inferer, autoencoder=autoencoder)
            diff_losses.append(diff_loss)
        
        # Save the unified model
        model_filename = f"unified_energy_conditioned_model_res{res}_energies{len(self.energies)}.ckpt"
        model_path = os.path.join(os.getcwd(), model_filename)
        
        torch.save({
            'autoencoder': autoencoder.state_dict(),
            'unet': unet.state_dict(),
            'scheduler': scheduler.state_dict(),
            'scale_factor': scale_factor,
            'resolutions': [res],
            'energies': self.energies,
            'training_config': {
                'batch_size': self.batch_size,
                'num_epochs': self.num_epochs,
                'learning_rate': self.learning_rate,
                'cube_size': self.cube_size
            }
        }, model_path)
        
        logger.info(f"✅ Unified energy-conditioned model saved to: {model_path}")
        logger.info("=== ENERGY-CONDITIONED TRAINING COMPLETED ===")
        
        return {
            'model_path': model_path,
            'autoencoder': autoencoder,
            'unet': unet,
            'scheduler': scheduler,
            'scale_factor': scale_factor,
            'losses': {
                'ae_train': ae_train_losses,
                'ae_val': ae_val_losses,
                'gen': gen_losses,
                'disc': disc_losses,
                'diff': diff_losses
            }
        }
