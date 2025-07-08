# system_manager.py

import shutil
import os
import torch
import log_config
import logging
logger = logging.getLogger(__name__)

from data_management import DataLoaderModule
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
    EnsureTyped, Orientationd, Spacingd, SpatialPadd,
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
    def __init__(self, root_dir, transforms, resolutions, energies, quad_energies, quad_weights, batch_size, device, num_epochs, learning_rate, patience, cube_size, seed=42):
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

    def save_models(self, autoencoder, unet, optimizer_diff, optimizer_g, optimizer_d, epoch):
        checkpoint_path = f"model_res{self.resolutions}_energy{self.energies}.ckpt"
        torch.save(
            {
                "autoencoder": autoencoder.state_dict(),
                "unet": unet.state_dict(),
                "optimizer_diff": optimizer_diff.state_dict(),
                "optimizer_g": optimizer_g.state_dict(),
                "optimizer_d": optimizer_d.state_dict(),
                "epoch": epoch,
            },
            checkpoint_path,
        )

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
        logger.info("Starting training with hyperparameters:")
        logger.info(f"  batch_size={self.batch_size}, num_epochs={self.num_epochs}, learning_rate={self.learning_rate}, patience={self.patience}, cube_size={self.cube_size}")
        logger.info(f"  resolutions={self.resolutions}, energies={self.energies}")
        for res in self.resolutions:
            if self.stop_training:
                logger.info("Training aborted by user.")
                return
            for energy in self.energies:
                if self.stop_training:
                    logger.info("Training aborted by user.")
                    return
                logger.info(f"--- Training at resolution={res}, energy={energy} eV ---")
                self.transforms = Compose([
                    LoadImaged(keys=["input", "target"], reader=NumpyReader),
                    EnsureChannelFirstd(keys=["input", "target"]),
                    EnsureTyped(keys=["input", "target"]),
                    Orientationd(keys=["input", "target"], axcodes="RAS"),
                    Spacingd(keys=["input", "target"], pixdim=res, mode= ("bilinear", "nearest")),
                    SpatialPadd(keys=["input", "target"], spatial_size=self.cube_size, method="symmetric"),
                    CenterSpatialCropd(keys=["input", "target"], roi_size=self.cube_size),
                    ScaleIntensityRangePercentilesd(
                        keys="input", lower=0, upper=99.5, b_min=0, b_max=1
                    ),
                    ToTensord(keys=["input", "target"]),
                    EnsureTyped(keys=["energy"]),
                    ToTensord  (keys=["energy"])
                    ])
                
                # initialize history lists for plotting
                ae_train_losses = []
                ae_val_losses   = []
                gen_losses      = []
                disc_losses     = []
                diff_losses     = []

                
                data_module = DataLoaderModule(
                    root_dir=self.root_dir,
                    transforms=self.transforms
                )
                # load complete dataset
                ds_full = data_module.load_dataset(section=None)
                # sample["energy"] delivers a tensor, so we need to convert it to float
                ds_full = [s for s in ds_full if float(s["energy"].item()) == energy]
                
                train_ds, val_ds = data_module.split_dataset(ds_full)
                train_loader = data_module.create_data_loader(train_ds, self.batch_size, shuffle=True)
                val_loader   = data_module.create_data_loader(val_ds, self.batch_size, shuffle=False)
                logger.info(f"Found {len(ds_full)} samples for energy={energy}, resolution={res}")
                
                # instantiate models
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
                    with_conditioning=True,
                    cross_attention_dim=2,
                    num_res_blocks=1,
                    num_channels=(32, 64, 64),
                    attention_levels=(False, True, True),
                    num_head_channels=(0, 64, 64),
                ).to(self.device)


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


                logger.info(f"Scaling factor set to {1/torch.std(z)}")
                scale_factor = 1 / torch.std(z)

                inferer = LatentDiffusionInferer(scheduler, scale_factor=scale_factor)

                # optimizers
                opt_g = Adam(autoencoder.parameters(), lr=self.learning_rate)
                opt_d = Adam(discriminator.parameters(), lr=self.learning_rate)
                opt_diff = Adam(unet.parameters(), lr=self.learning_rate)
                
                # trainers and early stopping
                ae_trainer = AutoencoderTrainer(autoencoder, discriminator, opt_g, opt_d, self.device)
                logger.info(f"Starting autoencoder training for resolution={res}, energy={energy}")
                stopper = EarlyStopping(patience=self.patience)
                # autoencoder training loop
                for epoch in range(self.num_epochs):
                    if self.stop_training:
                        logger.info(f"Autoencoder training aborted by user at epoch {epoch} for resolution={res}, energy={energy}")
                        break
                    train_loss, gen_loss, disc_loss = ae_trainer.train_one_epoch(train_loader, epoch)
                    val_loss = ae_trainer.validate(val_loader)
                    # record losses
                    ae_train_losses.append(train_loss)
                    ae_val_losses.append(val_loss)
                    gen_losses.append(gen_loss)
                    disc_losses.append(disc_loss)
                    if stopper.update(val_loss):
                        logger.info(f"Early stopping autoencoder at epoch {epoch+1} for resolution={res}, energy={energy}")
                        break
                
                # now diffusion training
                logger.info(f"Starting diffusion training for resolution={res}, energy={energy}")
                diff_trainer = DiffusionTrainer(unet, opt_diff, self.device)
                
                # diffusion (UNet) training loop
                for epoch in range(self.num_epochs):
                    if self.stop_training:
                        logger.info(f"Diffusion training aborted by user at epoch {epoch} for resolution={res}, energy={energy}")
                        break
                    diff_loss = diff_trainer.train_one_epoch(train_loader, epoch, inferer, autoencoder)
                    # record diffusion loss
                    diff_losses.append(diff_loss)

                # plot loss curves for this config
                Visualization.plot_loss_curves(
                    ae_train_losses, ae_val_losses,
                    gen_losses, disc_losses,
                    diff_losses,
                    resolution=res, energy=energy
                )
                
                # save checkpoint for this config
                self.save_models(autoencoder, unet, opt_diff, opt_g, opt_d, epoch)

   
        # after loops
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
            logger.info("Rebuilding models from checkpoint state_dict...")
            ae = AutoencoderKL(spatial_dims=3, in_channels=2, out_channels=1,
                                num_channels=(32, 32, 32), latent_channels=2,
                                num_res_blocks=1, norm_num_groups=8,
                                attention_levels=(False, False, True)).to(self.device)
            ae.load_state_dict(ckpt['autoencoder'])
            logger.info("✓ Autoencoder loaded from checkpoint")
            
            # Determine cross_attention_dim from checkpoint UNet weights
            unet_state = ckpt['unet']
            cross_dim = None
            # Find any to_k.weight where input dim != output dim (identifies cross-attn)
            for key, tensor in unet_state.items():
                if 'to_k.weight' in key and tensor.dim() == 2 and tensor.shape[1] != tensor.shape[0]:
                    # tensor shape is [inner_dim, cross_attention_dim]
                    cross_dim = tensor.shape[1]
                    logger.info(f"Detected cross_attention_dim={cross_dim} from key: {key}")
                    break
            if cross_dim is None:
                # Fallback if detection fails
                logger.warning("could not determine cross_attention_dim from UNet checkpoint; defaulting to 2")
                cross_dim = 2
            
            un = DiffusionModelUNet(
                spatial_dims=3, in_channels=2, out_channels=2,
                with_conditioning=True, cross_attention_dim=cross_dim,
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
            if model_checkpoint:
                logger.info(f"Creating CorrectedInferenceModule with checkpoint: {model_checkpoint}")
                inf_mod = CorrectedInferenceModule(
                    model_path=model_checkpoint,
                    device=self.device
                )
            else:
                # Use the most recent checkpoint
                model_path = "unified_energy_conditioned_model_res16.0_energies3.ckpt"
                logger.info(f"Creating CorrectedInferenceModule with default checkpoint: {model_path}")
                inf_mod = CorrectedInferenceModule(
                    model_path=model_path,
                    device=self.device
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
        self.transforms = Compose([
            LoadImaged(keys=["input", "target"], reader=NumpyReader),
            EnsureChannelFirstd(keys=["input", "target"]),
            EnsureTyped(keys=["input", "target"]),
            Orientationd(keys=["input", "target"], axcodes="RAS"),
            Spacingd(keys=["input", "target"], pixdim=res, mode=("bilinear", "nearest")),
            SpatialPadd(keys=["input", "target"], spatial_size=self.cube_size, method="symmetric"),
            CenterSpatialCropd(keys=["input", "target"], roi_size=self.cube_size),
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
            in_channels=2,  # Latent space is 2 channels
            out_channels=2,
            with_conditioning=True,
            cross_attention_dim=2,
            num_res_blocks=1,
            num_channels=(32, 64, 64),
            attention_levels=(False, True, True),
            num_head_channels=(0, 32, 32),
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
        
        # Create optimizers
        optimizer_g = torch.optim.Adam(autoencoder.parameters(), lr=self.learning_rate)
        optimizer_d = torch.optim.Adam(discriminator.parameters(), lr=self.learning_rate)
        
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
