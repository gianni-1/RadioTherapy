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
        # determine model sources and load checkpoint dict if needed
        ckpt = None
        if model_checkpoint is None:
            if not self.models_by_energy:
                raise RuntimeError("No trained models found. Train the models first.")
        elif isinstance(model_checkpoint, str):
            import torch as _torch
            ckpt = _torch.load(model_checkpoint, map_location=self.device)
        elif isinstance(model_checkpoint, dict) and 'autoencoder' in model_checkpoint and 'unet' in model_checkpoint:
            ckpt = model_checkpoint
        else:
            raise ValueError("Invalid model checkpoint. Provide a path or a dict with 'autoencoder' and 'unet' keys.")
        # if we have a checkpoint dict, rebuild models from state_dict
        if ckpt is not None:
            ae = AutoencoderKL(spatial_dims=3, in_channels=2, out_channels=1,
                                num_channels=(32, 32, 32), latent_channels=2,
                                num_res_blocks=1, norm_num_groups=8,
                                attention_levels=(False, False, True)).to(self.device)
            ae.load_state_dict(ckpt['autoencoder'])
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
            sched = DDPMScheduler(num_train_timesteps=1000,
                                  schedule="scaled_linear_beta",
                                  beta_start=0.0015, beta_end=0.0195)
            self.models_by_energy = {energy: (ae, un, sched) for energy in self.quad_energies}
            self.autoencoder, self.unet, self.scheduler = ae, un, sched
        
        # lazy import to avoid circular
        import nibabel as nib
        import numpy as np
        from inference_module import InferenceModule

        # build tensor from CT file: support both NIfTI (.nii, .nii.gz) and NumPy (.npy)
        path_lower = ct_file_path.lower()
        if path_lower.endswith('.nii') or path_lower.endswith('.nii.gz'):
            nifti_img = nib.load(ct_file_path)
            arr = np.asarray(nifti_img.dataobj)
        elif path_lower.endswith('.npy'):
            arr = np.load(ct_file_path)
        else:
            raise ValueError(f"Unsupported CT file format: {ct_file_path}")
        ct_tensor = torch.from_numpy(arr).unsqueeze(0).to(self.device)  # shape [1, D, H, W]
        # initialize inference module with all energy-conditioned models
        inf_mod = InferenceModule(
            models_by_energy=self.models_by_energy,
            energies=self.quad_energies,
            energy_weights=self.quad_weights,
            device=self.device,
        )
        # run quadrature-based inference over all energies
        logger.info("Running inference...")
        dose = inf_mod.run_inference(ct_tensor)
        # convert to numpy and remove batch dim
        dose_np = dose.detach().cpu().numpy()
        if dose_np.ndim == 4 and dose_np.shape[0] == 1:
            dose_np = dose_np[0]
        # create affine
        import numpy as _np, nibabel as _nib, json as _json, os as _os
        from nibabel.nifti1 import Nifti1Extension
        affine = _np.eye(4)
        img = _nib.Nifti1Image(dose_np, affine)
        # attach cubes.json manifest if available
        manifest_path = _os.path.join(self.root_dir, 'cubes.json')
        if _os.path.exists(manifest_path):
            with open(manifest_path, 'r') as mf:
                manifest = _json.load(mf)
            # Encode manifest JSON to bytes for NIfTI extension
            ext = Nifti1Extension('comment', _json.dumps(manifest).encode('utf-8'))
            img.header.extensions.append(ext)
        # save NIfTI file
        out_path = _os.path.join(self.root_dir, 'inference_with_manifest.nii.gz')
        _nib.save(img, out_path)
        # set result path
        self.dose_result_path = out_path
        return out_path
