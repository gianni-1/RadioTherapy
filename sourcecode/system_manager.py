# system_manager.py

import shutil
import os
import torch
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
    def __init__(self, root_dir, transforms, resolutions, energies, batch_size, device, num_epochs, learning_rate, patience,cube_size, seed=42):
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
        self.progress_callback = None
        # placeholders for trained models
        self.autoencoder = None
        self.unet = None
        self.scheduler = None
        self.cube_size = cube_size

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
        for res in self.resolutions:
            for energy in self.energies:
                print(f"\n--- Training at resolution={res}, energy={energy} eV ---")
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

                # create data module for this combination
                data_module = DataLoaderModule(
                    root_dir=self.root_dir,
                    transforms=self.transforms
                )
                # Load dataset directly from root_dir, which contains energy subfolders
                ds_full = data_module.load_dataset(section=None)
                train_ds, val_ds = data_module.split_dataset(ds_full)
                train_loader = data_module.create_data_loader(train_ds, self.batch_size, shuffle=True)
                val_loader   = data_module.create_data_loader(val_ds, self.batch_size, shuffle=False)
                print(f"Found {len(ds_full)} samples for energy={energy}, resolution={res}")
                
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


                print(f"Scaling factor set to {1/torch.std(z)}")
                scale_factor = 1 / torch.std(z)

                inferer = LatentDiffusionInferer(scheduler, scale_factor=scale_factor)

                # optimizers
                opt_g = Adam(autoencoder.parameters(), lr=self.learning_rate)
                opt_d = Adam(discriminator.parameters(), lr=self.learning_rate)
                opt_diff = Adam(unet.parameters(), lr=self.learning_rate)
                
                # trainers and early stopping
                ae_trainer = AutoencoderTrainer(autoencoder, discriminator, opt_g, opt_d, self.device)
                stopper = EarlyStopping(patience=self.patience)
                # autoencoder training loop
                for epoch in range(self.num_epochs):
                    train_loss, gen_loss, disc_loss = ae_trainer.train_one_epoch(train_loader, epoch)
                    val_loss = ae_trainer.validate(val_loader)
                    # record losses
                    ae_train_losses.append(train_loss)
                    ae_val_losses.append(val_loss)
                    gen_losses.append(gen_loss)
                    disc_losses.append(disc_loss)
                    # emit autoencoder progress update
                    if self.progress_callback:
                        self.progress_callback(epoch+1, self.num_epochs)
                    if stopper.update(val_loss):
                        print("Early stopping at epoch", epoch+1)
                        break
                
                # now diffusion training
                diff_trainer = DiffusionTrainer(unet, opt_diff, self.device)
                
                # diffusion (UNet) training loop
                for epoch in range(self.num_epochs):
                    diff_loss = diff_trainer.train_one_epoch(train_loader, epoch, inferer, autoencoder)
                    # record diffusion loss
                    diff_losses.append(diff_loss)
                    # emit diffusion progress update
                    if self.progress_callback:
                        self.progress_callback(epoch+1, self.num_epochs)

                # plot loss curves for this config
                Visualization.plot_loss_curves(
                    ae_train_losses, ae_val_losses,
                    gen_losses, disc_losses,
                    diff_losses,
                    resolution=res, energy=energy
                )
                
                # save checkpoint for this config
                self.save_models(autoencoder, unet, opt_diff, opt_g, opt_d, epoch)
                # store trained models for inference
                self.autoencoder = autoencoder
                self.unet = unet
                self.scheduler = scheduler
        # after loops
        self.training_complete = True
        print("All training finished.")

    def run_inference(self, ct_file_path):
        """
        Load a CT scan and run inference to compute dose distribution.
        """
        if self.autoencoder is None or self.unet is None:
            raise RuntimeError("Keine trainierten Modelle gefunden. Bitte zuerst das Training ausführen.")
        # lazy import to avoid circular
        import nibabel as nib
        import numpy as np
        from inference_module import InferenceModule

        # load CT scan, assume single channel
        img = nib.load(ct_file_path)
        arr = np.asarray(img.dataobj)
        ct_tensor = torch.from_numpy(arr).unsqueeze(0).to(self.device)  # shape [1, D, H, W]

        # initialize inference module
        inf_mod = InferenceModule(self.autoencoder, self.unet, self.scheduler, self.device)
        # run inference (assumes run_inference returns tensor)
        dose = inf_mod.run_inference(ct_tensor)
        return dose
