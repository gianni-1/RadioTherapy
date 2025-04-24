# system_manager.py

import shutil
import os
import torch
from data_management import DataLoaderModule
from training_pipeline import AutoencoderTrainer, DiffusionTrainer, EarlyStopping
from inference_module import InferenceModule
from monai.transforms import Compose, LoadImaged, EnsureChannelFirstd, ToTensord
from visualization import Visualization
from monai.data import NibabelReader
from generative.networks.nets import AutoencoderKL, PatchDiscriminator, DiffusionModelUNet
from torch.optim import Adam

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
    def __init__(self, root_dir, transforms, resolutions, energies, batch_size, device, num_epochs, learning_rate, patience, seed=42):
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

    def save_models(self, unet, optimizer_diff, optimizer_g, optimizer_d, epoch):
        checkpoint_path = f"model_res{self.resolutions}_energy{self.energies}.ckpt"
        torch.save(
            {
                "autoencoder": self.state_dict(),
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
                ds_full = data_module.load_dataset(section="training")
                train_ds, val_ds = data_module.split_dataset(ds_full)
                train_loader = data_module.create_data_loader(train_ds, self.batch_size, shuffle=True)
                val_loader   = data_module.create_data_loader(val_ds, self.batch_size, shuffle=False)
                
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
                    num_res_blocks=1,
                    num_channels=(32, 64, 64),
                    attention_levels=(False, True, True),
                    num_head_channels=(0, 64, 64),
                ).to(self.device)
                
                # optimizers
                opt_g = Adam(autoencoder.parameters(), lr=self.learning_rate)
                opt_d = Adam(discriminator.parameters(), lr=self.learning_rate)
                opt_diff = Adam(unet.parameters(), lr=self.learning_rate)
                
                # trainers and early stopping
                ae_trainer = AutoencoderTrainer(autoencoder, discriminator, opt_g, opt_d, self.device)
                stopper = EarlyStopping(patience=self.patience)
                for epoch in range(self.num_epochs):
                    train_loss, gen_loss, disc_loss = ae_trainer.train_one_epoch(train_loader, epoch)
                    val_loss = ae_trainer.validate(val_loader)
                    # record losses
                    ae_train_losses.append(train_loss)
                    ae_val_losses.append(val_loss)
                    gen_losses.append(gen_loss)
                    disc_losses.append(disc_loss)
                    if stopper.update(val_loss):
                        print("Early stopping at epoch", epoch+1)
                        break
                
                # now diffusion training
                diff_trainer = DiffusionTrainer(unet, opt_diff, self.device)
                
                for epoch in range(self.num_epochs):
                    diff_loss = diff_trainer.train_one_epoch(train_loader, epoch)
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
                self.save_models(unet, opt_diff, opt_g, opt_d, epoch)
        # after loops
        self.training_complete = True
        print("All training finished.")

    def run_inference(self):
        """
        Executes the inference process if training is complete.
        This function should load the trained models and run the inference module to generate outputs.
        """
        if not self.training_complete:
            print("Training is not complete yet. Inference cannot be started.")
            return
        
        # Here you would load your trained models. For demonstration, we'll use dummy models.
        print("Running inference...")
        # Dummy example of inference:
        # from inference_module import InferenceModule
        # inferencer = InferenceModule(trained_autoencoder, trained_diffusion, scheduler, self.device)
        # dummy_noise = torch.randn((1, 3, 24, 24, 16))
        # output = inferencer.run_inference(dummy_noise)
        # print("Inference output shape:", output.shape)
        print("Inference module is not fully implemented in this example.")
