# system_manager.py

import torch
from data_management import DataLoaderModule
from training_pipeline import AutoencoderTrainer, DiffusionTrainer, EarlyStopping
from inference_module import InferenceModule
from monai.transforms import Compose, LoadImaged, EnsureChannelFirstd, ToTensord
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
    def __init__(self, root_dir, transforms, resolutions, energies, batch_size, device, seed=42):
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
        self.training_complete = False

    def run_training(self):
        """
        Executes the full training pipeline over all resolution and energy combinations.
        
        For each combination, the following steps are executed:
          1. Load and split the dataset filtered by the current energy.
          2. Create DataLoaders for training and validation.
          3. Initialize the autoencoder, discriminator, and diffusion models.
          4. Run the autoencoder training loop with early stopping based on validation loss.
        
        After all configurations have been processed, a flag is set to indicate that training is complete.
        """
        # Iterate over all resolution and energy combinations
        for res in self.resolutions:
            print(f"\n--- Starting training for resolution {res} (all energy levels integrated) ---")
            
            # Create a DataLoaderModule instance with the given energy
            data_module = DataLoaderModule(
                root_dir=self.root_dir,
                transforms=self.transforms,
                energy=None,
                train_ratio=0.8,
                seed=self.seed
            )
            # Load and split dataset
            ds_full = data_module.load_dataset(section="training")
            train_ds, val_ds = data_module.split_dataset(ds_full)
            train_loader = data_module.create_data_loader(train_ds, self.batch_size, shuffle=True)
            val_loader = data_module.create_data_loader(val_ds, self.batch_size, shuffle=False)
            
            print(f"Dataset loaded: {len(ds_full)} samples. Train: {len(train_ds)}, Val: {len(val_ds)}")
            sample_batch = next(iter(train_loader))
            print("Sample batch shape:", sample_batch["image"].shape)
            
            # Initialize models and optimizers here (this is a simplified placeholder)
            # You should replace these with your actual model initializations
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

            # Initialize optimizers
            optimizer_g = Adam(autoencoder.parameters(), lr=1e-4)
            optimizer_d = Adam(discriminator.parameters(), lr=1e-4)
            optimizer_diff = Adam(unet.parameters(), lr=1e-4)
            
            # Set early stopping parameters
            n_epochs = 5  # for testing purposes; adjust as needed
            patience = 3
            early_stopper = EarlyStopping(patience=patience)
            
            autoencoder_trainer = AutoencoderTrainer(autoencoder, discriminator,
                                                        optimizer_g, optimizer_d, self.device)
            
            print("Starting autoencoder training...")
            for epoch in range(n_epochs):
                train_loss, gen_loss, disc_loss = autoencoder_trainer.train_one_epoch(train_loader, epoch)
                val_loss = autoencoder_trainer.validate(val_loader)
                print(f"Epoch {epoch+1}/{n_epochs} - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
                
                # Early stopping check
                if early_stopper.update(val_loss):
                    print("Early stopping triggered.")
                    break
            # Mark training as complete for this configuration
            print(f"Training complete for resolution {res} (all energies integrated).\n")
            
            # Here we could save the best model for this configuration if needed.
            # torch.save(autoencoder.state_dict(), f"autoencoder_res{res}_energy{energy}.pt")
            # torch.save(unet.state_dict(), f"diffusion_res{res}_energy{energy}.pt")

        # End of training loop for all resolution-energy combinations
        self.training_complete = True
        print("All training configurations completed. System is ready for inference.")

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

