#training_pipeline.py

import torch
import torch.nn.functional as F
from torch.nn import L1Loss
from tqdm import tqdm
from monai.losses import PatchAdversarialLoss, PerceptualLoss

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
    def __init__(self, autoencoder, discriminator, optimizer_g, optimizer_d, device,
                 kl_weight=1e-6, adv_weight=0.01, perceptual_weight=0.001, warm_up_epochs=2):
        """
        Args:
            autoencoder (torch.nn.Module): The autoencoder model.
            discriminator (torch.nn.Module): The discriminator model (for adversarial training).
            optimizer_g (torch.optim.Optimizer): Optimizer for the generator.
            optimizer_d (torch.optim.Optimizer): Optimizer for the discriminator.
            device (str): Device to run the training on ('cpu' or 'cuda').
            kl_weight (float): Weight for KL divergence loss.
            adv_weight (float): Weight for adversarial loss.
            warm_up_epochs (int): Number of epochs to warm up the KL weight.
        """
        self.autoencoder = autoencoder
        self.discriminator = discriminator
        self.optimizer_g = optimizer_g
        self.optimizer_d = optimizer_d
        self.device = device
        self.kl_weight = kl_weight
        self.adv_weight = adv_weight
        self.perceptual_weight = perceptual_weight
        # adversarial and perceptual loss modules
        self.adv_loss = PatchAdversarialLoss(criterion="least_squares")
        self.perceptual_loss = PerceptualLoss(
            spatial_dims=3, network_type="squeeze", is_fake_3d=True, fake_3d_ratio=0.2
        ).to(self.device)
        self.warm_up_epochs = warm_up_epochs
        self.l1_loss = L1Loss()

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

        for step, batch in enumerate(tqdm(train_loader, desc=f"Autoencoder Epoch {epoch}")):
            #Move images to device
            images = batch['image'].to(self.device)  #Expected shape: [B, 1, D, H, W]

            #Check if energy information is available; if yes, condition the input.
            if "energy" in batch:
                energies = batch["energy"].to(self.device)  #Expected shape: [B]
                # Normalize the energy values (example normalization: divide by 100)
                normalized_energy = energies.float() / 100.0  # Shape: [B]
                B, C, D, H, W = images.shape
                #Reshape energies to [B, 1, 1, 1, 1] and expand to [B,1 ,D, H, W]
                energy_tensor = normalized_energy.view(B, 1, 1, 1, 1).expand(B, 1, D, H, W)
                #Concatenate along the channel dimension, resulting in a conditioned input of shape [B, 2, D, H, W]
                conditioned_input = torch.cat([images, energy_tensor], dim=1)
            else:
                conditioned_input = images

            #Zero the gradients for the generator 
            self.optimizer_g.zero_grad(set_to_none=True)
            # Pass the conditioned input through the autoencoder
            # (Note: autoencoder's in_channels should be updated to handle conditioned input)
            reconstruction, z_mu, z_sigma = self.autoencoder(conditioned_input)

            #calculate KL loss
            kl_loss = 0.5 * torch.sum(
                z_mu.pow(2) + z_sigma.pow(2) - torch.log(z_sigma.pow(2)) - 1
            ) /images.size(0)
            # Compute reconstruction loss (L1 loss)
            recons_loss = self.l1_loss(reconstruction.float(), images.float())
            # compute adversarial loss if using discriminator
            adv = 0.0
            if epoch > self.warm_up_epochs and self.discriminator is not None:
                logits_fake = self.discriminator(reconstruction.contiguous().float())[-1]
                adv = self.adv_loss(logits_fake, torch.ones_like(logits_fake))
            # compute perceptual loss
            perc = self.perceptual_loss(reconstruction, images)
            # total generator loss
            loss_g = recons_loss + self.kl_weight * kl_loss + self.adv_weight * adv + self.perceptual_weight * perc

            # Add adversarial loss if past warm-up phase and discriminator is used.
            if epoch > self.warm_up_epochs and self.discriminator is not None:
                gen_epoch_loss += adv.item()

            loss_g.backward()
            self.optimizer_g.step()

            if epoch > self.warm_up_epochs and self.discriminator is not None:
                self.optimizer_d.zero_grad(set_to_none=True)
                logits_fake = self.discriminator(reconstruction.contiguous().detach())[-1]
                loss_d_fake = F.mse_loss(logits_fake, torch.zeros_like(logits_fake))
                logits_real = self.discriminator(images.contiguous().detach())[-1]
                loss_d_real = F.mse_loss(logits_real, torch.ones_like(logits_real))
                discrimator_loss = (loss_d_fake + loss_d_real) / 2
                loss_d = self.adv_weight * discrimator_loss
                loss_d.backward()
                self.optimizer_d.step()
                disc_epoch_loss += discrimator_loss.item()
            
            epoch_loss += recons_loss.item()
        
        avg_loss = epoch_loss / (step + 1)
        avg_gen_loss = gen_epoch_loss / (step + 1) if self.discriminator is not None else 0
        avg_disc_loss = disc_epoch_loss / (step + 1) if self.discriminator is not None else 0

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
        with torch.no_grad():
            for batch in val_loader:
                images = batch["image"].to(self.device)
                # For validation, energy conditioning is optional.
                if "energy" in batch:
                    energies = batch["energy"].to(self.device)
                    normalized_energy = energies.float() / 100.0
                    B, C, D, H, W = images.shape
                    energy_tensor = normalized_energy.view(B, 1, 1, 1, 1).expand(B, 1, D, H, W)
                    conditioned_input = torch.cat([images, energy_tensor], dim=1)
                else:
                    conditioned_input = images
                
                reconstruction, z_mu, z_sigma = self.autoencoder(conditioned_input)
                loss = self.l1_loss(reconstruction, images)
                val_loss += loss.item()
        avg_val_loss = val_loss / len(val_loader)
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
    def __init__(self, diffusion_model, optimizer_diff, device):
        """
        Args:
            diffusion_model (torch.nn.Module): Das Diffusionsmodell (z. B. ein UNet).
            optimizer_diff (torch.optim.Optimizer): Optimierer für das Diffusionsmodell.
            device (torch.device): CPU oder GPU.
        """
        self.diffusion_model = diffusion_model
        self.optimizer_diff = optimizer_diff
        self.device = device

    def train_one_epoch(self, train_loader, epoch, inferer):
        """
        Args:
            diffusion_model (torch.nn.Module): The diffusion model (e.g. a UNet).
            optimizer_diff (torch.optim.Optimizer): Optimizer for the diffusion model.
            device (torch.device): CPU or GPU.
        """
        self.diffusion_model.train()
        epoch_loss = 0.0

        for step, batch in enumerate(tqdm(train_loader, desc=f"Diffusion Epoch {epoch}")):
            images = batch["image"].to(self.device)
            self.optimizer_diff.zero_grad(set_to_none=True)

            # generates random noise, same shape as images
            noise = torch.randn_like(images).to(self.device)

            #  generates random Timesteps
            timesteps = torch.randint(0, inferer.scheduler.num_train_timesteps, (images.shape[0],), device=self.device).long()

            # predicts the noise
            noise_pred = inferer(
                inputs=images,
                autoencoder_model=None,  # here not used 
                diffusion_model=self.diffusion_model,
                noise=noise,
                timesteps=timesteps,
            )
            loss = F.mse_loss(noise_pred.float(), noise.float())
            loss.backward()
            self.optimizer_diff.step()
            epoch_loss += loss.item()

        avg_loss = epoch_loss / (step + 1)
        return avg_loss