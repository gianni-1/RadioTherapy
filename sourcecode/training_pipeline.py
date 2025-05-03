#training_pipeline.py

import torch
import torch.nn.functional as F
from torch.nn import L1Loss
from tqdm import tqdm
from monai.losses import PatchAdversarialLoss, PerceptualLoss
import math

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
            images = batch["input"].to(self.device) #Expected shape: [B, 1, D, H, W]

            # Determine if the autoencoder expects an extra channel for energy
            first_conv = next((m for m in self.autoencoder.modules() if isinstance(m, torch.nn.Conv3d)), None)
            expected_in_channels = first_conv.in_channels if first_conv else images.shape[1]
            # print(f"Expected input channels: {expected_in_channels}, Actual input channels: {images.shape[1]}")
            # Apply energy conditioning only if the model was built with an extra input channel
            if "energy" in batch and expected_in_channels == images.shape[1] + 1:
                energies = batch["energy"].to(self.device)
                normalized_energy = energies.float() / 100.0
                B, C, D, H, W = images.shape
                energy_tensor = normalized_energy.view(B, 1, 1, 1, 1).expand(B, 1, D, H, W)
                conditioned_input = torch.cat([images, energy_tensor], dim=1)
            else:
                conditioned_input = images

            # Ensure only expected channels are passed
            if conditioned_input.ndim == 5 and conditioned_input.shape[1] > expected_in_channels:
                conditioned_input = conditioned_input[:, :expected_in_channels, ...]

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
                adv = self.adv_loss(logits_fake, target_is_real=True, for_discriminator=False)
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
                images = batch["input"].to(self.device)
                # Apply energy conditioning as in training
                first_conv = next((m for m in self.autoencoder.modules() if isinstance(m, torch.nn.Conv3d)), None)
                expected_in_channels = first_conv.in_channels if first_conv else images.shape[1]
                if "energy" in batch and expected_in_channels == images.shape[1] + 1:
                    energies = batch["energy"].to(self.device)
                    normalized_energy = energies.float() / 100.0
                    B, C, D, H, W = images.shape
                    energy_tensor = normalized_energy.view(B, 1, 1, 1, 1).expand(B, 1, D, H, W)
                    conditioned_input = torch.cat([images, energy_tensor], dim=1)
                else:
                    conditioned_input = images

                # DEBUG: check and clamp conditioned_input channels for validation
                print(f"[AE VAL DEBUG] conditioned_input.shape = {tuple(conditioned_input.shape)}")
                if conditioned_input.ndim == 5 and conditioned_input.shape[1] > expected_in_channels:
                    conditioned_input = conditioned_input[:, :expected_in_channels, ...]

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
            # Build conditioned input for autoencoder with energy channel if available
            if "energy" in batch:
                energies = batch["energy"].to(self.device)
                normalized_energy = energies.float() / 100.0
                B, C, D, H, W = images.shape
                energy_tensor = normalized_energy.view(B, 1, 1, 1, 1).expand(B, 1, D, H, W)
                conditioned_input = torch.cat([images, energy_tensor], dim=1)
            else:
                conditioned_input = images

            # Zero gradients for diffusion optimizer
            self.optimizer_diff.zero_grad(set_to_none=True)

            # Encode conditioned input to latents using the autoencoder
            with torch.no_grad():
                encoded = autoencoder.encode(conditioned_input)
                # If encode returns an object with a latent_dist attribute:
                if hasattr(encoded, "latent_dist"):
                    latents = encoded.latent_dist.sample()
                # If encode returns a tuple, assume the first element is the latent tensor:
                elif isinstance(encoded, tuple):
                    latents = encoded[0]
                # Otherwise, treat the return as the latent tensor directly:
                else:
                    latents = encoded
                latents = latents.to(self.device)
                
                # Crop latents to ensure spatial dimensions divisible by downsampling factor
                # Determine number of downsample steps in UNet (Conv3d with stride >1)
                down_convs = [m for m in self.diffusion_model.modules() if isinstance(m, torch.nn.Conv3d) and hasattr(m, "stride") and m.stride[0] > 1]
                n_down = len(down_convs)
                factor = 2 ** n_down if n_down > 0 else 1
                B, C, D, H, W = latents.shape
                new_D = (D // factor) * factor
                new_H = (H // factor) * factor
                new_W = (W // factor) * factor
                if new_D != D or new_H != H or new_W != W:
                    latents = latents[:, :, :new_D, :new_H, :new_W]
                # Ensure latent channel count matches diffusion model's expected in_channels
                first_conv_diff = next((m for m in self.diffusion_model.modules() if isinstance(m, torch.nn.Conv3d)), None)
                expected_latent_ch = first_conv_diff.in_channels if first_conv_diff else latents.shape[1]
                if latents.ndim == 5 and latents.shape[1] != expected_latent_ch:
                    # If model expects a single-channel latent, collapse via mean
                    if expected_latent_ch == 1:
                        latents = latents.mean(dim=1, keepdim=True)
                    else:
                        # Otherwise slice to match expected channels
                        latents = latents[:, :expected_latent_ch, ...]
           
            # generates random noise
            noise = torch.randn_like(latents)

            #  generates random Timesteps
            timesteps = torch.randint(0, inferer.scheduler.num_train_timesteps, (images.shape[0],), device=self.device).long()

            # Build context vector by pooling latent representation of CT
            with torch.no_grad():
                # Use the same conditioned_input (CT + energy) as for the autoencoder encoding
                encoded_ct = autoencoder.encode(conditioned_input)
                # if encode returns (mu, sigma), take mu
                if isinstance(encoded_ct, tuple):
                    latent_ct = encoded_ct[0]
                else:
                    latent_ct = encoded_ct
            # global average pooling over spatial dimensions -> [B, latent_channels]
            context_tensor = latent_ct.mean(dim=(2, 3, 4))
            # add sequence dimension for cross-attention: [B, 1, latent_channels]
            context_tensor = context_tensor.unsqueeze(1)
            noise_pred = inferer(
                inputs=latents,
                autoencoder_model=autoencoder,
                diffusion_model=self.diffusion_model,
                noise=noise,
                timesteps=timesteps,
                condition=context_tensor,
            )
            # Crop original noise to match predicted noise spatial dimensions
            if noise.ndim == noise_pred.ndim and noise.shape[2:] != noise_pred.shape[2:]:
                noise = noise[:, :, :noise_pred.shape[2], :noise_pred.shape[3], :noise_pred.shape[4]]
            loss = F.mse_loss(noise_pred.float(), noise.float())
            loss.backward()
            self.optimizer_diff.step()
            epoch_loss += loss.item()

        avg_loss = epoch_loss / (step + 1)
        return avg_loss