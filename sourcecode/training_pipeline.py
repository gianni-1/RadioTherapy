#training_pipeline.py

import torch
import torch.nn.functional as F
from torch.nn import L1Loss
from tqdm import tqdm

###############################################################################
# EarlyStopping
###############################################################################
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

###############################################################################
# AutoenconderTrainer
###############################################################################
class AutoenconderTrainer:
    """
    Encapsulates the training process for the autoencoder, including validation.
    """
    def __init__(self, autoenconder, discriminator, optim_g, optim_d, device,
                 kl_weight=1e-6, adv_weight=0.01, perceptual_weight=0.001, warm_up_epochs=2):
        """
        Args:
            autoenconder (torch.nn.Module): The autoencoder model.
            discriminator (torch.nn.Module): The discriminator model (for adversarial training).
            optim_g (torch.optim.Optimizer): Optimizer for the generator.
            optim_d (torch.optim.Optimizer): Optimizer for the discriminator.
            device (str): Device to run the training on ('cpu' or 'cuda').
            kl_weight (float): Weight for KL divergence loss.
            adv_weight (float): Weight for adversarial loss.
            perceptual_weight (float): Weight for perceptual loss.
            warm_up_epochs (int): Number of epochs to warm up the KL weight.
        """
        self.autoenconder = autoenconder
        self.discriminator = discriminator
        self.optim_g = optim_g
        self.optim_d = optim_d
        self.device = device
        self.kl_weight = kl_weight
        self.adv_weight = adv_weight
        self.perceptual_weight = perceptual_weight
        self.warm_up_epochs = warm_up_epochs
        self.l1_loss = L1Loss()

    def train_one_epoch(self, train_loader, epoch):
        """
        Trains the autoencoder and the discriminator for an epoch.
        Args:
            train_loader (DataLoader): DataLoader for training data.
            epoch (int): Current epoch.
        Returns:
            tuple: Average reconstruction loss, generator loss and discriminator loss.
        """
        self.autoenconder.train()
        self.discriminator.train()
        epoch_loss = 0.0
        gen_epoch_loss = 0.0
        disc_epoch_loss = 0.0

        for step, batch in enumerate(tqdm(train_loader, desc=f"Autoencoder Epoch {epoch}")):
            images = batch['image'].to(self.device)
            self.optimizer_g.zero_grad(set_to_none=True)

            reconstruction, z_mu, z_sigma = self.autoenconder(images)

            #calculate KL loss
            kl_loss = 0.5 * torch.sum(
                z_mu.pow(2) + z_sigma.pow(2) - torch.log(z_sigma.pow(2)) - 1
            ) /images.size(0)
            recons_loss = self.l1_loss(reconstruction.float(), images.float())
            loss_g = recons_loss + self.kl_weight * kl_loss

            if epoch > self.warm_up_epochs and self.discriminator is not None:
                logits_fake = self.discriminator(reconstruction.contiguous().float())[-1]
                #For the example: adversarial loss as MSE, target: ones (real)
                generator_loss = F.mse_loss(logits_fake, torch.ones_like(logits_fake))
                gen_epoch_loss += generator_loss.item()

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
                reconstruction, z_mu, z_sigma = self.autoencoder(images)
                loss = self.l1_loss(reconstruction, images)
                val_loss += loss.item()
        avg_val_loss = val_loss / len(val_loader)
        return avg_val_loss

###############################################################################
# DiffusionTrainer
###############################################################################
class DiffusionTrainer:
    """
    Encapsulates the training process for the diffusion model.    """
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