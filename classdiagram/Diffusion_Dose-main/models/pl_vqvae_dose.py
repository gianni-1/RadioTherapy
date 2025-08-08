import pytorch_lightning as pl
import torch
import torch.nn.functional as F
# import matplotlib
# matplotlib.use('Agg')
import matplotlib.pyplot as plt
from models.vqvae import VQVAE3D
import wandb


class LightningVQVAE(pl.LightningModule):
    def __init__(self, im_channels, model_config, learning_rate=1e-3, sample_log_interval=5):
        super().__init__()
        self.vqvae = VQVAE3D(im_channels, model_config)
        self.learning_rate = learning_rate
        self.sample_log_interval = sample_log_interval
        self.model_config = model_config

    def forward(self, x, context=None):
        return self.vqvae(x.to(self.device), context)

    def encode(self, x, context=None):
        return self.vqvae.encode(x.to(self.device), context)

    def decode(self, x, context=None, image_shape=None):
        return self.vqvae.decode(x.to(self.device), context, image_shape)

    def training_step(self, batch, batch_idx):
        x, cond = batch
        x = x.float()
        recon, _, quant_losses = self(x)
        loss = self.compute_losses(x, recon, quant_losses, prefix="train")
        return loss

    def validation_step(self, batch, batch_idx):
        x, cond = batch
        x = x.float()
        recon, _, quant_losses = self(x)
        self.compute_losses(x, recon, quant_losses, prefix="val")

    def compute_losses(self, x, recon, quant_losses, prefix="train"):
        reconstruction_loss = F.mse_loss(recon, x)
        codebook_loss = quant_losses['codebook_loss']
        commitment_loss = quant_losses['commitment_loss']
        total_loss = reconstruction_loss + codebook_loss + commitment_loss

        self.log(f'{prefix}/reconstruction_loss', reconstruction_loss)
        self.log(f'{prefix}/codebook_loss', codebook_loss)
        self.log(f'{prefix}/commitment_loss', commitment_loss)
        self.log(f'{prefix}/total_loss', total_loss)

        return total_loss

    def on_train_epoch_end(self):
        if self.current_epoch % self.sample_log_interval == 0:
            self.generate_and_plot_samples()

    def generate_and_plot_samples(self):
        x, cond = next(iter(self.trainer.datamodule.val_dataloader()))
        x = x.float()
        self.eval()
        with torch.no_grad():
            recon, z, _ = self(x)
        self._plot_samples(x, z, recon)
        self.train()

    def _plot_samples(self, originals, z, reconstructions, num_samples=2):
        """
        Visualize central slice of 3D volumes (assumes shape B x C x D x H x W).
        """
        originals = originals[:num_samples].cpu()
        z = z[:num_samples].cpu()
        reconstructions = reconstructions[:num_samples].cpu()

        fig, axes = plt.subplots(num_samples, 3, figsize=(9, 3 * num_samples))

        for i in range(num_samples):
            # Take center slice along depth dimension (D)
            mid_slice = originals.shape[2] // 2
            orig_slice = originals[i, 0, mid_slice, :, :]
            latent_slice = z[i, 0, mid_slice // 2**(sum(self.model_config['down_sample'])+1), :, :]
            recon_slice = reconstructions[i, 0, mid_slice, :, :]

            axes[i, 0].imshow(orig_slice, cmap="gray")
            axes[i, 0].set_title("Original")
            axes[i, 1].imshow(latent_slice, cmap="gray")
            axes[i, 1].set_title("Latent")
            axes[i, 2].imshow(recon_slice, cmap="gray")
            axes[i, 2].set_title("Reconstructed")

            for j in range(3):
                axes[i, j].axis("off")

        plt.tight_layout()
        self.logger.experiment.log({f"vqvae_outputs_epoch_{self.current_epoch}": wandb.Image(fig)})
        plt.close(fig)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)
