import os
from datetime import datetime
import torch
import torch.nn as nn
import pytorch_lightning as pl
# import matplotlib
# matplotlib.use('Agg')
import matplotlib.pyplot as plt
import wandb
import numpy as np
from utils.evaluation_utils import ImageSimilarityMetrics
from utils.resample import LossAwareSampler, UniformSampler
from torch_ema import ExponentialMovingAverage
from denoiser.karras_denoiser import karras_sample
from transformers import get_cosine_schedule_with_warmup


class LightningConsistencyModel(pl.LightningModule):
    def __init__(
        self,
        model: nn.Module,  # Student model
        vae_model: nn.Module,
        diffusion: any,
        teacher_model: nn.Module,
        teacher_diffusion: any,
        target_model: nn.Module,  # Target model
        schedule_sampler: any = None,
        training_mode="consistency_distillation",  # Options: "consistency_distillation", "consistency_training", "progdist"
        plot_example_images_epoch_start=30,
        weight_schedule="uniform",
        lr=1e-4,
        num_scales=10,
        weight_decay=0.0,
        ema_decay=0.999,
        sigma_min=0.002,
        sigma_max=80,
        rho=7,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=[
            'diffusion',  # Diffusion scheduler
            'schedule_sampler'  # Sampling schedule
        ])

        self.model = model
        self.vae = vae_model
        self.diffusion = diffusion

        self.teacher_model = teacher_model
        self.teacher_diffusion = teacher_diffusion
        self.target_model = target_model
        self.training_mode = training_mode
        self.weight_schedule = weight_schedule
        self.lr = lr
        self.weight_decay = weight_decay
        self.ema_decay = ema_decay
        self.plot_example_images_epoch_start = plot_example_images_epoch_start
        self.schedule_sampler = schedule_sampler or UniformSampler(diffusion)
        self.num_scales = num_scales

        if teacher_model:
            self.teacher_model.requires_grad_(False)
            self.teacher_model.eval()

        # Ensure UNet parameters are trainable
        for param in self.model.parameters():
            param.requires_grad = True

        self.similarity_calculator = ImageSimilarityMetrics()

        self.ema = ExponentialMovingAverage(self.model.parameters(), decay=ema_decay)

        ###############
        # MANUAL OPTIM CODE (to show it performs the same as the automatic including grad clipping
        self.automatic_optimization = False  # Set manual optimization
        self.scaler = torch.cuda.amp.GradScaler()  # Use GradScaler for mixed precision training
        optimizer_list, schedulers = self.configure_optimizers()
        self.scheduler = schedulers[0]["scheduler"]  # Extract the actual scheduler object
        self.optimizer = optimizer_list[0]
        self.best_val_loss = float("inf")  # Track the best validation loss
        self.start_time = datetime.now().strftime("%d_%m_%Y-%H-%M")  # Store the training start time once
        ################

        # Initialize the target model
        if self.target_model is not None:
            self.target_model.load_state_dict(self.model.state_dict())  # Copy initial weights
            self.target_model.requires_grad_(False)  # Ensure the target model is not trainable

        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.rho = rho

    def forward(self, noisy_latents, t, cond_input=None):
        return self.model(noisy_latents, t, cond_input)

    def training_step(self, batch, batch_idx, logging=True):
        images, cond_input = batch
        images = images.float()

        # Encode images to latent space
        with torch.no_grad():
            z, _ = self.vae.encode(images, None)

        t, weights = self.schedule_sampler.sample(images.shape[0], self.device)

        losses = self.diffusion.consistency_losses(
            self.model,
            z,
            self.num_scales,
            target_model=self.target_model,
            teacher_model=self.teacher_model,
            teacher_diffusion=self.teacher_diffusion,
            model_kwargs=cond_input,
        )

        if isinstance(self.schedule_sampler, LossAwareSampler):
            self.schedule_sampler.update_with_local_losses(
                t, losses["loss"].detach()
            )

        loss = (losses["loss"] * weights).mean()

        if logging:
            self.log("train/loss", loss, prog_bar=True)

        # ✅ Zero gradients before backward pass
        self.optimizer.zero_grad()

        # ✅ Compute scaled gradients using GradScaler
        self.scaler.scale(loss).backward()

        # ✅ Unscale gradients before clipping (must do this in manual optimization!)
        self.scaler.unscale_(self.optimizer)

        # ✅ Apply manual gradient clipping (Same as Trainer's `gradient_clip_val=1.0`)
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

        # ✅ Perform optimizer step with GradScaler
        self.scaler.step(self.optimizer)
        self.scaler.update()  # Updates scaling factor for next step

        # Update EMA weights for the student model
        self.ema.update()

        # Explicitly update the target model with dynamic EMA rate
        self.update_target_ema()

        # ✅ Step the LR scheduler manually
        self.scheduler.step()

        # ✅ Log learning rate and EMA decay after optimizer step
        if logging:
            self.log("lr", self.optimizer.param_groups[0]['lr'], prog_bar=True, on_step=True)
            self.log("ema_decay", self.ema_scale_fn(self.global_step), on_step=True)

        return loss

    def validation_step(self, batch, batch_idx, logging=True):
        # Apply EMA weights
        self.ema.store()
        self.ema.copy_to()

        images, cond_input = batch
        images = images.float()

        # Encode images to latent space
        with torch.no_grad():
            z, _ = self.vae.encode(images, None)

        t, weights = self.schedule_sampler.sample(images.shape[0], self.device)

        losses = self.diffusion.consistency_losses(
            self.model,
            z,
            self.num_scales,
            target_model=self.target_model,
            teacher_model=self.teacher_model,
            teacher_diffusion=self.teacher_diffusion,
            model_kwargs=cond_input,
        )

        if isinstance(self.schedule_sampler, LossAwareSampler):
            self.schedule_sampler.update_with_local_losses(
                t, losses["loss"].detach()
            )

        # Restore original weights
        self.ema.restore()

        loss = (losses["loss"] * weights).mean()

        if logging:
            self.log("val/loss", loss, prog_bar=True, on_step=False, on_epoch=True)

        return loss

    def test_step(self, batch, batch_idx):
        """Test step for generating and evaluating images"""
        images, cond_input = batch
        images = images.float()

        # Encode images to latent space
        with torch.no_grad():
            z, _ = self.vae.encode(images, None)

            # Generate samples using one-step generation
            _, _ = self.generate_samples(
                shape=z.shape,
                cond_input=cond_input,
                return_latent=True  # Set based on your model's behavior
            )

        return None

    def on_validation_epoch_end(self):
        # Update target model with EMA after validation
        self.update_target_ema()

        ########################################
        total_difference = 0
        total_params = 0

        for param1, param2 in zip(self.model.parameters(), self.target_model.parameters()):
            total_difference += torch.sum((param1 - param2) ** 2).item()
            total_params += param1.numel()

        mean_difference = total_difference / total_params
        print(f"Mean squared difference between model and target model: {mean_difference}")
        ########################################

        if self.current_epoch < self.plot_example_images_epoch_start:
            return

        val_batch = next(iter(self.trainer.datamodule.val_dataloader()))
        images, cond_input = val_batch
        images = images.float().to(self.device)

        with torch.no_grad():
            im, _ = self.vae.encode(images, None)
            _, _ = self.generate_samples(shape=im.shape, cond_input=cond_input, return_latent=True)


    def get_denoised_latent(self, noisy_latents, t, cond_input=None):
        """
        Wrapper for KarrasDiffusion's get_denoised_latent method.
        """
        return self.diffusion.get_denoised_latent(self.model, noisy_latents, t, cond_input)

    def generate_samples(self, shape, cond_input=None, return_latent=False, log_images=True):
        """
        Generate samples using the NoiseSampler.
        Args:
            shape (tuple): Shape of the latent space (e.g., (channels, height, width)).
            cond_input (dict): Conditioning inputs for the model.
            return_latent (bool): If true, return the consistency generated image without VAE decoding
            log_images (bool): If true, generate plots and log images to logger

        Returns:
            torch.Tensor: Decoded samples from the VAE.
        """
        self.model.eval()

        # Generate latent samples via diffusion sampling
        with torch.no_grad():
            sampled_latents = karras_sample(
                diffusion=self.diffusion,
                model=self.model,
                shape=shape,
                steps=self.num_scales,
                model_kwargs=cond_input,
                device=self.device,
                sigma_min=self.sigma_min,
                sigma_max=self.sigma_max,
                rho=self.rho,
            )

            generated_images = self.vae.decode(sampled_latents)

        if log_images:
            # Calculate all metrics using the calculate_all_metrics() method
            real_images, metrics = self.similarity_calculator.calculate_all_metrics(cond_input['tensor'], generated_images)

            # Compute the mean across all images for each metric
            mse_mean = metrics['mse']['mean']
            ssim_mean = metrics['ssim']['mean']
            psnr_mean = metrics['psnr']['mean']
            lpips_mean = metrics['lpips']['mean']

            # Log mean metrics to WandB
            wandb.log({
                'consistency_epoch': self.current_epoch,
                'consistency_mse_mean': mse_mean,
                'consistency_ssim_mean': ssim_mean,
                'consistency_psnr_mean': psnr_mean,
                'consistency_lpips_mean': lpips_mean
            })

            # Visualize and log the generated images
            fig, axs = plt.subplots(1, 2, figsize=(11, 5))
            for i, ax in enumerate(axs):
                ax.imshow(generated_images[i, 0].detach().cpu().numpy(), cmap="gray")
                ax.set_title(f"Cond: {np.around(cond_input['tensor'][i, :].cpu().numpy(), decimals=2)}", fontsize=10)
                ax.axis("off")
            plt.tight_layout()

            # Log the figure
            self.logger.log_image(
                key="consistency_generated_images_epoch",
                images=[wandb.Image(fig)],
                step=self.current_epoch
            )
            plt.close(fig)

            # Visualize and log the real images
            fig, axs = plt.subplots(1, 2, figsize=(11, 5))
            for i, ax in enumerate(axs):
                ax.imshow(real_images[i, 0].detach().cpu().numpy(), cmap="gray")
                ax.set_title(f"Cond: {np.around(cond_input['tensor'][i, :].cpu().numpy(), decimals=2)}", fontsize=10)
                ax.axis("off")
            plt.tight_layout()

            # Log the figure
            self.logger.log_image(
                key="real_images_epoch",
                images=[wandb.Image(fig)],
                step=self.current_epoch
            )
            plt.close(fig)

        if return_latent:
            return generated_images, sampled_latents

        self.model.train()

        return generated_images

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)

        # Use defaults if setup() hasn't run
        total_steps = getattr(self, 'total_steps', 10000)
        num_warmup_steps = getattr(self, 'num_warmup_steps', 1000)

        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=total_steps,
        )
        return [optimizer], [{"scheduler": scheduler, "interval": "step"}]

    def setup(self, stage=None):
        if self.trainer is not None:
            train_dataloader = self.trainer.datamodule.train_dataloader()
            self.total_steps = self.trainer.max_epochs * len(train_dataloader)

            # Save warmup steps too (optional from sweep config)
            warmup_ratio = getattr(self.hparams, "warmup_ratio", 0.1)
            self.num_warmup_steps = int(warmup_ratio * self.total_steps)

    def on_validation_end(self) -> None:
        if not self.automatic_optimization:
            """Manually save the best checkpoint when val_loss improves."""
            val_loss = self.trainer.callback_metrics.get("val/loss", None)

            if val_loss is None:
                print("⚠️ val/loss not found in callback metrics!")
                return

            val_loss = val_loss.item()  # Convert from tensor to float

            # Check if the new validation loss is better
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss  # Update best loss

                # Use the stored start_time
                checkpoint_path = os.path.join(self.trainer.default_root_dir, "./checkpoints/consistency_model/",
                                               f"best_model_{self.start_time}.ckpt")
                print(f"✅ New best model found! Saving checkpoint to {checkpoint_path}")

                self.trainer.save_checkpoint(checkpoint_path)

    def on_save_checkpoint(self, checkpoint):
        checkpoint["ema"] = self.ema.state_dict()

    def state_dict(self, *args, **kwargs):
        state = super().state_dict(*args, **kwargs)  # Ensure compatibility with PyTorch's state_dict
        if hasattr(self.ema, "state_dict") and callable(getattr(self.ema, "state_dict")):
            state["ema"] = self.ema.state_dict()
        return state
    def on_load_checkpoint(self, checkpoint):
        if "ema" in checkpoint:
            self.ema.load_state_dict(checkpoint["ema"])
            print("✅ EMA weights successfully loaded!")
        else:
            print("⚠️ No EMA weights found in checkpoint.")

    def load_state_dict(self, state_dict, strict=True, *args, **kwargs):
        if "ema" in state_dict:
            self.ema.load_state_dict(state_dict["ema"])  # Load EMA weights
            print("✅ EMA weights restored from checkpoint!")
        else:
            print("⚠️ No EMA weights found in checkpoint!")

        # Remove "ema" from state_dict to avoid conflicts with strict loading
        state_dict = {k: v for k, v in state_dict.items() if k != "ema"}

        super().load_state_dict(state_dict, strict=strict, *args, **kwargs)

    def update_target_ema(self):
        target_ema = self.ema_scale_fn(self.global_step)
        with torch.no_grad():
            for param_target, param_student in zip(
                    self.target_model.parameters(), self.model.parameters()
            ):
                param_target.data.mul_(target_ema).add_(
                    param_student.data, alpha=(1 - target_ema)
                )

    def ema_scale_fn(self, step):
        # Example: Decrease EMA rate over time
        base_rate = self.ema_decay  # Start with a high rate
        min_rate = 0.0  # Ensure it doesn't drop below this
        progress = step / max(self.trainer.max_steps, 1)
        return max(base_rate * (1 - progress), min_rate)
