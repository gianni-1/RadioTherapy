import os
import shutil
import tempfile

import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from torch.utils.data import random_split
from monai import transforms
from monai.apps.datasets import CustomDataset
from monai.apps.datasets import DecathlonDataset
from monai.apps.utils import check_hash
from monai.config import print_config
from monai.data import DataLoader
from monai.transforms.intensity.dictionary import ScaleIntensityd
from monai.transforms.utility.dictionary import EnsureChannelFirstd, ToTensord
from monai.utils import first, set_determinism
from monai.transforms import LoadImaged, Randomizable, Compose
from monai.data import NibabelReader
from monai.transforms import SpatialPadd
from torch.cuda.amp import GradScaler, autocast
from torch.amp import autocast
from torch.nn import L1Loss
from tqdm import tqdm
from monai.data import NumpyReader
from generative.inferers import LatentDiffusionInferer
from generative.losses import PatchAdversarialLoss, PerceptualLoss
from generative.networks.nets import (
    AutoencoderKL,
    DiffusionModelUNet,
    PatchDiscriminator,
)
from generative.networks.schedulers.ddpm import DDPMScheduler
import multiprocessing

# Setzt die Startmethode auf 'spawn', was auf macOS und Windows notwendig ist
multiprocessing.set_start_method("spawn", force=True)

# Definiere eine globale Funktion anstelle einer Lambda-Funktion
def select_channel(image, channel=0):
    return image[channel, :, :, :]

print_config()
# -


resolutions = [
    (8.0, 8.0, 8.0),  # sehr grob
    (6.0, 6.0, 6.0),  # grob
]

energies = [62, 75, 90] #Beispielwerte in keV

if __name__ == '__main__':
    # for reproducibility purposes set a seed
    set_determinism(42) 

    # ### Setup a data directory and download dataset
    # Specify a MONAI_DATA_DIRECTORY variable, where the data will be downloaded. If not specified a temporary directory will be used.
    directory = os.environ.get("MONAI_DATA_DIRECTORY")
    root_dir = "/Users/maximilianpalm/Documents/GitHub/RadioTherapy/data"
    print(root_dir)



    # ### Prepare data loader for the training set
    # Here we will download the Brats dataset using MONAI's `DecathlonDataset` class, and we prepare the data loader for the training set.

    # +
    batch_size = 1
    channel = 0  # 0 = Flair
    assert channel in [0, 1, 2, 3], "Choose a valid channel"

    for res in resolutions:
        for energy in energies:
            print(f"Training for resolution: {res}, energy: {energy}")

            train_transforms = transforms.Compose(
                [
                    transforms.LoadImaged(keys=["image"], reader=NibabelReader),
                    transforms.EnsureChannelFirstd(keys=["image"]),
                    transforms.Lambdad(keys="image", func=select_channel),
                    transforms.EnsureChannelFirstd(keys=["image"], channel_dim="no_channel"),
                    transforms.EnsureTyped(keys=["image"]),
                    transforms.Orientationd(keys=["image"], axcodes="RAS"),
                    transforms.Spacingd(keys=["image"], pixdim=(2.4, 2.4, 2.2), mode=("bilinear")),
                    SpatialPadd(keys=["image"], spatial_size=(32, 32, 32), method='symmetric'),
                    transforms.CenterSpatialCropd(keys=["image"], roi_size=(32, 32, 32)),
                    transforms.ScaleIntensityRangePercentilesd(
                        keys="image", lower=0, upper=99.5, b_min=0, b_max=1
                    ),
                ]
            )       
            train_ds = CustomDataset(
                root_dir=root_dir,
                section="training",  # validation
                cache_rate=0.0,  # you may need a few Gb of RAM... Set to 0 otherwise
                num_workers=0,  # Set download to True  if the dataset hasnt been downloaded yet
                transform=train_transforms,
                download=False,
                seed=0,
            )
            train_loader = DataLoader(
                train_ds,
                batch_size=batch_size,
                shuffle=True,
                num_workers=0,
                persistent_workers=False,
            )
            print(f'Image shape {train_ds[0]["image"].shape}')
            # -

            # ## Autoencoder KL
            #
            # ### Define Autoencoder KL network
            #
            # In this section, we will define an autoencoder with KL-regularization for the LDM. The autoencoder's primary purpose is to transform input images into a latent representation that the diffusion model will subsequently learn. By doing so, we can decrease the computational resources required to train the diffusion component, making this approach suitable for learning high-resolution medical images.
            #

            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            print(f"Using {device}")

            # +
            if res == resolutions[0]:
                autoencoder = AutoencoderKL(
                    spatial_dims=3,
                    in_channels=1,
                    out_channels=1,
                    num_channels=(32, 32, 32),
                    latent_channels=2,
                    num_res_blocks=1,
                    norm_num_groups=8,
                    attention_levels=(False, False, True),
                )
                autoencoder.to(device)


                discriminator = PatchDiscriminator(
                    spatial_dims=3, num_layers_d=3, num_channels=32, in_channels=1, out_channels=1
                )
                discriminator.to(device)
                # -

                # ### Defining Losses
                #
                # We will also specify the perceptual and adversarial losses, including the involved networks, and the optimizers to use during the training process.

                # +
                l1_loss = L1Loss()
                adv_loss = PatchAdversarialLoss(criterion="least_squares")
                loss_perceptual = PerceptualLoss(
                    spatial_dims=3, network_type="squeeze", is_fake_3d=True, fake_3d_ratio=0.2
                )
                loss_perceptual.to(device)


                def KL_loss(z_mu, z_sigma):
                    kl_loss = 0.5 * torch.sum(
                        z_mu.pow(2) + z_sigma.pow(2) - torch.log(z_sigma.pow(2)) - 1, dim=[1, 2, 3, 4]
                    )
                    return torch.sum(kl_loss) / kl_loss.shape[0]


            adv_weight = 0.01
            perceptual_weight = 0.001
            kl_weight = 1e-6
            # -
            if res == resolutions[0]:
                optimizer_g = torch.optim.Adam(params=autoencoder.parameters(), lr=1e-4)
                optimizer_d = torch.optim.Adam(params=discriminator.parameters(), lr=1e-4)

            # ### Train model

            # +
            n_epochs = 2   #hier war ursprünglich 100, dann 10 
            autoencoder_warm_up_n_epochs = 2
            val_interval = 10
            epoch_recon_loss_list = []
            epoch_gen_loss_list = []
            epoch_disc_loss_list = []
            val_recon_epoch_loss_list = []
            intermediary_images = []
            n_example_images = 4

            for epoch in range(n_epochs):
                autoencoder.train()
                discriminator.train()
                epoch_loss = 0
                gen_epoch_loss = 0
                disc_epoch_loss = 0
                progress_bar = tqdm(enumerate(train_loader), total=len(train_loader), ncols=110)
                progress_bar.set_description(f"Epoch {epoch}")
                for step, batch in progress_bar:
                    images = batch["image"].to(device)  # choose only one of Brats channels

                    # Generator part
                    optimizer_g.zero_grad(set_to_none=True)
                    reconstruction, z_mu, z_sigma = autoencoder(images)
                    kl_loss = KL_loss(z_mu, z_sigma)

                    recons_loss = l1_loss(reconstruction.float(), images.float())
                    p_loss = loss_perceptual(reconstruction.float(), images.float())
                    loss_g = recons_loss + kl_weight * kl_loss + perceptual_weight * p_loss

                    if epoch > autoencoder_warm_up_n_epochs:
                        logits_fake = discriminator(reconstruction.contiguous().float())[-1]
                        generator_loss = adv_loss(
                            logits_fake, target_is_real=True, for_discriminator=False
                        )
                        loss_g += adv_weight * generator_loss

                    loss_g.backward()
                    optimizer_g.step()

                    if epoch > autoencoder_warm_up_n_epochs:
                        # Discriminator part
                        optimizer_d.zero_grad(set_to_none=True)
                        logits_fake = discriminator(reconstruction.contiguous().detach())[-1]
                        loss_d_fake = adv_loss(
                            logits_fake, target_is_real=False, for_discriminator=True
                        )
                        logits_real = discriminator(images.contiguous().detach())[-1]
                        loss_d_real = adv_loss(
                            logits_real, target_is_real=True, for_discriminator=True
                        )
                        discriminator_loss = (loss_d_fake + loss_d_real) * 0.5

                        loss_d = adv_weight * discriminator_loss

                        loss_d.backward()
                        optimizer_d.step()

                    epoch_loss += recons_loss.item()
                    if epoch > autoencoder_warm_up_n_epochs:
                        gen_epoch_loss += generator_loss.item()
                        disc_epoch_loss += discriminator_loss.item()

                    progress_bar.set_postfix(
                        {
                            "recons_loss": epoch_loss / (step + 1),
                            "gen_loss": gen_epoch_loss / (step + 1),
                            "disc_loss": disc_epoch_loss / (step + 1),
                        }
                    )
                epoch_recon_loss_list.append(epoch_loss / (step + 1))
                epoch_gen_loss_list.append(gen_epoch_loss / (step + 1))
                epoch_disc_loss_list.append(disc_epoch_loss / (step + 1))

            # Validation - Autoencoder
            if (epoch + 1) % val_interval == 0:
                autoencoder.eval()
                discriminator.eval()
                val_loss = 0
                with torch.no_grad():
                    for step, batch in enumerate(val_loader):
                        images = batch["image"].to(device)
                        reconstruction, z_mu, z_sigma = autoencoder(images)
                        loss = F.l1_loss(reconstruction, images)
                        val_loss += loss.item()
                val_loss /= len(val_loader)
                print(f"[Autoencoder] Epoch {epoch+1}/{n_epochs} | "
                    f"Train Loss: {epoch_loss/(step+1):.4f} | Val Loss: {val_loss:.4f}")

        del discriminator
        del loss_perceptual
        torch.cuda.empty_cache()
        # -

            plt.style.use("ggplot")
            plt.title("Learning Curves", fontsize=20)
            plt.plot(epoch_recon_loss_list)
            plt.yticks(fontsize=12)
            plt.xticks(fontsize=12)
            plt.xlabel("Epochs", fontsize=16)
            plt.ylabel("Loss", fontsize=16)
            plt.legend(prop={"size": 14})
            plt.show()
            plt.savefig(f'learning_curves_res{res}_energy{energy}.png')

            plt.title("Adversarial Training Curves", fontsize=20)
            plt.plot(epoch_gen_loss_list, color="C0", linewidth=2.0, label="Generator")
            plt.plot(epoch_disc_loss_list, color="C1", linewidth=2.0, label="Discriminator")
            plt.yticks(fontsize=12)
            plt.xticks(fontsize=12)
            plt.xlabel("Epochs", fontsize=16)
            plt.ylabel("Loss", fontsize=16)
            plt.legend(prop={"size": 14})
            plt.show()
            plt.savefig(f'adversarial_training_curves_res{res}_energy{energy}.png') 

            # ### Visualise reconstructions

            # Plot axial, coronal and sagittal slices of a training sample
            idx = 0
            img = reconstruction[idx, channel].detach().cpu().numpy()
            fig, axs = plt.subplots(nrows=1, ncols=3)
            for ax in axs:
                ax.axis("off")
            ax = axs[0]
            ax.imshow(img[..., img.shape[2] // 2], cmap="gray")
            ax = axs[1]
            ax.imshow(img[:, img.shape[1] // 2, ...], cmap="gray")
            ax = axs[2]
            ax.imshow(img[img.shape[0] // 2, ...], cmap="gray")

            # ## Diffusion Model
            #
            # ### Define diffusion model and scheduler
            #
            # In this section, we will define the diffusion model that will learn data distribution of the latent representation of the autoencoder. Together with the diffusion model, we define a beta scheduler responsible for defining the amount of noise tahat is added across the diffusion's model Markov chain.

            # +
            if res == resolutions[0]:
                unet = DiffusionModelUNet(
                    spatial_dims=3,
                    in_channels=2,
                    out_channels=2,
                    num_res_blocks=1,
                    num_channels=(32, 64, 64),
                    attention_levels=(False, True, True),
                    num_head_channels=(0, 64, 64),
                )
                unet.to(device)


                scheduler = DDPMScheduler(
                    num_train_timesteps=1000,
                    schedule="scaled_linear_beta",
                    beta_start=0.0015,
                    beta_end=0.0195,
                )
            # -

                # ### Scaling factor
                #
                # As mentioned in Rombach et al. [1] Section 4.3.2 and D.1, the signal-to-noise ratio (induced by the scale of the latent space) can affect the results obtained with the LDM, if the standard deviation of the latent space distribution drifts too much from that of a Gaussian. For this reason, it is best practice to use a scaling factor to adapt this standard deviation.
                #
                # _Note: In case where the latent space is close to a Gaussian distribution, the scaling factor will be close to one, and the results will not differ from those obtained when it is not used._
                #

                # +
                with torch.no_grad():
                    with autocast('cuda', enabled=True):
                        first_batch = first(train_loader)
                        z = autoencoder.encode_stage_2_inputs(first_batch["image"].to(device))

                print(f"Scaling factor set to {1/torch.std(z)}")
                scale_factor = 1 / torch.std(z)
                # -

                # We define the inferer using the scale factor:

                inferer = LatentDiffusionInferer(scheduler, scale_factor=scale_factor)

                optimizer_diff = torch.optim.Adam(params=unet.parameters(), lr=1e-4)

            # ### Train diffusion model

            # +
            n_epochs = 2       #hier waren ursprünglich 150 und dann 25
            epoch_loss_list = []
            autoencoder.eval()
            scaler = GradScaler()

            first_batch = first(train_loader)
            z = autoencoder.encode_stage_2_inputs(first_batch["image"].to(device))

            for epoch in range(n_epochs):
                unet.train()
                epoch_loss = 0
                progress_bar = tqdm(enumerate(train_loader), total=len(train_loader), ncols=70)
                progress_bar.set_description(f"Epoch {epoch}")
                for step, batch in progress_bar:
                    images = batch["image"].to(device)
                    optimizer_diff.zero_grad(set_to_none=True)

                    with autocast(device_type="cpu", enabled=True):
                        # Generate random noise
                        noise = torch.randn_like(z).to(device)

                        # Create timesteps
                        timesteps = torch.randint(
                            0,
                            inferer.scheduler.num_train_timesteps,
                            (images.shape[0],),
                            device=images.device,
                        ).long()

                        # Get model prediction
                        noise_pred = inferer(
                            inputs=images,
                            autoencoder_model=autoencoder,
                            diffusion_model=unet,
                            noise=noise,
                            timesteps=timesteps,
                        )

                        loss = F.mse_loss(noise_pred.float(), noise.float())

                    scaler.scale(loss).backward()
                    scaler.step(optimizer_diff)
                    scaler.update()

                    epoch_loss += loss.item()

                progress_bar.set_postfix({"loss": epoch_loss / (step + 1)})
            epoch_loss_list.append(epoch_loss / (step + 1))

            # Validation - Diffusion Model
            if (epoch + 1) % val_interval == 0:
                autoencoder.eval()
                discriminator.eval()
                val_loss = 0
                with torch.no_grad():
                    for step, batch in enumerate(val_loader):
                        images = batch["image"].to(device)
                        reconstruction, z_mu, z_sigma = autoencoder(images)
                        loss = F.l1_loss(reconstruction, images)
                        val_loss += loss.item()
                val_loss /= len(val_loader)
                print(f"[Diffusion] Epoch {epoch+1}/{n_epochs} | "
                    f"Train Loss: {epoch_loss/(step+1):.4f} | Val Loss: {val_loss:.4f}")
        # -

            plt.plot(epoch_loss_list, label='Training Loss')
            plt.title("Learning Curves", fontsize=20)
            plt.plot(epoch_loss_list)
            plt.yticks(fontsize=12)
            plt.xticks(fontsize=12)
            plt.xlabel("Epochs", fontsize=16)
            plt.ylabel("Loss", fontsize=16)
            plt.legend(prop={"size": 14})
            plt.show()
            plt.savefig(f'training_loss_res{res}_energy{energy}.png')
            # ### Plotting sampling example
            #
            # Finally, we generate an image with our LDM. For that, we will initialize a latent representation with just noise. Then, we will use the `unet` to perform 1000 denoising steps. In the last step, we decode the latent representation and plot the sampled image.

            # +
            autoencoder.eval()
            unet.eval()

            noise = torch.randn((1, 3, 24, 24, 16))
            noise = noise.to(device)
            scheduler.set_timesteps(num_inference_steps=1000)
            if noise.shape[1] == 3:
                noise = noise[:, :2, :, :, :]
            synthetic_images = inferer.sample(
                input_noise=noise,
                autoencoder_model=autoencoder,
                diffusion_model=unet,
                scheduler=scheduler,
            )
            # -

            # ### Visualise synthetic data

            idx = 0
            img = synthetic_images[idx, channel].detach().cpu().numpy()  # images
            fig, axs = plt.subplots(nrows=1, ncols=3)
            for ax in axs:
                ax.axis("off")
            ax = axs[0]
            ax.imshow(img[..., img.shape[2] // 2], cmap="gray")
            ax = axs[1]
            ax.imshow(img[:, img.shape[1] // 2, ...], cmap="gray")
            ax = axs[2]
            ax.imshow(img[img.shape[0] // 2, ...], cmap="gray")
            plt.savefig(f'synthetic_sample_res{res}_energy{energy}.png')

    # ## Clean-up data

    if directory is None:
        shutil.rmtree(root_dir)
