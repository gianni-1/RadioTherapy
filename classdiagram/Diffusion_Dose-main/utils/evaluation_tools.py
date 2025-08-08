import torch
import torch.nn.functional as F
import numpy as np
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
import lpips


class ImageSimilarityMetrics:
    def __init__(self, device='cuda'):
        self.device = device
        # Initialize LPIPS model
        self.lpips_model = lpips.LPIPS(net='alex').to(device)

    def calculate_ssim(self, real_images, generated_images):
        """
        Calculate SSIM between real and generated images.

        Args:
            real_images (torch.Tensor): Shape [B, C, H, W]
            generated_images (torch.Tensor): Shape [B, C, H, W]

        Returns:
            float: Mean SSIM across batch
            torch.Tensor: SSIM per image
        """
        batch_size = real_images.shape[0]
        ssim_scores = []

        # Convert to numpy and ensure proper range [0, 1]
        real_np = real_images.cpu().numpy()
        gen_np = generated_images.cpu().numpy()

        for i in range(batch_size):
            score = ssim(real_np[i, 0], gen_np[i, 0],
                         data_range=1.0,
                         gaussian_weights=True,
                         sigma=1.5,
                         use_sample_covariance=False)
            ssim_scores.append(score)

        ssim_scores = torch.tensor(ssim_scores, device=self.device)
        return torch.mean(ssim_scores).item(), ssim_scores

    def calculate_psnr(self, real_images, generated_images):
        """
        Calculate PSNR between real and generated images.

        Args:
            real_images (torch.Tensor): Shape [B, C, H, W]
            generated_images (torch.Tensor): Shape [B, C, H, W]

        Returns:
            float: Mean PSNR across batch
            torch.Tensor: PSNR per image
        """
        batch_size = real_images.shape[0]
        psnr_scores = []

        # Convert to numpy and ensure proper range [0, 1]
        real_np = real_images.cpu().numpy()
        gen_np = generated_images.cpu().numpy()

        for i in range(batch_size):
            score = psnr(real_np[i, 0], gen_np[i, 0], data_range=1.0)
            psnr_scores.append(score)

        psnr_scores = torch.tensor(psnr_scores, device=self.device)
        return torch.mean(psnr_scores).item(), psnr_scores

    def calculate_lpips(self, real_images, generated_images):
        """
        Calculate LPIPS distance between real and generated images.

        Args:
            real_images (torch.Tensor): Shape [B, C, H, W]
            generated_images (torch.Tensor): Shape [B, C, H, W]

        Returns:
            float: Mean LPIPS across batch
            torch.Tensor: LPIPS per image
        """
        with torch.no_grad():
            # Repeat grayscale channel to match LPIPS input requirements
            if real_images.shape[1] == 1:
                real_images = real_images.repeat(1, 3, 1, 1)
                generated_images = generated_images.repeat(1, 3, 1, 1)

            distances = self.lpips_model(real_images, generated_images)
            distances = distances.squeeze()

        return torch.mean(distances).item(), distances

    def calculate_all_metrics(self, real_images, generated_images):
        """
        Calculate all similarity metrics between real and generated images.

        Args:
            real_images (torch.Tensor): Shape [B, C, H, W]
            generated_images (torch.Tensor): Shape [B, C, H, W]

        Returns:
            dict: Dictionary containing mean and per-image metrics
        """
        mean_mse, mse_per_image = calculate_batch_mse(real_images, generated_images)
        mean_ssim, ssim_per_image = self.calculate_ssim(real_images, generated_images)
        mean_psnr, psnr_per_image = self.calculate_psnr(real_images, generated_images)
        mean_lpips, lpips_per_image = self.calculate_lpips(real_images, generated_images)

        return {
            'mse': {'mean': mean_mse, 'per_image': mse_per_image},
            'ssim': {'mean': mean_ssim, 'per_image': ssim_per_image},
            'psnr': {'mean': mean_psnr, 'per_image': psnr_per_image},
            'lpips': {'mean': mean_lpips, 'per_image': lpips_per_image}
        }

    def log_metrics(self, metrics_dict, epoch, batch_idx=None):
        """
        Log the metrics in a formatted way.

        Args:
            metrics_dict (dict): Dictionary containing the metrics
            epoch (int): Current epoch
            batch_idx (int, optional): Current batch index
        """
        log_str = f"Epoch {epoch}"
        if batch_idx is not None:
            log_str += f", Batch {batch_idx}"

        log_str += f" | MSE: {metrics_dict['mse']['mean']:.4f}"
        log_str += f" | SSIM: {metrics_dict['ssim']['mean']:.4f}"
        log_str += f" | PSNR: {metrics_dict['psnr']['mean']:.4f} dB"
        log_str += f" | LPIPS: {metrics_dict['lpips']['mean']:.4f}"

        print(log_str)