#inference_module.py

import torch

class InferenceModule:
    """
    This module encapsulates the inference process using the trained autoencoder and diffusion model.
    It takes a noise tensor as input, uses the diffusion process (guided by a scheduler) to generate a latent
    representation, and returns the final output. Optionally, the autoencoder can be used to decode the latent 
    representation into a full image.
    """
    def __init__(self, autoencoder, diffusion_model, scheduler, device):
        """
        Initializes the InferenceModule.

        Args:
            autoencoder (torch.nn.Module): The trained autoencoder model.
            diffusion_model (torch.nn.Module): The trained diffusion model (e.g., a UNet).
            scheduler (object): The scheduler that manages the diffusion process (e.g., DDPM scheduler).
            device (torch.device): The device (CPU or GPU) to run inference on.
        """
        self.autoencoder = autoencoder
        self.diffusion_model = diffusion_model
        self.scheduler = scheduler
        self.device = device

    def run_inference(self, input_noise):
        """
        Runs the inference process using the diffusion model.

        This function takes a noise tensor, sends it through the diffusion model using the scheduler's
        settings, and returns the generated output. Optionally, the autoencoder's decoder can be used
        to transform the latent output into a full image.

        Args:
            input_noise (torch.Tensor): A noise tensor that will be transformed into an output image.
        
        Returns:
            torch.Tensor: The generated output (latent representation or decoded image).
        """
        # Move the noise tensor to the appropriate device
        input_noise = input_noise.to(self.device)

        # Run the diffusion model on the noise input.
        # This is a simplified example. In a real scenario, you might need to iterate over timesteps,
        # update the scheduler, and adjust the noise accordingly.
        latent_output = self.diffusion_model(input_noise)

        # Optionally, decode the latent output using the autoencoder's decoder.
        # Uncomment the next two lines if your autoencoder has a decode method.
        # decoded_output = self.autoencoder.decode(latent_output)
        # return decoded_output

        return latent_output