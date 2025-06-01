#import torch
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.functional import interpolate
import log_config
import logging
logger = logging.getLogger(__name__)

class InferenceModule:
    """
    The InferenceModule performs the dose inference process using the trained autoencoder 
    and diffusion model. It computes the dose distribution from the uploaded patient CT data.
    
    The process is as follows:
      1. Preprocess the input CT scan by resizing or cropping it to a specified cube size.
      2. Optionally encode the CT scan with the autoencoder to obtain a latent representation.
      3. Pass the (latent) representation through the diffusion model to generate a voxel-wise dose distribution.
      
    Additional parameters (e.g. cube size) are taken into account during preprocessing.
    """
    def __init__(self, models_by_energy: dict, energies: list, energy_weights: list, device):
        """
        Args:
            models_by_energy (dict): Mapping from energy value (float) to tuple (autoencoder, diffusion_model, scheduler).
            energies (list of float): List of quadrature energy levels.
            energy_weights (list of float): Corresponding quadrature weights.
            device (torch.device): Device for inference.
        """
        self.models_by_energy = models_by_energy
        self.energies = energies
        self.energy_weights = energy_weights
        self.device = device
        # Build projection for cross-attention context from 2 dims to model’s context dimension
        example_unet = next(iter(models_by_energy.values()))[1]
        try:
            context_dim = example_unet.to_k.in_features
        except AttributeError:
            for m in example_unet.modules():
                if hasattr(m, "to_k"):
                    context_dim = m.to_k.in_features
                    break
            else:
                raise RuntimeError("Cannot determine context projection dimension")
        self.context_proj = nn.Linear(2, context_dim).to(self.device)
    
    def preprocess_ct(self, ct_tensor, target_cube_size=(64, 64, 64)):
        """
        Preprocesses the input CT scan to match the required cube size.
        This includes resizing the volume using trilinear interpolation.
        
        Args:
            ct_tensor (torch.Tensor): Input CT scan as a tensor. Expected shape: [C, D, H, W].
            target_cube_size (tuple): The desired spatial dimensions (D, H, W).
        
        Returns:
            torch.Tensor: The preprocessed CT scan, with a batch dimension added. Shape: [1, C, D, H, W].
        """
        # Ensure the tensor has a batch dimension (if not, add one)
        if ct_tensor.dim() == 4:
            ct_tensor = ct_tensor.unsqueeze(0)

        # Resize the volume to the target cube size using trilinear interpolation.
        # This is where the cube size parameter is applied.
        preprocessed = interpolate(ct_tensor, size=target_cube_size, mode='trilinear', align_corners=False)
        return preprocessed

    def run_inference(self, ct_tensor, target_cube_size=(64, 64, 64)):
        """
        Runs the complete dose inference on the provided CT scan.
        
        The process includes aggregating the dose distributions over the Gaussian quadrature energies and weights.
        
        Args:
            ct_tensor (torch.Tensor): The input CT scan with shape [C, D, H, W].
            target_cube_size (tuple): The target spatial dimensions (D, H, W) for inference.
        
        Returns:
            torch.Tensor: The aggregated predicted dose distribution as a tensor.
        """
        logger.info("Starting full inference over energies")
        result = self.run_inference_over_energies(ct_tensor, target_cube_size, self.energies, self.energy_weights)
        logger.info("Completed full inference over energies")
        return result
    
    def run_inference_conditioned_on_energy(self, ct_tensor, energy_value, target_cube_size=(64, 64, 64)):
        """
        Runs inference on the given CT scan while conditioning on a specified energy value.
        
        The conditioning is achieved by creating an additional channel that is filled with a normalized
        energy value and concatenating it to the CT scan data. The autoencoder and diffusion model should be 
        adapted to accept an extra input channel.
        
        Args:
            ct_tensor (torch.Tensor): Input CT scan with shape [C, D, H, W]. Typically C=1.
            target_cube_size (tuple): Desired spatial dimensions (D, H, W).
            energy_value (float): The energy level (in keV) to condition the inference on.
        
        Returns:
            torch.Tensor: The predicted dose distribution based on the energy-conditioned input.
        """
        logger.info(f"Starting inference conditioned on energy: {energy_value} keV")
        # select the models corresponding to this energy; fallback to index if key missing
        logger.debug(f"Models by energy keys: {list(self.models_by_energy.keys())}")
        try:
            autoencoder, unet, scheduler = self.models_by_energy[energy_value]
        except KeyError:
            raise KeyError(f"No models loaded for energy {energy_value}. Available energies: {list(self.models_by_energy.keys())}")

        # Preprocess CT scan
        input_data = self.preprocess_ct(ct_tensor, target_cube_size=target_cube_size).to(self.device)
        # Get original shape: expected shape [B, C, D, H, W] (C usually equals 1)
        B, C, D, H, W = input_data.shape
        
        # Normalize the energy value (example normalization: divide by 100)
        # FIXED: Use named constant instead of hardcoded value
        ENERGY_NORMALIZATION_FACTOR = 100.0  # keV
        normalized_energy = energy_value / ENERGY_NORMALIZATION_FACTOR
        # Build and project cross-attention context from energy and weight
        idx = self.energies.index(energy_value)
        energy_weight = self.energy_weights[idx]
        raw_context = torch.tensor(
            [[normalized_energy, energy_weight]],
            dtype=torch.float32,
            device=self.device
        )  # shape [1,2]
        # Create an energy conditioning tensor with shape [B, 1, D, H, W]
        energy_tensor = torch.full((B, 1, D, H, W), normalized_energy, device=self.device)
        
        # Concatenate the energy channel to the input data.
        # New input shape becomes [B, C+1, D, H, W] (e.g., from [B, 1, D, H, W] to [B, 2, D, H, W]).
        conditioned_input = torch.cat((input_data, energy_tensor), dim=1)

        
        # Encode the conditioned CT to latent space
        encoded_output = autoencoder.encode(conditioned_input)
        # Unpack encode result: if a tuple, assume first element is latent sample; otherwise sample from the distribution
        if isinstance(encoded_output, tuple):
            latent = encoded_output[0].to(self.device)
        else:
            latent = encoded_output.latent_dist.sample().to(self.device)
        
        # Run diffusion sampling in latent space
        from generative.inferers import LatentDiffusionInferer
        inferer = LatentDiffusionInferer(scheduler=scheduler, scale_factor=1.0)
        sampled_latent = inferer.sample(
            input_noise=latent,
            autoencoder_model=autoencoder,
            diffusion_model=unet,
            scheduler=scheduler,
            conditioning=raw_context.unsqueeze(1),
            mode="crossattn"
        )
        
        # If inferer returns image-space output (1 channel), use it directly; otherwise decode latent
        if sampled_latent.dim() == 5 and sampled_latent.shape[1] == 1:
            dose_distribution = sampled_latent
        else:
            dose_distribution = autoencoder.decode(sampled_latent)
        logger.info(f"Completed inference for energy: {energy_value} keV")
        return dose_distribution

    def run_inference_over_energies(self, ct_tensor, target_cube_size, energies, energy_weights):
        """
        Runs inference on the input CT scan for multiple energy levels and aggregates the results.
        
        For each energy value provided, this method performs inference (using the conditioned inference
        method, if available) and then computes a weighted sum of the dose distributions according to the
        provided energy weights.
        
        Args:
            ct_tensor (torch.Tensor): Input CT scan tensor with shape [C, D, H, W].
            target_cube_size (tuple): Desired spatial dimensions for inference.
            energies (list of float): A list of energy levels (e.g., [62, 75, 90]).
            energy_weights (list of float): Corresponding weights for each energy level.
        
        Returns:
            torch.Tensor: The aggregated dose distribution computed as:
                          dose_distribution = sum_i (weight_i * N(E_i))
        """
        logger.info(f"Running inference over energies: {energies}")
        dose_list = []
        # Loop over each energy level.
        for energy in energies:
            logger.info(f"Running inference for energy: {energy} keV")
            # Run inference conditioned on the given energy.
            dose = self.run_inference_conditioned_on_energy(
                ct_tensor,
                energy,
                target_cube_size
            )
            dose_list.append(dose)
        
        # Stack the dose outputs along a new dimension (energy dimension).
        dose_stack = torch.stack(dose_list, dim=0)
        
        # Convert energy weights to a tensor for broadcasting.
        weights_tensor = torch.tensor(energy_weights, dtype=dose_stack.dtype, device=self.device).view(-1, 1, 1, 1, 1)
        
        # Compute the weighted sum across energy levels.
        aggregated_dose = torch.sum(dose_stack * weights_tensor, dim=0)
        logger.info("Aggregated dose distribution computed over all energies")
        return aggregated_dose