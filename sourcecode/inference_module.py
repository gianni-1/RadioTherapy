#inference_module.py

import torch
import torch.nn.functional as F
from torch.nn.functional import interpolate

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
        return self.run_inference_over_energies(ct_tensor, target_cube_size, self.energies, self.energy_weights)
    
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
        # select the models corresponding to this energy
        print(f"Keys in models_by_energy: {self.models_by_energy.keys()}")
        autoencoder, unet, scheduler = self.models_by_energy[0]

        # Preprocess CT scan
        input_data = self.preprocess_ct(ct_tensor, target_cube_size=target_cube_size).to(self.device)
        # Get original shape: expected shape [B, C, D, H, W] (C usually equals 1)
        B, C, D, H, W = input_data.shape
        
        # Normalize the energy value (example normalization: divide by 100)
        normalized_energy = energy_value / 100.0
        # Create an energy conditioning tensor with shape [B, 1, D, H, W]
        energy_tensor = torch.full((B, 1, D, H, W), normalized_energy, device=self.device)
        
        # Concatenate the energy channel to the input data.
        # New input shape becomes [B, C+1, D, H, W] (e.g., from [B, 1, D, H, W] to [B, 2, D, H, W]).
        conditioned_input = torch.cat((input_data, energy_tensor), dim=1)
        
        # Pass the conditioned input through the autoencoder.
        #latent_representation = autoencoder(conditioned_input)
        
        # Use the diffusion model to compute the dose distribution based on the latent representation.
        #dose_distribution = diffusion_model(latent_representation)

        # Obtain latent representation (stage 2 inputs) from autoencoder
        latent = autoencoder.encode_stage_2_inputs(conditioned_input)
        # use diffusion inferer to sample dose distribution
        from generative.inferers import LatentDiffusionInferer
        inferer = LatentDiffusionInferer(scheduler=scheduler, scale_factor=1.0)
        # perform sampling (default num steps)
        dose_distribution = inferer.sample(
            input_noise=latent,
            autoencoder_model=autoencoder,
            diffusion_model=unet,
            scheduler=scheduler,
            save_intermediates=False,
            conditioning=None,
        )
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
        dose_list = []
        # Loop over each energy level.
        for energy in energies:
            print(f"Running inference for energy: {energy} keV")
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
        return aggregated_dose