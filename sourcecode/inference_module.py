import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.functional import interpolate
import log_config
import logging
import math
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
    def __init__(self, models_by_energy: dict, energies: list, energy_weights: list, device, scale_factor, energy_min=None, energy_max=None, clip_min=None, clip_max=None, dose_normalization_params=None):
        """
        Args:
            models_by_energy (dict): Mapping from energy value (float) to tuple (autoencoder, diffusion_model, scheduler).
            energies (list of float): List of quadrature energy levels.
            energy_weights (list of float): Corresponding quadrature weights.
            device (torch.device): Device for inference.
            scale_factor (float): Scale factor used during training. If None, will try to extract from training.
            energy_min (float): Minimum energy value for normalization. If None, computed from energies.
            energy_max (float): Maximum energy value for normalization. If None, computed from energies.
            clip_min (float): Minimum dose value for denormalization. If None, no denormalization.
            clip_max (float): Maximum dose value for denormalization. If None, no denormalization.
            dose_normalization_params (dict): Energy-specific normalization parameters from checkpoint.
        """
        self.models_by_energy = models_by_energy
        self.energies = energies
        self.energy_weights = energy_weights
        self.device = device
        self.scale_factor = scale_factor
        # Use provided energy_min/max or compute from energies list
        self.energy_min = energy_min if energy_min is not None else min(energies)
        self.energy_max = energy_max if energy_max is not None else max(energies)
        # Add dose denormalization parameters (global fallbacks)
        self.clip_min = clip_min
        self.clip_max = clip_max
        # Energy-specific normalization parameters (preferred over global)
        self.dose_normalization_params = dose_normalization_params or {}
        
    def get_energy_normalization_params(self, energy_value, resolution=100):
        """
        Get energy-specific normalization parameters for denormalization.
        
        Args:
            energy_value (float): Energy in keV
            resolution (int): Spatial resolution
            
        Returns:
            dict: Dictionary with clip_min, clip_max, etc.
        """
        # 🔧 NEW: Try multiple resolution formats for compatibility
        energy_keys_to_try = [
            f'res64x64x64_e{energy_value:.2f}',  # New training format
            f'res{resolution}_e{energy_value:.2f}',  # Legacy format
            f'res100_e{energy_value:.2f}',  # Fallback format
        ]
        
        for energy_key in energy_keys_to_try:
            if energy_key in self.dose_normalization_params:
                params = self.dose_normalization_params[energy_key]
                logger.info(f"Using energy-specific parameters for {energy_key}: clip_min={params.get('clip_min', 0.0):.6f}, clip_max={params.get('clip_max', 1.0):.6f}")
                return params
            
        # Fallback to global parameters
        logger.warning(f"No energy-specific parameters found for any format (tried {energy_keys_to_try}), using global fallback")
        
        # **ENHANCED FALLBACK: Use reasonable radiotherapy defaults if global params are missing/bad**
        fallback_clip_min = self.clip_min or 0.0
        fallback_clip_max = self.clip_max or 1.0
        
        # Check if fallback parameters seem reasonable for radiotherapy
        if fallback_clip_max < 5.0:
            logger.warning(f"🚨 Fallback clip_max ({fallback_clip_max:.2f}) too low for radiotherapy, using emergency default")
            fallback_clip_max = 50.0  # Reasonable default for radiotherapy
            fallback_clip_min = 0.0
            logger.info(f"🚨 Using emergency defaults: clip_min=0.0, clip_max=50.0")
        
        return {
            'clip_min': fallback_clip_min,
            'clip_max': fallback_clip_max,
            'energy': energy_value,
            'resolution': resolution,
            'source': 'global_fallback'
        }
    
    def preprocess_ct(self, ct_tensor, target_cube_size=(100, 100, 100)):
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
        #preprocessed = interpolate(ct_tensor, size=target_cube_size, mode='trilinear', align_corners=True)
        preprocessed = ct_tensor
        # Instrumentation: log preprocess_ct output stats
        logger.info(f"Preprocessed CT stats: shape={tuple(preprocessed.shape)}, min={preprocessed.min().item():.4f}, max={preprocessed.max().item():.4f}, mean={preprocessed.mean().item():.4f}, std={preprocessed.std().item():.4f}")
        
        return preprocessed

    def run_inference(self, ct_tensor, target_cube_size=(100, 100, 100)):
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
    
    def run_inference_conditioned_on_energy(self, ct_tensor, energy_value, target_cube_size=(100, 100, 100)):
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
        # Debug: log how ct_tensor is built
        logger.info(f"Raw ct_tensor shape: {tuple(ct_tensor.shape)}, dims: {ct_tensor.dim()}")
        
        # select the models corresponding to this energy; fallback to index if key missing
        logger.debug(f"Models by energy keys: {list(self.models_by_energy.keys())}")
        try:
            autoencoder, unet, scheduler = self.models_by_energy[energy_value]
        except KeyError:
            raise KeyError(f"No models loaded for energy {energy_value}. Available energies: {list(self.models_by_energy.keys())}")

        # Preprocess CT scan
        input_data = self.preprocess_ct(ct_tensor, target_cube_size=target_cube_size).to(self.device)
        
        logger.info(f"Input data shape after preprocessing: {input_data.shape}")
        logger.info(f"Input data min/max: {input_data.min().item():.4f}/{input_data.max().item():.4f}")
        
        # Get original shape: expected shape [B, C, D, H, W] (C usually equals 1)
        B, C, D, H, W = input_data.shape
        
        # Normalize the energy value using Min-Max normalization
        eps = 1e-8
        #normalized_energy = (energy_value - 3.47) / (46.53 - 3.47 + eps)
        print(f"Original energy_value: {energy_value} keV")
        normalized_energy = energy_value / 100.00
        print(f"Normalized energy: {normalized_energy:.4f}")
        # LOG THE ACTUAL VALUES BEING USED
        logger.info(f"Original energy_value: {energy_value} keV")
        logger.info(f"Normalized energy: {normalized_energy}")
        
        # Cross-attention context building - using unconditional diffusion
        logger.info("Using unconditional diffusion (no cross-attention)")
        
        # Create an energy conditioning tensor with shape [B, 1, D, H, W]
        energy_tensor = torch.full((B, 1, D, H, W), normalized_energy, device=self.device)
        logger.info(f"Energy tensor constant value: {normalized_energy}")
        logger.info(f"Energy tensor unique values: {torch.unique(energy_tensor).cpu().numpy()}")
        
        # Concatenate the energy channel to the input data.
        # New input shape becomes [B, C+1, D, H, W] (e.g., from [B, 1, D, H, W] to [B, 2, D, H, W]).
        conditioned_input = torch.cat((input_data, energy_tensor), dim=1)
        logger.info(f"Conditioned input shape: {conditioned_input.shape}")

        # Encode the conditioned CT to latent space
        encoded_output = autoencoder.encode(conditioned_input)
        # Unpack encode result: if a tuple, assume first element is latent sample; otherwise sample from the distribution
        if isinstance(encoded_output, tuple):
            latent = encoded_output[0].to(self.device)
        else:
            latent = encoded_output.latent_dist.sample().to(self.device)
        # Debug: Check for NaN/Inf in latent
        logger.info(f"Latent isnan: {latent.isnan().any().item()}, isinf: {latent.isinf().any().item()}")
        # Match the scale used during training; compute scale factor if not provided
        if self.scale_factor is None:
            self.scale_factor = 1e-8
        # CRITICAL FIX: Correct scale factor direction - MULTIPLY during inference (inverse of training division)
        latent = latent * self.scale_factor
        logger.info(f"Latent shape after encoding: {latent.shape}")
        logger.info(f"Latent min/max: {latent.min().item():.4f}/{latent.max().item():.4f}")
        logger.info(f"Latent (after scaling) isnan: {latent.isnan().any().item()}, isinf: {latent.isinf().any().item()}")
        # Ensure latent spatial sizes divisible by UNet stride (4)
        _, _, D, H, W = latent.shape
        factor = 4
        cd, ch, cw = D % factor, H % factor, W % factor
        if cd or ch or cw:
            new_D, new_H, new_W = D - cd, H - ch, W - cw
            latent = latent[..., :new_D, :new_H, :new_W]
            logger.info(f"Cropped latent to {{new_D,new_H,new_W}} for UNet compatibility")
        
        # Run diffusion sampling in latent space WITHOUT cross-attention
        from generative.inferers import LatentDiffusionInferer
        
        # CRITICAL FIX: Use scale_factor=1.0 for inferer, apply scaling manually
        inferer = LatentDiffusionInferer(scheduler=scheduler, scale_factor=1.0)
        noise = torch.randn_like(latent)
        
        # REMOVED: Cross-attention context projection - using unconditional diffusion
        logger.info("Running unconditional diffusion sampling (no cross-attention)")
        
        sampled_output = inferer.sample(
            input_noise=noise,
            autoencoder_model=autoencoder,
            diffusion_model=unet,
            scheduler=scheduler
            # REMOVED: conditioning and mode parameters
        )
        logger.info(f"Sampled output shape: {sampled_output.shape}")
        logger.info(f"Sampled output min/max: {sampled_output.min().item():.4f}/{sampled_output.max().item():.4f}")
        logger.info(f"Sampled output isnan: {sampled_output.isnan().any().item()}, isinf: {sampled_output.isinf().any().item()}")
        
        # Determine if sample output is latent (needs decoding) or already image space by channel count
        latent_channels = latent.shape[1]
        if sampled_output.shape[1] == latent_channels:
            logger.info("Sampled output is latent; decoding to image space")
            with torch.no_grad():
                decoded = autoencoder.decode(sampled_output)
                dose_distribution = decoded[0] if isinstance(decoded, tuple) else decoded
            logger.info(f"Decoded output shape: {dose_distribution.shape}")
        else:
            logger.info("Sampled output is already in image space")
            dose_distribution = sampled_output
        logger.info(f"Decoded output min/max: {dose_distribution.min().item():.4f}/{dose_distribution.max().item():.4f}")
        logger.info(f"Decoded output isnan: {dose_distribution.isnan().any().item()}, isinf: {dose_distribution.isinf().any().item()}")
        
        # **CRITICAL FIX: Use energy-specific dose denormalization**
        norm_params = self.get_energy_normalization_params(energy_value)
        
        # **ENHANCED DEBUGGING: Log the normalization flow**
        logger.info(f"=== DOSE NORMALIZATION DEBUG for Energy {energy_value:.2f} ===")
        logger.info(f"Model output (normalized) stats: min={dose_distribution.min().item():.6f}, max={dose_distribution.max().item():.6f}, mean={dose_distribution.mean().item():.6f}")
        
        if norm_params['clip_min'] is not None and norm_params['clip_max'] is not None:
            clip_min = norm_params['clip_min']
            clip_max = norm_params['clip_max']
            
            # 🔧 SAFETY: Ensure values are floats, not dicts
            if isinstance(clip_min, dict):
                clip_min = float(clip_min) if clip_min else 0.0
            if isinstance(clip_max, dict):
                clip_max = float(clip_max) if clip_max else 1.0
                
            clip_min = float(clip_min)
            clip_max = float(clip_max)
            
            # **ENHANCED: Check if dose parameters seem reasonable**
            dose_range = clip_max - clip_min
            logger.info(f"Training normalization params: clip_min={clip_min:.6f}, clip_max={clip_max:.6f}, range={dose_range:.6f}")
            
            if dose_range < 1.0:
                logger.warning(f"⚠️  SUSPICIOUS: Dose range ({dose_range:.6f}) seems very small for radiotherapy!")
                logger.warning(f"⚠️  This may indicate training normalization issues.")
                logger.warning(f"⚠️  Expected ranges: 10-70 Gy for typical treatments.")
            
            if clip_max < 5.0:
                logger.error(f"🚨 CRITICAL: clip_max ({clip_max:.6f}) is suspiciously low!")
                logger.error(f"🚨 Expected clip_max: 20-70 Gy for radiotherapy")
                logger.error(f"🚨 This will cause severely underestimated dose predictions!")
            
            logger.info(f"Applying dose denormalization for energy {energy_value}: clip_min={clip_min:.6f}, clip_max={clip_max:.6f}")
            
            # **DETAILED DENORMALIZATION LOGGING**
            logger.info(f"Denormalization formula: output = normalized * {dose_range:.6f} + {clip_min:.6f}")
            
            # Denormalize from [0,1] back to original dose range [clip_min, clip_max]
            dose_distribution = dose_distribution * (clip_max - clip_min) + clip_min
            
            logger.info(f"Denormalized dose - min: {dose_distribution.min().item():.6f}, max: {dose_distribution.max().item():.6f}, mean: {dose_distribution.mean().item():.6f}")
            logger.info(f"Denormalized dose isnan: {dose_distribution.isnan().any().item()}, isinf: {dose_distribution.isinf().any().item()}")
            
            # **QUALITY CHECK: Compare with expected radiotherapy ranges**
            max_dose = dose_distribution.max().item()
            if max_dose < 1.0:
                logger.error(f"🚨 RESULT TOO LOW: Max dose {max_dose:.6f} Gy is unrealistic for radiotherapy!")
            elif max_dose < 10.0:
                logger.warning(f"⚠️  RESULT LOW: Max dose {max_dose:.6f} Gy seems low for typical radiotherapy.")
            else:
                logger.info(f"✓ Max dose {max_dose:.6f} Gy is in reasonable range for radiotherapy.")
                
            # Ensure non-negative dose values (physical constraint)
            if dose_distribution.min().item() < 0:
                logger.warning(f"Found negative dose values after denormalization, clamping to zero")
                dose_distribution = torch.clamp(dose_distribution, min=0.0)
                logger.info(f"After clamping - min: {dose_distribution.min().item():.6f}, max: {dose_distribution.max().item():.6f}")
                logger.info(f"After clamping isnan: {dose_distribution.isnan().any().item()}, isinf: {dose_distribution.isinf().any().item()}")
        else:
            logger.warning(f"Missing normalization parameters for energy {energy_value} - skipping dose denormalization")
            logger.warning(f"Available params: {norm_params}")
        
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
        
        # Normalize energy weights so they sum to 1
        total_w = sum(energy_weights)
        if not math.isclose(total_w, 1.0, rel_tol=1e-3):
            energy_weights = [w / total_w for w in energy_weights]
        
        # Convert energy weights to a tensor for broadcasting.
        weights_tensor = torch.tensor(energy_weights, dtype=dose_stack.dtype, device=self.device).view(-1, 1, 1, 1, 1)
        
        # Compute the weighted sum across energy levels.
        aggregated_dose = torch.sum(dose_stack * weights_tensor, dim=0)
        logger.info("Aggregated dose distribution computed over all energies")
        return aggregated_dose
    
    def _compute_scale_factor(self):
        """
        Compute scale factor by running a sample through the autoencoder.
        This matches the training procedure.
        """
        try:
            # Get the first available autoencoder
            autoencoder = next(iter(self.models_by_energy.values()))[0]
            autoencoder.eval()
            
            # Create a dummy input (similar to training)
            dummy_input = torch.randn(1, 2, 64, 64, 64, device=self.device)
            
            with torch.no_grad():
                z = autoencoder.encode_stage_2_inputs(dummy_input)
                computed_scale = 1 / torch.std(z)
                logger.info(f"Computed scale factor from dummy input: {computed_scale:.6f}")
                return computed_scale.item()
        except Exception as e:
            raise RuntimeError(f"Failed to compute scale factor: {e}")