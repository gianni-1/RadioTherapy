#import torch
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
        self.scale_factor = 0.18215  # Same latent scale factor used during training
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
        preprocessed = interpolate(ct_tensor, size=target_cube_size, mode='trilinear', align_corners=True)
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
        logger.info(f"Input data shape after preprocessing: {input_data.shape}")
        logger.info(f"Input data min/max: {input_data.min().item():.4f}/{input_data.max().item():.4f}")
        
        # Get original shape: expected shape [B, C, D, H, W] (C usually equals 1)
        B, C, D, H, W = input_data.shape
        
        # Normalize the energy value (example normalization: divide by 100)
        # FIXED: Use named constant instead of hardcoded value
        ENERGY_NORMALIZATION_FACTOR = 100.0  # keV
        normalized_energy = energy_value / ENERGY_NORMALIZATION_FACTOR
        
        # LOG THE ACTUAL VALUES BEING USED
        logger.info(f"Original energy_value: {energy_value} keV")
        logger.info(f"Normalized energy: {normalized_energy}")
        
        # Build and project cross-attention context from energy and weight
        idx = self.energies.index(energy_value)
        energy_weight = self.energy_weights[idx]
        logger.info(f"Energy weight: {energy_weight}")
        
        raw_context = torch.tensor(
            [[normalized_energy, energy_weight]],
            dtype=torch.float32,
            device=self.device
        )  # shape [1,2]
        logger.info(f"Raw context tensor: {raw_context}")
        
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
        # Match the scale used during training
        latent = latent * self.scale_factor
        logger.info(f"Latent shape after encoding: {latent.shape}")
        logger.info(f"Latent min/max: {latent.min().item():.4f}/{latent.max().item():.4f}")
        
        # Run diffusion sampling in latent space
        from generative.inferers import LatentDiffusionInferer
        inferer = LatentDiffusionInferer(scheduler=scheduler, scale_factor=1.0)
        noise = torch.randn_like(latent)
        sampled_output = inferer.sample(
            input_noise=noise,
            autoencoder_model=autoencoder,
            diffusion_model=unet,
            scheduler=scheduler,
            conditioning=raw_context.unsqueeze(1),
            mode="crossattn"
        )
        logger.info(f"Sampled output shape: {sampled_output.shape}")
        logger.info(f"Sampled output min/max: {sampled_output.min().item():.4f}/{sampled_output.max().item():.4f}")
        
        # CRITICAL: Check if sampled_output is still in latent space and needs decoding
        # Compare with the ORIGINAL input shape (without energy channel), not the conditioned input
        original_shape = input_data.shape  # [1, 1, 64, 64, 64]
        
        # The LatentDiffusionInferer should return decoded results with the same spatial dims as input
        # but might have different number of channels. If spatial dims match, it's already decoded.
        if (sampled_output.shape[2:] == original_shape[2:] and 
            sampled_output.shape[0] == original_shape[0]):
            logger.info("Sampled output is already in image space (spatial dimensions match)")
            dose_distribution = sampled_output
        else:
            logger.info(f"Sampled output appears to be in latent space, decoding... (spatial dims: {sampled_output.shape[2:]} vs expected: {original_shape[2:]})")
            # Decode from latent space to image space
            with torch.no_grad():
                decoded_output = autoencoder.decode(sampled_output)
                if isinstance(decoded_output, tuple):
                    dose_distribution = decoded_output[0]
                else:
                    dose_distribution = decoded_output
            logger.info(f"Decoded output shape: {dose_distribution.shape}")
            logger.info(f"Decoded output min/max: {dose_distribution.min().item():.4f}/{dose_distribution.max().item():.4f}")
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

    def debug_inference_pipeline(self, ct_tensor, energy_value, target_cube_size=(64, 64, 64)):
        """
        Debug-Version der Inference, die jeden Schritt detailliert loggt.
        Verwendet INFO-Level, damit die Nachrichten im Log erscheinen.
        """
        logger.info(f"=== DEBUG INFERENCE START for energy {energy_value} keV ===")
        
        try:
            autoencoder, unet, scheduler = self.models_by_energy[energy_value]
        except KeyError:
            raise KeyError(f"No models loaded for energy {energy_value}")

        # 1. Preprocessing
        input_data = self.preprocess_ct(ct_tensor, target_cube_size=target_cube_size).to(self.device)
        logger.info(f"1. Preprocessed input: shape={input_data.shape}, min={input_data.min():.4f}, max={input_data.max():.4f}")
        
        B, C, D, H, W = input_data.shape
        
        # 2. Energy conditioning
        ENERGY_NORMALIZATION_FACTOR = 100.0
        normalized_energy = energy_value / ENERGY_NORMALIZATION_FACTOR
        energy_tensor = torch.full((B, 1, D, H, W), normalized_energy, device=self.device)
        conditioned_input = torch.cat((input_data, energy_tensor), dim=1)
        logger.info(f"2. Conditioned input: shape={conditioned_input.shape}, min={conditioned_input.min():.4f}, max={conditioned_input.max():.4f}")
        
        # 3. Encoding
        autoencoder.eval()
        with torch.no_grad():
            encoded_output = autoencoder.encode(conditioned_input)
            if isinstance(encoded_output, tuple):
                latent = encoded_output[0].to(self.device)
            else:
                latent = encoded_output.latent_dist.sample().to(self.device)
            latent = latent * self.scale_factor
        logger.info(f"3. Encoded latent: shape={latent.shape}, min={latent.min():.4f}, max={latent.max():.4f}")
        
        # 4. Test direct autoencoder reconstruction (bypass diffusion)
        with torch.no_grad():
            direct_reconstruction = autoencoder.decode(latent / self.scale_factor)
            if isinstance(direct_reconstruction, tuple):
                direct_reconstruction = direct_reconstruction[0]
        logger.info(f"4. Direct AE reconstruction: shape={direct_reconstruction.shape}, min={direct_reconstruction.min():.4f}, max={direct_reconstruction.max():.4f}")
        
        # 5. Diffusion sampling
        from generative.inferers import LatentDiffusionInferer
        inferer = LatentDiffusionInferer(scheduler=scheduler, scale_factor=1.0)
        
        # Context for cross-attention
        idx = self.energies.index(energy_value)
        energy_weight = self.energy_weights[idx]
        raw_context = torch.tensor([[normalized_energy, energy_weight]], dtype=torch.float32, device=self.device)
        
        noise = torch.randn_like(latent)
        logger.info(f"5. Starting diffusion sampling with noise: min={noise.min():.4f}, max={noise.max():.4f}")
        
        sampled_output = inferer.sample(
            input_noise=noise,
            autoencoder_model=autoencoder,
            diffusion_model=unet,
            scheduler=scheduler,
            conditioning=raw_context.unsqueeze(1),
            mode="crossattn"
        )
        logger.info(f"6. Diffusion sampled output: shape={sampled_output.shape}, min={sampled_output.min():.4f}, max={sampled_output.max():.4f}")
        
        # 6.5. Check if decoding is needed - compare with original input spatial dims
        original_shape = input_data.shape  # [1, 1, D, H, W]
        if (sampled_output.shape[2:] == original_shape[2:] and 
            sampled_output.shape[0] == original_shape[0]):
            logger.info("7. Sampled output is already in image space")
            final_output = sampled_output
        else:
            logger.info("7. Sampled output appears to be in latent space, decoding...")
            with torch.no_grad():
                decoded_output = autoencoder.decode(sampled_output)
                if isinstance(decoded_output, tuple):
                    final_output = decoded_output[0]
                else:
                    final_output = decoded_output
            logger.info(f"8. Final decoded output: shape={final_output.shape}, min={final_output.min():.4f}, max={final_output.max():.4f}")
        
        logger.info(f"=== DEBUG INFERENCE END ===")
        
        return {
            'input_data': input_data,
            'conditioned_input': conditioned_input, 
            'latent': latent,
            'direct_reconstruction': direct_reconstruction,
            'diffusion_output': sampled_output,
            'final_output': final_output
        }
    
    def debug_inference_with_console_output(self, ct_tensor, energy_value, target_cube_size=(64, 64, 64)):
        """
        Debug-Version der Inference mit direkter Konsolen-Ausgabe (zusätzlich zum Log).
        """
        print(f"\n=== DEBUG INFERENCE START for energy {energy_value} keV ===")
        logger.info(f"=== DEBUG INFERENCE START for energy {energy_value} keV")
        
        try:
            autoencoder, unet, scheduler = self.models_by_energy[energy_value]
        except KeyError:
            raise KeyError(f"No models loaded for energy {energy_value}")

        # 1. Preprocessing
        input_data = self.preprocess_ct(ct_tensor, target_cube_size=target_cube_size).to(self.device)
        msg = f"1. Preprocessed input: shape={input_data.shape}, min={input_data.min():.4f}, max={input_data.max():.4f}"
        print(msg)
        logger.info(msg)
        
        B, C, D, H, W = input_data.shape
        
        # 2. Energy conditioning
        ENERGY_NORMALIZATION_FACTOR = 100.0
        normalized_energy = energy_value / ENERGY_NORMALIZATION_FACTOR
        energy_tensor = torch.full((B, 1, D, H, W), normalized_energy, device=self.device)
        conditioned_input = torch.cat((input_data, energy_tensor), dim=1)
        msg = f"2. Conditioned input: shape={conditioned_input.shape}, min={conditioned_input.min():.4f}, max={conditioned_input.max():.4f}"
        print(msg)
        logger.info(msg)
        
        # 3. Encoding
        autoencoder.eval()
        with torch.no_grad():
            encoded_output = autoencoder.encode(conditioned_input)
            if isinstance(encoded_output, tuple):
                latent = encoded_output[0].to(self.device)
            else:
                latent = encoded_output.latent_dist.sample().to(self.device)
            latent = latent * self.scale_factor
        msg = f"3. Encoded latent: shape={latent.shape}, min={latent.min():.4f}, max={latent.max():.4f}"
        print(msg)
        logger.info(msg)
        
        # 4. Test direct autoencoder reconstruction (bypass diffusion)
        with torch.no_grad():
            direct_reconstruction = autoencoder.decode(latent / self.scale_factor)
            if isinstance(direct_reconstruction, tuple):
                direct_reconstruction = direct_reconstruction[0]
        msg = f"4. Direct AE reconstruction: shape={direct_reconstruction.shape}, min={direct_reconstruction.min():.4f}, max={direct_reconstruction.max():.4f}"
        print(msg)
        logger.info(msg)
        
        # 5. Diffusion sampling
        from generative.inferers import LatentDiffusionInferer
        inferer = LatentDiffusionInferer(scheduler=scheduler, scale_factor=1.0)
        
        # Context for cross-attention
        idx = self.energies.index(energy_value)
        energy_weight = self.energy_weights[idx]
        raw_context = torch.tensor([[normalized_energy, energy_weight]], dtype=torch.float32, device=self.device)
        
        noise = torch.randn_like(latent)
        msg = f"5. Starting diffusion sampling with noise: min={noise.min():.4f}, max={noise.max():.4f}"
        print(msg)
        logger.info(msg)
        
        sampled_output = inferer.sample(
            input_noise=noise,
            autoencoder_model=autoencoder,
            diffusion_model=unet,
            scheduler=scheduler,
            conditioning=raw_context.unsqueeze(1),
            mode="crossattn"
        )
        msg = f"6. Diffusion sampled output: shape={sampled_output.shape}, min={sampled_output.min():.4f}, max={sampled_output.max():.4f}"
        print(msg)
        logger.info(msg)
        
        # 6.5. Check if decoding is needed - compare with original input spatial dims
        original_shape = input_data.shape  # [1, 1, D, H, W]
        if (sampled_output.shape[2:] == original_shape[2:] and 
            sampled_output.shape[0] == original_shape[0]):
            msg = "7. Sampled output is already in image space"
            print(msg)
            logger.info(msg)
            final_output = sampled_output
        else:
            msg = "7. Sampled output appears to be in latent space, decoding..."
            print(msg)
            logger.info(msg)
            with torch.no_grad():
                decoded_output = autoencoder.decode(sampled_output)
                if isinstance(decoded_output, tuple):
                    final_output = decoded_output[0]
                else:
                    final_output = decoded_output
            msg = f"8. Final decoded output: shape={final_output.shape}, min={final_output.min():.4f}, max={final_output.max():.4f}"
            print(msg)
            logger.info(msg)
        
        print(f"=== DEBUG INFERENCE END ===\n")
        logger.info(f"=== DEBUG INFERENCE END ===")
        
        return {
            'input_data': input_data,
            'conditioned_input': conditioned_input, 
            'latent': latent,
            'direct_reconstruction': direct_reconstruction,
            'diffusion_output': sampled_output,
            'final_output': final_output
        }
    
    def enable_debug_logging(self):
        """
        Aktiviert Debug-Logging für detaillierte Ausgaben.
        """
        logging.getLogger().setLevel(logging.DEBUG)
        logging.getLogger('inference_module').setLevel(logging.DEBUG)
        logger.setLevel(logging.DEBUG)
        logger.info("Debug logging enabled")
    
    def disable_debug_logging(self):
        """
        Deaktiviert Debug-Logging und setzt es zurück auf INFO.
        """
        logging.getLogger().setLevel(logging.INFO)
        logging.getLogger('inference_module').setLevel(logging.INFO)
        logger.setLevel(logging.INFO)
        logger.info("Debug logging disabled")
    
    def test_energy_conditioning_effect(self, ct_tensor, target_cube_size=(64, 64, 64)):
        """
        Testet, ob die Energy-Konditionierung tatsächlich einen Effekt hat.
        """
        logger.info("=== TESTING ENERGY CONDITIONING EFFECT ===")
        
        # Test mit verschiedenen Energien
        test_energies = [0.99, 20.0, 49.0]  # Niedrig, Mittel, Hoch
        
        results = {}
        for energy in test_energies:
            try:
                autoencoder, unet, scheduler = self.models_by_energy[energy]
            except KeyError:
                # Wenn exakte Energie nicht verfügbar, nimm nächstliegende
                available_energies = list(self.models_by_energy.keys())
                closest_energy = min(available_energies, key=lambda x: abs(x - energy))
                autoencoder, unet, scheduler = self.models_by_energy[closest_energy]
                logger.info(f"Using closest available energy {closest_energy} for test energy {energy}")
            
            # Preprocessing
            input_data = self.preprocess_ct(ct_tensor, target_cube_size=target_cube_size).to(self.device)
            B, C, D, H, W = input_data.shape
            
            # Energy conditioning
            ENERGY_NORMALIZATION_FACTOR = 100.0
            normalized_energy = energy / ENERGY_NORMALIZATION_FACTOR
            energy_tensor = torch.full((B, 1, D, H, W), normalized_energy, device=self.device)
            conditioned_input = torch.cat((input_data, energy_tensor), dim=1)
            
            logger.info(f"Energy {energy}: normalized_energy={normalized_energy:.4f}")
            logger.info(f"Energy {energy}: energy_tensor unique values={torch.unique(energy_tensor).cpu().numpy()}")
            logger.info(f"Energy {energy}: conditioned_input channel 0 (CT) min/max={conditioned_input[0,0].min():.4f}/{conditioned_input[0,0].max():.4f}")
            logger.info(f"Energy {energy}: conditioned_input channel 1 (Energy) min/max={conditioned_input[0,1].min():.4f}/{conditioned_input[0,1].max():.4f}")
            
            # Test encoding only
            autoencoder.eval()
            with torch.no_grad():
                encoded_output = autoencoder.encode(conditioned_input)
                if isinstance(encoded_output, tuple):
                    latent = encoded_output[0].to(self.device)
                else:
                    latent = encoded_output.latent_dist.sample().to(self.device)
            
            logger.info(f"Energy {energy}: latent min/max={latent.min().item():.6f}/{latent.max().item():.6f}")
            logger.info(f"Energy {energy}: latent mean/std={latent.mean().item():.6f}/{latent.std().item():.6f}")
            
            results[energy] = {
                'normalized_energy': normalized_energy,
                'latent_stats': (latent.min().item(), latent.max().item(), latent.mean().item(), latent.std().item())
            }
        
        # Vergleiche die Unterschiede
        logger.info("=== COMPARISON OF ENERGY EFFECTS ===")
        energies = list(results.keys())
        for i, energy1 in enumerate(energies):
            for energy2 in energies[i+1:]:
                stats1 = results[energy1]['latent_stats']
                stats2 = results[energy2]['latent_stats']
                diff_mean = abs(stats1[2] - stats2[2])
                diff_std = abs(stats1[3] - stats2[3])
                logger.info(f"Energy {energy1} vs {energy2}: mean_diff={diff_mean:.8f}, std_diff={diff_std:.8f}")
        
        logger.info("=== END ENERGY CONDITIONING TEST ===")
        return results
    
    def diagnose_model_architecture(self, energy_value=None):
        """
        Diagnostiziert die Modell-Architektur, insbesondere ob der Autoencoder 
        Energy-Konditionierung unterstützt (2 Input-Kanäle vs. 1 Input-Kanal).
        """
        logger.info("=== DIAGNOSING MODEL ARCHITECTURE ===")
        
        # Wenn keine Energie angegeben, nimm die erste verfügbare
        if energy_value is None:
            energy_value = list(self.models_by_energy.keys())[0]
        
        try:
            autoencoder, unet, scheduler = self.models_by_energy[energy_value]
        except KeyError:
            raise KeyError(f"No models loaded for energy {energy_value}")
        
        autoencoder.eval()
        
        # Test 1: Versuche 1-Kanal Input
        logger.info("Testing autoencoder with 1-channel input...")
        test_input_1ch = torch.randn(1, 1, 64, 64, 64, device=self.device)
        try:
            with torch.no_grad():
                encoded_1ch = autoencoder.encode(test_input_1ch)
                if isinstance(encoded_1ch, tuple):
                    latent_1ch = encoded_1ch[0]
                else:
                    latent_1ch = encoded_1ch.latent_dist.sample()
            logger.info(f"✓ 1-channel input SUCCESSFUL: input={test_input_1ch.shape} -> latent={latent_1ch.shape}")
            success_1ch = True
        except Exception as e:
            logger.info(f"✗ 1-channel input FAILED: {str(e)}")
            success_1ch = False
        
        # Test 2: Versuche 2-Kanal Input
        logger.info("Testing autoencoder with 2-channel input...")
        test_input_2ch = torch.randn(1, 2, 64, 64, 64, device=self.device)
        try:
            with torch.no_grad():
                encoded_2ch = autoencoder.encode(test_input_2ch)
                if isinstance(encoded_2ch, tuple):
                    latent_2ch = encoded_2ch[0]
                else:
                    latent_2ch = encoded_2ch.latent_dist.sample()
            logger.info(f"✓ 2-channel input SUCCESSFUL: input={test_input_2ch.shape} -> latent={latent_2ch.shape}")
            success_2ch = True
        except Exception as e:
            logger.info(f"✗ 2-channel input FAILED: {str(e)}")
            success_2ch = False
        
        # Test 3: Wenn beide funktionieren, teste ob unterschiedliche Inputs verschiedene Outputs produzieren
        if success_1ch and success_2ch:
            logger.info("Both 1-channel and 2-channel inputs work. Testing if energy channel makes a difference...")
            
            # Erstelle 2-Kanal Input mit verschiedenen Energy-Werten
            base_ct = torch.randn(1, 1, 64, 64, 64, device=self.device)
            
            # Energy-Kanal mit niedrigem Wert
            energy_low = torch.full((1, 1, 64, 64, 64), 0.01, device=self.device)
            input_low = torch.cat([base_ct, energy_low], dim=1)
            
            # Energy-Kanal mit hohem Wert
            energy_high = torch.full((1, 1, 64, 64, 64), 0.5, device=self.device)
            input_high = torch.cat([base_ct, energy_high], dim=1)
            
            with torch.no_grad():
                # Encode mit niedrigem Energy-Wert
                encoded_low = autoencoder.encode(input_low)
                if isinstance(encoded_low, tuple):
                    latent_low = encoded_low[0]
                else:
                    latent_low = encoded_low.latent_dist.sample()
                
                # Encode mit hohem Energy-Wert
                encoded_high = autoencoder.encode(input_high)
                if isinstance(encoded_high, tuple):
                    latent_high = encoded_high[0]
                else:
                    latent_high = encoded_high.latent_dist.sample()
            
            # Vergleiche die Latent-Repräsentationen
            diff_mean = torch.abs(latent_low.mean() - latent_high.mean()).item()
            diff_std = torch.abs(latent_low.std() - latent_high.std()).item()
            max_abs_diff = torch.abs(latent_low - latent_high).max().item()
            
            logger.info(f"Latent comparison (low vs high energy):")
            logger.info(f"  Mean difference: {diff_mean:.8f}")
            logger.info(f"  Std difference: {diff_std:.8f}")
            logger.info(f"  Max absolute difference: {max_abs_diff:.8f}")
            
            if max_abs_diff < 1e-6:
                logger.info("⚠️  PROBLEM: Energy channel appears to be IGNORED by the autoencoder!")
                logger.info("    The model produces identical latent representations regardless of energy input.")
            else:
                logger.info("✓ Energy channel appears to affect the latent representation.")
        
        # Test 4: Inspiziere erste Layer des Encoders
        logger.info("Inspecting autoencoder's first layer...")
        try:
            # Suche nach dem ersten Conv3D Layer
            first_conv = None
            for name, module in autoencoder.named_modules():
                if isinstance(module, torch.nn.Conv3d):
                    first_conv = module
                    first_conv_name = name
                    break
            
            if first_conv is not None:
                in_channels = first_conv.in_channels
                logger.info(f"First Conv3D layer '{first_conv_name}': in_channels={in_channels}")
                
                if in_channels == 1:
                    logger.info("🔍 Model expects 1 input channel - NOT designed for energy conditioning!")
                elif in_channels == 2:
                    logger.info("🔍 Model expects 2 input channels - DESIGNED for energy conditioning!")
                else:
                    logger.info(f"🔍 Model expects {in_channels} input channels - unusual configuration")
            else:
                logger.info("Could not find Conv3D layer in autoencoder")
        except Exception as e:
            logger.info(f"Error inspecting model layers: {str(e)}")
        
        # Zusammenfassung
        logger.info("=== DIAGNOSIS SUMMARY ===")
        logger.info(f"1-channel input support: {'✓' if success_1ch else '✗'}")
        logger.info(f"2-channel input support: {'✓' if success_2ch else '✗'}")
        
        if success_1ch and not success_2ch:
            logger.info("🚨 CONCLUSION: Model was trained WITHOUT energy conditioning!")
            logger.info("   -> You need to retrain the model with 2-channel input (CT + Energy)")
        elif success_2ch and not success_1ch:
            logger.info("✓ CONCLUSION: Model was trained WITH energy conditioning")
            logger.info("   -> But something else might be wrong with the implementation")
        elif success_1ch and success_2ch:
            logger.info("⚠️  CONCLUSION: Model accepts both 1 and 2 channels")
            logger.info("   -> Check if energy channel actually affects the output (see tests above)")
        else:
            logger.info("❌ CONCLUSION: Model has issues - neither 1 nor 2 channels work")
        
        logger.info("=== END DIAGNOSIS ===")
        
        return {
            'supports_1_channel': success_1ch,
            'supports_2_channels': success_2ch,
            'energy_value_tested': energy_value
        }

if __name__ == "__main__":
    print("Running simplified model architecture diagnosis...")
    
    # Teste nur mit den einfachen Mock-Modellen
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Erstelle Simple Test ohne alle die Model-Loading-Komplikationen
    from generative.networks.nets import AutoencoderKL
    
    # Teste Autoencoder mit 1 und 2 Kanälen
    print("\n=== TESTING AUTOENCODER ARCHITECTURE ===")
    
    # Test 1: Autoencoder mit 1 Input-Kanal (wie ursprünglich trainiert?)
    try:
        ae_1ch = AutoencoderKL(
            spatial_dims=3, 
            in_channels=1,  # Nur CT
            out_channels=1,
            num_channels=(32, 32, 32), 
            latent_channels=2,
            num_res_blocks=1, 
            norm_num_groups=8,
            attention_levels=(False, False, True)
        ).to(device)
        
        test_input_1ch = torch.randn(1, 1, 64, 64, 64, device=device)
        with torch.no_grad():
            encoded_1ch = ae_1ch.encode(test_input_1ch)
            if isinstance(encoded_1ch, tuple):
                latent_1ch = encoded_1ch[0]
            else:
                latent_1ch = encoded_1ch.latent_dist.sample()
        
        print(f"✓ 1-channel autoencoder: {test_input_1ch.shape} -> {latent_1ch.shape}")
        success_1ch = True
        
    except Exception as e:
        print(f"✗ 1-channel autoencoder failed: {e}")
        success_1ch = False
    
    # Test 2: Autoencoder mit 2 Input-Kanälen (CT + Energy)
    try:
        ae_2ch = AutoencoderKL(
            spatial_dims=3, 
            in_channels=2,  # CT + Energy
            out_channels=1,
            num_channels=(32, 32, 32), 
            latent_channels=2,
            num_res_blocks=1, 
            norm_num_groups=8,
            attention_levels=(False, False, True)
        ).to(device)
        
        test_input_2ch = torch.randn(1, 2, 64, 64, 64, device=device)
        with torch.no_grad():
            encoded_2ch = ae_2ch.encode(test_input_2ch)
            if isinstance(encoded_2ch, tuple):
                latent_2ch = encoded_2ch[0]
            else:
                latent_2ch = encoded_2ch.latent_dist.sample()
        
        print(f"✓ 2-channel autoencoder: {test_input_2ch.shape} -> {latent_2ch.shape}")
        success_2ch = True
        
    except Exception as e:
        print(f"✗ 2-channel autoencoder failed: {e}")
        success_2ch = False
    
    print(f"\n=== RESULTS ===")
    print(f"1-channel autoencoder: {'✓ Works' if success_1ch else '✗ Failed'}")
    print(f"2-channel autoencoder: {'✓ Works' if success_2ch else '✗ Failed'}")
    
    if success_2ch:
        print("\n🔍 Based on system_manager.py code analysis:")
        print("   - The trained models were designed for in_channels=2 (CT + Energy)")
        print("   - Autoencoder: in_channels=2, out_channels=1")
        print("   - UNet: in_channels=2, out_channels=2")
        print("   ➜ The architecture SUPPORTS energy conditioning!")
        print("\n⚠️  But the INFERENCE shows identical outputs for all energies!")
        print("   This means either:")
        print("   1. Training data didn't include proper energy variation")
        print("   2. Training loss didn't encourage energy-dependent learning")
        print("   3. Energy normalization/preprocessing is incorrect")
        print("   4. Model weights are not responding to energy input correctly")
        print("\n🎯 NEXT STEPS:")
        print("   - Check training data: Do dose distributions actually vary with energy?")
        print("   - Check training logs: Did the model learn energy-dependent features?")
        print("   - Check training pipeline: Was energy conditioning implemented correctly?")
        print("   - Consider retraining with explicit energy-aware loss function")
    else:
        print("\n❌ Neither architecture works - there might be import/setup issues")
    
    print(f"\nCheck inference.log for any additional details.")