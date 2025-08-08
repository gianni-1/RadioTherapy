# parameter_manager.py
import torch
import logging

logger = logging.getLogger(__name__)

class ParameterManager:
    """
    The ParameterManager class centralizes configuration parameters for the RadioTherapy project.
    It stores key settings such as spatial resolutions, energy levels, batch size, cube size,
    learning rate, number of epochs, and additional parameters.
    """

    def __init__(self, energies, batch_size, cube_size, learning_rate, num_epochs, patience, other_parameters=None):
        """
        Initializes the ParameterManager with the required configuration parameters.
        
        Args:
            resolutions (list of tuple): List of spatial resolutions, e.g. [(64, 64, 64), (32, 32, 32)].
            energies (list of int): List of energy levels (in keV) for which separate training might be executed.
            batch_size (int): The batch size used during training.
            cube_size (tuple): The cube size to which CT data will be resized (e.g., (64, 64, 64)).
            learning_rate (float): The learning rate used during training.
            num_epochs (int): The number of epochs for training.
            other_parameters (dict, optional): Any additional parameters to store.
        """
        # Convert energies to tensor if it's a list
        if isinstance(energies, list):
            import torch
            self.energies = torch.tensor(energies, dtype=torch.float32)
        else:
            self.energies = energies
            
        self.quad_energies = []
        self.quad_weights = []
        self.batch_size = batch_size
        
        # Validate and constrain cube_size for optimal performance
        self.cube_size = cube_size
        
        # CORRECTED: Use SINGLE resolution (64x64x64) instead of multi-resolution for stability
        if isinstance(self.cube_size, tuple):
            # Use only the original cube_size resolution for stability
            self.resolutions = [self.cube_size]  # Only one resolution
            logger.info(f"🔧 CORRECTED: Using SINGLE resolution {self.cube_size} for stability (was multi-resolution)")
        else:
            # Handle single integer cube_size (backward compatibility) 
            self.resolutions = [self.cube_size]  # Only one resolution
            logger.info(f"🔧 CORRECTED: Using SINGLE resolution {self.cube_size} for stability (was multi-resolution)")
        
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs
        self.patience = patience
        self.other_parameters = other_parameters if other_parameters is not None else {}

        min_e, max_e = float('inf'), float('-inf')
        print("Energies:" , self.energies)
        min_e = min(min_e, self.energies.min().item())
        max_e = max(max_e, self.energies.max().item())
        print("Min energy:", min_e, "Max energy:", max_e)
        
        # Store as instance attributes
        self.energy_min = min_e
        self.energy_max = max_e
        self.energy_normalization = (self.energies.float() - min_e) / (max_e - min_e)
    

    def get_parameters(self):
        """
        Returns all configuration parameters as a dictionary.
        
        Returns:
            dict: A dictionary containing resolutions, energies, batch_size, and any additional parameters.
        """
        params = {
            "resolutions": self.resolutions,
            "energies": self.energies,
            "energy_min": self.energy_min,
            "energy_max": self.energy_max,
            "energy_normalization": self.energy_normalization,
            "quadrature_weights": self.quadrature_weights,
            "batch_size": self.batch_size,
            "cube_size": self.cube_size,
            "learning_rate": self.learning_rate,
            "num_epochs": self.num_epochs
        }
        params.update(self.other_parameters)
        return params

    def update_parameter(self, key, value):
        """
        Updates a specific parameter.
        
        Args:
            key (str): The parameter name to update.
            value: The new value for the parameter.
        """
        if key == "cube_size":
            value = self._validate_cube_size(value)
            self.cube_size = value
            # Update resolutions based on new cube_size
            if isinstance(value, tuple):
                self.resolutions = [
                    tuple(dim // 4 for dim in value),
                    tuple(dim // 2 for dim in value),
                    value
                ]
            else:
                self.resolutions = [value//4, value//2, value]
        elif key in ["resolutions", "energies", "quadrature_weights", "batch_size", "learning_rate", "num_epochs"]:
            setattr(self, key, value)
        else:
            self.other_parameters[key] = value
