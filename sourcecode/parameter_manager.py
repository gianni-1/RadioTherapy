# parameter_manager.py

class ParameterManager:
    """
    The ParameterManager class centralizes configuration parameters for the RadioTherapy project.
    It stores key settings such as spatial resolutions, energy levels, batch size, and additional parameters.
    """

    def __init__(self, resolutions, energies, batch_size, cube_size, other_parameters=None):
        """
        Initializes the ParameterManager with the required configuration parameters.
        
        Args:
            resolutions (list of tuple): List of spatial resolutions, e.g. [(64, 64, 64), (32, 32, 32)].
            energies (list of int): List of energy levels (in keV) for which separate trainings are executed.
            batch_size (int): The batch size used during training.
            cube_size (tuple): The cube size to which CT data will be resized (e.g., (64, 64, 64)).
            other_parameters (dict, optional): Any additional parameters to store.
        """
        self.resolutions = resolutions
        self.energies = energies
        self.batch_size = batch_size
        self.cube_size = cube_size
        self.other_parameters = other_parameters if other_parameters is not None else {}

    def get_parameters(self):
        """
        Returns all configuration parameters as a dictionary.
        
        Returns:
            dict: A dictionary containing resolutions, energies, batch_size, and any additional parameters.
        """
        params = {
            "resolutions": self.resolutions,
            "energies": self.energies,
            "batch_size": self.batch_size,
            "cube_size": self.cube_size
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
        if key in ["resolutions", "energies", "batch_size", "cube_size"]:
            setattr(self, key, value)
        else:
            self.other_parameters[key] = value