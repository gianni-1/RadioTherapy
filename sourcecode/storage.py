# storage.py

import os
import torch
import numpy as np

class Storage:
    """
    The Storage module handles persistence tasks for the project.
    It provides methods to save and load model checkpoints, save dose distributions,
    log errors, and export results.
    """

    def __init__(self, model_dir="models", log_file="training_log.txt"):
        """
        Initializes the Storage module.
        
        Args:
            model_dir (str): Directory where model checkpoints and results are saved.
            log_file (str): File path for the training log.
        """
        self.model_dir = model_dir
        self.log_file = log_file
        # Create the model directory if it doesn't exist
        os.makedirs(self.model_dir, exist_ok=True)

    def save_model(self, model, filename):
        """
        Saves the model's state dictionary to a file.
        
        Args:
            model (torch.nn.Module): The model to be saved.
            filename (str): The filename (relative to model_dir) to save the model.
        """
        path = os.path.join(self.model_dir, filename)
        torch.save(model.state_dict(), path)
        print(f"Model saved to {path}")

    def load_model(self, model, filename, device=torch.device("cpu")):
        """
        Loads a saved model's state dictionary from a file and loads it into the given model.
        
        Args:
            model (torch.nn.Module): The model instance to load the state dictionary into.
            filename (str): The filename (relative to model_dir) to load the model from.
            device (torch.device): The device to map the model to.
        
        Returns:
            torch.nn.Module: The model loaded with the saved state dictionary.
        """
        path = os.path.join(self.model_dir, filename)
        if not os.path.exists(path):
            raise FileNotFoundError(f"Model file {path} not found!")
        state_dict = torch.load(path, map_location=device)
        model.load_state_dict(state_dict)
        print(f"Model loaded from {path}")
        return model

    def save_dose_distribution(self, dose_distribution, filename):
        """
        Saves the computed dose distribution as a numpy file.
        
        Args:
            dose_distribution (numpy.ndarray): The dose distribution array.
            filename (str): The filename to save the dose distribution (relative to model_dir).
        """
        path = os.path.join(self.model_dir, filename)
        np.save(path, dose_distribution)
        print(f"Dose distribution saved to {path}")

    def log_error(self, message):
        """
        Appends an error message to the log file.
        
        Args:
            message (str): The error message to log.
        """
        with open(self.log_file, "a") as log_file:
            log_file.write(message + "\n")
        print(f"Logged error: {message}")

    def export_results(self, results, filename):
        """
        Exports training or inference results to a file.
        
        Args:
            results (str): The results in string format.
            filename (str): The filename to export the results (relative to model_dir).
        """
        path = os.path.join(self.model_dir, filename)
        with open(path, "w") as f:
            f.write(results)
        print(f"Results exported to {path}")