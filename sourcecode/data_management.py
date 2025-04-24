#data_management.py

import os
import torch
from monai.data import DataLoader
from monai.apps.datasets import CustomDataset
from torch.utils.data import random_split

class DataLoaderModule:
    """
    This module takes care of loading, splitting and creating DataLoaders
    for training and validation data from a CustomDataset.
    """
    def __init__(self, root_dir, transforms, train_ratio=0.8, seed=42):
        """
        Args:
            root_dir (str): The root directory where the data is located.
            transforms (monai.transforms.Compose): Transforms that are applied to the data.
            energy (optional, int or float): Filter to load only samples of a certain energy.
            train_ratio (float): Ratio of the data used for training.
            seed (int): Seed for the reproducibility of the split.
        """
        self.root_dir = root_dir
        self.transforms = transforms
        self.train_ratio = train_ratio
        self.seed = seed

    def load_dataset(self, section="training"):
        """
      Loads the complete dataset as a CustomDataset.
        
        Args:
            section (str): Specifies which part of the dataset is to be loaded (e.g. “training”).
        
        Returns:
            CustomDataset: The loaded dataset.
        """
        dsfull = CustomDataset(
            data_dir=self.root_dir,
            section=section,
            cache_rate=0.0,             # set to 0 to keep RAM consumption low
            num_workers=0,              # use 0 if multiprocessing causes problems
            transform=self.transforms,
        )
        return dsfull
    
    def split_dataset(self, ds_full):
        """
        Splits the complete dataset into training and validation sets.
        
        Args:
            ds_full (Dataset): The complete dataset.
        
        Returns:
            tuple: (train_ds, val_ds)
        """
        dataset_size = len(ds_full)
        train_size = int(dataset_size * self.train_ratio)
        val_size = dataset_size - train_size
        train_ds, val_ds = random_split(
            ds_full,
            [train_size, val_size],
            generator=torch.Generator().manual_seed(self.seed)
        )
        return train_ds, val_ds
    
    def custom_collate(self, batch):
        """
        Custom collate function that checks and collates 'input', 'target', and 'energy' if available.
        """
        for i, sample in enumerate(batch):
            if "input" not in sample or "target" not in sample:
                print(f"Warning: Sample {i} is missing required keys 'input' or 'target'.")
        # Use the default collate function to combine the dictionaries, which will combine
        # any common keys (including 'energy', if present)
        return torch.utils.data.dataloader.default_collate(batch)

    
    def create_data_loader(self, dataset, batch_size, shuffle, num_workers=0):
        """
         Creates a DataLoader from the given dataset.
        
        Args:
            dataset (Dataset): The dataset from which data is to be loaded.
            batch_size (int): Number of samples per batch.
            shuffle (bool): Whether the data should be shuffled before each run.
            num_workers (int): Number of parallel processes for loading the data.
        
        Returns:
            DataLoader: The DataLoader created.
        """
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            persistent_workers=False
        )