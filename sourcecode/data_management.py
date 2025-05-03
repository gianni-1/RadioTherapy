#data_management.py

import os
import torch
from monai.data import DataLoader
from monai.apps.datasets import CustomDataset
from torch.utils.data import random_split
import glob
import numpy as np
from torch.utils.data import Dataset
from monai.transforms import Compose, LoadImaged
from monai.data import NumpyReader

class DoseNpyDataset(Dataset):
    """
    Dataset that loads input/output .npy cubes and parses the energy level from the folder name.
    Expects directory structure:
      root_dir/
        <section>/         # e.g. "training"
          <energy_folder>/ # e.g. "11_5"
            inputcube/     # contains input .npy files
            outputcube/    # contains target .npy files
    """
    def __init__(self, root_dir, section=None, transforms=None):
        base_dir = root_dir if section is None else os.path.join(root_dir, section)
        self.samples = []
        for energy_folder in sorted(os.listdir(base_dir)):
            folder_path = os.path.join(base_dir, energy_folder)
            # Skip non-directories and hidden entries
            if not os.path.isdir(folder_path) or energy_folder.startswith('.'):
                continue
            # Parse energy value from folder name
            try:
                energy = float(energy_folder.replace("_", "."))
            except ValueError:
                continue
            in_dir = os.path.join(folder_path, "inputcube")
            out_dir = os.path.join(folder_path, "outputcube")
            # Skip if expected subdirectories do not exist
            if not os.path.isdir(in_dir) or not os.path.isdir(out_dir):
                continue
            for fname in sorted(os.listdir(in_dir)):
                if not fname.endswith(".npy"):
                    continue
                in_fp = os.path.join(in_dir, fname)
                out_fp = os.path.join(out_dir, fname)
                self.samples.append((in_fp, out_fp, energy))
        self.transforms = transforms
        print(f"Energy levels found: {sorted(set([s[2] for s in self.samples]))}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        in_fp, out_fp, energy = self.samples[idx]
        # Determine if LoadImaged is expected by checking transforms
        use_load_imaged = False
        if self.transforms and isinstance(self.transforms, Compose):
            from monai.transforms import LoadImaged
            use_load_imaged = any(isinstance(tr, LoadImaged) for tr in self.transforms.transforms)

        if use_load_imaged:
            # Return file paths for MONAI LoadImaged
            sample = {
                "input": in_fp,
                "target": out_fp,
                "energy": torch.tensor([energy], dtype=torch.float32),
            }
        else:
            # Load arrays directly for tensor-based pipeline
            arr_in = np.load(in_fp)    # shape [D, H, W]
            arr_out = np.load(out_fp)
            sample = {
                "input": torch.from_numpy(arr_in)[None].float(),    # [1,D,H,W]
                "target": torch.from_numpy(arr_out)[None].float(),  # [1,D,H,W]
                "energy": torch.tensor([energy], dtype=torch.float32),
            }
        # Now apply transforms (e.g., EnsureChannelFirstd, spacing) which act on keys
        if self.transforms:
            sample = self.transforms(sample)
        return sample

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
        # Pass through full transform pipeline (LoadImaged remains for file loading)
        return DoseNpyDataset(self.root_dir, section=section, transforms=self.transforms)
    
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