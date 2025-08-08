# Import Dataset before using it
from torch.utils.data import Dataset
from hotspot_patch_sampler import extract_hotspot_patches
# --- HotspotPatchDataset: Patch-basierte Trainingsdaten für Hotspot-Lernen ---
class HotspotPatchDataset(Dataset):
    """
    Dataset, das aus jedem Input/Target-Cube Hotspot-Patches extrahiert und als einzelne Trainingssamples zurückgibt.
    Nutzt extract_hotspot_patches(volume, ...), um Patch-Startpositionen zu bestimmen.
    Jeder __getitem__ gibt einen Patch (Input, Target, Energy) zurück.
    """
    def __init__(self, root_dir, section=None, patch_size=(32,32,32), min_hotspot_voxels=1000, dose_threshold=0.5, max_patches=16, random_patches=0, transforms=None, energy=None, max_patches_per_energy=80):
        base_dir = root_dir if section is None else os.path.join(root_dir, section)
        self.patch_size = patch_size
        self.transforms = transforms
        self.patch_records = []  # Liste: (in_fp, out_fp, energy, (z,y,x))
        self.selected_energy = energy
        self.patch_stats = []  # Liste: (n_voxels_gt0, n_voxels_gt_thresh, max_dose, mean_dose)
        self.max_patches_per_energy = max_patches_per_energy
        for energy_folder in sorted(os.listdir(base_dir)):
            folder_path = os.path.join(base_dir, energy_folder)
            if not os.path.isdir(folder_path) or energy_folder.startswith('.'):
                continue
            try:
                folder_energy = float(energy_folder.replace("_", "."))
            except ValueError:
                continue
            # Filter: Nur gewünschte Energie verwenden, falls gesetzt
            if self.selected_energy is not None and abs(folder_energy - self.selected_energy) > 0.01:
                continue
            in_dir = os.path.join(folder_path, "inputcube")
            out_dir = os.path.join(folder_path, "outputcube")
            if not os.path.isdir(in_dir) or not os.path.isdir(out_dir):
                continue
            for fname in sorted(os.listdir(in_dir)):
                if not fname.endswith(".npy"):
                    continue
                in_fp = os.path.join(in_dir, fname)
                out_fp = os.path.join(out_dir, fname)
                # Lade Target-Dosis-Array, um Hotspot-Patches zu bestimmen
                try:
                    dose_array = np.load(out_fp)
                except Exception as e:
                    logger.warning(f"Fehler beim Laden von {out_fp}: {e}")
                    continue
                patch_indices = extract_hotspot_patches(
                    dose_array,
                    patch_size=patch_size,
                    min_hotspot_voxels=min_hotspot_voxels,
                    dose_threshold=dose_threshold,
                    max_patches=max_patches,
                    random_patches=random_patches
                )
                
                # **PATCH BALANCING: Limit patches per energy to prevent imbalance**
                if len(patch_indices) > self.max_patches_per_energy:
                    logger.info(f"Energy {folder_energy:.2f}: Limiting patches from {len(patch_indices)} to {self.max_patches_per_energy}")
                    # Random selection for balanced distribution
                    import random
                    patch_indices = random.sample(patch_indices, self.max_patches_per_energy)
                
                for idx in patch_indices:
                    # Patch-Statistiken berechnen
                    z, y, x = idx
                    dz, dy, dx = patch_size
                    patch = dose_array[z:z+dz, y:y+dy, x:x+dx]
                    n_voxels_gt0 = int((patch > 0).sum())
                    n_voxels_gt_thresh = int((patch > dose_threshold).sum())
                    max_dose = float(patch.max())
                    mean_dose = float(patch.mean())
                    self.patch_records.append((in_fp, out_fp, folder_energy, idx))
                    self.patch_stats.append((n_voxels_gt0, n_voxels_gt_thresh, max_dose, mean_dose))
        # Logging der Patch-Statistiken
        if len(self.patch_stats) > 0:
            n_voxels_gt0_list = [s[0] for s in self.patch_stats]
            n_voxels_gt_thresh_list = [s[1] for s in self.patch_stats]
            max_dose_list = [s[2] for s in self.patch_stats]
            mean_dose_list = [s[3] for s in self.patch_stats]
            logger.info(f"Patch-Statistiken: Mittelwert Voxel>0: {np.mean(n_voxels_gt0_list):.1f}, Mittelwert Voxel>thresh: {np.mean(n_voxels_gt_thresh_list):.1f}, MaxDose Mittelwert: {np.mean(max_dose_list):.3f}, MeanDose Mittelwert: {np.mean(mean_dose_list):.3f}")
            logger.info(f"Patch-Statistiken: Min/Max Voxel>0: {np.min(n_voxels_gt0_list)}/{np.max(n_voxels_gt0_list)}, Min/Max Voxel>thresh: {np.min(n_voxels_gt_thresh_list)}/{np.max(n_voxels_gt_thresh_list)}")
        logger.info(f"HotspotPatchDataset: {len(self.patch_records)} Patches extrahiert.")
        # Debug: Zeige die Energie der ersten 5 Patches
        if len(self.patch_records) > 0:
            logger.debug(f"Energie der ersten 5 Patches: {[self.patch_records[i][2] for i in range(min(5, len(self.patch_records)))]}")

    def __len__(self):
        return len(self.patch_records)

    def __getitem__(self, idx):
        in_fp, out_fp, energy, (z, y, x) = self.patch_records[idx]
        arr_in = np.load(in_fp)    # [D,H,W]
        arr_out = np.load(out_fp)
        dz, dy, dx = self.patch_size
        patch_in = arr_in[z:z+dz, y:y+dy, x:x+dx]
        patch_out = arr_out[z:z+dz, y:y+dy, x:x+dx]
        sample = {
            "input": torch.from_numpy(patch_in)[None].float(),    # [1,D,H,W]
            "target": torch.from_numpy(patch_out)[None].float(),  # [1,D,H,W]
            "energy": torch.tensor([energy], dtype=torch.float32),
        }
        if self.transforms:
            sample = self.transforms(sample)
        return sample
#data_management.py

import os
import torch
from monai.data import DataLoader
try:
    from monai.apps.datasets import CustomDataset
except ImportError:
    # Fallback for newer MONAI versions where CustomDataset moved
    from monai.data import Dataset as CustomDataset
from torch.utils.data import random_split
import glob
import numpy as np
from torch.utils.data import Dataset
from monai.transforms import Compose, LoadImaged
from monai.data import NumpyReader
import log_config
import logging

logger = logging.getLogger(__name__)

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
                
                # **CRITICAL FIX: Validate sample quality before adding**
                if self.validate_sample_quality(out_fp):
                    self.samples.append((in_fp, out_fp, energy))
                else:
                    logger.debug(f"Rejected sample {fname} (insufficient dose)")
                    
        self.transforms = transforms
        logger.info(f"Energy levels found: {sorted(set([s[2] for s in self.samples]))}")
        logger.info(f"Total valid samples: {len(self.samples)}")
    
    def validate_sample_quality(self, dose_path, min_dose_voxels=20, min_max_dose=0.01):
        """
        Check if a dose sample is suitable for training.
        
        Args:
            dose_path: Path to dose .npy file
            min_dose_voxels: Minimum number of dose voxels required
            min_max_dose: Minimum maximum dose value required
            
        Returns:
            bool: True if sample is valid for training
        """
        try:
            dose_array = np.load(dose_path)
            
            # Count dose voxels
            dose_mask = dose_array > 1e-6
            dose_voxel_count = dose_mask.sum()
            max_dose = dose_array.max()
            
            # Check quality criteria
            is_valid = (dose_voxel_count >= min_dose_voxels) and (max_dose >= min_max_dose)
            
            if not is_valid:
                logger.debug(f"Sample validation failed: dose_voxels={dose_voxel_count} (min={min_dose_voxels}), "
                           f"max_dose={max_dose:.6f} (min={min_max_dose})")
            
            return is_valid
            
        except Exception as e:
            logger.warning(f"Failed to validate sample {dose_path}: {e}")
            return False
        self.transforms = transforms
        logger.info(f"Energy levels found: {sorted(set([s[2] for s in self.samples]))}")

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
                logger.warning(f"Sample {i} is missing required keys 'input' or 'target'.")
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