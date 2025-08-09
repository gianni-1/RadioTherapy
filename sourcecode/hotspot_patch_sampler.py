import numpy as np

def extract_hotspot_patches(volume, patch_size=(32,32,32), min_hotspot_voxels=10, dose_threshold=0.1, max_patches=16, random_patches=2):
    """
    Extracts patches from a 3D volume where the number of voxels above a certain threshold is sufficient.
    Args:
        volume: 3D numpy array representing the dose distribution
        patch_size: Size of the patches to extract (z, y, x)
        min_hotspot_voxels: Minimum number of voxels in a patch that must be above the dose threshold
        dose_threshold: Dose threshold to consider a voxel as part of a hotspot
        max_patches: Maximum number of patches to extract
        random_patches: Number of additional random patches to add
    Returns:
        List of Patch-Start-Indices (e.g. [(z,y,x), ...])
    """
    D, H, W = volume.shape
    dz, dy, dx = patch_size
    patches = []
    # Iterate over the volume with a step size of half the patch size
    for z in range(0, D-dz+1, dz//2):
        for y in range(0, H-dy+1, dy//2):
            for x in range(0, W-dx+1, dx//2):
                patch = volume[z:z+dz, y:y+dy, x:x+dx]
                if np.count_nonzero(patch > dose_threshold) >= min_hotspot_voxels:
                    patches.append((z, y, x))
    # if there are too many patches, randomly sample
    if len(patches) > max_patches:
        import random
        patches = random.sample(patches, min(max_patches, len(patches)))
    # Add some random patches
    for _ in range(random_patches):
        z = np.random.randint(0, D-dz+1)
        y = np.random.randint(0, H-dy+1)
        x = np.random.randint(0, W-dx+1)
        patches.append((z, y, x))
    return patches
