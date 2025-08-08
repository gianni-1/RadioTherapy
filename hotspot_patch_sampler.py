import numpy as np

def extract_hotspot_patches(volume, patch_size=(32,32,32), min_hotspot_voxels=10, dose_threshold=0.1, max_patches=16, random_patches=2):
    """
    Extrahiere Patches aus einem 3D-Volumen, die Hotspots enthalten.
    Args:
        volume (np.ndarray): 3D-Dosis-Array (z.B. [D,H,W])
        patch_size (tuple): Größe der Patches (z.B. (32,32,32))
        min_hotspot_voxels (int): Mindestanzahl Dosis-Voxel > threshold im Patch
        dose_threshold (float): Schwellenwert für Dosis (z.B. 0.1 in normalisierter Skala)
        max_patches (int): Maximale Anzahl Hotspot-Patches pro Volumen
        random_patches (int): Anzahl zusätzlicher Zufallspatches (für Diversität)
    Returns:
        List of Patch-Start-Indices (z.B. [(z,y,x), ...])
    """
    D, H, W = volume.shape
    dz, dy, dx = patch_size
    patches = []
    # Suche alle möglichen Patch-Startpositionen
    for z in range(0, D-dz+1, dz//2):
        for y in range(0, H-dy+1, dy//2):
            for x in range(0, W-dx+1, dx//2):
                patch = volume[z:z+dz, y:y+dy, x:x+dx]
                if np.count_nonzero(patch > dose_threshold) >= min_hotspot_voxels:
                    patches.append((z, y, x))
    # Wenn zu viele, zufällig auswählen
    if len(patches) > max_patches:
        import random
        patches = random.sample(patches, min(max_patches, len(patches)))
    # Füge einige Zufallspatches hinzu
    for _ in range(random_patches):
        z = np.random.randint(0, D-dz+1)
        y = np.random.randint(0, H-dy+1)
        x = np.random.randint(0, W-dx+1)
        patches.append((z, y, x))
    return patches
