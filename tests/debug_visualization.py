import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go
from scipy import ndimage

def visualize_dose_distribution(cube_path, smooth_sigma=0.5, title="Dose Distribution"):
    """
    Verbesserte Visualisierung für Dosisverteilungen.
    
    Args:
        cube_path: Pfad zur .npy Datei
        smooth_sigma: Sigma für Gaussian-Filter (0 = kein Smoothing)
        title: Titel für die Visualisierung
    """
    # Lade die Daten
    cube = np.load(cube_path)
    
    # Handle verschiedene Dimensionen
    if cube.ndim == 4:
        cube = cube[0]
    elif cube.ndim == 5:
        cube = cube[0, 0]  # Entferne Batch und Channel-Dimensionen
    
    print(f"Cube shape: {cube.shape}")
    print(f"Cube min/max: {cube.min():.4f}/{cube.max():.4f}")
    print(f"Cube mean/std: {cube.mean():.4f}/{cube.std():.4f}")
    
    # Normalisierung
    cube_norm = (cube - cube.min()) / (cube.max() - cube.min() + 1e-8)
    
    # Optional: Smoothing anwenden
    if smooth_sigma > 0:
        cube_smoothed = ndimage.gaussian_filter(cube_norm, sigma=smooth_sigma)
    else:
        cube_smoothed = cube_norm
    
    # Berechne Percentile für bessere Threshold-Wahl
    percentiles = np.percentile(cube_smoothed, [5, 10, 25, 50, 75, 90, 95, 99])
    print(f"Percentiles: {dict(zip([5, 10, 25, 50, 75, 90, 95, 99], percentiles))}")
    
    # 3D Koordinaten
    x, y, z = np.mgrid[0:cube.shape[0], 0:cube.shape[1], 0:cube.shape[2]]
    
    # Erstelle Volume Rendering
    fig = go.Figure(
        data=go.Volume(
            x=x.flatten(),
            y=y.flatten(),
            z=z.flatten(),
            value=cube_smoothed.flatten(),
            opacity=0.15,
            surface_count=12,
            isomin=percentiles[2],  # 25th percentile
            isomax=percentiles[6],  # 95th percentile
            colorscale="Hot",
            caps=dict(x_show=False, y_show=False, z_show=False),
            colorbar=dict(title="Normalized Dose")
        )
    )
    
    fig.update_layout(
        title=title,
        scene=dict(
            aspectmode="data",
            xaxis=dict(title="X", range=[0, cube.shape[0]]),
            yaxis=dict(title="Y", range=[0, cube.shape[1]]),
            zaxis=dict(title="Z", range=[0, cube.shape[2]])
        )
    )
    
    return fig

def create_slice_comparison(cube_path1, cube_path2, slice_axis=2):
    """
    Vergleiche zwei Dosisverteilungen slice-weise.
    """
    cube1 = np.load(cube_path1)
    cube2 = np.load(cube_path2)
    
    if cube1.ndim == 4:
        cube1 = cube1[0]
    if cube2.ndim == 4:
        cube2 = cube2[0]
    
    mid_slice = cube1.shape[slice_axis] // 2
    
    if slice_axis == 2:
        slice1 = cube1[:, :, mid_slice]
        slice2 = cube2[:, :, mid_slice]
    elif slice_axis == 1:
        slice1 = cube1[:, mid_slice, :]
        slice2 = cube2[:, mid_slice, :]
    else:
        slice1 = cube1[mid_slice, :, :]
        slice2 = cube2[mid_slice, :, :]
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    im1 = axes[0].imshow(slice1, cmap='hot')
    axes[0].set_title('Expected')
    axes[0].axis('off')
    plt.colorbar(im1, ax=axes[0])
    
    im2 = axes[1].imshow(slice2, cmap='hot')
    axes[1].set_title('Actual')
    axes[1].axis('off')
    plt.colorbar(im2, ax=axes[1])
    
    # Differenz
    diff = slice1 - slice2
    im3 = axes[2].imshow(diff, cmap='RdBu_r')
    axes[2].set_title('Difference')
    axes[2].axis('off')
    plt.colorbar(im3, ax=axes[2])
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # Beispiel-Nutzung
    cube_path = "/Users/giannigagliardi/Documents/Git/RadioTherapy/traindata/11_5/outputcube/235101017859661465075472232303048949736_10.npy"
    
    fig = visualize_dose_distribution(cube_path, smooth_sigma=0.5)
    fig.show()
