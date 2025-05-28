# visualization.py

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
import nibabel as nib
import numpy as np

class Visualization:
    """
    The Visualization module provides methods to plot training curves and display sample images.
    This is useful for monitoring the training process and visualizing inference results.
    """

    def __init__(self):
        """
        Initializes the Visualization module.
        """
        pass

    def plot_learning_curve(self, epochs, loss_list, title="Learning Curve", filename=None):
        """
        Plots the learning curve over epochs.

        Args:
            epochs (list or array): Epoch numbers.
            loss_list (list or array): Loss values corresponding to each epoch.
            title (str): Title of the plot.
            filename (str, optional): If provided, the plot is saved to this file.
        """
        plt.figure(figsize=(10, 6))
        plt.plot(epochs, loss_list, marker='o', label='Training Loss')
        plt.title(title, fontsize=20)
        plt.xlabel("Epochs", fontsize=16)
        plt.ylabel("Loss", fontsize=16)
        plt.legend(prop={"size": 14})
        plt.grid(True)
        if filename:
            plt.savefig(filename)
        else:
            # save to default file
            plt.savefig('learning_curve.png')
        plt.close()

    def plot_adversarial_curves(self, epochs, gen_loss, disc_loss, title="Adversarial Training Curves", filename=None):
        """
        Plots the adversarial training curves for generator and discriminator.

        Args:
            epochs (list or array): Epoch numbers.
            gen_loss (list or array): Generator loss values.
            disc_loss (list or array): Discriminator loss values.
            title (str): Title of the plot.
            filename (str, optional): If provided, the plot is saved to this file.
        """
        plt.figure(figsize=(10, 6))
        plt.plot(epochs, gen_loss, marker='o', linewidth=2, label='Generator Loss')
        plt.plot(epochs, disc_loss, marker='o', linewidth=2, label='Discriminator Loss')
        plt.title(title, fontsize=20)
        plt.xlabel("Epochs", fontsize=16)
        plt.ylabel("Loss", fontsize=16)
        plt.legend(prop={"size": 14})
        plt.grid(True)
        if filename:
            plt.savefig(filename)
        else:
            plt.savefig('adv_curves.png')
        plt.close()

    def display_sample_images(self, images, title="Sample Images", filename=None):
        """
        Displays sample images from the given batch. For 3D images, it shows three orthogonal slices.

        Args:
            images (torch.Tensor or numpy.ndarray): Images to display. Expected shape is either 
                [batch, channels, depth, height, width] for 3D images or [batch, channels, height, width] for 2D.
            title (str): Title for the plot.
            filename (str, optional): If provided, the figure is saved to this file.
        """
        # Convert to numpy if input is a tensor
        if hasattr(images, 'cpu'):
            images = images.cpu().numpy()

        # For simplicity, take the first image from the batch and its first channel
        sample = images[0, 0]

        # Check dimensions for 3D image
        if sample.ndim == 3:
            depth, height, width = sample.shape
            # Select slices: middle slice along each axis
            slice_depth = sample[depth // 2, :, :]
            slice_height = sample[:, height // 2, :]
            slice_width = sample[:, :, width // 2]

            fig, axs = plt.subplots(1, 3, figsize=(15, 5))
            axs[0].imshow(slice_depth, cmap='gray')
            axs[0].set_title("Depth Slice")
            axs[0].axis("off")
            axs[1].imshow(slice_height, cmap='gray')
            axs[1].set_title("Height Slice")
            axs[1].axis("off")
            axs[2].imshow(slice_width, cmap='gray')
            axs[2].set_title("Width Slice")
            axs[2].axis("off")
            plt.suptitle(title, fontsize=20)
        else:
            # For 2D images, simply show the first image
            plt.figure(figsize=(6, 6))
            plt.imshow(sample, cmap='gray')
            plt.title(title, fontsize=20)
            plt.axis("off")

        if filename:
            plt.savefig(filename)
        else:
            plt.savefig('sample_images.png')
        plt.close()

    @staticmethod
    def plot_loss_curves(ae_train, ae_val, gen, disc, diff, resolution, energy):
        """
        Plot training and validation losses for autoencoder and diffusion model.

        Args:
            ae_train (list of float): Autoencoder training losses per epoch.
            ae_val   (list of float): Autoencoder validation losses per epoch.
            gen      (list of float): Generator losses per epoch.
            disc     (list of float): Discriminator losses per epoch.
            diff     (list of float): Diffusion model losses per epoch.
            resolution (tuple): Spatial resolution used for this run.
            energy   (float): Energy level used for this run.
        """
        import matplotlib.pyplot as plt
        plt.switch_backend('Agg')
        plt.figure(figsize=(10, 6))
        plt.plot(ae_train, marker='o', label='AE Train')
        plt.plot(ae_val,   marker='o', label='AE Val')
        plt.plot(gen,      marker='o', label='Generator Loss')
        plt.plot(disc,     marker='o', label='Discriminator Loss')
        plt.plot(diff,     marker='o', label='Diffusion Loss')
        plt.title(f'Loss Curves (res={resolution}, energy={energy} eV)', fontsize=18)
        plt.xlabel('Epoch', fontsize=14)
        plt.ylabel('Loss', fontsize=14)
        plt.legend(prop={"size": 12})
        plt.grid(True)
        # save loss curves to file
        fname = f'loss_curves_res{resolution}_energy{energy}.png'
        plt.savefig(fname)
        plt.close()


class MplCanvas(FigureCanvasQTAgg):
    """
    A custom widget that converts a matplotlib figure to a Qt widget.
    This enables embedding matplotlib visualizations directly in the GUI.
    """
    def __init__(self, fig):
        super().__init__(fig)
        self.setMinimumSize(400, 300)

def load_and_visualize(nifti_path, ct_scan=None, title="Dose distribution - Axial View"):
    """
    Loads a Nifti dose distribution file and visualizes its middle axial slice.

    Args:
        nifti_path: Path to the NIfTI file
        title: Title for the visualization

    Returns:
        matplotlib.figure.Figure: A figure object that can be displayed
    """
    if nifti_path.lower().endswith(('.nii', '.nii.gz')):
        img = nib.load(nifti_path)
        cube = np.asarray(img.dataobj)
    else:
        cube = np.load(nifti_path, allow_pickle=True)
            # Aggregate over energy dimension and ensure 3D volume
    if cube.ndim == 5 and cube.shape[1] == 1:
        cube = cube.squeeze(1)
    if cube.ndim == 4:
        cube = np.sum(cube, axis=0)
    cube = np.squeeze(cube)
            # Call visualization module for interactive slice viewer and volume rendering
    interactive_slice_viewer(cube, ct_volume=ct_scan)
    volume_rendering(cube)


def get_visualization_widget(figure):
    """
    Converts a matplotlib figure to a Qt widget for embedding in the GUI.

    Args:
        figure: matplotlib.figure.Figure object

    Returns:
        MplCanvas: A Qt widget containing the visualization
    """
    return MplCanvas(figure)


def interactive_slice_viewer(cube, ct_volume=None, title="Interactive Slice Viewer", show=True):
    """
    Launches an interactive Matplotlib window with a slider to browse through axial slices of a 3D volume.
    """
    if cube.ndim != 3:
        raise ValueError("Volume must be a 3D array for interactive slice viewing.")
    
    # Ensure interactive backend is set properly
    import matplotlib
    matplotlib.use('QtAgg')
    import matplotlib.pyplot as plt
    from matplotlib.widgets import Slider
    
    # Clear any existing figures to prevent conflicts
    plt.close('all')
    plt.ion()
    plt.switch_backend('QtAgg')
    # Initial slice
    z0 = cube.shape[2] // 2
    fig, ax = plt.subplots(figsize=(8, 7))
    
    # Adjust layout for sliders
    if ct_volume is not None:
        fig.subplots_adjust(bottom=0.25)
    else:
        fig.subplots_adjust(bottom=0.15)
    
    # Initialize images
    if ct_volume is not None:
        base_im = ax.imshow(ct_volume[:, :, z0], cmap='gray')
        overlay_im = ax.imshow(cube[:, :, z0], cmap='jet', alpha=0.5)
    else:
        overlay_im = ax.imshow(cube[:, :, z0], cmap='gray')
    
    ax.set_title(f"{title} - Slice {z0}")
    ax.axis('off')

    # Create slice slider
    slider_ax = fig.add_axes([0.15, 0.02, 0.7, 0.03])
    slider = Slider(slider_ax, 'Slice', 0, cube.shape[2] - 1, valinit=z0, valstep=1)
    
    # Create alpha slider if CT volume is provided
    alpha_slider = None
    if ct_volume is not None:
        alpha_ax = fig.add_axes([0.15, 0.10, 0.7, 0.03])
        alpha_slider = Slider(alpha_ax, 'Alpha', 0.0, 1.0, valinit=0.5, valstep=0.01)

    # Keep references to prevent garbage collection
    slider._fig = fig
    if alpha_slider is not None:
        alpha_slider._fig = fig

    # Define callback functions with error handling
    def update_slice(val):
        try:
            z = int(slider.val)
            if z < 0 or z >= cube.shape[2]:
                return
            
            if ct_volume is not None:
                base_im.set_data(ct_volume[:, :, z])
                overlay_im.set_data(cube[:, :, z])
            else:
                overlay_im.set_data(cube[:, :, z])
            ax.set_title(f"{title} - Slice {z}")
            fig.canvas.draw_idle()
        except Exception as e:
            print(f"Error in update_slice: {e}")

    def update_alpha(val):
        try:
            if ct_volume is not None and alpha_slider is not None:
                alpha_val = float(alpha_slider.val)
                if 0.0 <= alpha_val <= 1.0:
                    overlay_im.set_alpha(alpha_val)
                    fig.canvas.draw_idle()
        except Exception as e:
            print(f"Error in update_alpha: {e}")
    
    # Connect callbacks
    slider.on_changed(update_slice)
    if alpha_slider is not None:
        alpha_slider.on_changed(update_alpha)
    
    # Ensure widgets are properly initialized
    fig.canvas.draw()
    
    # Return objects for testing when show=False
    if not show:
        if ct_volume is not None:
            return fig, slider, alpha_slider, overlay_im
        return fig, slider, overlay_im
    
    if show:
        try:
            plt.show(block=True)
        except KeyboardInterrupt:
            plt.close(fig)


def volume_rendering(cube, opacity=0.2, surface_count=20):
    """
    Performs a 3D volume rendering of a normalized 3D array using Plotly.
    """
    if cube.ndim != 3:
        raise ValueError("Volume must be a 3D array for volume rendering.")
    import numpy as _np
    import plotly.graph_objects as go

    # normalize
    cube_norm = (cube - _np.min(cube)) / (_np.max(cube) - _np.min(cube) + 1e-8)
    # Downsample volume for performance if too large
    max_points = 200_000  # limit total points to ~200k
    n_voxels = cube_norm.size
    if n_voxels > max_points:
        factor = int(_np.ceil((n_voxels / max_points) ** (1/3)))
        cube_norm = cube_norm[::factor, ::factor, ::factor]
    # thresholds
    p_low, p_high = _np.percentile(cube_norm, [1, 99])
    # coordinate grid
    x, y, z = _np.mgrid[0:cube_norm.shape[0],
                        0:cube_norm.shape[1],
                        0:cube_norm.shape[2]]
    fig = go.Figure(data=go.Volume(
        x=x.flatten(), y=y.flatten(), z=z.flatten(),
        value=cube_norm.flatten(),
        opacity=opacity,
        surface_count=surface_count,
        isomin=p_low, isomax=p_high,
        colorscale='Viridis'
    ))
    fig.update_layout(scene=dict(aspectmode='data'))
    fig.show(renderer="browser")
