#visualization.py

import matplotlib.pyplot as plt

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
        plt.show()

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
        plt.plot(epochs, gen_loss, marker='o', color='C0', linewidth=2, label='Generator Loss')
        plt.plot(epochs, disc_loss, marker='o', color='C1', linewidth=2, label='Discriminator Loss')
        plt.title(title, fontsize=20)
        plt.xlabel("Epochs", fontsize=16)
        plt.ylabel("Loss", fontsize=16)
        plt.legend(prop={"size": 14})
        plt.grid(True)
        if filename:
            plt.savefig(filename)
        plt.show()

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
            plt.figure(figsize=(6,6))
            plt.imshow(sample, cmap='gray')
            plt.title(title, fontsize=20)
            plt.axis("off")
        
        if filename:
            plt.savefig(filename)
        plt.show()