import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
import cv2
import os
import torch
from PIL import Image

def load_image(image_path):
    """Load an image from path, supporting various formats"""
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image file not found: {image_path}")
    
    # Try different methods to load the image
    try:
        # Try loading with PIL first
        img = Image.open(image_path)
        # Convert to grayscale if needed
        if img.mode != 'L':
            img = img.convert('L')
        img_array = np.array(img)
    except Exception as e:
        print(f"PIL loading failed, trying OpenCV: {e}")
        try:
            # Try OpenCV if PIL fails
            img_array = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            if img_array is None:
                raise ValueError("OpenCV could not load the image")
        except Exception as e:
            print(f"OpenCV loading failed: {e}")
            raise ValueError(f"Failed to load image using both PIL and OpenCV: {image_path}")
    
    return img_array

def normalize_image(img):
    """Normalize image to [0, 1] range"""
    if img.max() == img.min():
        return np.zeros_like(img, dtype=np.float32)
    return (img - img.min()) / (img.max() - img.min())

def apply_threshold(img, threshold):
    """Apply a threshold to an image"""
    return (img > threshold).astype(np.float32)

def auto_thresholds(img, num_intervals=10):
    """Generate thresholds automatically based on image min/max values"""
    min_val = img.min()
    max_val = img.max()
    
    # Generate thresholds
    if min_val == max_val:  # Handle the case where all pixels have the same value
        return [min_val]
    
    thresholds = np.linspace(min_val, max_val, num_intervals)
    return thresholds

def visualize_thresholds(img, thresholds, output_dir=None, filename_prefix="threshold"):
    """Visualize the results of applying different thresholds"""
    num_thresholds = len(thresholds)
    
    # Setup output directory
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    else:
        output_dir = "."  # Current directory
    
    # Create combined visualization
    cols = min(4, num_thresholds)
    rows = (num_thresholds + cols - 1) // cols
    
    # Create figure and subplots
    fig, axes = plt.subplots(rows, cols, figsize=(15, 3 * rows))
    if rows * cols == 1:
        axes = np.array([axes])
    axes = axes.flatten()
    
    # Original image
    if rows * cols > num_thresholds:
        axes[0].imshow(img, cmap='gray')
        axes[0].set_title('No Threshold')
        axes[0].axis('off')
        start_idx = 1
        
        # Save original image
        # plt.imsave(os.path.join(output_dir, f"{filename_prefix}_original.png"), img, cmap='gray')
    else:
        start_idx = 0
    
    # Thresholded images
    for i, threshold in enumerate(thresholds):
        if start_idx + i >= len(axes):
            break
            
        thresholded = apply_threshold(img, threshold)
        
        # Add to combined visualization
        axes[start_idx + i].imshow(thresholded, cmap='gray')
        axes[start_idx + i].set_title(f'Threshold: {threshold:.4f}')
        axes[start_idx + i].axis('off')
        
        # Save individual thresholded image
        # threshold_str = f"{threshold:.4f}".replace(".", "_")
        # plt.imsave(os.path.join(output_dir, f"{filename_prefix}_thresh_{threshold_str}.png"), 
        #           thresholded, cmap='gray')
    
    # Hide any remaining subplots
    for i in range(start_idx + len(thresholds), len(axes)):
        axes[i].axis('off')
        
    # Save combined visualization
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{filename_prefix}_combined.png"), dpi=200, bbox_inches='tight')
    plt.close()

def interactive_threshold(img):
    """Create an interactive slider to adjust threshold in real-time"""
    min_val = img.min()
    max_val = img.max()
    
    # Create figure and subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Display original image
    ax1.imshow(img, cmap='gray')
    ax1.set_title('No Threshold')
    ax1.axis('off')
    
    # Initial threshold
    initial_threshold = (min_val + max_val) / 2
    thresholded = apply_threshold(img, initial_threshold)
    
    # Display thresholded image
    img_display = ax2.imshow(thresholded, cmap='gray')
    ax2.set_title(f'Threshold: {initial_threshold:.4f}')
    ax2.axis('off')
    
    # Create slider
    ax_slider = plt.axes([0.2, 0.02, 0.65, 0.03])
    threshold_slider = Slider(
        ax=ax_slider,
        label='Threshold',
        valmin=min_val,
        valmax=max_val,
        valinit=initial_threshold,
    )
    
    # Update function for the slider
    def update(val):
        threshold = threshold_slider.val
        thresholded = apply_threshold(img, threshold)
        img_display.set_data(thresholded)
        ax2.set_title(f'Threshold: {threshold:.4f}')
        fig.canvas.draw_idle()
    
    # Register the update function with the slider
    threshold_slider.on_changed(update)
    
    plt.subplots_adjust(bottom=0.2)
    plt.show()

def visualize_pytorch_tensor(tensor, num_intervals=10, output_dir=None, filename_prefix="tensor"):
    """Visualize thresholds for a PyTorch tensor"""
    # Convert tensor to numpy
    if isinstance(tensor, torch.Tensor):
        tensor = tensor.detach().cpu().numpy()
        
    # Squeeze to remove singleton dimensions
    tensor = np.squeeze(tensor)
    
    # Generate thresholds and visualize
    thresholds = auto_thresholds(tensor, num_intervals)
    print(f"Image range: min={tensor.min()}, max={tensor.max()}")
    print(f"Generated thresholds: {[f'{t:.4f}' for t in thresholds]}")
    
    visualize_thresholds(tensor, thresholds, output_dir=output_dir, filename_prefix=filename_prefix)

# Main function with kwargs instead of command line arguments
def threshold_viz(
    image_path=None,
    tensor=None,
    mode='auto',
    intervals=10,
    thresholds=None,
    show_original=True,
    output_dir=None,
    filename_prefix="threshold",
    **kwargs
):
    """
    Visualize thresholds on an image with customizable parameters
    
    Parameters:
    -----------
    image_path : str, optional
        Path to the image file
    tensor : torch.Tensor, optional
        PyTorch tensor to visualize (alternative to image_path)
    mode : str, default='auto'
        Visualization mode: 'auto', 'interactive', or 'custom'
    intervals : int, default=10
        Number of threshold intervals in auto mode
    thresholds : list, optional
        Custom threshold values for 'custom' mode
    show_original : bool, default=True
        Whether to show the original image alongside thresholded versions
    output_dir : str, optional
        Directory where images will be saved (default: current directory)
    filename_prefix : str, default="threshold"
        Prefix for saved image filenames
    """
    # Handle input source
    if tensor is not None:
        # PyTorch tensor provided
        if isinstance(tensor, torch.Tensor):
            img_array = tensor.detach().cpu().numpy()
            img_array = np.squeeze(img_array)  # Remove singleton dimensions
        else:
            img_array = np.asarray(tensor)
    elif image_path:
        # Load image from path
        img_array = load_image(image_path)
    else:
        raise ValueError("Either image_path or tensor must be provided")
    
    # Normalize image to [0, 1] for consistent thresholding
    normalized_img = normalize_image(img_array)
    
    # Process based on the mode
    if mode == 'auto':
        auto_thresholds_values = auto_thresholds(normalized_img, intervals)
        print(f"Image range: min={normalized_img.min()}, max={normalized_img.max()}")
        print(f"Generated thresholds: {[f'{t:.4f}' for t in auto_thresholds_values]}")
        visualize_thresholds(normalized_img, auto_thresholds_values, output_dir=output_dir, filename_prefix=filename_prefix)
    elif mode == 'interactive':
        interactive_threshold(normalized_img)
    elif mode == 'custom':
        if not thresholds:
            raise ValueError("thresholds list must be provided when mode is 'custom'")
        visualize_thresholds(normalized_img, thresholds, output_dir=output_dir, filename_prefix=filename_prefix)
    else:
        raise ValueError(f"Unsupported mode: {mode}")
    
    return normalized_img

def example_with_image():
    # Example with a sample image - replace with your actual image path
    image_path = "/path/to/your/image.jpg"
    
    print("1. Auto mode with 5 threshold intervals")
    threshold_viz(
        image_path=image_path,
        intervals=5
    )
    
    print("2. Interactive mode")
    threshold_viz(
        image_path=image_path,
        mode="interactive"
    )
    
    print("3. Custom thresholds")
    threshold_viz(
        image_path=image_path,
        mode="custom",
        thresholds=[0.2, 0.3, 0.4, 0.5, 0.6]
    )

def example_with_tensor():
    # Create a sample PyTorch tensor
    # This creates a gradient from dark to light
    h, w = 256, 256
    y, x = torch.meshgrid(torch.linspace(0, 1, h), torch.linspace(0, 1, w))
    gradient = x + y  # Simple gradient from 0 to 2
    gradient = gradient / gradient.max()  # Normalize to [0, 1]
    
    # Add batch and channel dimensions [1, 1, 256, 256]
    tensor = gradient.unsqueeze(0).unsqueeze(0)
    
    print("1. Auto mode with tensor")
    threshold_viz(
        tensor=tensor,
        intervals=10
    )
    
    print("2. Interactive mode with tensor")
    threshold_viz(
        tensor=tensor,
        mode="interactive"
    )
    
    # Example with model output
    print("3. Simulating a model output (with sigmoid values)")
    # Create a tensor with values in the small range (like your case)
    simulated_output = torch.rand(1, 1, 256, 256) * 0.04 + 0.51  # Range ~[0.51, 0.55]
    print(f"Simulated output range: {simulated_output.min().item():.4f} to {simulated_output.max().item():.4f}")
    
    threshold_viz(
        tensor=simulated_output,
        mode="custom",
        thresholds=[0.505, 0.51, 0.515, 0.52, 0.525, 0.53, 0.535, 0.54]
    )

# Example usages:
if __name__ == "__main__":
    threshold_viz(
        image_path="runs/model_2/inference_results/ISIC_0012650_0.56/prediction_2.png",
        mode="custom",
        thresholds=np.linspace(0.48, 0.62, 5).tolist(),
        # output_dir=os.path.join(os.getcwd(), "threshold_test"),
        output_dir="."
    )