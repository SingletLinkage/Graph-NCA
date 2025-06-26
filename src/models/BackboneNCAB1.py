import torch
import torch.nn as nn
import torch.nn.functional as F
from colorama import Fore, Back, Style
from matplotlib import pyplot as plt
import numpy as np
from src.utils.helper import log_message

def visualize_heatmap(image_tensor, num_samples=4):
    """Visualize activation heatmaps"""
    # Create figure for this visualization
    fig = plt.figure(figsize=(10, 10))
    
    # Convert tensor to numpy image
    img = image_tensor.cpu()
    if img.shape[0] == 1:
        img = img.repeat(3, 1, 1)  # Convert to RGB
    img = img.permute(1, 2, 0).numpy()
    plt.imshow(img)
    
    # Plot heatmap overlay
    plt.colorbar(label='Activation Intensity')
    plt.title('NCA Activation Heatmap')
    plt.axis('off')
    
    return fig

class BasicNCA(nn.Module):
    def __init__(self, hidden_channels=16, n_channels=1, fire_rate=0.5, device=None, log_enabled=True):
        super(BasicNCA, self).__init__()
        
        self.hidden_channels = hidden_channels
        self.n_channels = n_channels
        self.fire_rate = fire_rate
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.log_enabled = log_enabled
        
        # Encoder module
        self.encoder = nn.Sequential(
            nn.Conv2d(n_channels, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, hidden_channels, 3, padding=1),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU(inplace=True)
        )
        
        # Perception module (standard convolution instead of graph)
        self.perception = nn.Conv2d(hidden_channels, hidden_channels, 3, padding=1)
        
        # Update network
        self.update_net = nn.Sequential(
            nn.Conv2d(hidden_channels, 128, 1),
            nn.ReLU(),
            nn.Conv2d(128, hidden_channels, 1)
        )
        
        # Output layers
        self.to_output = nn.Conv2d(hidden_channels, n_channels, 1)
        
    def perception_process(self, fmap, return_vis=False):
        try:
            module = f"{__name__}:perception_process" if self.log_enabled else ""
            
            # Apply perception convolution
            perception_vector = self.perception(fmap)
            perception_vector = F.relu(perception_vector)
            
            if return_vis:
                # For visualization, we'll return the raw perception vector
                return perception_vector, perception_vector[0, 0]  # Return first channel of first batch for vis
            else:
                return perception_vector
                
        except Exception as e:
            log_message(f"Error in perception_process: {str(e)}", "ERROR", module, self.log_enabled)
            import traceback
            traceback.print_exc()
            # Return input as fallback
            if return_vis:
                return fmap, None
            else:
                return fmap
        
    def forward(self, x, steps=1, return_vis=False):
        """
        Forward pass with error handling and progress tracking
        
        Args:
            x: Input tensor
            steps: Number of NCA steps to run
            return_vis: If True, returns visualization data along with output
        """
        try:
            module = f"{__name__}:forward" if self.log_enabled else ""
            
            # Initial encoding
            h = self.encoder(x)
            
            # Store visualization data if requested
            vis_data = None
            
            # Run NCA steps
            for step in range(steps):
                # Apply perception to create perception vector
                if return_vis and step == steps-1:  # Only keep the last step's visualization
                    p, vis_data = self.perception_process(h, return_vis=True)
                else:
                    p = self.perception_process(h)
                
                # Update cell states using perception vector
                update = self.update_net(p)
                
                # Stochastic update with fire rate
                mask = torch.rand_like(update[:, :1], device=self.device) < self.fire_rate
                mask = mask.float().repeat(1, self.hidden_channels, 1, 1)
                h = h + mask * update
            
            # Generate output
            out = self.to_output(h)
            output = torch.sigmoid(out)
            
            if return_vis:
                # Return the output and visualization data
                return output, (x[0], vis_data)
            else:
                return output
            
        except Exception as e:
            module = f"{__name__}:forward" if self.log_enabled else ""
            log_message(f"Error in BasicNCA forward pass: {str(e)}", "ERROR", module, self.log_enabled)
            log_message(f"Input shape: {x.shape}, dtype: {x.dtype}", "WARNING", module, self.log_enabled)
            import traceback
            traceback.print_exc()
            
            # Return dummy outputs based on return_vis parameter
            if return_vis:
                return torch.zeros_like(x), (x[0], None)
            else:
                return torch.zeros_like(x)