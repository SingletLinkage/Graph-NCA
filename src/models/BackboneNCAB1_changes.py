import torch
import torch.nn as nn
import torch.nn.functional as F

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

        # Perception module: identity + conv3x3 + conv1x1
        self.perception_conv1 = nn.Conv2d(hidden_channels, hidden_channels, 3, padding=1)
        self.perception_conv2 = nn.Conv2d(hidden_channels, hidden_channels, 1)

        # Update network (matches Dense128 → ReLU → DenseN)
        # Note: Input channels = hidden_channels * 3 (identity + conv1 + conv2)
        self.update_net = nn.Sequential(
            nn.Conv2d(hidden_channels * 3, 128, 1),
            nn.ReLU(),
            nn.Conv2d(128, hidden_channels, 1)
        )

        # Output layer
        self.to_output = nn.Conv2d(hidden_channels, n_channels, 1)

    def perception_process(self, fmap, return_vis=False):
        try:
            module = f"{__name__}:perception_process" if self.log_enabled else ""

            # Identity
            identity = fmap

            # Learned convolutions
            conv1_out = self.perception_conv1(fmap)
            conv2_out = self.perception_conv2(fmap)

            # Concatenate all
            perception_vector = torch.cat([identity, conv1_out, conv2_out], dim=1)
            perception_vector = F.relu(perception_vector)

            if return_vis:
                return perception_vector, conv1_out[0, 0]  # for visualization
            else:
                return perception_vector

        except Exception as e:
            log_message(f"Error in perception_process: {str(e)}", "ERROR", module, self.log_enabled)
            import traceback
            traceback.print_exc()
            if return_vis:
                return fmap, None
            else:
                return fmap

    def forward(self, x, steps=1, return_vis=False):
        try:
            module = f"{__name__}:forward" if self.log_enabled else ""

            # Initial encoding
            h = self.encoder(x)
            vis_data = None

            for step in range(steps):
                if return_vis and step == steps - 1:
                    p, vis_data = self.perception_process(h, return_vis=True)
                else:
                    p = self.perception_process(h)

                # Get update
                update = self.update_net(p)

                # Apply stochastic fire_rate mask
                mask = torch.rand_like(update[:, :1], device=self.device) < self.fire_rate
                mask = mask.float().repeat(1, self.hidden_channels, 1, 1)

                # Residual update
                h = h + mask * update

            out = self.to_output(h)
            output = torch.sigmoid(out)

            if return_vis:
                return output, (x[0], vis_data)
            else:
                return output

        except Exception as e:
            log_message(f"Error in BasicNCA forward pass: {str(e)}", "ERROR", module, self.log_enabled)
            log_message(f"Input shape: {x.shape}, dtype: {x.dtype}", "WARNING", module, self.log_enabled)
            import traceback
            traceback.print_exc()
            if return_vis:
                return torch.zeros_like(x), (x[0], None)
            else:
                return torch.zeros_like(x)
            

