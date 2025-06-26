import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
 
class BasicNCA(nn.Module):
    r"""Basic implementation of an NCA using a sobel x and y filter for the perception
    This model also uses a 3x3 Laplacian filter for perception, which is a common choice for edge detection.
    """
    def __init__(self, hidden_channels, fire_rate, device, hidden_size=128, n_channels=1, init_method="standard", log_enabled=True):
        r"""Init function
            #Args:
                hidden_channels: number of channels per cell
                fire_rate: random activation of each cell
                device: device to run model on
                hidden_size: hidden size of model
                n_channels: number of input channels
                init_method: Weight initialisation function
        """
        super(BasicNCA, self).__init__()

        self.device = device
        self.hidden_channels = hidden_channels
        self.input_channels = n_channels
        self.log_enabled = log_enabled

        self.fc0 = nn.Linear(hidden_channels*3, hidden_size)
        self.fc1 = nn.Linear(hidden_size, hidden_channels, bias=False)
        with torch.no_grad():
            self.fc1.weight.zero_()

        if init_method == "xavier":
            torch.nn.init.xavier_uniform(self.fc0.weight)
            torch.nn.init.xavier_uniform(self.fc1.weight)

        self.p0 = nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, stride=1, padding=1, padding_mode="reflect")
        self.p1 = nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, stride=1, padding=1, padding_mode="reflect")

        self.fire_rate = fire_rate
        self.output_conv = nn.Conv2d(hidden_channels, n_channels, kernel_size=1)
        self.to(self.device)

    @DeprecationWarning
    def _perceive(self, x):
        r"""Perceptive function, combines 2 sobel x and y and laplacian outputs with the identity of the cell
            #Args:
                x: image
        """
        def _perceive_with(x, weight):
            conv_weights = torch.from_numpy(weight.astype(np.float32)).to(self.device)
            conv_weights = conv_weights.view(1,1,3,3).repeat(self.hidden_channels, 1, 1, 1)
            return F.conv2d(x, conv_weights, padding=1, groups=self.hidden_channels)

        dx = np.outer([1, 2, 1], [-1, 0, 1]) / 8.0  # Sobel filter
        dy = dx.T
        dlaplacian = np.array([[1, 2, 1], [2, -12, 2], [1, 2, 1]]) / 12.0  # Laplacian filter

        y1 = _perceive_with(x, dx)
        y2 = _perceive_with(x, dy)
        y3 = _perceive_with(x, dlaplacian)
        y = torch.cat((x,y1,y2,y3),1)

        return y

    def perceive(self, x):
        r"""Perception function, applies perception to each cell of an image
            #Args:
                x: image
        """
        y1 = self.p0(x)
        y2 = self.p1(x)
        y = torch.cat((x,y1,y2),1)
        return y

    def update(self, x_in, fire_rate):
        r"""Update function runs same nca rule on each cell of an image with a random activation
            #Args:
                x_in: image
                fire_rate: random activation of cells
        """
        x = x_in.transpose(1,3)

        dx = self.perceive(x)
        dx = dx.transpose(1,3)
        dx = self.fc0(dx)
        dx = F.relu(dx)
        dx = self.fc1(dx)

        if fire_rate is None:
            fire_rate=self.fire_rate
        stochastic = torch.rand([dx.size(0),dx.size(1),dx.size(2),1])>fire_rate
        stochastic = stochastic.float().to(self.device)
        dx = dx * stochastic

        x = x+dx.transpose(1,3)

        x = x.transpose(1,3)

        return x

    @DeprecationWarning
    def _forward(self, x, steps=64, fire_rate=0.5):
        r"""Forward function applies update function s times leaving input channels unchanged
            #Args:
                x: image
                steps: number of steps to run update
                fire_rate: random activation rate of each cell
        """
        # print(f"Input shape: {x.shape}, Input channels: {self.input_channels}, Channel n: {self.hidden_channels}")
        if x.shape[1] < self.hidden_channels:
            padding = self.hidden_channels - x.shape[1]
            x = torch.cat((x, torch.zeros(x.shape[0], padding, x.shape[2], x.shape[3]).to(self.device)), dim=1)
            x = x.permute(0, 2, 3, 1)
        # print(f"Input shape after padding: {x.shape}")

        for step in range(steps):
            x2 = self.update(x, fire_rate).clone() #[...,3:][...,3:]
            x = torch.concat((x[...,:self.input_channels], x2[...,self.input_channels:]), 3)
        
        # print(f"Output shape: {x.shape}")

        # slice first hidden channel and permute to original format
        # out = x[..., self.input_channels:self.input_channels+1]
        # out = out.permute(0, 3, 1, 2)

        out = self.output_conv(x.permute(0, 3, 1, 2))  # [B, C, H, W]

        # print(f"Output shape after slicing: {x.shape}")
        return out

    def forward(self, x, steps=64, fire_rate=0.5):
        # If input has fewer channels than hidden, pad
        if x.shape[1] < self.hidden_channels:
            padding = self.hidden_channels - x.shape[1]
            x = torch.cat((x, torch.zeros(x.shape[0], padding, x.shape[2], x.shape[3], device=self.device)), dim=1)

        x = x.permute(0, 2, 3, 1)  # [B, H, W, C]

        for _ in range(steps):
            x2 = self.update(x, fire_rate).clone() #[...,3:][...,3:]
            x = torch.concat((x[...,:self.input_channels], x2[...,self.input_channels:]), 3)

        out = x[..., self.input_channels:self.input_channels+1]
        out = out.permute(0, 3, 1, 2)  # [B, 1, H, W]
        return out
