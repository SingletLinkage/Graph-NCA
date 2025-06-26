import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, knn_graph
from torch_geometric.data import Data
from colorama import Fore, Back, Style
from src.utils.helper import log_message
from matplotlib import pyplot as plt
import numpy as np
import random

class GrapherModule(nn.Module):
    def __init__(self, in_channels, out_channels, num_heads=4):
        super(GrapherModule, self).__init__()
        self.gat = GATConv(in_channels, out_channels // num_heads, heads=num_heads)

    def forward(self, x, edge_index):
        x = self.gat(x, edge_index)
        return F.relu(x)

def visualize_graph(image_tensor, edge_index, num_nodes=200):
    """Visualize graph connections on image patches"""
    # Create figure for this visualization
    fig = plt.figure(figsize=(10, 10))
    
    # Convert tensor to numpy image (handle both 1-channel and 3-channel)
    img = image_tensor.cpu()
    if img.shape[0] == 1:
        img = img.repeat(3, 1, 1)  # Convert to RGB
    img = img.permute(1, 2, 0).numpy()
    plt.imshow(img)
    
    # Get node positions
    H, W = image_tensor.shape[1], image_tensor.shape[2]
    coords = torch.stack(torch.meshgrid(
        torch.linspace(0, 1, H, device='cpu'),
        torch.linspace(0, 1, W, device='cpu'),
        indexing='ij'
    ), dim=-1).view(H*W, 2)
    
    # Plot subset of nodes and connections
    nodes = np.random.choice(len(coords), min(num_nodes, len(coords)))
    for node in nodes:
        x, y = coords[node]
        plt.scatter(y*W, x*H, c='red', s=10)
        
        # Plot connections (limit to 4 for clarity)
        neighbors = edge_index[1][edge_index[0] == node]
        for neighbor in neighbors[:4]:
            nx, ny = coords[neighbor]
            plt.plot([y*W, ny*W], [x*H, nx*H], 'r-', alpha=0.3)
            
    plt.title('Graph Connections')
    plt.axis('off')
    
    # Instead of saving, return the figure for embedding in subplots
    return fig

def image_to_graph(feature_map, k=8, log_enabled=False):
    B, C, H, W = feature_map.size()
    graphs = []
    module = f"{__name__}:image_to_graph" if log_enabled else ""

    for i in range(B):
        fmap = feature_map[i]
        nodes = fmap.view(C, -1).permute(1, 0)
        edge_index = knn_graph(nodes, k=k)
        graphs.append(Data(x=nodes, edge_index=edge_index))

    return graphs
def extract_grid_patches(images, labels, patch_size=64):
    """
    Extract non-overlapping grid patches (e.g., 64x64) across the entire image.
    
    Args:
        images: Input tensor [B, C, H, W]
        labels: Label tensor [B, C, H, W]
        patch_size: Size of each patch (default: 64)

    Returns:
        patches: [B*num_patches, C, patch_size, patch_size]
        patch_labels: [B*num_patches, C, patch_size, patch_size]
    """
    B, C, H, W = images.shape
    assert H % patch_size == 0 and W % patch_size == 0, "Image must be divisible by patch size"

    patches = []
    patch_labels = []

    for b in range(B):
        for i in range(0, H, patch_size):
            for j in range(0, W, patch_size):
                patch = images[b:b+1, :, i:i+patch_size, j:j+patch_size]
                # print("Shape of Patch:", patch.shape)
                label_patch = labels[b:b+1, :, i:i+patch_size, j:j+patch_size]
                patches.append(patch)
                patch_labels.append(label_patch)

    patches = torch.cat(patches, dim=0)
    patch_labels = torch.cat(patch_labels, dim=0)
    return patches, patch_labels

def extract_random_patches(images, labels, patch_size=16, num_patches=4):
    """
    Extract random patches from images and corresponding labels
    
    Args:
        images: Input images tensor [B, C, H, W]
        labels: Ground truth labels tensor [B, C, H, W]
        patch_size: Size of patches to extract
        num_patches: Number of patches to extract per image
    
    Returns:
        patches: Extracted image patches [B*num_patches, C, patch_size, patch_size]
        patch_labels: Corresponding label patches [B*num_patches, C, patch_size, patch_size]
    """
    B, C, H, W = images.shape
    patches = []
    patch_labels = []
    
    for b in range(B):
        for _ in range(num_patches):
            # Random top-left corner
            max_h = H - patch_size
            max_w = W - patch_size
            
            if max_h <= 0 or max_w <= 0:
                # If image is smaller than patch size, use the entire image
                patch = F.interpolate(images[b:b+1], size=(patch_size, patch_size), mode='bilinear', align_corners=False)
                label_patch = F.interpolate(labels[b:b+1], size=(patch_size, patch_size), mode='nearest')
            else:
                top = random.randint(0, max_h)
                left = random.randint(0, max_w)
                
                # Extract patches
                patch = images[b:b+1, :, top:top+patch_size, left:left+patch_size]
                label_patch = labels[b:b+1, :, top:top+patch_size, left:left+patch_size]
            
            patches.append(patch)
            patch_labels.append(label_patch)
    
    patches = torch.cat(patches, dim=0)
    patch_labels = torch.cat(patch_labels, dim=0)
    
    return patches, patch_labels
class ImprovedEncoder(nn.Module):
    def __init__(self, in_channels=3, out_channels=16):
        super(ImprovedEncoder, self).__init__()

        self.block1 = nn.Sequential(
            nn.Conv2d(in_channels, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )

        self.block2 = nn.Sequential(
            nn.Conv2d(64, 128, 3, padding=1, stride=2),  # Downsample
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )

        self.block3 = nn.Sequential(
            nn.Conv2d(128, out_channels, 3, padding=1, stride=2),  # Downsample again
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        return x
class DeepDecoder16to64(nn.Module):
    def __init__(self, in_channels=1, out_channels=1):
        super(DeepDecoder16to64, self).__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(64, 64, kernel_size=4, stride=2, padding=1),  # 16x16 -> 32x32
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(32, 16, kernel_size=4, stride=2, padding=1),  # 32x32 -> 64x64
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),

            nn.Conv2d(16, out_channels, kernel_size=3, padding=1)
        )

    def forward(self, x):
        return self.net(x)
class UNetDecoder16to256(nn.Module):
    def __init__(self, in_channels=1, out_channels=1):
        super(UNetDecoder16to256, self).__init__()

        def conv_block(in_ch, out_ch):
            return nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True)
            )

        self.enc1 = conv_block(in_channels, 64)            # -> [B, 64, 16, 16]
        self.enc2 = conv_block(64, 128)                    # -> [B, 128, 8, 8]
        self.enc3 = conv_block(128, 256)                   # -> [B, 256, 4, 4]
        self.enc4 = conv_block(256, 512)                   # -> [B, 512, 2, 2]

        self.bottleneck = conv_block(512, 1024)            # -> [B, 1024, 2, 2] (no downsampling to 1x1)

        self.up4 = nn.ConvTranspose2d(1024, 512, 2, stride=2)  # -> [B, 512, 4, 4]
        self.dec4 = conv_block(768, 512)

        self.up3 = nn.ConvTranspose2d(512, 256, 2, stride=2)   # -> [B, 256, 8, 8]
        self.dec3 = conv_block(384, 256)

        self.up2 = nn.ConvTranspose2d(256, 128, 2, stride=2)   # -> [B, 128, 16, 16]
        self.dec2 = conv_block(192, 128)

        self.up1 = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1),   # -> [B, 64, 32, 32]
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1),    # -> [B, 32, 64, 64]
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(32, 16, 4, stride=2, padding=1),    # -> [B, 16, 128, 128]
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(16, out_channels, 4, stride=2, padding=1),  # -> [B, 1, 256, 256]
        )

    def forward(self, x):
        # Encoder pathway with downsampling
        e1 = self.enc1(x)                         # 16x16
        e2 = self.enc2(F.max_pool2d(e1, 2))       # 8x8
        e3 = self.enc3(F.max_pool2d(e2, 2))       # 4x4
        e4 = self.enc4(F.max_pool2d(e3, 2))       # 2x2

        b = self.bottleneck(e4)                  # 2x2

        d4 = self.up4(b)                         # 4x4
        d4 = self.dec4(torch.cat([d4, e3], dim=1))

        d3 = self.up3(d4)                        # 8x8
        d3 = self.dec3(torch.cat([d3, e2], dim=1))

        d2 = self.up2(d3)                        # 16x16
        d2 = self.dec2(torch.cat([d2, e1], dim=1))

        out = self.up1(d2)                       # 256x256
        return out



class GraphMedNCA(nn.Module):
    def __init__(self, hidden_channels=16, n_channels=1, fire_rate=0.5, 
                 device=None, log_enabled=True, patch_size=64, num_patches=16):
        super(GraphMedNCA, self).__init__()
        
        self.hidden_channels = hidden_channels
        self.n_channels = n_channels
        self.fire_rate = fire_rate
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.log_enabled = log_enabled
        self.patch_size = patch_size
        self.num_patches = num_patches
        
        # Universal encoder that works for both backbone sizes
        # For backbone 1 (64x64) and backbone 2 (16x16)
        # self.encoder = nn.Sequential(
        #     nn.Conv2d(n_channels, 64, 3, padding=1),
        #     nn.BatchNorm2d(64),
        #     nn.ReLU(inplace=True),
        #     nn.Conv2d(64, hidden_channels, 3, padding=1),
        #     nn.BatchNorm2d(hidden_channels),
        #     nn.ReLU(inplace=True)
        # )
        self.encoder = ImprovedEncoder(in_channels=n_channels, out_channels=hidden_channels)
        self.encoder_features = ImprovedEncoder(in_channels=1, out_channels=hidden_channels)
        self.grapher = GrapherModule(hidden_channels, hidden_channels)
        self.update_net = nn.Sequential(
            nn.Conv2d(hidden_channels, 128, 1),
            nn.ReLU(),
            nn.Conv2d(128, hidden_channels, 1)
        )
        self.b1_decoder = UNetDecoder16to256(in_channels=1, out_channels=1)
        # Decoder for backbone1 output (16x16to 256x256)
        # Universal output layer
        self.to_output = nn.Conv2d(hidden_channels, 1, 1)# here the last second parameter is the OUT PUT CHANNELS, which is 1 for segmentation
        #Decoder for backbone2 output (16x16 to 256x256)
        self.b2_decoder = DeepDecoder16to64(in_channels=1, out_channels=1)
    def upsample_b1_output(self, x):
        return self.b1_decoder(x)
    def graph_process(self, fmap, return_graph=False):
        """Process feature maps through graph module"""
        try:
            module = f"{__name__}:graph_process" if self.log_enabled else ""
            graphs = image_to_graph(fmap, log_enabled=self.log_enabled)
            outputs = []
            
            # Store the first graph's edge_index for visualization if requested
            edge_index = None
            if return_graph and len(graphs) > 0:
                edge_index = graphs[0].edge_index

            for i, g in enumerate(graphs):
                out = self.grapher(g.x.to(fmap.device), g.edge_index.to(fmap.device))
                H, W = fmap.shape[2], fmap.shape[3]
                out = out.permute(1, 0).view(-1, H, W)
                outputs.append(out)

            result = torch.stack(outputs)
            
            if return_graph:
                return result, edge_index
            else:
                return result
                
        except Exception as e:
            log_message(f"Error in graph_process: {str(e)}", "ERROR", module, self.log_enabled)
            import traceback
            traceback.print_exc()
            # Return input as fallback
            if return_graph:
                return fmap, None
            else:
                return fmap
    
    def forward_nca_steps(self, x, steps=1, return_graph=False):
        """
        Run NCA steps on input tensor (works for any size)
        
        Args:
            x: Input tensor [B, C, H, W]
            steps: Number of NCA steps
            return_graph: If True, returns graph data from last step
        
        Returns:
            output: Segmentation output [B, C, H, W]
            graph_data: (optional) Graph visualization data
        """
        # Initial encoding
        h = self.encoder(x)
        
        edge_index = None
        
        # Run NCA steps
        for step in range(steps):
            # Apply graph-based perception
            if return_graph and step == steps - 1:  # Only get graph data on last step
                p, edge_index = self.graph_process(h, return_graph=True)
            else:
                p = self.graph_process(h, return_graph=False)
            
            # Update cell states
            update = self.update_net(p)
            
            # Stochastic update with fire rate
            mask = torch.rand_like(update[:, :1], device=self.device) < self.fire_rate
            mask = mask.float().repeat(1, self.hidden_channels, 1, 1)
            h = h + mask * update
        
        # Generate output
        out = self.to_output(h)
        output = torch.sigmoid(out)
        
        if return_graph:
            return output, edge_index
        else:
            return output
    
    def forward_backbone1(self, x, steps=1, return_graph=False):
        """
        Backbone 1: Process 256x256 -> downsample to 64x64 -> NCA -> upsample to 256x256
        
        Args:
            x: Input tensor [B, C, 256, 256]
            steps: Number of NCA steps
            return_graph: If True, returns graph visualization data
        
        Returns:
            output: Upsampled output [B, C, 256, 256]
            graph_data: (optional) Graph data for visualization
        """
        # Downsample to 64x64
        x_low = F.interpolate(x, size=(64, 64), mode='bilinear', align_corners=False)
        print("Done !")
        # Run NCA on low resolution
        # Run NCA on low resolution
        if return_graph:
            output_low, edge_index = self.forward_nca_steps(x_low, steps, return_graph=True)
            output = self.upsample_b1_output(output_low)
            return output, (x_low[0], edge_index)
        else:
            output_low = self.forward_nca_steps(x_low, steps, return_graph=False)
        # Check and upsample to 64x64 if needed (e.g., NCA gave 16x16)
            if output_low.shape[2:] != (64, 64):
                # print("Shape of Output Low:", output_low.shape)
                # output = F.interpolate(output_low, size=(64, 64), mode='bilinear', align_corners=False)
                output = self.upsample_b1_output(output_low)
            return output
    
    
    def forward_backbone2(self, patches, steps=1, return_graph=False):
        """
        Backbone 2: Process 16x16 patches directly
        
        Args:
            patches: Input patches [B*num_patches, C, 16, 16]
            steps: Number of NCA steps
            return_graph: If True, returns graph visualization data
        
        Returns:
            output: Patch outputs [B*num_patches, C, 16, 16]
            graph_data: (optional) Graph data for visualization
        """
        # Run NCA on patches directly
        if return_graph:
            output, edge_index = self.forward_nca_steps(patches, steps, return_graph=True)
            return output, (patches[0], edge_index)
        else:
            output = self.forward_nca_steps(patches, steps, return_graph=False)
            # print("Shape of Output:", output.shape)
            output_low = self.b2_decoder(output)
            return output_low
    
    def forward(self, x, labels=None, steps=1, mode='dual', return_graph=False):
        """
        Complete forward pass following the exact architecture:
        Dataset -> 256x256 -> downsample 64x64 -> backbone1 -> upsample 256x256 
        -> add to original -> extract 16x16 patches -> backbone2
        
        Args:
            x: Input images [B, C, H, W] (will be resized to 256x256)
            labels: Ground truth labels [B, C, H, W] (will be resized to 256x256)
            steps: Number of NCA steps
            mode: 'dual', 'backbone1', 'backbone2'
            return_graph: If True, returns graph visualization data
        
        Returns:
            Dictionary containing outputs from different processing paths
        """
        results = {}
        
        try:
            module = f"{__name__}:forward" if self.log_enabled else ""
            
            # Resize input to 256x256 (original image)
            x_256 = F.interpolate(x, size=(256, 256), mode='bilinear', align_corners=False)
            if labels is not None:
                labels_256 = F.interpolate(labels, size=(256, 256), mode='nearest')
            else:
                labels_256 = None
            
            if mode in ['dual', 'backbone1']:
                # Backbone 1: 256x256 -> 64x64 -> NCA -> 256x256
                if return_graph and mode == 'backbone1':
                    b1_output, graph_data = self.forward_backbone1(x_256, steps, return_graph=True)
                    results['graph_data'] = graph_data
                else:
                    b1_output = self.forward_backbone1(x_256, steps, return_graph=False)
                
                results['b1_output'] = b1_output
                results['b1_target'] = labels_256
            
            if mode in ['dual', 'backbone2']:
                # Prepare input for backbone 2
                if mode == 'dual' and 'b1_output' in results:
                    # Add backbone1 output to original 256x256 image
                    combined_256 = x_256 + results['b1_output']
                elif mode == 'backbone2':
                    # For backbone2 only mode, use original image
                    combined_256 = x_256
                else:
                    combined_256 = x_256
                
                if labels_256 is not None:
                    # Extract 16x16 patches from combined image
                    patches, patch_labels = extract_grid_patches(
                        combined_256, labels_256, 
                        self.patch_size
                    )
                    
                    # Backbone 2: Process 16x16 patches
                    if return_graph and mode == 'backbone2':
                        b2_output, graph_data = self.forward_backbone2(patches, steps, return_graph=True)
                        results['graph_data'] = graph_data
                    else:
                        b2_output = self.forward_backbone2(patches, steps, return_graph=False)
                    
                    results['b2_output'] = b2_output
                    results['b2_target'] = patch_labels
                    results['patches'] = patches
            
            return results
            
        except Exception as e:
            module = f"{__name__}:forward" if self.log_enabled else ""
            log_message(f"Error in GraphMedNCA forward pass: {str(e)}", "ERROR", module, self.log_enabled)
            log_message(f"Input shape: {x.shape}, dtype: {x.dtype}", "WARNING", module, self.log_enabled)
            import traceback
            traceback.print_exc()
            
            # Return dummy outputs
            dummy_output = torch.zeros_like(x)
            results = {'b1_output': dummy_output}
            if return_graph:
                results['graph_data'] = (x[0], None)
            return results
    
    def get_training_outputs(self, x, labels, steps=1):
        """
        Get outputs specifically for training with proper loss calculation
        Following the exact pipeline: Dataset -> 256x256 -> backbone1 -> add -> patches -> backbone2
        
        Args:
            x: Input images [B, C, H, W] (any size, will be resized to 256x256)
            labels: Ground truth labels [B, C, H, W] (any size, will be resized to 256x256)
            steps: Number of NCA steps
        
        Returns:
            Dictionary with outputs for loss calculation
        """
        # results = self.forward(x, labels, steps, mode='dual')
        
        # training_outputs = {}
        
        # # Backbone 1: 256x256 predictions and targets
        # if 'b1_output' in results:
        #     training_outputs['b1_pred'] = results['b1_output']
        #     training_outputs['b1_target'] = results['b1_target']
        
        # # Backbone 2: 16x16 patch predictions and targets
        # if 'b2_output' in results:
        #     training_outputs['b2_pred'] = results['b2_output']
        #     training_outputs['b2_target'] = results['b2_target']
        
        # return training_outputs

        results = self.forward(x, labels, steps, mode='dual')

        training_outputs = {}

        if 'b1_output' in results:
            training_outputs['b1_pred'] = results['b1_output']
            training_outputs['b1_target'] = results['b1_target']
            # Save intermediate feature for fusion
            # training_outputs['b1_features'] = self.encoder(
            #     F.interpolate(x, size=(64, 64), mode='bilinear', align_corners=False)
            # )
            # Resize output back to 64x64 before passing to encoder

            # Downsample 256x256 prediction back to 64x64 before feature extraction
            b1_down = F.interpolate(results['b1_output'], size=(64, 64), mode='bilinear', align_corners=False)
            training_outputs['b1_features'] = self.encoder_features(b1_down)

        if 'b2_output' in results:
            training_outputs['b2_pred'] = results['b2_output']
            training_outputs['b2_target'] = results['b2_target']
            training_outputs['patches'] = results['patches']
            # Save b2 features as encoded patches
            # training_outputs['b2_features'] = self.encoder(
            #     F.interpolate(results['patches'], size=(64, 64), mode='bilinear', align_corners=False)
            # )
            training_outputs['b2_features'] = self.encoder_features(results['b2_output'])

        return training_outputs

    
class EnhancedGraphMedNCA(nn.Module):
    """Enhanced wrapper that ensures proper dual backbone integration"""
    def __init__(self, base_model):
        super(EnhancedGraphMedNCA, self).__init__()
        self.base_model = base_model
        
        # Feature integration module (implements the backbone model from diagram)
        self.feature_integrator = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),  # Assuming 16 channels from each backbone
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 1, kernel_size=1),  # Final segmentation output
        )
        # self.upsample_decoder = nn.Sequential(
        #     nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),  # 64 → 128
        #     nn.Conv2d(1, 8, kernel_size=3, padding=1),   # Now from 1 → 8
        #     nn.BatchNorm2d(8),
        #     nn.ReLU(inplace=True),
        #     nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),  # 128 → 256
        #     nn.Conv2d(8, 1, kernel_size=1)
        # )
        # Decoder with intermediate skip connections
        self.upsample_decoder = nn.ModuleDict({
            # First upsample: 64 → 128
            "up1": nn.Sequential(
                nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
                nn.Conv2d(1, 32, kernel_size=3, padding=1),
                nn.BatchNorm2d(32),
                nn.ReLU(inplace=True)
            ),
            
            # Skip 1: from input resized to 128x128
            "skip1": nn.Sequential(
                nn.Conv2d(3, 32, kernel_size=3, padding=1),  # use 1 if grayscale input
                nn.BatchNorm2d(32),
                nn.ReLU(inplace=True)
            ),
            
            # Second upsample: 128 → 256
            "up2": nn.Sequential(
                nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
                nn.Conv2d(32, 16, kernel_size=3, padding=1),
                nn.BatchNorm2d(16),
                nn.ReLU(inplace=True)
            ),
            
            # Skip 2: from input resized to 256x256
            "skip2": nn.Sequential(
                nn.Conv2d(3, 16, kernel_size=3, padding=1),
                nn.BatchNorm2d(16),
                nn.ReLU(inplace=True)
            ),
            
            # Final prediction head
            "final": nn.Sequential(
                nn.Conv2d(16, 8, kernel_size=3, padding=1),
                nn.BatchNorm2d(8),
                nn.ReLU(inplace=True),
                nn.Conv2d(8, 1, kernel_size=1)
            )
        })

        # Adaptive pooling for feature size matching
        self.adaptive_pool = nn.AdaptiveAvgPool2d((64, 64))
         # Prediction heads for individual backbones
        self.b1_head = nn.Sequential(
            nn.Upsample(scale_factor=4, mode='bilinear', align_corners=False),  # 64 → 256
            nn.Conv2d(16, 1, kernel_size=1)
        )
        self.b2_head = nn.Sequential(
            nn.Upsample(scale_factor=4, mode='bilinear', align_corners=False),  # 64 → 256
            nn.Conv2d(16, 1, kernel_size=1)
        )
    def forward(self, x, target=None, steps=4):
        """
        Fixed forward method that properly handles shape mismatches
        """
        # Get outputs from base GraphMedNCA model
        base_outputs = self.base_model.get_training_outputs(x, target, steps=steps)
        
        enhanced_outputs = {}
        batch_size = x.shape[0]
        target_size = x.shape[-2:]  # Use input size as reference
        
        # Extract backbone features with proper shape handling
        if 'b1_features' in base_outputs:
            b1_features = base_outputs['b1_features']
            # Ensure correct batch size
            if b1_features.shape[0] != batch_size:
                b1_features = b1_features[:batch_size]
            enhanced_outputs['b1_features'] = b1_features
            
        if 'b2_features' in base_outputs:
            b2_features = base_outputs['b2_features']
            # Ensure correct batch size
            if b2_features.shape[0] != batch_size:
                b2_features = b2_features[:batch_size]
            enhanced_outputs['b2_features'] = b2_features
            
        # Preserve individual backbone predictions with proper upsampling
        if 'b1_pred' in base_outputs:
            b1_pred = base_outputs['b1_pred']
            
            # Fix batch size mismatch
            if b1_pred.shape[0] != batch_size:
                b1_pred = b1_pred[:batch_size]
            
            # Upsample to target size if needed
            if b1_pred.shape[-2:] != target_size:
                b1_pred = F.interpolate(b1_pred, size=target_size, mode='bilinear', align_corners=True)
            
            enhanced_outputs['b1_pred'] = b1_pred
            
        if 'b2_pred' in base_outputs:
            b2_pred = base_outputs['b2_pred']
            
            # Fix batch size mismatch
            if b2_pred.shape[0] != batch_size:
                b2_pred = b2_pred[:batch_size]
            
            # Upsample to target size if needed
            if b2_pred.shape[-2:] != target_size:
                b2_pred = F.interpolate(b2_pred, size=target_size, mode='bilinear', align_corners=True)
            
            enhanced_outputs['b2_pred'] = b2_pred
        
        # Feature integration (following the orange line in diagram)
        if 'b1_features' in enhanced_outputs and 'b2_features' in enhanced_outputs:
            b1_feats = enhanced_outputs['b1_features']
            b2_feats = enhanced_outputs['b2_features']
            
            # Ensure batch sizes match
            min_batch_size = min(b1_feats.shape[0], b2_feats.shape[0], batch_size)
            b1_feats = b1_feats[:min_batch_size]
            b2_feats = b2_feats[:min_batch_size]
            
            # Resize features to common size
            common_size = (64, 64)  # Common feature map size
            b1_resized = F.interpolate(b1_feats, size=common_size, mode='bilinear', align_corners=True)
            b2_resized = F.interpolate(b2_feats, size=common_size, mode='bilinear', align_corners=True)
            
            # Concatenate features (implements the combination shown in diagram)
            combined_features = torch.cat([b1_resized, b2_resized], dim=1)
            enhanced_outputs['combined_features'] = combined_features
            
            # Generate final integrated prediction (main output following orange line)
            # Step 1: Fuse channels to 1-channel low-res prediction
            low_res_pred = self.feature_integrator(combined_features)  # [B, 1, 64, 64]
            # Decode step-by-step with skip connections

            # Step 1: upsample from 64 → 128
            x_up1 = self.upsample_decoder["up1"](low_res_pred)

            # Skip connection from input @128x128
            skip1 = F.interpolate(x, size=x_up1.shape[-2:], mode='bilinear', align_corners=True)
            skip1 = self.upsample_decoder["skip1"](skip1)

            # Fuse
            x_fused1 = x_up1 + skip1

            # Step 2: upsample to 256
            x_up2 = self.upsample_decoder["up2"](x_fused1)

            # Skip connection from input @256x256
            skip2 = F.interpolate(x, size=x_up2.shape[-2:], mode='bilinear', align_corners=True)
            skip2 = self.upsample_decoder["skip2"](skip2)

            # Fuse again
            x_fused2 = x_up2 + skip2

            # Final prediction
            combined_pred = self.upsample_decoder["final"](x_fused2)

            # Step 2: Upsample to full resolution using learnable decoder
            # combined_pred = self.upsample_decoder(low_res_pred)  # [B, 1, 256, 256]
                        
            enhanced_outputs['combined_pred'] = combined_pred
        
        return enhanced_outputs
