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

class GraphMedNCA(nn.Module):
    def __init__(self, hidden_channels=16, n_channels=1, fire_rate=0.5, 
                 slice_dim=None, device=None, log_enabled=True,
                 low_res_size=256, patch_size=16, num_patches=4, combine_inputs = True):
        super(GraphMedNCA, self).__init__()
        
        self.hidden_channels = hidden_channels
        self.n_channels = n_channels
        self.fire_rate = fire_rate
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.log_enabled = log_enabled
        self.low_res_size = low_res_size
        self.patch_size = patch_size
        self.num_patches = num_patches
        
        # Backbone 1: Low Resolution Processing (256x256)
        self.encoder_b1 = nn.Sequential(
            nn.Conv2d(n_channels, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, hidden_channels, 3, padding=1),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU(inplace=True)
        )
        
        # Grapher module for backbone 1
        self.grapher_b1 = GrapherModule(hidden_channels, hidden_channels)
        
        # Update network for backbone 1
        self.update_net_b1 = nn.Sequential(
            nn.Conv2d(hidden_channels, 128, 1),
            nn.ReLU(),
            nn.Conv2d(128, hidden_channels, 1)
        )
        
        # Output layer for backbone 1
        self.to_output_b1 = nn.Conv2d(hidden_channels, n_channels, 1)
        
        # Backbone 2: High Resolution Patch Processing (16x16 patches)
        self.encoder_b2 = nn.Sequential(
            nn.Conv2d(n_channels*2, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, hidden_channels, 3, padding=1),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU(inplace=True)
        )
        
        # Grapher module for backbone 2
        self.grapher_b2 = GrapherModule(hidden_channels, hidden_channels)
        
        # Update network for backbone 2
        self.update_net_b2 = nn.Sequential(
            nn.Conv2d(hidden_channels, 128, 1),
            nn.ReLU(),
            nn.Conv2d(128, hidden_channels, 1)
        )
        
        # Output layer for backbone 2
        self.to_output_b2 = nn.Conv2d(hidden_channels, n_channels, 1)
        
        # Combination layer for merging low and high res features
        self.combine_features = nn.Sequential(
            nn.Conv2d(hidden_channels * 2, hidden_channels, 1),
            nn.ReLU(),
            nn.Conv2d(hidden_channels, n_channels, 1)
        )

        
    def extract_random_patches_dual(images, labels, patch_size=16, num_patches=4):
        """
        Extract random patches from combined images (original + b1_output) and corresponding labels
        
        Args:
            images: Combined input tensor [B, C*2, H, W] (original + b1_output concatenated)
            labels: Ground truth labels tensor [B, C, H, W] (original labels only)
            patch_size: Size of patches to extract
            num_patches: Number of patches to extract per image
        
        Returns:
            patches: Extracted image patches [B*num_patches, C*2, patch_size, patch_size]
            patch_labels: Corresponding label patches [B*num_patches, C, patch_size, patch_size]
        """
        B, combined_C, H, W = images.shape
        _, label_C, _, _ = labels.shape
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
                    
                    # Extract patches - combined input has double channels, labels have original channels
                    patch = images[b:b+1, :, top:top+patch_size, left:left+patch_size]  # [1, C*2, patch_size, patch_size]
                    label_patch = labels[b:b+1, :, top:top+patch_size, left:left+patch_size]  # [1, C, patch_size, patch_size]
                
                patches.append(patch)
                patch_labels.append(label_patch)
        
        patches = torch.cat(patches, dim=0)  # [B*num_patches, C*2, patch_size, patch_size]
        patch_labels = torch.cat(patch_labels, dim=0)  # [B*num_patches, C, patch_size, patch_size]
        
        return patches, patch_labels  
    def graph_process(self, fmap, grapher_module, return_graph=False):
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
                out = grapher_module(g.x.to(fmap.device), g.edge_index.to(fmap.device))
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
    
    def forward_backbone1(self, x, steps=1):
        """
        Forward pass for backbone 1 (low resolution processing)
        
        Args:
            x: Input tensor [B, C, H, W] - will be resized to low_res_size
            steps: Number of NCA steps
        
        Returns:
            output: Low resolution segmentation [B, C, low_res_size, low_res_size]
            features: Feature representation for combination
        """
        original_size = x.shape[2:]
        
        # Downsample to low resolution
        x_low = F.interpolate(x, size=(self.low_res_size, self.low_res_size), 
                             mode='bilinear', align_corners=False)
        
        # Initial encoding
        h = self.encoder_b1(x_low)
        
        # Run NCA steps
        for step in range(steps):
            # Apply graph-based perception
            p = self.graph_process(h, self.grapher_b1)
            
            # Update cell states
            update = self.update_net_b1(p)
            
            # Stochastic update with fire rate
            mask = torch.rand_like(update[:, :1], device=self.device) < self.fire_rate
            mask = mask.float().repeat(1, self.hidden_channels, 1, 1)
            h = h + mask * update
        
        # Generate output
        out = self.to_output_b1(h)
        output = torch.sigmoid(out)
        
        # Upsample back to original size
        output_upsampled = F.interpolate(output, size=original_size, 
                                       mode='bilinear', align_corners=False)
        features_upsampled = F.interpolate(h, size=original_size, 
                                         mode='bilinear', align_corners=False)
        
        return output_upsampled, features_upsampled
    
    def forward_backbone2(self, patches, steps=1):
        """
        Forward pass for backbone 2 (high resolution patch processing)
        
        Args:
            patches: Input patches [B*num_patches, C, patch_size, patch_size]
            steps: Number of NCA steps
        
        Returns:
            output: Patch segmentations [B*num_patches, C, patch_size, patch_size]
            features: Feature representations for patches
        """
        # Initial encoding
        h = self.encoder_b2(patches)
        
        # Run NCA steps
        for step in range(steps):
            # Apply graph-based perception
            p = self.graph_process(h, self.grapher_b2)
            
            # Update cell states
            update = self.update_net_b2(p)
            
            # Stochastic update with fire rate
            mask = torch.rand_like(update[:, :1], device=self.device) < self.fire_rate
            mask = mask.float().repeat(1, self.hidden_channels, 1, 1)
            h = h + mask * update
        
        # Generate output
        out = self.to_output_b2(h)
        output = torch.sigmoid(out)
        
        return output, h
    
    def forward(self, x, labels=None, steps=1, mode='dual', return_graph=False):
        """
        Complete forward pass with dual backbone architecture
        
        Args:
            x: Input images [B, C, H, W] (varying sizes)
            labels: Ground truth labels [B, C, H, W] (for patch extraction)
            steps: Number of NCA steps
            mode: 'dual' for both backbones, 'b1' for low-res only, 'b2' for patches only
            return_graph: If True, returns graph visualization data
        
        Returns:
            Dictionary containing outputs from different processing paths
        """
        results = {}
        
        try:
            module = f"{__name__}:forward" if self.log_enabled else ""
            
            if mode in ['dual', 'b1']:
                # Backbone 1: Low resolution processing
                b1_output, b1_features = self.forward_backbone1(x, steps)
                results['b1_output'] = b1_output
                results['b1_features'] = b1_features
            
            if mode in ['dual', 'b2'] and labels is not None:
                # Combine original with upsampled b1 output
                if 'b1_output' in results:
                    combined_input = torch.cat([x, results['b1_output']], dim=1)  # Concat along channel dim
                else:
                    combined_input = x
                labels_for_patches = labels
                # Extract random patches from combined input
                patches, patch_labels = extract_random_patches(
                    combined_input, labels_for_patches, 
                    self.patch_size, self.num_patches
                )
                
                # Process patches through backbone 2
                b2_output, b2_features = self.forward_backbone2(patches, steps)
                
                results['b2_output'] = b2_output
                results['b2_features'] = b2_features
                results['patch_labels'] = patch_labels
                results['patches'] = patches
            
            # If dual mode, combine features (optional - for future enhancement)
            if mode == 'dual' and 'b1_features' in results and 'b2_features' in results:
                # This is a placeholder for feature combination
                # In practice, you might want to implement spatial attention or other combination methods
                results['combined_output'] = results['b1_output']  # For now, use b1 output
            
            # Handle graph visualization
            if return_graph and 'b1_output' in results:
                # Return graph data from the last processing step
                _, graph_data = self.graph_process(results['b1_features'][:1], 
                                                 self.grapher_b1, return_graph=True)
                results['graph_data'] = (x[0], graph_data)
            
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
        
        Args:
            x: Input images [B, C, H, W]
            labels: Ground truth labels [B, C, H, W]
            steps: Number of NCA steps
        
        Returns:
            Dictionary with outputs for loss calculation
        """
        results = self.forward(x, labels, steps, mode='dual')
        
        training_outputs = {}
        
        # Low resolution output and target
        if 'b1_output' in results:
            training_outputs['b1_pred'] = results['b1_output']
            training_outputs['b1_target'] = labels
        
        # High resolution patch outputs and targets
        if 'b2_output' in results:
            training_outputs['b2_pred'] = results['b2_output']
            training_outputs['b2_target'] = results['patch_labels']
        
        return training_outputs