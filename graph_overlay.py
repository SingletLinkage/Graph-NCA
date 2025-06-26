import torch
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

def visualize_graph(image, edge_index, num_nodes=200, max_connections=4, 
                  node_color='red', node_size=10, edge_color='red', edge_alpha=0.3,
                  figsize=(10, 10), random_seed=None):
    """
    Visualize a graph overlaid on an image.
    
    Args:
        image: Input image to overlay graph on
        edge_index: Tensor of shape [2, num_edges] containing edge indices
        num_nodes: Maximum number of nodes to visualize
        max_connections: Maximum number of connections to show per node
        node_color: Color of the graph nodes
        node_size: Size of the graph nodes
        edge_color: Color of the graph edges
        edge_alpha: Transparency of the graph edges
        figsize: Figure size in inches (width, height)
        random_seed: Random seed for node selection
    
    Returns:
        fig: Matplotlib figure containing the visualization
    """
    # Set random seed if provided for reproducibility
    if random_seed is not None:
        np.random.seed(random_seed)
        
    fig = plt.figure(figsize=figsize)
    
    # Get node positions
    H, W = image.shape[0], image.shape[1]  # Fixed: use correct dimensions from image
    total_nodes = H * W
    coords = torch.stack(torch.meshgrid(
        torch.linspace(0, 1, H, device='cpu'),
        torch.linspace(0, 1, W, device='cpu'),
        indexing='ij'
    ), dim=-1).view(total_nodes, 2)
    
    # Ensure edge_index references valid node indices
    max_idx = total_nodes - 1
    valid_edges = (edge_index[0] <= max_idx) & (edge_index[1] <= max_idx)
    edge_index = edge_index[:, valid_edges]
    
    # If no valid edges found, just plot the image
    if edge_index.shape[1] == 0:
        plt.title('No valid graph connections for this image size')
        plt.imshow(image)
        return fig
    
    # Display the image as background
    plt.imshow(image)
    
    # Plot subset of nodes and connections
    nodes = np.random.choice(total_nodes, min(num_nodes, total_nodes))
    for node in nodes:
        x, y = coords[node]
        plt.scatter(y*W, x*H, c=node_color, s=node_size)
        
        # Find valid neighbors only
        valid_neighbors_mask = edge_index[0] == node
        if valid_neighbors_mask.any():
            neighbors = edge_index[1][valid_neighbors_mask]
            # Ensure all neighbors are within valid range
            valid_neighbors = neighbors[neighbors < total_nodes]
            for neighbor in valid_neighbors[:max_connections]:
                nx, ny = coords[neighbor]
                plt.plot([y*W, ny*W], [x*H, nx*H], color=edge_color, alpha=edge_alpha)
            
    plt.title('Graph Connections')
    plt.axis('off')
    
    # Instead of saving, return the figure for embedding in subplots
    return fig


def main(graph_path, image_path, output_path="graph_overlay.png", dpi=300, **kwargs):
    """
    Main function to generate a graph overlay visualization.
    
    Args:
        graph_path: Path to the graph edge_index tensor file (.pt)
        image_path: Path to the image file
        output_path: Path to save the output visualization
        dpi: Resolution of the output image
        **kwargs: Additional parameters to pass to visualize_graph function
    """
    
    # Load data
    edge_index = torch.load(graph_path)
    # img = np.array(Image.open(image_path).convert("RGB"))
    img_tensor = torch.load(image_path).cpu()
    if img_tensor.shape[0] == 1:
        img_tensor = img_tensor.repeat(3, 1, 1)
    img = img_tensor.permute(1, 2, 0).numpy()
    
    # Default visualization parameters
    viz_params = {
        'num_nodes': 200,
        'max_connections': 4,
        'node_color': 'red',
        'node_size': 10,
        'edge_color': 'red',
        'edge_alpha': 0.3,
        'figsize': (10, 10),
        'random_seed': 42
    }
    
    # Update with any user-provided parameters
    viz_params.update(kwargs)
    
    # Generate visualization
    graph_fig = visualize_graph(img, edge_index, **viz_params)
    
    # Save output
    plt.savefig(output_path, bbox_inches='tight', dpi=dpi)
    print(f"Visualization saved to {output_path}")
    return graph_fig


# Run the main function if script is executed directly
if __name__ == "__main__":
    main(
        graph_path="/home/teaching/group21/final/test-mednca-temp/runs/model_2/inference_results/ISIC_0012373_0.67/graph_0.pt",
        image_path="/home/teaching/group21/final/test-mednca-temp/runs/model_2/inference_results/ISIC_0012373_0.67/image_tensor_0.pt",
        num_nodes=200,
        max_connections=3,
        node_color='yellow',
        node_size=20,
        edge_color='red',
        edge_alpha=0.25,
        output_path="graph_overlay.png",
        dpi=300,
    )
    # import argparse
    
    # parser = argparse.ArgumentParser(description="Generate graph overlay visualization")
    # parser.add_argument("--graph", type=str, help="Path to graph edge_index tensor file")
    # parser.add_argument("--image", type=str, help="Path to image file")
    # parser.add_argument("--output", type=str, default="graph_overlay.png", help="Output path")
    # parser.add_argument("--dpi", type=int, default=300, help="Output image resolution")
    # parser.add_argument("--nodes", type=int, default=200, help="Number of nodes to visualize")
    # parser.add_argument("--connections", type=int, default=4, help="Max connections per node")
    # parser.add_argument("--node-color", type=str, default="red", help="Color of nodes")
    # parser.add_argument("--node-size", type=int, default=10, help="Size of nodes")
    # parser.add_argument("--edge-color", type=str, default="red", help="Color of edges")
    # parser.add_argument("--edge-alpha", type=float, default=0.3, help="Transparency of edges")
    # parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    
    # args = parser.parse_args()
    
    # # Call main with parsed arguments
    # main(
    #     graph_path=args.graph,
    #     image_path=args.image,
    #     output_path=args.output,
    #     dpi=args.dpi,
    #     num_nodes=args.nodes,
    #     max_connections=args.connections,
    #     node_color=args.node_color,
    #     node_size=args.node_size,
    #     edge_color=args.edge_color,
    #     edge_alpha=args.edge_alpha,
    #     random_seed=args.seed
    # )

