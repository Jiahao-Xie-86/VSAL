import torch
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
import torch.nn.functional as F
from torchvision.transforms import GaussianBlur
from torchvision.transforms.functional import to_pil_image

def plot_graph_layout(layout, adj_matrix=None, size=(224, 224), return_tensor=True):
    fig, ax = plt.subplots(figsize=(3, 3))  # 224x224 pixels corresponds to 3 inches at 72 DPI

       # Ensure layout is a tensor (if it's still a numpy array, convert to tensor first)
    if isinstance(layout, np.ndarray):
        layout = torch.tensor(layout)
    
    # Remove padded nodes (assume actual nodes have non-zero coordinates)
    actual_nodes = torch.norm(layout, dim=1) > 0  # Filter out padded nodes based on non-zero coordinates
    layout = layout[actual_nodes]  # Only keep the actual nodes

    # Convert layout back to NumPy after filtering
    layout = layout.detach().cpu().numpy()
    
    # Normalize coordinates to spread out the layout within [0, 224]
    layout_min, layout_max = layout.min(axis=0), layout.max(axis=0)
    layout_range = layout_max - layout_min
    layout = (layout - layout_min) / layout_range  # Normalize to [0, 1]
    layout = layout * (size[0] - 20) + 10  # Scale to fit within the plot, add padding

    # Check if the adjacency matrix is provided
    if adj_matrix is not None:
        # Ensure the adjacency matrix matches the number of nodes in the layout
        adj_matrix = adj_matrix[:layout.shape[0], :layout.shape[0]]  # Crop to match the number of nodes

        # Draw edges
        num_nodes = layout.shape[0]
        for i in range(num_nodes):
            for j in range(i + 1, num_nodes):
                if adj_matrix[i, j] == 1:  # If there's an edge between node i and node j
                    ax.plot([layout[i, 0], layout[j, 0]],  # Draw edge from node i to node j
                            [layout[i, 1], layout[j, 1]], 
                            color="blue", lw=1.0, alpha=0.7)  # Blue edges with thicker lines

    # Draw nodes with red markers (s=50 for marker size)
    ax.scatter(layout[:, 0], layout[:, 1], color='red', s=10, edgecolor='black', zorder=5)

    # Set limits and hide axis
    ax.set_xlim(0, size[0])
    ax.set_ylim(0, size[1])
    plt.axis('off')
    fig.canvas.draw()

    # Convert plot to RGB image
    image = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    image = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    plt.close(fig)

    # Resize to 224x224 pixels
    pil_img = Image.fromarray(image)
    pil_img_resized = pil_img.resize(size, Image.Resampling.LANCZOS)
    
    if return_tensor:
        image_tensor = transforms.ToTensor()(pil_img_resized)
        return image_tensor
    else:
        return pil_img_resized


def visualize_graph_layout(layout, adj_matrix, size=(224, 224), line_thickness=1, edge_alpha=1, edge_thickness=0):

    # Remove padded nodes (assume actual nodes have non-zero coordinates)
    actual_nodes = torch.norm(layout, dim=1) > 0  # Filter out padded nodes based on non-zero coordinates
    layout = layout[actual_nodes]  # Only keep the actual nodes
    
    num_nodes = layout.size(0)
    device = layout.device
    image = torch.ones(3, size[1], size[0], device=device)  # White background

    # Normalize and scale layout to fit with padding
    layout_min = layout.min(dim=0, keepdim=True)[0]
    layout_max = layout.max(dim=0, keepdim=True)[0]
    layout = (layout - layout_min) / (layout_max - layout_min + 1e-8)  # Normalize to [0, 1]
    layout = layout * (torch.tensor(size, device=device) - 20) + 10  # Scale to size with padding

    # Draw edges with blue, semi-transparent lines and set thickness
    edge_indices = (adj_matrix > 0).nonzero(as_tuple=False)
    start_points = layout[edge_indices[:, 0]]
    end_points = layout[edge_indices[:, 1]]
    
    # Increase number of steps for smoother lines
    # max_steps = int((start_points - end_points).norm(dim=1).max().item()) * 100  # Increase for smoother lines
    raw_steps = int((start_points - end_points).norm(dim=1).max().item() * 100)
    max_steps = min(raw_steps, 1000)  # set an upper limit
    t_values = torch.linspace(0, 1, steps=max_steps, device=device)
    points = (1 - t_values.view(-1, 1, 1)) * start_points.view(1, -1, 2) + t_values.view(-1, 1, 1) * end_points.view(1, -1, 2)
    x = points[:, :, 0].long().clamp(0, size[0] - 1)
    y = points[:, :, 1].long().clamp(0, size[1] - 1)

    # Draw thicker edges by expanding points around each interpolated edge point
    for dx in range(-edge_thickness, edge_thickness + 1):
        for dy in range(-edge_thickness, edge_thickness + 1):
            if dx**2 + dy**2 <= edge_thickness**2:  # Circular mask for line thickness
                x_idx = x + dx
                y_idx = y + dy
                valid_idx = (0 <= x_idx) & (x_idx < size[0]) & (0 <= y_idx) & (y_idx < size[1])
                image[2, y_idx[valid_idx], x_idx[valid_idx]] = edge_alpha  # Set blue channel for edges
                image[0, y_idx[valid_idx], x_idx[valid_idx]] *= (1 - edge_alpha)  # Dim the red channel
                image[1, y_idx[valid_idx], x_idx[valid_idx]] *= (1 - edge_alpha)  # Dim the green channel

    # Draw nodes as small red circles
    x = layout[:, 0].long().clamp(0, size[0] - 1)
    y = layout[:, 1].long().clamp(0, size[1] - 1)
    for dx in range(-line_thickness, line_thickness + 1):
        for dy in range(-line_thickness, line_thickness + 1):
            x_idx = x + dx
            y_idx = y + dy
            valid_idx = (0 <= x_idx) & (x_idx < size[0]) & (0 <= y_idx) & (y_idx < size[1])
            image[0, y_idx[valid_idx], x_idx[valid_idx]] = 1.0  # Red channel for nodes
            image[1, y_idx[valid_idx], x_idx[valid_idx]] = 0.0
            image[2, y_idx[valid_idx], x_idx[valid_idx]] = 0.0

    # Apply Gaussian blur for smoother appearance
    blur = GaussianBlur(kernel_size=(5, 5), sigma=(1, 1))
    image = blur(image.unsqueeze(0)).squeeze(0)

    # Normalize image for consistent appearance
    mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=device).view(3, 1, 1)
    image = (image - mean) / std

    return image  # Return as differentiable tensor if needed




# def save_tensor_as_pdf(layout, adj_matrix, save_path, size=(224, 224)):
#     """
#     Plots a graph layout with nodes and edges as specified, and saves it as a PDF file.

#     Args:
#     - layout (torch.Tensor): Node coordinates, shape (num_nodes, 2).
#     - adj_matrix (torch.Tensor): Adjacency matrix, shape (num_nodes, num_nodes).
#     - save_path (str): The file path to save the PDF.
#     - size (tuple): Size of the figure (width, height) in pixels.
#     """
#     # Convert layout to numpy for plotting
#     layout = layout.detach().cpu().numpy()
#     adj_matrix = adj_matrix.detach().cpu().numpy()

#     # Set up figure with size corresponding to pixel dimensions
#     fig, ax = plt.subplots(figsize=(size[0] / 72, size[1] / 72), dpi=72)  # Convert pixels to inches

#     # Draw edges
#     num_nodes = layout.shape[0]
#     for i in range(num_nodes):
#         for j in range(i + 1, num_nodes):
#             if adj_matrix[i, j] == 1:  # If there's an edge between node i and node j
#                 ax.plot(
#                     [layout[i, 0], layout[j, 0]],  # X coordinates
#                     [layout[i, 1], layout[j, 1]],  # Y coordinates
#                     color="blue", lw=1.0, alpha=0.7
#                 )

#     # Draw nodes with red markers and black edge color
#     ax.scatter(layout[:, 0], layout[:, 1], color='red', s=50, edgecolor='black', zorder=5)

#     # Set limits and hide axis
#     ax.set_xlim(0, size[0])
#     ax.set_ylim(0, size[1])
#     ax.axis('off')

#     # Save the plot as a PDF
#     plt.savefig(save_path, format="pdf", bbox_inches="tight", pad_inches=0)
#     plt.close(fig)


def save_tensor_as_pdf(image_tensor, save_path):
    """
    Saves a single image tensor as a PDF file.

    Args:
        image_tensor (torch.Tensor): Image tensor of shape (3, H, W).
        save_path (str): Path to save the PDF file.
    """
    try:
        # Convert tensor to PIL Image
        pil_image = to_pil_image(image_tensor.cpu())
        
        # Save as PDF
        pil_image.save(save_path, format='PDF')
        print(f"Image saved as PDF at {save_path}")
    except Exception as e:
        print(f"Error saving image as PDF: {e}")


def wasserstein_loss(real_validity, fake_validity):
    return torch.mean(fake_validity) - torch.mean(real_validity)


def contrastive_layout_loss(ref_embeddings, gen_embeddings, temperature=0.1):
    """
    Contrastive Layout Loss (InfoNCE style).
    
    For each graph G, treat the pair:
    - Reference layout r_ref(G) as the positive
    - Generated layout g_θ_gen(G, z) as the anchor
    
    Negatives are layouts of other graphs in the batch.
    
    Args:
        ref_embeddings: Reference layout embeddings (batch_size, embedding_dim)
        gen_embeddings: Generated layout embeddings (batch_size, embedding_dim)
        temperature: Temperature parameter τ for softmax
        
    Returns:
        contrastive_loss: InfoNCE-style contrastive loss
    """
    batch_size = ref_embeddings.size(0)
    
    # Normalize embeddings for cosine similarity
    ref_embeddings = F.normalize(ref_embeddings, p=2, dim=1)
    gen_embeddings = F.normalize(gen_embeddings, p=2, dim=1)
    
    # Compute cosine similarity matrix
    # sim_matrix[i,j] = similarity between ref_embeddings[i] and gen_embeddings[j]
    sim_matrix = torch.mm(ref_embeddings, gen_embeddings.t()) / temperature
    
    # For each reference layout, the positive pair is with the corresponding generated layout
    # (diagonal elements are positive pairs)
    positive_scores = torch.diag(sim_matrix)
    
    # Compute log-sum-exp for denominator (all similarities in the batch)
    log_sum_exp = torch.logsumexp(sim_matrix, dim=1)
    
    # Contrastive loss: -log(exp(positive_score) / sum(exp(all_scores)))
    contrastive_loss = -torch.mean(positive_scores - log_sum_exp)
    
    return contrastive_loss


def perceptual_layout_loss(ref_embeddings, gen_embeddings):
    """
    Perceptual Layout Loss.
    
    Using the same encoder h(·), enforce feature-space similarity 
    between generated and reference layouts.
    
    Args:
        ref_embeddings: Reference layout embeddings (batch_size, embedding_dim)
        gen_embeddings: Generated layout embeddings (batch_size, embedding_dim)
        
    Returns:
        perceptual_loss: L2 distance between embeddings
    """
    # L2 distance between reference and generated embeddings
    perceptual_loss = F.mse_loss(ref_embeddings, gen_embeddings)
    
    return perceptual_loss


def remove_module_prefix(state_dict):
    new_state_dict = {}
    for key, value in state_dict.items():
        new_key = key.replace('module.', '')  # Remove 'module.' from keys
        new_state_dict[new_key] = value
    return new_state_dict