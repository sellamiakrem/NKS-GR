# -*- coding: utf-8 -*-
"""
@author: Akrem Sellami
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.colors import Normalize, to_hex
import matplotlib.patches as mpatches
import pandas as pd
import seaborn as sns
import networkx as nx
from matplotlib import cm


def plot_rgb_image(img, r_idx, g_idx, b_idx, title="RGB Composite", figsize=(5,5)):
    """
    Display RGB composite from hyperspectral image by selecting 3 bands.

    Args:
        img (ndarray): Hyperspectral image of shape (height, width, n_bands).
        r_idx (int): Index of the band to use for Red.
        g_idx (int): Index of the band to use for Green.
        b_idx (int): Index of the band to use for Blue.
        title (str): Title of the plot.
        figsize (tuple): Size of the figure in inches (width, height).
    """
    # Stack the 3 selected bands
    rgb = np.stack([img[:,:,r_idx],
                    img[:,:,g_idx],
                    img[:,:,b_idx]], axis=-1)
    
    # Normalize to [0, 1] for display
    rgb = (rgb - rgb.min()) / (rgb.max() - rgb.min())    
    
    fig, ax = plt.subplots(figsize=figsize)
    ax.imshow(rgb, interpolation='nearest')
    ax.set_title(title, fontsize=10, weight='bold')
    ax.axis('off')
    ax.set_aspect('equal')  # Keep pixel proportions
    plt.show()

def plot_ground_truth_map(gt, class_names=None, title="Ground Truth Map", figsize=(6, 6)):
    """
    Display ground truth map with custom colors and a more aesthetic legend.

    Parameters:
        gt (ndarray): 2D array of groundtruth labels.
        class_names (list, optional): List of class names. The first should be background.
        title (str): Title of the plot.
        figsize (tuple): Size of the plot.
    """
    num_classes = int(gt.max()) + 1
    
    # Define colors: black for background then tab20 colors for the rest
    base_colors = ['black'] + [plt.cm.tab20(i % 20) for i in range(1, num_classes)]

    cmap = mcolors.ListedColormap(base_colors)

    bounds = np.arange(-0.5, num_classes + 0.5, 1)
    norm = mcolors.BoundaryNorm(bounds, cmap.N)

    fig, ax = plt.subplots(figsize=figsize)
    ax.imshow(gt, cmap=cmap, norm=norm, interpolation='nearest')
    ax.set_title(title, fontsize=14, weight='bold')
    ax.axis('off')

    if class_names is not None and len(class_names) == num_classes:
        patches = [mpatches.Patch(color=base_colors[i],
                                   label=f"{i} - {class_names[i]}")
                   for i in range(num_classes)]

        ax.legend(handles=patches,
                   bbox_to_anchor=(1.05, 1),
                   loc='upper left',
                   borderaxespad=0,
                   frameon=False)

        fig.subplots_adjust(right=0.7)  # laisse de la place à la légende

    plt.show()

def plot_class_distribution(y, class_names, figsize=(8,6), title='Class Distribution'):

    # Filter out background
    y_filtered = y[y != 0]

    # Count class occurrences
    unique_labels, counts = np.unique(y_filtered, return_counts=True)
    mapped_class_names = [class_names[int(i)-1] for i in unique_labels]  

    # Normalize counts for colormap
    norm = Normalize(vmin=min(counts), vmax=max(counts))
    cmap = cm.get_cmap("coolwarm")
    colors = [to_hex(cmap(norm(c))) for c in counts]

    # DataFrame
    df = pd.DataFrame({'Class': mapped_class_names, 'Pixel Count': counts})

    # Plot
    plt.figure(figsize=figsize)
    sns.barplot(data=df, x='Class', y='Pixel Count', palette=colors)

    plt.xticks(rotation=45, ha='right')
    plt.title(title)
    plt.xlabel('Classes')
    plt.ylabel('Number of pixels')

    # Add counts
    for i, count in enumerate(counts):
        plt.text(i, count, str(count), ha='center', va='bottom', fontsize=8)

    plt.tight_layout()
    plt.grid()
    plt.show()


def plot_average_spectral_signatures(data, labels, class_names=None, title="Mean Spectral Signatures"):
    """
    Display the mean spectral signatures per class.

    Parameters:
        data (ndarray): Image hyperspectrale de forme (H, W, B).
        labels (ndarray): Vérité terrain (H, W).
        class_names (list): Liste des noms de classes (facultatif).
        title (str): Titre du graphique.
    """
    data_2d = data
    labels_1d = labels.flatten()
    classes = np.unique(labels_1d)
    classes = classes[classes > 0]  # Exclure le fond (classe 0)

    plt.figure(figsize=(10, 6))
    for c in classes:
        idx = labels_1d == c
        mean_spectrum = data_2d[idx].mean(axis=0)
        name = class_names[c-1] if class_names else f"Class {c}"
        plt.plot(mean_spectrum, label=name)

    plt.title(title)
    plt.xlabel("Spectral Band")
    plt.ylabel("Reflectance")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def plot_band_correlation_matrix(data, title="Band Correlation Matrix"):
    """
    Display the correlation matrix between spectral bands.

    Parameters:
        data (ndarray): HSI (height* width, bands).
        title (str): Title of the plot.
    """
    df = pd.DataFrame(data)
    corr = df.corr()
    plt.figure(figsize=(8, 8))
    sns.heatmap(corr, cmap='coolwarm', square=True, cbar_kws={"shrink": 0.7})
    plt.title(title)
    plt.xlabel("Band")
    plt.ylabel("Band")
    plt.tight_layout()
    plt.show()

def plot_OA_surface(param1_grid, param1_name, param2_grid, param2_name,  OA_matrix, alpha, save_path=None):
    """
    Plots a 3D surface of Overall Accuracy (OA) as a function of hyperparameters gamma and lambda.

    Parameters:
    - param1__grid : array-like, values of gamma_ (X-axis)
    - param2_grid : array-like, values of lambda (Y-axis)
    - param1_name : name of the first parameter
    - parame2_name : name of the second paramter
    - OA_matrix : 2D array, matrix of OA scores (shape: len(param2_grid) x len(param1_grid))
    - save_path : str or None, path to save the figure (optional)
    """

    # Create grid for surface plot
    X, Y = np.meshgrid(param1_grid, param2_grid)
    z_min = OA_matrix.min() - 0.01

    # Find the best lambdas for each gamma and the corresponding best OA
    best_lambda_per_gamma_idx = np.argmax(OA_matrix, axis=0)  # axis=0 because lambda is along rows
    best_lambdas = param2_grid[best_lambda_per_gamma_idx]
    best_OAs = OA_matrix[best_lambda_per_gamma_idx, np.arange(len(param1_grid))]

    # Plot setup
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')

    # Surface plot without transpose
    surf = ax.plot_surface(X, Y, OA_matrix, cmap='viridis', alpha=0.6, edgecolor='k', linewidth=0.3)
    cbar = fig.colorbar(surf, shrink=0.5, aspect=12, pad=0.1)
    cbar.set_label('Overall Accuracy (OA)', fontsize=12, weight='bold')
    cbar.ax.tick_params(labelsize=10)

    # Line showing best lambdas per gamma(in red)
    ax.plot(param1_grid, best_lambdas, best_OAs + 0.001, color='red', marker="*", linestyle='--',
            linewidth=1.5, alpha=1.0, label='Best per gamma', zorder=10)

    # Projection of the red line on the base plane
    ax.plot(param1_grid, best_lambdas, [z_min] * len(param1_grid), color='darkred',
            linestyle=':', linewidth=1.5, alpha=0.6)

    # Global max point (best OA overall)
    best_idx = np.unravel_index(np.argmax(OA_matrix), OA_matrix.shape)
    best_gamma = param1_grid[best_idx[1]]  # gamma: column
    best_lambda = param2_grid[best_idx[0]]  # lambda: row
    best_OA = OA_matrix[best_idx]

    ax.scatter(best_gamma, best_lambda, best_OA,
               color='orange', s=120, edgecolor='black', linewidth=1.2, label='Global Best', zorder=10)

    # Projection of the global max point on the base plane
    ax.scatter(best_gamma, best_lambda, z_min,
               color='navy', s=80, marker='x', label='Global Best (projection)', zorder=5)

    # Labels and style
    if alpha==True:
        ax.set_xlabel(r'$\alpha$', labelpad=10, fontsize=12, fontweight='bold')
    else:
        ax.set_xlabel(param2_name, labelpad=10, fontsize=12, fontweight='bold')
        
    
    ax.set_ylabel(param2_name, labelpad=10, fontsize=12, fontweight='bold')
    ax.set_zlabel(r'$\mathrm{mean\ OA}$', labelpad=10, fontsize=12, fontweight='bold')
    ax.set_title('OA Surface with Projections and Best Values', fontsize=16, fontweight='bold', pad=20)
    ax.tick_params(axis='both', which='major', labelsize=10)

    # Grid styling
    ax.xaxis._axinfo['grid'].update(color='gray', linestyle='--', linewidth=0.5)
    ax.yaxis._axinfo['grid'].update(color='gray', linestyle='--', linewidth=0.5)
    ax.zaxis._axinfo['grid'].update(color='lightgray', linestyle=':', linewidth=0.5)
    ax.set_xticks(param1_grid)
    ax.set_xticklabels([f'{a:.1f}' for a in param1_grid])
    ax.set_yticks(param2_grid)
    ax.set_xticklabels([f'{l:.1f}' for l in param2_grid])

    ax.legend(loc='upper left', fontsize=11)

    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()

def plot_graph_adj(W, title="Graph Visualization", max_edges=500, selected_nodes=None, node_size=80):
    """
    Visualize a graph from an adjacency matrix W using NetworkX and highlight selected nodes.

    Parameters:
        W (ndarray): Adjacency matrix (symmetric).
        title (str): Title of the graph.
        max_edges (int): Maximum number of edges to display.
        selected_nodes (list or array): Indices of selected nodes to highlight and annotate.
        node_size (int): Size of the nodes.
    """
    G = nx.from_numpy_array(W)

    # Select some edges
    if G.number_of_edges() > max_edges:
        edges_sorted = sorted(G.edges(data=True), key=lambda x: x[2]['weight'], reverse=True)
        edges_to_keep = edges_sorted[:max_edges]
        G_filtered = nx.Graph()
        G_filtered.add_nodes_from(G.nodes)
        G_filtered.add_edges_from([(u, v, d) for u, v, d in edges_to_keep])
        G = G_filtered

    pos = nx.spring_layout(G, seed=42, k=0.3)
    weights = [G[u][v]['weight'] for u, v in G.edges()]
    weights_scaled = [0.3 + 2.5 * (w - min(weights)) / (max(weights) - min(weights) + 1e-6) for w in weights]

    # node color
    if selected_nodes is not None:
        selected_nodes_set = set(selected_nodes)
        node_colors = ['red' if i in selected_nodes_set else 'green' for i in G.nodes()]
    else:
        node_colors = 'green'

    fig, ax = plt.subplots(figsize=(10, 8))

    nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=node_size, alpha=0.9, ax=ax)
    nx.draw_networkx_edges(G, pos, width=weights_scaled, edge_color=weights, edge_cmap=plt.cm.viridis,
                           edge_vmin=min(weights), edge_vmax=max(weights), alpha=0.75, ax=ax)

    # labels of selected nodes
    if selected_nodes is not None:
        labels = {i: str(i) for i in selected_nodes if i in G.nodes}
        nx.draw_networkx_labels(G, pos, labels=labels, font_size=10, font_color='black', font_weight='bold', ax=ax)

    # Colorbar
    sm = plt.cm.ScalarMappable(cmap=plt.cm.viridis, norm=plt.Normalize(vmin=min(weights), vmax=max(weights)))
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax)
    cbar.set_label("Edge Weight", fontsize=12)

    ax.set_title(title, fontsize=16)
    ax.axis('off')
    plt.tight_layout()
    plt.show()
