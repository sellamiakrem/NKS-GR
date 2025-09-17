# -*- coding: utf-8 -*-
"""
@author: Akrem Sellami
"""
from sklearn.neighbors import kneighbors_graph
import numpy as np
def compute_laplacian_band(X, n_neighbors):
    """
    Compute the graph Laplacian matrix based on band-wise similarity.
    Also returns the adjacency matrix W_band for visualization.

    Parameters:
        X: Data.
        n_neighbors (int): Number of neighbors for graph construction.

    Returns:
        L_band (ndarray): Laplacian matrix for spectral bands.
        W_band (ndarray): Weighted adjacency matrix.
    """
    bands = X.T  # Each row is a band
    band_graph = kneighbors_graph(bands, n_neighbors=n_neighbors, mode='connectivity', include_self=False)
    W_band = 0.5 * (band_graph.toarray() + band_graph.toarray().T)  # symmetrize
    D_band = np.diag(W_band.sum(axis=1))
    L_band = D_band - W_band
    return L_band, W_band