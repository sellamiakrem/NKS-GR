# -*- coding: utf-8 -*-
"""
@author: Akrem Sellami
"""
import numpy as np
from sklearn.metrics.pairwise import rbf_kernel
from numpy.linalg import pinv
def compute_laplacian_pixel_nystrom(X, m_ratio=0.1, gamma_rbf=1e-1, seed=42):
    """
    Approximate Laplacian using the Nyström method with adaptive sampling.
    
    Parameters:
        X (ndarray): Hyperspectral data.
        m_ratio (float): Proportion of data points to sample (0 < m_ratio < 1).
        gamma_rbf (float): RBF kernel width.
        seed (int): Random seed.
    
    Returns:
        L_pixel (ndarray): Approximate Laplacian matrix.
    """
    np.random.seed(seed)
    m = int(m_ratio * X.shape[0])
    sample_idx = np.random.choice(X.shape[0], m, replace=False)

    C = rbf_kernel(X, X[sample_idx], gamma=gamma_rbf)
    W = rbf_kernel(X[sample_idx], gamma=gamma_rbf)
    W_inv = pinv(W)
    K_approx = C @ W_inv @ C.T
    D_approx = np.diag(K_approx.sum(axis=1))
    return D_approx - K_approx