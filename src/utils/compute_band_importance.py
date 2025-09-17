# -*- coding: utf-8 -*-
"""
@author: Akrem Sellami
"""

import numpy as np
def compute_band_importance(Beta):
    """
    Compute the importance of each spectral band from Beta coefficients.

    Parameters:
        Beta (ndarray): Coefficient matrix of shape (n_bands, n_classes).

    Returns:
        importance (ndarray): Importance score for each band (1D array).
    """
    # Importance as the L2 norm of each row (band)
    importance = np.linalg.norm(Beta, axis=1)
    return importance