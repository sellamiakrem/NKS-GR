# -*- coding: utf-8 -*-
"""
@author: Akrem Sellami
"""
import numpy as np
def weights_l2(Beta, epsilon=1e-3):
    n = Beta.shape[0]
    return np.eye(n)

def weights_l1(Beta, epsilon=1e-3):
    row_norms = np.linalg.norm(Beta, ord=2, axis=1)
    w = 1.0 / (row_norms + epsilon)
    return np.diag(w)

def weights_l0_approx(Beta, epsilon=1e-3):
    row_norms = np.linalg.norm(Beta, axis=1)
    w = 1.0 / (row_norms**2 + epsilon)
    return np.diag(w)

def weights_log_sum(Beta, epsilon=1e-3):
    row_norms = np.linalg.norm(Beta, axis=1)
    w = 2 * row_norms / (row_norms**2 + epsilon)
    return np.diag(1.0 / (w + 1e-8))  