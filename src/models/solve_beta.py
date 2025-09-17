# -*- coding: utf-8 -*-
"""
Created on Tue Sep 16 10:57:14 2025

@author: 283902
"""


from utils.penalty_functions import weights_l1, weights_l2, weights_log_sum, weights_l0_approx
import numpy as np

def solve_beta(X, Y_labeled, L_pixel, L_band, alpha, lambda_reg, gamma, 
               labeled_indices, penalty='log_sum', epsilon=1e-5, max_iter=50):
    n_features = X.shape[1]
    X_l = X[labeled_indices]
    Y_l = Y_labeled

    penalty_functions = {
        'l1': weights_l1,
        'l2': weights_l2,
        'log_sum': weights_log_sum,
        'l0_approx': weights_l0_approx
    }

    if penalty not in penalty_functions:
        raise ValueError(f"Penalty '{penalty}' is not implemented.")
    
    weight_func = penalty_functions[penalty]
    
    A_init = (X_l.T @ X_l +
              lambda_reg * ((1 - alpha) * L_band + alpha * (X.T @ L_pixel @ X)) +
              gamma * np.eye(n_features))
    
    Beta = np.linalg.solve(A_init, X_l.T @ Y_l)
    convergence = []

    for _ in range(max_iter):
        W_diag = weight_func(Beta, epsilon)
        A = (X_l.T @ X_l +
             lambda_reg * ((1 - alpha) * L_band + alpha * (X.T @ L_pixel @ X)) +
             gamma * W_diag)
        Beta_new = np.linalg.solve(A, X_l.T @ Y_l)

        diff = np.linalg.norm(Beta_new - Beta, ord='fro')
        convergence.append(diff)
        if diff < 1e-4:
            break
        Beta = Beta_new

    return Beta, convergence