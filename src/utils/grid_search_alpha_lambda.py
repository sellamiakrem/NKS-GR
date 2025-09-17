# -*- coding: utf-8 -*-
"""
Created on Wed Sep 17 08:41:48 2025

@author: 283902
"""

import numpy as np

from utils.run_grid_search_alpha_lambda import run_grid_search_alpha_lambda
from utils.plots import plot_OA_surface


def grid_search_alpha_lambda(X, y, k, n_repeats, alpha_grid, lambda_grid, best_gamma, penalty,  labeled_ratio, unlabeled_ratio, n_neighbors, m_ratio, plot_oa=True):

    all_OAs = []
    all_OA_matrices = []
    all_best_alpha = []
    all_best_lambdas = []

    for seed in range(n_repeats):
        print(f"\n==== Repeat {seed+1}/{n_repeats} ====")

        OA_matrix, best_alpha, best_lambda, best_OA = run_grid_search_alpha_lambda(X, y,alpha_grid,
            lambda_grid, best_gamma, k, penalty,labeled_ratio, unlabeled_ratio,   n_neighbors, m_ratio,
            random_state=seed )
        all_OAs.append(best_OA)
        all_best_alpha.append(best_alpha)
        all_best_lambdas.append(best_lambda)
        all_OA_matrices.append(OA_matrix)

    # Pointwise average
    mean_OA_matrix = np.mean(np.array(all_OA_matrices), axis=0)
    mean_OA = np.mean(all_OAs)
    std_OA = np.std(all_OAs)
    mean_alpha = np.mean(all_best_alpha)
    mean_lambda = np.mean(all_best_lambdas)

    print("\n==== Overall results over 5 repeats ====")
    print(f"Mean OA: {mean_OA:.4f}")
    print(f"Std OA : {std_OA:.4f}")
    print(f"Mean Alpha: {mean_alpha:.2f}")
    print(f"Mean Lambda: {mean_lambda:.2f}")
    best_idx = np.unravel_index(np.argmax(mean_OA_matrix), mean_OA_matrix.shape)
    best_alpha_global = alpha_grid[best_idx[1]]
    best_lambda_global = lambda_grid[best_idx[0]]
    print(f"Best Alpha Global: {best_alpha_global:.2f}")
    print(f"Best Lambda Global: {best_lambda_global:.2f}")
    if plot_oa==True:
        plot_OA_surface(alpha_grid, '$alpha$', lambda_grid, '$\lambda$',  mean_OA_matrix, alpha=True, save_path=None)
        
    return mean_OA_matrix, alpha_grid, lambda_grid, best_alpha_global, best_lambda_global
