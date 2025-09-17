# -*- coding: utf-8 -*-
"""
Created on Tue Sep 16 11:46:09 2025

@author: 283902
"""



import numpy as np

from utils.run_grid_search_gamma_lambda import run_grid_search_gamma_lambda
from utils.plots import plot_OA_surface


def grid_search_gamma_lambda(X, y, k, n_repeats, gamma_grid, lambda_grid,  penalty,  labeled_ratio, unlabeled_ratio, n_neighbors, m_ratio, plot_oa=True):

    all_OAs = []
    all_OA_matrices = []
    all_best_gamma = []
    all_best_lambdas = []

    for seed in range(n_repeats):
        print(f"\n==== Repeat {seed+1}/{n_repeats} ====")

        OA_matrix, best_gamma, best_lambda, best_OA = run_grid_search_gamma_lambda(X, y,gamma_grid,
            lambda_grid, k, penalty,labeled_ratio, unlabeled_ratio,   n_neighbors, m_ratio,
            random_state=seed )
        all_OAs.append(best_OA)
        all_best_gamma.append(best_gamma)
        all_best_lambdas.append(best_lambda)
        all_OA_matrices.append(OA_matrix)

    # Pointwise average
    mean_OA_matrix = np.mean(np.array(all_OA_matrices), axis=0)
    mean_OA = np.mean(all_OAs)
    std_OA = np.std(all_OAs)
    mean_gamma = np.mean(all_best_gamma)
    mean_lambda = np.mean(all_best_lambdas)

    print("\n==== Overall results over 5 repeats ====")
    print(f"Mean OA: {mean_OA:.4f}")
    print(f"Std OA : {std_OA:.4f}")
    print(f"Mean Gamma: {mean_gamma:.2f}")
    print(f"Mean Lambda: {mean_lambda:.2f}")
    best_idx = np.unravel_index(np.argmax(mean_OA_matrix), mean_OA_matrix.shape)
    best_gamma_global = gamma_grid[best_idx[1]]
    best_lambda_global = lambda_grid[best_idx[0]]
    print(f"Best Gamma Global: {best_gamma_global:.2f}")
    print(f"Best Lambda Global: {best_lambda_global:.2f}")
    if plot_oa==True:
        plot_OA_surface(gamma_grid, '$\gamma$', lambda_grid, '$\lambda$',  mean_OA_matrix, alpha=False, save_path=None)
        
    return mean_OA_matrix, gamma_grid, lambda_grid, best_gamma_global, best_lambda_global


    