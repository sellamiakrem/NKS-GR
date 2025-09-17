# -*- coding: utf-8 -*-  
"""
Main script for hyperspectral image band selection and classification.

This script loads a hyperspectral dataset, performs visualization (ground truth,
class distributions, spectral signatures, correlation matrix), and runs grid search
experiments to optimize regularization parameters for semi-supervised learning and classification.

Author: Akrem Sellami
Date: 14/09/2025
Usage:
    python main.py

Dependencies:
    - numpy
    - matplotlib
    - seaborn
    - scikit-learn
    - custom modules: data.load_hyperspectral_data, utils.grid_search_*, utils.plots
"""
import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Ensure relative imports work when running the script directly
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import custom modules for data loading, band selection, Laplacian computation, and model evaluation
from data.load_hyperspectral_data import load_data
from utils.grid_search_gamma_lambda import grid_search_gamma_lambda
from utils.grid_search_alpha_lambda import grid_search_alpha_lambda
from utils.stratified_semi_supervised_split import stratified_semi_supervised_split
from utils.compute_band_importance import compute_band_importance  
from graphs.compute_laplacian_band import compute_laplacian_band
from graphs.compute_laplacian_pixel_nystrom import compute_laplacian_pixel_nystrom
from models.solve_beta import solve_beta
from models.svm_model import evaluate_svm_model
from models.knn_model import evaluate_knn_model
from utils.plots import (
    plot_ground_truth_map,
    plot_class_distribution,
    plot_average_spectral_signatures,
    plot_band_correlation_matrix,
    plot_graph_adj
)

if __name__ == '__main__':
    # ====== Load dataset ======
    # Path to the hyperspectral dataset
    path = r"C:/Users/283902/Downloads/"
    # Load hyperspectral data:
    # X : spectral features (H x W x bands)
    # y : class labels for each pixel
    # class_names : full class names
    # abrev : abbreviated class names
    # labels : ground-truth map
    X, y, class_names, abrev, labels = load_data(path, "botswana", rgb=True, r_idx=40, g_idx=30, b_idx=20)

    # ====== Data Visualization ======
    plot_ground_truth_map(labels, class_names)          # Visualize ground-truth map
    plot_class_distribution(y, abrev)                  # Show class distribution
    plot_average_spectral_signatures(X, y)             # Plot average spectral signatures for each class
    plot_band_correlation_matrix(X)                    # Display band correlation matrix

    # ====== Experimental Parameters ======
    gamma_grid = np.arange(0, 1.1, 0.1)   # Candidate gamma values for regularization
    lambda_grid = np.arange(0, 1.1, 0.1)  # Candidate lambda values for regularization
    alpha_grid = np.arange(0, 1.1, 0.1)   # Candidate alpha values (trade-off between Laplacians)
    
    k = 30                # Number of spectral bands to select
    n_repeats = 5         # Number of repetitions for robust evaluation
    penalty = 'log_sum'   # Penalty function used in Beta solver
    labeled_ratio = 0.1   # Fraction of labeled samples
    unlabeled_ratio = 0.2 # Fraction of unlabeled samples
    n_neighbors = 5       # Number of neighbors for graph Laplacian
    m_ratio = 0.01        # Ratio for Nyström approximation
    do_grid_search = True # Flag to enable/disable hyperparameter search

    # ====== Grid Search: Hyperparameter Optimization ======
    if do_grid_search:
        # 1) Grid search over (gamma, lambda)
        mean_OA_matrix, _, _, best_gamma, best_lambda = grid_search_gamma_lambda(
            X, y, k, n_repeats, gamma_grid, lambda_grid, penalty,
            labeled_ratio, unlabeled_ratio, n_neighbors, m_ratio,
            plot_oa=True
        )
        print(f"Best gamma={best_gamma:.2f}, best lambda={best_lambda:.2f}")

        # 2) Grid search over (alpha, lambda) with gamma fixed
        mean_OA_matrix, _, _, best_alpha, best_lambda = grid_search_alpha_lambda(
            X, y, k, n_repeats, alpha_grid, lambda_grid, best_gamma, penalty,
            labeled_ratio, unlabeled_ratio, n_neighbors, m_ratio,
            plot_oa=True
        )
        print(f"Best alpha={best_alpha:.2f}, best lambda={best_lambda:.2f}") 

    # ====== Initialize Variables ======
    alpha = best_alpha       # Trade-off between band and pixel Laplacians
    lambda_reg = best_lambda # Regularization coefficient
    gamma_reg = best_gamma   # Gamma used in Beta solver
    n_bands = X.shape[1]     # Total number of spectral bands
    band_selection_counter = np.zeros(n_bands, dtype=int) # Count frequency of band selection
    top_k_selections = []    # Track top-k bands for stability analysis
    all_OA, all_kappa, all_AA = [], [], []  # SVM performance metrics
    all_OA_knn, all_kappa_knn, all_AA_knn = [], [], []  # k-NN performance metrics

    # ====== Repeat Experiments ======
    for seed in range(n_repeats):
        print(f"\n==== Repeat {seed+1}/{n_repeats} ====")
        np.random.seed(seed)

        # --- Step 1: Split data (stratified semi-supervised) ---
        labeled_indices, unlabeled_indices, test_indices = stratified_semi_supervised_split(
            y, labeled_ratio, unlabeled_ratio, random_state=seed
        )
        train_indices = np.concatenate([labeled_indices, unlabeled_indices])
        X_train = X[train_indices]
        y_train = y[train_indices]
        Y_labeled = np.eye(np.max(y) + 1)[y[labeled_indices]] # One-hot encoding
        labeled_indices_in_train = np.array([np.where(train_indices == idx)[0][0] for idx in labeled_indices])

        # --- Step 2: Compute Laplacian matrices ---
        L_band, W_band = compute_laplacian_band(X_train, n_neighbors=n_neighbors)
        L_pixel = compute_laplacian_pixel_nystrom(X_train, m_ratio)

        # --- Step 3: Solve Beta coefficients ---
        Beta, _ = solve_beta(
            X_train, Y_labeled, L_pixel, L_band,
            alpha, lambda_reg, gamma_reg,
            labeled_indices=labeled_indices_in_train,
            penalty='log_sum'
        )

        # --- Step 4: Band importance and feature selection ---
        importance = compute_band_importance(Beta)
        top_k = np.argsort(importance)[-k:]  # Select top-k bands
        X_train_selected = X_train[:, top_k]
        X_test_selected = X[test_indices][:, top_k]

        # --- Step 5: Evaluate classifiers ---
        # SVM
        acc_svm, kappa_svm, aa_svm, _ = evaluate_svm_model(X_train_selected, y_train, X_test_selected, y[test_indices])
        all_OA.append(acc_svm); all_AA.append(aa_svm); all_kappa.append(kappa_svm)
        print(f"Accuracy (SVM): {acc_svm:.4f}, AA: {aa_svm:.4f}, Kappa: {kappa_svm:.4f}")
        # Count band selections
        band_selection_counter[top_k] += 1
        top_k_selections.append(set(top_k))

        # k-NN
        acc_knn, kappa_knn, aa_knn, _ = evaluate_knn_model(X_train_selected, y_train, X_test_selected, y[test_indices])
        all_OA_knn.append(acc_knn); all_AA_knn.append(aa_knn); all_kappa_knn.append(kappa_knn)
        print(f"Accuracy (k-NN): {acc_knn:.4f}, AA: {aa_knn:.4f}, Kappa: {kappa_knn:.4f}")

    # ====== Summary of Results ======
    print(f"\nSummary (SVM) over {n_repeats} runs: OA={np.mean(all_OA):.2%} ± {np.std(all_OA):.2%}, AA={np.mean(all_AA):.2%} ± {np.std(all_AA):.2%}, Kappa={np.mean(all_kappa):.4f} ± {np.std(all_kappa):.4f}")
    print(f"Summary (k-NN) over {n_repeats} runs: OA={np.mean(all_OA_knn):.2%} ± {np.std(all_OA_knn):.2%}, AA={np.mean(all_AA_knn):.2%} ± {np.std(all_AA_knn):.2%}, Kappa={np.mean(all_kappa_knn):.4f} ± {np.std(all_kappa_knn):.4f}")

    # ====== Band Stability Analysis ======
    stable_bands = np.where(band_selection_counter >= 3)[0]
    print(f"Number of stable bands (selected ≥3 times): {len(stable_bands)}")
    plot_graph_adj(W_band, title="Band Similarity Graph (Stable Bands ≥3)", selected_nodes=stable_bands)

    # Boxplot of SVM accuracies over repetitions
    plt.boxplot(all_OA)
    plt.title("SVM Accuracy Distribution over Runs")
    plt.ylabel("Accuracy")
    plt.show()

    # Most frequently selected top-k bands
    stable_top_k = np.argsort(band_selection_counter)[-k:]
    print("Number of frequently selected bands after 5 runs:", len(stable_top_k))
    plot_graph_adj(W_band, title="Band Similarity Graph (Top-k Frequent Bands)", selected_nodes=stable_top_k)

    # Jaccard similarity between top-k selections for stability assessment
    jaccard_matrix = np.zeros((n_repeats, n_repeats))
    for i in range(n_repeats):
        for j in range(n_repeats):
            inter = len(top_k_selections[i].intersection(top_k_selections[j]))
            union = len(top_k_selections[i].union(top_k_selections[j]))
            jaccard_matrix[i, j] = inter / union if union != 0 else 0.0

    plt.figure(figsize=(6, 5))
    sns.heatmap(jaccard_matrix, annot=True, fmt=".2f", cmap='viridis',
                xticklabels=[f"Run {i+1}" for i in range(n_repeats)],
                yticklabels=[f"Run {i+1}" for i in range(n_repeats)])
    plt.title(f"Jaccard Similarity Between Top-{k} Bands ({n_repeats} runs)")
    plt.tight_layout()
    plt.show()

    # Global stability score
    stability_score = len(stable_bands) / k
    print(f"Stability score (bands ≥3 times): {stability_score:.2f}")

    

