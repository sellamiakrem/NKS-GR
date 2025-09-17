# -*- coding: utf-8 -*-
"""
@author: Akrem Sellami
"""
import numpy as np
from utils.stratified_semi_supervised_split import stratified_semi_supervised_split
from models.solve_beta import solve_beta
from utils.compute_band_importance import compute_band_importance  
from graphs.compute_laplacian_band import compute_laplacian_band
from graphs.compute_laplacian_pixel_nystrom import compute_laplacian_pixel_nystrom
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
def run_grid_search_gamma_lambda(X, y, gamma_grid, lambda_grid, k, penalty, labeled_ratio, unlabeled_ratio,   n_neighbors, m_ratio, random_state=None):    
    OA_matrix = np.zeros((len(lambda_grid), len(gamma_grid)))
    best_OA = -np.inf
    best_gamma = None
    best_lambda = None

    for i, lam in enumerate(lambda_grid):
        for j, al in enumerate(gamma_grid):
            n_classes = np.max(y) + 1

            labeled_indices, unlabeled_indices, test_indices = stratified_semi_supervised_split(
                y, labeled_ratio, unlabeled_ratio,random_state=random_state
            )
        
            train_indices = np.concatenate([labeled_indices, unlabeled_indices])
            
            X_train = X[train_indices]
            y_train = y[train_indices]
            Y_labeled = np.eye(n_classes)[y[labeled_indices]]
            L_band, W_band = compute_laplacian_band(X_train, n_neighbors=n_neighbors)
            L_pixel = compute_laplacian_pixel_nystrom(X_train, m_ratio=m_ratio)
            
            labeled_indices_in_train = np.array([np.where(train_indices == idx)[0][0] for idx in labeled_indices])
            
            Beta, convergence = solve_beta(
                X_train, Y_labeled, L_pixel, L_band,
                alpha=0.5, lambda_reg=lam, gamma=al,
                labeled_indices=labeled_indices_in_train,
                penalty=penalty
            )

            importance = compute_band_importance(Beta)
            top_band_indices = np.argsort(importance)[-k:]
            
            X_train_selected = X_train[:, top_band_indices]
            X_test_selected = X[test_indices][:, top_band_indices]
            
            clf = SVC(kernel='rbf', C=100, gamma='scale')
            clf.fit(X_train_selected, y_train)
            y_pred = clf.predict(X_test_selected)
            acc = accuracy_score(y_pred, y[test_indices])
            OA_matrix[i, j] = acc
            print(f"lambda={lam:.2f}, gamma={al:.2f} => OA={acc:.4f}")

            if acc > best_OA:
                best_OA = acc
                best_gamma = al
                best_lambda = lam
    return OA_matrix, best_gamma, best_lambda, best_OA