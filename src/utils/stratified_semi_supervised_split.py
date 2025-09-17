# -*- coding: utf-8 -*-
"""

@author: Akrem Sellami
"""
import numpy as np

    
def stratified_semi_supervised_split(y, labeled_ratio, unlabeled_ratio, min_labeled=5, random_state=42):
    np.random.seed(random_state)
    classes = np.unique(y)
    labeled_indices = []
    unlabeled_indices = []
    test_indices = []

    for c in classes:
        class_idx = np.where(y == c)[0]
        n_samples = len(class_idx)

        n_train_total = int(np.round(unlabeled_ratio * n_samples))  # train total (label + unlabel)
        n_labeled = max(int(np.round(labeled_ratio * n_samples)), min_labeled)  # labeled minimal 5

        # Shuffle indices
        np.random.shuffle(class_idx)

        train_total_idx = class_idx[:n_train_total]
        test_idx = class_idx[n_train_total:]

        labeled_idx = train_total_idx[:n_labeled]
        unlabeled_idx = train_total_idx[n_labeled:]

        labeled_indices.extend(labeled_idx)
        unlabeled_indices.extend(unlabeled_idx)
        test_indices.extend(test_idx)

    return np.array(labeled_indices), np.array(unlabeled_indices), np.array(test_indices)