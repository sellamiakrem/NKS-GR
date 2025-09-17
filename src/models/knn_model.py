# -*- coding: utf-8 -*-
"""
@author: Akrem Sellami
"""

from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, f1_score, classification_report, cohen_kappa_score, recall_score

def evaluate_knn_model(X_train_selected, y_train, X_test_selected, y_test, n_neighbors=5):
    """
    Evaluate classification performance using a k-NN classifier.

    Parameters:
        X_train_selected (ndarray): Feature matrix for training (after feature selection).
        y_train (ndarray): Training class labels.
        X_test_selected (ndarray): Feature matrix for testing (after feature selection).
        y_test (ndarray): Testing class labels.
        n_neighbors (int): Number of neighbors to use.

    Returns:
        acc (float): Overall classification accuracy.
        kappa (float): Cohen's Kappa score.
        aa (float): Average Accuracy (mean recall per class).
        report (str): Detailed classification report.
    """
    clf = KNeighborsClassifier(n_neighbors=n_neighbors)
    clf.fit(X_train_selected, y_train)
    y_pred = clf.predict(X_test_selected)
    acc = accuracy_score(y_test, y_pred)
    kappa = cohen_kappa_score(y_test, y_pred)
    aa = recall_score(y_test, y_pred, average='macro')
    report = classification_report(y_test, y_pred)

    return acc, kappa, aa, report