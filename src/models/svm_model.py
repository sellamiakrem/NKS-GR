# -*- coding: utf-8 -*-
"""

@author: Akrem Sellami
"""
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, f1_score, classification_report, cohen_kappa_score, recall_score

def evaluate_svm_model(X_train_selected, y_train, X_test_selected, y_test, C=100):
    """
    Train and evaluate an SVM classifier with RBF kernel.

    Parameters:
        X_train_selected (ndarray): Feature matrix for training (after feature selection).
        y_train (ndarray): Training class labels.
        X_test_selected (ndarray): Feature matrix for testing (after feature selection).
        y_test (ndarray): Testing class labels.
        C (float): Regularization parameter for SVM (default=100).

    Returns:
        acc (float): Overall classification accuracy.
        kappa (float): Cohen's Kappa score.
        aa (float): Average Accuracy (mean recall per class).
        report (str): Detailed classification report.
    """
    clf = SVC(kernel='rbf', C=C, gamma='scale')
    clf.fit(X_train_selected, y_train)
    y_pred = clf.predict(X_test_selected)

    acc = accuracy_score(y_test, y_pred)
    kappa = cohen_kappa_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    aa = recall_score(y_test, y_pred, average='macro')

    return acc, kappa, aa, report
