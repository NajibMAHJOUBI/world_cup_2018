# -*- coding: utf-8 -*-

"""
 Get the classifiers models available
"""


def get_classification_models(to_remove=None):
    classifiers = ["logistic_regression", "k_neighbors", "gaussian_classifier", "decision_tree", "random_forest",
                   "mlp_classifier", "ada_boost", "gaussian", "quadratic", "svc", "sgd"]
    for classifier in to_remove:
        classifiers.remove(classifier)
    return classifiers
