import numpy as np
import pytest

from sklearn.datasets import load_iris
from sklearn.multiclass import OneVsRestClassifier
from src import SubgradientSVMClassifier


@pytest.fixture
def data():
    return load_iris(return_X_y=True)


def test_svm_linear_multiclass(data):
    X, y = data
    svm = SubgradientSVMClassifier(iterations=5000, regularizer=1, alpha=0.01)
    ovr = OneVsRestClassifier(svm)
    ovr.fit(X, y)
    # Assume that the accuracy is higher than 75% (actually ~80%)
    assert ovr.score(X, y) > 0.75


def test_svm_kernelized_multiclass(data):
    X, y = data
    svm = SubgradientSVMClassifier(iterations=1000, regularizer=1, alpha=0.01, kernel="rbf", gamma=0.1)
    ovr = OneVsRestClassifier(svm)
    ovr.fit(X, y)
    # Assume that the accuracy is higher than 90% (actually ~91%)
    assert ovr.score(X, y) > 0.9
