import numpy as np
import pytest

from sklearn.datasets import load_breast_cancer
from src import SubgradientSVMClassifier


@pytest.fixture
def data():
    return load_breast_cancer(return_X_y=True)


def test_linear_svm(data):
    X, y = data
    svm = SubgradientSVMClassifier()
    svm.fit(X, y)
    # Check fit variables
    assert hasattr(svm, 'classes_')
    assert hasattr(svm, 'coef_')
    assert hasattr(svm, 'history_')

    y_pred = svm.predict(X)
    assert y_pred.shape == (X.shape[0],)
    # Assume that the accuracy is higher than 85% (actually ~89%)
    assert svm.score(X, y) > 0.85


def test_kernelized_svm(data):
    X, y = data
    svm = SubgradientSVMClassifier(kernel="rbf")
    svm.fit(X, y)
    y_pred = svm.predict(X)

    assert y_pred.shape == (X.shape[0],)
    # Assume that the accuracy is higher than 95% (actually ~98%)
    assert svm.score(X, y) > 0.95
