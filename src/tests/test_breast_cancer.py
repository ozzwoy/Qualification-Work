import numpy as np
import pytest

from sklearn.datasets import load_breast_cancer
from src import SubgradientSVMClassifier


@pytest.fixture
def data():
    return load_breast_cancer(return_X_y=True)


def test_template_classifier(data):
    X, y = data
    svm = SubgradientSVMClassifier(alpha=0.01)
    svm.fit(X, y)
    assert hasattr(svm, 'classes_')
    assert hasattr(svm, 'coef_')
    assert hasattr(svm, 'history_')

    y_pred = svm.predict(X)
    assert y_pred.shape == (X.shape[0],)
    assert np.sum(y_pred != y) / len(y) < 0.2
