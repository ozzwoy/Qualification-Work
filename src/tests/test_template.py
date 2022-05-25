import pytest

from sklearn.datasets import load_iris
from src import SubgradientSVMClassifier


@pytest.fixture
def data():
    return load_iris(return_X_y=True)


def test_template_classifier(data):
    X, y = data
    svm = SubgradientSVMClassifier()
    assert svm.demo_param == 'demo'

    svm.fit(X, y)
    assert hasattr(svm, 'classes_')
    assert hasattr(svm, 'X_')
    assert hasattr(svm, 'y_')

    y_pred = svm.predict(X)
    assert y_pred.shape == (X.shape[0],)
