import pytest

from sklearn.utils.estimator_checks import check_estimator

from src import SubgradientSVMClassifier


@pytest.mark.parametrize(
    "estimator",
    [SubgradientSVMClassifier()]
)
def test_all_estimators(estimator):
    return check_estimator(estimator)
