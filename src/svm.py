import math
from enum import Enum

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted

from src.subgradient_descent import SubgradientDescent


class LossFunction(Enum):
    HINGE = 0
    LOGISTIC = 1
    QUADRATIC = 2

    @staticmethod
    def by_name(name):
        if name == "hinge":
            return LossFunction.HINGE
        elif name == "logistic":
            return LossFunction.LOGISTIC
        elif name == "quadratic":
            return LossFunction.QUADRATIC
        else:
            raise ValueError("invalid loss function")

    def value_at(self, X, y, w, _lambda):
        size = len(y)
        regular_term = w.dot(w) * _lambda
        mean = 0

        if self == LossFunction.HINGE:
            for i in range(size):
                margin = X[i].dot(w) * y[i]
                if margin < 1:
                    mean += (1 - margin) / size
        elif self == LossFunction.LOGISTIC:
            for i in range(size):
                margin = X[i].dot(w) * y[i]
                mean += math.log(1 + math.exp(-margin)) / size
        elif self == LossFunction.QUADRATIC:
            for i in range(size):
                margin = X[i].dot(w) * y[i]
                mean += (1 - margin) ** 2 / size
        else:
            raise ValueError("invalid loss function")

        return regular_term + mean

    def subgradient_at(self, X, y, w, _lambda):
        size = len(y)
        regular_term = 2 * _lambda * w
        mean = 0

        if self == LossFunction.HINGE:
            for i in range(size):
                margin = X[i].dot(w) * y[i]
                if margin < 1:
                    mean += -X[i] * y[i] / size
        elif self == LossFunction.LOGISTIC:
            for i in range(size):
                margin = X[i].dot(w) * y[i]
                mean += -(y[i] / (1 + math.exp(margin))) * X[i] / size
        elif self == LossFunction.QUADRATIC:
            for i in range(size):
                margin = X[i].dot(w) * y[i]
                mean += -2 * (1 - margin) * y[i] * X[i] / size
        else:
            raise ValueError("invalid loss function")

        return regular_term + mean


class StepSizeRule(Enum):
    CONSTANT = 0
    DIMINISHING = 1
    POLYAK = 2
    BACKTRACKING = 3

    @staticmethod
    def by_name(name):
        if name == "constant":
            return StepSizeRule.CONSTANT
        elif name == "diminishing":
            return StepSizeRule.DIMINISHING
        elif name == "polyak":
            return StepSizeRule.POLYAK
        elif name == "backtracking":
            return StepSizeRule.BACKTRACKING
        else:
            raise ValueError("invalid step size rule")

    def next_step(self, alpha, beta, iteration, X, y, f_opt, w, loss, regularizer):
        if self == StepSizeRule.CONSTANT:
            return alpha
        elif self == StepSizeRule.DIMINISHING:
            return alpha / (iteration + 1)
        elif self == StepSizeRule.POLYAK:
            subgradient = loss.subgradient_at(X, y, w, regularizer)
            return (loss.value_at(X, y, w, regularizer) - f_opt + (alpha / iteration)) / (subgradient.dot(subgradient))
        elif self == StepSizeRule.BACKTRACKING:
            return backtracking_line_search(alpha, beta, X, y, w, loss, regularizer)


class SubgradientSVMClassifier(ClassifierMixin, BaseEstimator):

    def __init__(self,
                 loss=LossFunction.HINGE,
                 iterations=1000,
                 batch_size=None,
                 regularizer=1e-4,
                 step_size_rule=StepSizeRule.DIMINISHING,
                 alpha=0.1,
                 beta=0.1):

        self.loss = loss
        self.iterations = iterations
        self.batch_size = batch_size
        self.regularizer = regularizer
        self.step_size_rule = step_size_rule
        self.alpha = alpha
        self.beta = beta

    def fit(self, X, y):
        # Check that X and y have correct shape
        X, y = check_X_y(X, y)

        # Store the classes seen during fit, y stores the indices of the classes in range [0, n_classes)
        self.classes_, y = np.unique(y, return_inverse=True)

        if len(self.classes_) != 2:
            raise ValueError("use sklearn.multiclass for multiple classes")

        # The training algorithm requires the labels to be -1 and +1.
        y[y == 0] = -1

        descent = SubgradientDescent(self.loss, self.iterations, self.batch_size, self.regularizer,
                                     self.step_size_rule, self.alpha, self.beta)
        self.coef_ = descent.execute(X, y)
        self.history_ = descent.get_last_search_history()

        # Return the classifier
        return self

    def predict(self, X):
        # Check is fit had been called
        check_is_fitted(self, ['coef_', 'history_'])

        # # Check input
        # X = check_array(X)

        y = self.decision_function(X)
        y[y > 0] = self.classes_[1]
        y[y <= 0] = self.classes_[0]

        return y

    def score(self, X, y, sample_weight=None):
        # Check is fit had been called
        check_is_fitted(self, ['coef_', 'history_'])

        # Check that X and y have correct shape
        X, y = check_X_y(X, y)

        # Store the classes seen during fit, y stores the indices of the classes in range [0, n_classes)
        _, y = np.unique(y, return_inverse=True)

        # The training algorithm requires the labels to be -1 and +1.
        y[y == 0] = -1

        total = len(y)
        predicted = self.decision_function(X)
        predicted[predicted > 0] = 1
        predicted[predicted <= 0] = -1
        right = 0

        for i in range(total):
            margin = predicted[i] * y[i]
            if margin > 0:
                right += 1

        return right / total

    def decision_function(self, X):
        return X.dot(self.coef_)

    def _more_tags(self):
        return {'binary_only': True}


def backtracking_line_search(alpha, beta, x, y, w, loss, regularizer):
    t = 1
    f_x = loss.value_at(x, y, w, regularizer)
    d_x = loss.subgradient_at(x, y, w, regularizer)
    d_x_squared = d_x.dot(d_x)

    while loss.value_at(x, y, w - t * d_x) > f_x - t * alpha * d_x_squared:
        t = beta * t

    return t
