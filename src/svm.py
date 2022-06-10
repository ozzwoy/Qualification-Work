import math
from enum import Enum

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted, as_float_array

from src.subgradient_descent import SubgradientDescent


class LossFunction(Enum):
    HINGE = "hinge"
    LOGISTIC = "logistic"
    QUADRATIC = "quadratic"

    @staticmethod
    def by_name(name):
        try:
            return LossFunction(name)
        except ValueError:
            raise ValueError(name + " is not a valid loss function.")

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

        return regular_term + mean


class StepSizeRule(Enum):
    CONSTANT = "constant"
    DIMINISHING = "diminishing"
    POLYAK = "polyak"
    BACKTRACKING = "backtracking"

    @staticmethod
    def by_name(name):
        try:
            return StepSizeRule(name)
        except ValueError:
            raise ValueError(name + " is not a valid step size rule.")

    def next_step(self, alpha, beta, iteration, X, y, f_opt, w, loss, regularizer):
        if self == StepSizeRule.CONSTANT:
            return alpha
        elif self == StepSizeRule.DIMINISHING:
            return alpha / (iteration + 1)
        elif self == StepSizeRule.POLYAK:
            subgradient = loss.subgradient_at(X, y, w, regularizer)
            return (loss.value_at(X, y, w, regularizer) - f_opt + (alpha / (iteration + 1))) / \
                   (subgradient.dot(subgradient))
        elif self == StepSizeRule.BACKTRACKING:
            return backtracking_line_search(alpha, beta, X, y, w, loss, regularizer)


class Kernel(Enum):
    LINEAR="linear"
    QUADRATIC="quadratic"
    RBF="rbf"

    @staticmethod
    def by_name(name):
        try:
            return Kernel(name)
        except ValueError:
            raise ValueError(name + " is not a valid kernel.")

    def map(self, x1, x2, gamma):
        if self == Kernel.LINEAR:
            return x1.dot(x2)
        elif self == Kernel.QUADRATIC:
            return (x1.dot(x2)) ** 2
        elif self == Kernel.RBF:
            return math.exp(-gamma * (x1 - x2).dot(x1 - x2))


class SubgradientSVMClassifier(ClassifierMixin, BaseEstimator):

    def __init__(self,
                 loss="hinge",
                 iterations=1000,
                 batch_size=None,
                 regularizer=1e-4,
                 step_size_rule="diminishing",
                 alpha=0.1,
                 beta=0.1,
                 kernel="linear",
                 gamma=1e-3):

        self.loss = loss
        self.iterations = iterations
        self.batch_size = batch_size
        self.regularizer = regularizer
        self.step_size_rule = step_size_rule
        self.alpha = alpha
        self.beta = beta
        self.kernel = kernel
        self.gamma = gamma

    def fit(self, X, y):
        # Check that X and y have correct shape
        X, y = check_X_y(X, y)
        X = as_float_array(X)

        self.n_features_in_ = len(X[0])

        # Store the classes seen during fit, y stores the indices of the classes in range [0, n_classes)
        self.classes_, y = np.unique(y, return_inverse=True)

        if len(self.classes_) != 2:
            if len(self.classes_) == 1:
                raise ValueError("Classifier can't train when only one class is present.")
            else:
                raise ValueError("Unknown label type: " + str(self.classes_[2]) + ". "
                                 "Use sklearn.multiclass for multiple classes.")

        # The training algorithm requires the labels to be -1 and +1.
        y[y == 0] = -1

        # If kernelized, store the training data
        if self.kernel != Kernel.LINEAR:
            self.X_train_ = X
            self.y_train_ = y

        # Get loss function, step size rule and kernel objects by their names
        self.loss_ = LossFunction.by_name(self.loss)
        self.step_size_rule_ = StepSizeRule.by_name(self.step_size_rule)
        self.kernel_ = Kernel.by_name(self.kernel)

        descent = SubgradientDescent(self.loss_, self.iterations, self.batch_size, self.regularizer,
                                     self.step_size_rule_, self.alpha, self.beta, self.kernel_, self.gamma)
        self.coef_ = descent.execute(X, y)
        self.history_ = descent.get_last_search_history()

        # Return the classifier
        return self

    def predict(self, X):
        # Check is fit had been called
        check_is_fitted(self, ['coef_', 'history_'])

        # Check input
        X = as_float_array(check_array(X))

        y = self.decision_function(X)
        y = np.array([self.classes_[1] if el > 0 else self.classes_[0] for el in y])

        return y

    def predict_proba(self, X):
        # Check is fit had been called
        check_is_fitted(self, ['coef_', 'history_'])

        # Check input
        X = as_float_array(check_array(X))

        y = self.decision_function(X)
        y = np.exp(y) / (1.0 + np.exp(y))

        return np.vstack([1 - y, y]).T

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
        # Check is fit had been called
        check_is_fitted(self, ['coef_', 'history_'])

        # Check input
        X = as_float_array(check_array(X))

        if self.kernel_ == Kernel.LINEAR:
            return X.dot(self.coef_)
        else:
            result = []

            for x in X:
                decision = 0
                for i in range(len(self.y_train_)):
                    decision += self.coef_[i] * self.y_train_[i] * self.kernel_.map(self.X_train_[i], x, self.gamma)
                result.append(decision)

            return np.array(result)

    def _more_tags(self):
        return {'binary_only': True}


def backtracking_line_search(alpha, beta, X, y, w, loss, regularizer):
    t = 1
    f_x = loss.value_at(X, y, w, regularizer)
    d_x = loss.subgradient_at(X, y, w, regularizer)
    d_x_squared = d_x.dot(d_x)

    while loss.value_at(X, y, w - t * d_x) > f_x - t * alpha * d_x_squared:
        t = beta * t

    return t
