import random

import numpy as np
from sklearn.utils import check_random_state


class SubgradientDescent:

    def __init__(self,
                 loss,
                 iterations,
                 batch_size,
                 reularizer,
                 step_size_rule,
                 alpha,
                 beta):

        self.loss = loss
        self.iterations = iterations
        self.batch_size = batch_size
        self.regularizer = reularizer
        self.step_size_rule = step_size_rule
        self.alpha = alpha
        self.beta = beta

        self.history = []

    def execute(self, X, y):
        joined = []
        current = np.zeros(len(X[0]))
        f_min = float("inf")
        self.history.append(current)

        for i in range(self.iterations):
            if self.batch_size is None:
                subgradient = self.loss.subgradient_at(X, y, current, self.regularizer)
            elif self.batch_size == 1:
                j = random.randint(0, len(X) - 1)
                subgradient = self.loss.subgradient_at([X[j]], [y[j]], current, self.regularizer)
            else:
                if i == 0:
                    joined = np.append(X, np.array([y]).transpose(), axis=1).tolist()
                batch = self.__get_batch(joined)
                subgradient = self.loss.subgradient_at(batch[:, :-1], np.array(batch[:, -1]).flatten(), current,
                                                       self.regularizer)

            if self.step_size_rule.value == "polyak":
                f = self.loss.value_at(X, y, current, self.regularizer)
                if f < f_min:
                    f_min = f

            step = self.step_size_rule.next_step(self.alpha, self.beta, i, X, y, f_min, current, self.loss,
                                                 self.regularizer)
            current = current - step * subgradient
            self.history.append(current)

        return current

    def __get_batch(self, data, random_state=0):
        return np.array(random.sample(data, self.batch_size))
        # random_state = check_random_state(random_state)
        # return np.array(random_state.sample(data, self.batch_size))

    def get_last_search_history(self):
        return self.history
