import time

import matplotlib.pyplot as plt
import numpy as np

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import cross_validate
from sklearn.svm import LinearSVC
from sklearn.svm import SVC

from data_loader import load_data
from ..svm import SubgradientSVMClassifier


def evaluate_estimator(estimator, X, y):
    cv_results = cross_validate(estimator, X, y, cv=5, scoring="accuracy", return_train_score=True)
    return np.mean(cv_results["fit_time"]), np.mean(cv_results["train_score"]), np.mean(cv_results["test_score"])


def create_measures_plot(title, x_label, y_label):
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)
    plt.grid()


def adult_test():
    measures_num = 30
    step = 1000

    svm_results = []
    svc_results = []
    lsvc_results = []

    X, y = load_data("adult")

    svm = SubgradientSVMClassifier(batch_size=100, iterations=1000, regularizer=1e-2, step_size_rule="diminishing",
                                   alpha=0.001)
    svc = SVC()
    lsvc = LinearSVC(max_iter=1000, C=1e-2)

    sample_sizes = [step * (i + 1) for i in range(measures_num)]

    for n in sample_sizes:
        print(n)
        svm_results.append(evaluate_estimator(svm, X[:n], y[:n]))
        print(svm_results[-1])
        svc_results.append(evaluate_estimator(svc, X[:n], y[:n]))
        print(svc_results[-1])
        lsvc_results.append(evaluate_estimator(lsvc, X[:n], y[:n]))
        print(lsvc_results[-1])

    svm_results = np.array(svm_results).T
    svc_results = np.array(svc_results).T
    lsvc_results = np.array(lsvc_results).T

    statistics = ["mean training time (s)", "mean accuracy", "mean validation accuracy"]
    for i in range(len(statistics)):
        create_measures_plot("adult dataset", "number of samples", statistics[i])
        if i != 0:
            plt.ylim([0, 1])
        plt.plot(sample_sizes, svm_results[i], color="red", label="Subgradient SVM")
        plt.plot(sample_sizes, svc_results[i], color="green", label="SVC")
        plt.plot(sample_sizes, lsvc_results[i], color="blue", label="Linear SVC")
        plt.legend()
        plt.savefig("plots/adult-statistics-" + str(i) + ".png")
        plt.show()


def breast_cancer_test():
    X, y = load_breast_cancer(return_X_y=True)

    svm = SubgradientSVMClassifier(iterations=50, regularizer=1e-5)
    start = time.time()
    svm.fit(X, y)
    end = time.time()
    accuracy = svm.score(X, y)
    print("Subgradient SVM: " + str(end - start) + "s; " + str(accuracy) + "%")

    svc = SVC()
    start = time.time()
    svc.fit(X, y)
    end = time.time()
    accuracy = svc.score(X, y)
    print("SVC: " + str(end - start) + "s; " + str(accuracy) + "%")

    lsvc = LinearSVC(max_iter=1000, C=1e-5)
    start = time.time()
    lsvc.fit(X, y)
    end = time.time()
    accuracy = lsvc.score(X, y)
    print("Linear SVC: " + str(end - start) + "s; " + str(accuracy) + "%")


if __name__ == "__main__":
    adult_test()
    # breast_cancer_test()