import time

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap

from sklearn.datasets import load_breast_cancer, load_iris
from sklearn.model_selection import cross_validate
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import LinearSVC
from sklearn.svm import SVC

from data_loader import load_data
from src.svm import SubgradientSVMClassifier


def evaluate_estimator(estimator, X, y):
    cv_results = cross_validate(estimator, X, y, cv=5, scoring="accuracy", return_train_score=True)
    return np.mean(cv_results["fit_time"]), np.mean(cv_results["train_score"]), np.mean(cv_results["test_score"])


def create_measures_plot(title, x_label, y_label):
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)
    plt.grid()


def adult_plots():
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
        plt.savefig("plots/adult/adult-statistics-" + str(i) + ".png")
        plt.show()


def breast_cancer_demo():
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


def create_decision_boundary_plot(X, y, svm, title):
    colourmap = ListedColormap(("red", "green", "blue"))

    X1, X2 = np.meshgrid(np.arange(start=X[:, 0].min() - 1,
                                   stop=X[:, 0].max() + 1,
                                   step=0.01),
                         np.arange(start=X[:, 1].min() - 1,
                                   stop=X[:, 1].max() + 1,
                                   step=0.01))

    plt.contourf(X1, X2, svm.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape), alpha=0.65,
                 cmap=colourmap)
    plt.xlim(X1.min(), X1.max())
    plt.ylim(X2.min(), X2.max())
    for i, j in enumerate(np.unique(y)):
        plt.scatter(X[y == j, 0], X[y == j, 1], color=colourmap(i), label=j)

    plt.title(title)
    plt.xlabel("x1")
    plt.ylabel("x2")
    plt.legend()

    return plt.show()


def iris_plots():
    X, y = load_iris(return_X_y=True)

    X, y = load_data("custom")

    svm = SubgradientSVMClassifier(iterations=100, regularizer=0.01, alpha=0.1, kernel="rbf", gamma=1)
    ovr = OneVsRestClassifier(svm)
    ovr.fit(X, y)
    print(ovr.score(X, y))
    create_decision_boundary_plot(X, y, ovr, "SVM Decision Boundary (iris dataset)")


def generate_points():
    file = open("data/custom.data", "w")

    x1 = np.random.normal(0, 0.8, 25)
    x2 = np.random.uniform(-1.2, 1, 25)
    class1 = np.vstack([x1, x2]).T
    temp = class1.copy()
    temp[:, [1, 0]] = temp[:, [0, 1]]
    class1 = np.concatenate((class1, temp))
    class1 = np.append(class1, np.full((50, 1), [0]), axis=1)

    class2 = np.random.normal([1.8, 1.8], [0.5, 0.5], [20, 2])
    class2 = np.concatenate((class2, np.random.normal([-1.8, -1.8], [0.5, 0.5], [20, 2])))
    class2 = np.append(class2, np.full((40, 1), [1]), axis=1)

    class3 = np.random.normal([1.8, -1.8], [0.5, 0.5], [20, 2])
    class3 = np.concatenate((class3, np.random.normal([-1.8, 1.8], [0.5, 0.5], [20, 2])))
    class3 = np.append(class3, np.full((40, 1), [2]), axis=1)

    full = np.concatenate((class1, class2, class3), axis=0)
    np.random.shuffle(full)

    for p in full:
        p[0] = np.around(p[0], decimals=3)
        p[1] = np.around(p[1], decimals=3)
        file.write(str(p[0]) + ", " + str(p[1]) + ", " + str(p[2]) + "\n")

    file.close()


def plot_points():
    X, y = load_data("custom")

    for i in range(len(y)):
        if y[i] == 0:
            plt.scatter(X[i][0], X[i][1], c="r")
        elif y[i] == 1:
            plt.scatter(X[i][0], X[i][1], c="g")
        else:
            plt.scatter(X[i][0], X[i][1], c="b")

    plt.show()


def custom_plots():
    # generate_points()
    # plot_points()

    X, y = load_data("custom")

    svm = SubgradientSVMClassifier(iterations=30, regularizer=0.1, alpha=0.1, kernel="rbf", gamma=1)
    ovr = OneVsRestClassifier(svm)
    ovr.fit(X, y)
    print(ovr.score(X, y))
    create_decision_boundary_plot(X, y, ovr, "SVM Decision Boundary (RBF kernel, custom dataset)")


if __name__ == "__main__":
    # adult_plots()
    # breast_cancer_demo()
    # iris_plots()
    custom_plots()
