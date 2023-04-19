import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from svm import SVM

def gen_non_linear_data():
    n_samples = 1000
    X, y = make_classification(n_samples=n_samples, n_features=2, 
                            n_informative=2, n_redundant=0, n_repeated=0, 
                            n_classes=2, n_clusters_per_class=1, 
                            random_state=92)
    y = y * 2 - 1 # (y = 0 or 1) => (y = -1 or 1)
    X_train, X_test = np.split(X, (n_samples * 8 // 10, ))
    y_train, y_test = np.split(y, (n_samples * 8 // 10, ))

    plt.scatter(X[y > 0, 0], X[y > 0, 1], c = 'red', s=10)
    plt.scatter(X[y < 0, 0], X[y < 0, 1], c = 'blue', s=10)
    plt.savefig('img/data.png')
    plt.show()

    return (X_train, y_train, X_test, y_test)


def plot_contour(X1_train, X2_train, clf, kernel_name):
    plt.scatter(clf.sv[:, 0], clf.sv[:, 1], c='green', s=45)
    plt.scatter(X1_train[:, 0], X1_train[:, 1], c='red', s=10)
    plt.scatter(X2_train[:, 0], X2_train[:, 1], c='blue', s=10)
    

    X1, X2 = np.meshgrid(np.linspace(-4.5, 4.5, 50), np.linspace(-4.5, 2.5, 50))
    X = np.array([[x1, x2] for x1, x2 in zip(np.ravel(X1), np.ravel(X2))])
    Z = clf.project(X).reshape(X1.shape)
    plt.contour(X1, X2, Z, [0.0], colors='k', linewidths=1, origin='lower')
    plt.contour(X1, X2, Z + 1, [0.0], colors='grey', linewidths=1, origin='lower')
    plt.contour(X1, X2, Z - 1, [0.0], colors='grey', linewidths=1, origin='lower')

    plt.axis("tight")
    plt.title(kernel_name + ' kernel')
    plt.savefig('img/' + kernel_name + '.png')
    plt.show()


def test_non_linear(data, kernel):
    X_train, y_train, X_test, y_test = data

    clf = SVM(kernel=kernel, C=1)    # change kernel here
    clf.fit(X_train, y_train)

    y_predict = clf.predict(X_test)
    correct = np.sum(y_predict == y_test)
    print("%d out of %d predictions correct" % (correct, len(y_predict)))
    print("Test accuracy: %.4f%%" % (correct / len(y_predict)))

    plot_contour(X_train[y_train == 1], X_train[y_train == -1], clf, kernel_name=kernel)

if __name__ == "__main__":
    data = gen_non_linear_data()

    test_non_linear(data, 'linear')
    test_non_linear(data, 'polynomial')
    test_non_linear(data, 'gaussian')