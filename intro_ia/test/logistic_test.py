import numpy as np
import matplotlib.pyplot as plt

import model
from dataset import Data


if __name__ == '__main__':

    dataset = Data('clase_6_dataset.txt')

    X_train, X_test, y_train, y_test  = dataset.split(0.8)

    X_train_expanded = np.vstack((X_train[:, 0], X_train[:, 1], np.ones(len(X_train)))).T
    X_test_expanded = np.vstack((X_test[:, 0], X_test[:, 1], np.ones(len(X_test)))).T

    logreg = model.LogisticRegression()

    lr = 0.001
    epochs = 50000
    logreg.fit(X_train_expanded, y_train.reshape(-1, 1), lr=lr, epochs=epochs)


    y_hat = logreg.predict(X_test_expanded)
    print('Predicted (y_hat)')
    print(y_hat)
    print('Real (y_test)')
    print(y_test)

    # y_hat = w0 * x1 + w1 * x2 + w3(b)
    # 0 = w0 * x1 + w1 * x2 + w3(b) = w0 * x + w1 * y + w3(b)
    # y = ( -w0 *x - w3) / w1
    W = logreg.model
    x_regression = np.linspace(30, 100, 70)
    y_regression = (-x_regression*W[0] - W[2])/W[1] 

    zeros = y_train < 0.5
    ones = y_train >= 0.5

    X_train_zeros = X_train[zeros]
    y_train_zeros = y_train[zeros]

    X_train_ones = X_train[ones]
    y_train_ones = y_train[ones]

    plt.scatter(X_train_zeros[:,0], X_train_zeros[:,1], marker='*')
    plt.scatter(X_train_ones[:,0], X_train_ones[:,1], marker='+')
    plt.plot(x_regression, y_regression, c='r')
    plt.show()