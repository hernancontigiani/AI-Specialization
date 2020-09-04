'''
1. Leer dataset "clase_6_dataset"
2. Aplicar Logistic Regresion
3. Hacer fit con y = w2 *x1 + w1 * x2 + 1
4. Graficar el resultado
'''

import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold

from sklearn.linear_model import LogisticRegression


class Data(object):

    def __init__(self, path):
        self.dataset = self._build_dataset(path)
        
    def _build_dataset(self, path):
        # usar numpy esctructurado
        data = np.genfromtxt(path, delimiter=',')
        return data
        # np.fromiter

    def shape(self):
        return self.dataset.shape

    def add(self, offset):
        self.dataset += offset

    def split(self, percentage):
        # retornar train y test segun el %
        dim = self.dataset.ndim
        X = self.dataset[:,0:dim]
        y = self.dataset[:,dim]
        #train_test_split(X, y, test_size=(1-percentage))
        X_train, X_test, y_train, y_test  = train_test_split(X, y, test_size=(1-percentage))
        return X_train, X_test, y_train, y_test

class CsvData(Data):
    def _build_dataset(self, path):
        # usar numpy esctructurado
        data = np.genfromtxt(path, delimiter=',')
        data = data[1:,1:]
        return data
        # np.fromiter


class BaseModel(object):

    def __init__(self):
        self.model = None

    def fit(self, X, Y):
        # train model
        return NotImplemented

    def predict(self, X):
        # retorna y hat
        return NotImplemented


class ConstantModel(BaseModel):

    def fit(self, X, Y):
        # Calcular los W y guadarlos en el modelo
        W = Y.mean()
        self.model = W

    def predict(self, X):
        # usar el modelo (self.model) y predecir
        # y hat a partir de X e W
        return np.ones(len(X)) * self.model


class LinearRegresion(BaseModel):

    def fit(self, X, Y):
        # Calcular los W y guadarlos en el modelo
        if X.ndim == 1:
            W = X.T.dot(Y) / X.T.dot(X)
        else:
            W = np.linalg.inv(X.T.dot(X)).dot(X.T).dot(Y)
        self.model = W

    def predict(self, X):
        # usar el modelo (self.model) y predecir
        # y_hat a partir de X e W
        
        if X.ndim == 1:
            return self.model * X
        else:
            return np.matmul(X, self.model)


class Metric(object):
    def __call__(self, target, prediction):
        # target --> Y
        # prediction --> Y hat
        return NotImplemented


class MSE(Metric):
    def __call__(self, target, prediction):
        # Implementar el error cuadratico medio MSE
        return np.square(target-prediction).mean()

def gradient_descent(X_train, y_train, lr=0.01, amt_epochs=100):
    """
    shapes:
        X_t = nxm
        y_t = nx1
        W = mx1
    """
    n = X_train.shape[0]
    m = X_train.shape[1]

    # initialize random weights
    W = np.random.randn(m).reshape(m, 1)

    for i in range(amt_epochs):
        prediction = np.matmul(X_train, W)  # nx1
        error = y_train - prediction  # nx1

        grad_sum = np.sum(error * X_train, axis=0)
        grad_mul = -2/n * grad_sum  # 1xm
        gradient = np.transpose(grad_mul).reshape(-1, 1)  # mx1

        W = W - (lr * gradient)

    return W


def stochastic_gradient_descent(X_train, y_train, lr=0.01, amt_epochs=100):
    """
    shapes:
        X_t = nxm
        y_t = nx1
        W = mx1
    """
    n = X_train.shape[0]
    m = X_train.shape[1]

    # initialize random weights
    W = np.random.randn(m).reshape(m, 1)

    for i in range(amt_epochs):
        idx = np.random.permutation(X_train.shape[0])
        X_train = X_train[idx]
        y_train = y_train[idx]

        for j in range(n):
            prediction = np.matmul(X_train[j].reshape(1, -1), W)  # 1x1
            error = y_train[j] - prediction  # 1x1

            grad_sum = error * X_train[j]
            grad_mul = -2/n * grad_sum  # 2x1
            gradient = np.transpose(grad_mul).reshape(-1, 1)  # 2x1

            W = W - (lr * gradient)

    return W


def mini_batch_gradient_descent(X_train, y_train, lr=0.01, amt_epochs=100):
    """
    shapes:
        X_t = nxm
        y_t = nx1
        W = mx1
    """
    b = 16
    n = X_train.shape[0]
    m = X_train.shape[1]

    # initialize random weights
    W = np.random.randn(m).reshape(m, 1)

    for i in range(amt_epochs):
        idx = np.random.permutation(X_train.shape[0])
        X_train = X_train[idx]
        y_train = y_train[idx]

        batch_size = int(len(X_train) / b)
        for i in range(0, len(X_train), batch_size):
            end = i + batch_size if i + batch_size <= len(X_train) else len(X_train)
            batch_X = X_train[i: end]
            batch_y = y_train[i: end]

            prediction = np.matmul(batch_X, W)  # nx1
            error = batch_y - prediction  # nx1

            grad_sum = np.sum(error * batch_X, axis=0)
            grad_mul = -2/n * grad_sum  # 1xm
            gradient = np.transpose(grad_mul).reshape(-1, 1)  # mx1

            W = W - (lr * gradient)

    return W


def mini_batch_logistic_regression(X_train, y_train, lr=0.01, amt_epochs=100):
    """
    shapes:
        X_t = nxm
        y_t = nx1
        W = mx1
    """
    b = 16
    m = X_train.shape[1]

    # initialize random weights
    W = np.random.randn(m).reshape(m, 1)

    for i in range(amt_epochs):
        idx = np.random.permutation(X_train.shape[0])
        X_train = X_train[idx]
        y_train = y_train[idx]

        batch_size = int(len(X_train) / b)
        for i in range(0, len(X_train), batch_size):
            end = i + batch_size if i + batch_size <= len(X_train) else len(X_train)
            batch_X = X_train[i: end]
            batch_y = y_train[i: end]

            exponent = np.sum(np.transpose(W) * batch_X, axis=1)
            prediction = 1/(1 + np.exp(-exponent))
            error = prediction.reshape(-1, 1) - batch_y.reshape(-1, 1)

            grad_mul = (1/b) * np.matmul(batch_X.T, error)    # 1xm
            gradient = np.transpose(grad_mul).reshape(-1, 1)  # mx1

            W = W - (lr * gradient)

    return W


def predict(X, W):
    wx1 = np.matmul(X, W)
    wx = np.concatenate(wx1, axis=0 )
    sigmoid = 1/(1 + np.exp(-wx))
    #sigmoid = [1 if x >= 0.5 else 0 for x in sigmoid]
    sigmoid = np.where(sigmoid >= 0.5, 1.0, 0.0)
    return sigmoid


if __name__ == '__main__':

    dataset = Data('clase_6_dataset.txt')

    X_train, X_test, y_train, y_test  = dataset.split(0.8)

    X_train_expanded = np.vstack((X_train[:, 0], X_train[:, 1], np.ones(len(X_train)))).T
    X_test_expanded = np.vstack((X_test[:, 0], X_test[:, 1], np.ones(len(X_test)))).T

    lr = 0.001
    epochs = 50000
    W = mini_batch_logistic_regression(X_train_expanded, y_train.reshape(-1, 1), lr=lr, amt_epochs=epochs)


    y_hat = predict(X_test_expanded, W)
    print('Predicted (y_hat)')
    print(y_hat)
    print('Real (y_test)')
    print(y_test)

    # y_hat = w0 * x1 + w1 * x2 + w3(b)
    # 0 = w0 * x1 + w1 * x2 + w3(b) = w0 * x + w1 * y + w3(b)
    # y = ( -w0 *x - w3) / w1
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

