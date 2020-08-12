import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold


def k_means(X, n_clusters, iterations=100):
    centroids = np.random.rand(n_clusters, X.shape[1])
    for i in range(iterations):
        centroids, cluster_ids = k_means_loop(X, centroids)
    
    return centroids, cluster_ids


def k_means_loop(X, centroids):
    # find labels for rows in X based in centroids values
    expanded_centroids = centroids[:, None]
    distances = np.sqrt(np.sum((expanded_centroids - X) ** 2, axis=2))
    arg_min = np.argmin(distances, axis=0)
    # recompute centroids
    i = 5
    for i in range(centroids.shape[0]):
        val = X[arg_min == i, :]
        if val.size > 0:
            centroids[i] = np.mean(X[arg_min == i, :], axis=0)

    return centroids, arg_min


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


class BaseModel(object):

    def __init__(self):
        self.model = None

    def fit(self, X, Y):
        # train model
        return NotImplemented

    def predict(self, X):
        # retorna y hat
        return NotImplemented



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


class LogisticRegresion(BaseModel):

    def fit(self, X, Y, lr=0.01, epochs=100):
        self.model = mini_batch_logistic_regression(X, Y, lr, epochs)

    def predict(self, X):
        wx1 = np.matmul(X, self.model)
        wx = np.concatenate(wx1, axis=0 )
        sigmoid = 1/(1 + np.exp(-wx))
        y_hat = np.where(sigmoid >= 0.5, 1.0, 0.0)
        return y_hat
