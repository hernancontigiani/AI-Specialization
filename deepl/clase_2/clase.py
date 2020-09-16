import os
import numpy as np
import matplotlib.pyplot as plt

# Librerias creadas durante la materia
import model
from dataset import CsvData

# Modelo para computar el error
mse = model.MSE()

def sigmoid(z):
    return 1 / (1+np.exp(-z))


class inputLayer():
    def __init__(self, inputs, outputs):
        self.inputs = inputs
        self.n = outputs

        self.W = np.random.rand(inputs, outputs)
        self.b = np.random.rand(outputs).reshape(-1, 1)

        # print('shape w:', self.W.shape)
        # print('shape b:', self.b.shape)

        self.last_input = 0
        self.a = 0
        self.z = 0
        self.dz = 0

    def fowardpropagation(self, X):
        self.last_input = X
        # print('shape w:', self.W.shape)
        # print('shape b:', self.b.shape)
        # print('shape X:', self.last_input.shape)
        self.a = self.W.T @ self.last_input + self.b
        self.z = sigmoid(self.a)

    def backwardpropagation(self, W, dz):
        self.dz = (W @ dz) * (sigmoid(self.z) * (1-sigmoid(self.z)))
        
        self.delta_W = (1/self.dz.shape[1]) * (self.dz @ self.last_input.T)
        self.delta_b = (1/self.dz.shape[1]) * (np.sum(self.dz, axis=1, keepdims=True))


    def update(self, lr):
        self.W = self.W - lr * self.delta_W.T
        self.b = self.b - lr * self.delta_b


class outputLayer(inputLayer):
    def backwardpropagation(self, Y):
        self.dz = -2 * (Y-self.a) * sigmoid(self.z) * (1-sigmoid(self.z))

        self.delta_W = (1/self.dz.shape[1]) * (self.dz @ self.last_input.T)
        self.delta_b = (1/self.dz.shape[1]) * (np.sum(self.dz, axis=1, keepdims=True))


def rnn_batch(X, y, epochs, lr=0.1):

    n = X.shape[0]
    m = X.shape[1]

    # Init
    l1 = inputLayer(m, 3)
    l2 = inputLayer(3, 2)
    out = outputLayer(2, 1)

    batch_size = 32

    #y_hat = np.zeros(n)
    err_mse = []

    for i in range(epochs):
        for j in range(0, n, batch_size):
            X_batch = X[j:(j+batch_size), :].T
            y_batch = y[j:(j+batch_size)]

            #forward_propagation
            l1.fowardpropagation(X_batch)
            l2.fowardpropagation(l1.a)
            out.fowardpropagation(l2.a)

            # Calcular y almacenar el error
            err = y_batch - out.a
            err_j = (1/batch_size) * np.sum(np.power(err,2))
            err_mse.append(err_j)
            #print(err)

            # backward_propagation
            out.backwardpropagation(y_batch)
            l2.backwardpropagation(out.W, out.dz)
            l1.backwardpropagation(l2.W, l2.dz)

            # update
            out.update(lr)
            l2.update(lr)
            l1.update(lr)


    fig = plt.figure()
    ax = fig.add_subplot()
    ax.plot(err_mse, color='b', label=f'mse, lr={lr}')
    ax.legend()
    plt.show()


if __name__ == "__main__":
    # Obtener el path y el nombre del dataset
    script_path = os.path.dirname(os.path.realpath(__file__))
    dataset_path_name = os.path.join(script_path, 'train_data.csv')

    dataset = np.genfromtxt(dataset_path_name, delimiter=',')
    dim = dataset.shape[1]
    if dim > 2:
        X = dataset[:,0:(dim-1)]
    else:
        X = dataset[:,0]
    y = dataset[:,dim-1]

    rnn_batch(X, y, 100, 0.5)
