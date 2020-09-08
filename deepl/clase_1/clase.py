import numpy as np
import matplotlib.pyplot as plt

# Librerias creadas durante la materia
import model
#from dataset import Data

# Modelo para computar el error
mse = model.MSE()

def sigmoid(z):
    return 1 / (1+np.exp(-z))

def rnn_sgd(X, y, epochs, lr=0.1):

    n = X.shape[0]
    m = X.shape[1]

    # Random init
    W11 = np.random.rand(m)
    W12 = np.random.rand(m)
    W2 = np.random.rand(m)

    y_hat = np.zeros(n)
    err_mse = []

    for i in range(epochs):
        for j in range(n):

            #forward_propagation
            z11 = W11.T @ X[j, :]
            a11 = sigmoid(z11)
            z12 = W12.T @ X[j, :]
            a12 = sigmoid(z12)

            x2_exp = np.array([a11, a12, 1])
            z2 = W2.T @ x2_exp
            a2 = sigmoid(z2)

            # Calcular y almacenar el error
            y_hat[j] = a2
            err = y[j] - y_hat[j]

            # back_propagation
            dj_dw2 = np.array([(-2*err) * a11, (-2*err) * a12, (-2*err) * 1])
            dj_dz11 = (-2*err) * W2[0] * sigmoid(z11) * (1-sigmoid(z11))
            dj_dz12 = (-2*err) * W2[1] * sigmoid(z12) * (1-sigmoid(z12))
            dj_dw11 = np.array([dj_dz11 * X[j,0], dj_dz11 * X[j,1], dj_dz11 * 1])
            dj_dw12 = np.array([dj_dz12 * X[j,0], dj_dz12 * X[j,1], dj_dz12 * 1])
            
            W2 = W2 - lr * (dj_dw2)
            W11 = W11 - lr * (dj_dw11)
            W12 = W12 - lr * (dj_dw12)

        # Calcular el error cuadrático medio
        err_mse.append(mse(y, y_hat))

    fig = plt.figure()
    ax = fig.add_subplot()
    ax.plot(err_mse, color='b', label=f'mse, lr={lr}')
    ax.set_title("RNN fit")
    ax.set_ylabel("mse")
    ax.set_xlabel("epoch")
    ax.legend()
    plt.show()


if __name__ == "__main__":
    # XOR con un modelo lineal
    #             x1, x2, b=1
    x = np.array([[0, 0, 1], 
                  [0, 1, 1], 
                  [1, 0, 1], 
                  [1, 1, 1]])
    
    # Salida de la XOR con LinearRegresion
    y = np.array([0, 1, 1, 0])
    lr = model.LinearRegresion()
    lr.fit(x, y)
    print(lr.model)

    # Respuesta w = 0 0 0.5

    # Salidar de la XOR con RNN
    rnn_sgd(x, y, 2000)