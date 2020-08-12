import numpy as np
import matplotlib.pyplot as plt

import model
from dataset import CsvData


if __name__ == '__main__':

    dataset = CsvData('income.data.csv')
    offset = np.zeros(dataset.shape())
    offset[:, 1] = 3*np.ones(dataset.shape()[0])
    dataset.add(offset)

    X_train, X_test, y_train, y_test  = dataset.split(0.8)

    plt.scatter(X_train, y_train, color='b', label='dataset')
    
    lr = model.LinearRegresion()

    for i in range(4):

        if i == 0:
            # Constante model
            ni = 0
            nf = 1
        elif i == 1:
            # Linear Model
            ni = 1
            nf = 2
        elif i == 2:
            # Linearl+B model
            ni = 0
            nf = 2
        elif i == 3:
            # Linearl+B model gradient descent
            ni = 0
            nf = 2


        # Armamos el polinomio de grado n
        X_train_list = [np.power(X_train, i) for i in range(ni,nf)]
        X_test_list = [np.power(X_test, i) for i in range(ni,nf)]
    
        X_train_exp = np.vstack((X_train_list)).T
        X_test_exp = np.vstack((X_test_list)).T

        if i != 3:
            lr.fit(X_train_exp, y_train)
        else:
            lr = model.LinearRegresionGradientDescent(model.mini_batch_gradient_descent)
            lr.fit(X_train_exp, y_train, 0.001, 10000)

        mse = model.MSE()
        
        lr_y_hat = lr.predict(X_test_exp)
        lr_mse = mse(y_test, lr_y_hat)

        x_plot = np.linspace(0,10,10)
        x_plot_list = [np.power(x_plot, i) for i in range(ni,nf)]
        x_plot_exp = np.vstack((x_plot_list)).T
        lr_y_plot = np.matmul(x_plot_exp, lr.model)

        plt.plot(x_plot, lr_y_plot, label=f'Model={i}, MSE={lr_mse:.3f}')

    plt.legend()
    plt.show()
