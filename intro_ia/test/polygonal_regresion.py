import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold

import model
from dataset import Data


if __name__ == '__main__':

    # Simular señal senoidal con desvio 0.4
    sample_size = 1000
    X = np.linspace(0, 4*np.pi, sample_size)
    y = np.sin(X) + np.random.normal(loc=0, scale=0.40, size=sample_size)

    #X_train, X_test, y_train, y_test  = train_test_split(X, y, test_size=(0.2))
    X_train, X_test, y_train, y_test  = Data.split_static(X, y, 0.8)

    # Graficar la señal
    # plt.scatter(X, y)
    # plt.show()

    mse = model.MSE()
    best_models = []

    for n in range(0,11):
        # Armamos el polinomio de grado n
        X_train_list = [np.power(X_train, i) for i in range(0,n+1)]
        X_train_exp = np.vstack((X_train_list)).T

        # Utilizo kf para dividir mi train dataset en slots
        kf = KFold(n_splits=5, shuffle=True)
        best_model = []
        #print("Polinomio grado:", n)
        for train_index, test_index in kf.split(X_train_exp):
            #print("TRAIN:", train_index, "TEST:", test_index)
            Xf_train, Xf_test = X_train_exp[train_index], X_train_exp[test_index]
            yf_train, yf_test = y_train[train_index], y_train[test_index]

            # fit model with fold "k"
            lr = model.LinearRegresion()
            lr.fit(Xf_train, yf_train)
            #lr = model.LinearRegresionGradientDescent(model.mini_batch_gradient_descent)
            #lr.fit(Xf_train, yf_train, 0.0001, 1000)
        
            y_hat = lr.predict(Xf_test)
        
            mse_test = mse(yf_test, y_hat)

            if not best_model:
                # No hay mejor modelo almacenado
                best_model.append(mse_test)
                best_model.append(lr)
                best_model.append(n)
            else:
                if mse_test < best_model[0]:
                    best_model[0] = mse_test
                    best_model[1] = lr
                    best_model[2] = n
        
        # Almaceno el mejor modelo del polinomio evaluado
        print("Polinomio grado:", n, 'best MSE', best_model[0])
        best_models.append(best_model)
    
    best_model = best_models[0]
    for model in best_models[1:]:
        if model[0] < best_model[0]:
            best_model = model

    print('El modelo de mejor polinomio es n=', best_model[2])

    # Recupero el modelo
    lr = best_model[1]
    n = best_model[2]

    # Creo el polinomio de test de grado n
    X_test_list = [np.power(X_test, i) for i in range(0,n+1)]
    X_test_exp = np.vstack((X_test_list)).T

    y_hat = lr.predict(X_test_exp)
    mse_test = mse(y_test, y_hat)

    print('Test MSE:', mse_test)
    
    # Graficar la señal
    plt.scatter(X, y, label='dataset')
    plt.scatter(X_test, y_hat, label=f'poly n={n}')
    plt.legend()
    plt.show()

