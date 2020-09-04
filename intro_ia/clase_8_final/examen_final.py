import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold

# Librerias creadas durante la materia
import model
from dataset import Data

show_plots = True

# Modelo para computar el error
mse = model.MSE()

def k_fold_train(X_train_exp, y_train, k_split):
    
    # Utilizo kf para dividir mi train dataset en slots
    kf = KFold(n_splits=k_split, shuffle=True)
    best_model = []

    for train_index, test_index in kf.split(X_train_exp):
        Xf_train, Xf_val = X_train_exp[train_index], X_train_exp[test_index]
        yf_train, yf_val = y_train[train_index], y_train[test_index]

        # fit model with fold "k"
        lr = model.LinearRegresion()
        lr.fit(Xf_train, yf_train)
    
        y_hat = lr.predict(Xf_val)
    
        mse_val = mse(yf_val, y_hat)

        # Por cada set de k-Folds evaluo si el modelo actual supera en
        # performance a los anteriores. Para eso lo compare contra el error
        # de validacion en cada caso
        if not best_model:
            # No hay mejor modelo almacenado
            best_model.append(mse_val)
            best_model.append(lr)
        else:
            if mse_val < best_model[0]:
                best_model[0] = mse_val
                best_model[1] = lr
    
    # Almaceno el mejor modelo del polinomio evaluado
    return best_model

def polinomic_regresion(X_train, X_test, y_train, y_test, start_n, end_n):

    k_parts = 5
    best_models = []

    for n in range(start_n, end_n+1):
        # Armamos el polinomio de grado n
        X_train_list = [np.power(X_train, i) for i in range(0,n+1)]
        X_train_exp = np.vstack((X_train_list)).T

        # Se ejecuta el fit con k-Fold y se busca el mejor modelo
        # para ese polinomio. El mejor modelo se lo almacena
        # en la lista de modelos, uno por cada "n"
        best_model = k_fold_train(X_train_exp, y_train, k_parts)
        best_model.append(n) # Agrego la información de a que "n" corresponde
        print("Polinomio grado:", n, 'best Val MSE', best_model[0])
        best_models.append(best_model)


    # En best_models se encuentran los mejores modelos para cada "n"
    # Ahora se calcula el error de test para cada uno de ellos:
    for i in range(len(best_models)):
        # Obtenemos el objeto lr y el n
        lr = best_models[i][1]
        n = best_models[i][2]
        X_test_list = [np.power(X_test, i) for i in range(0,n+1)]
        X_test_exp = np.vstack((X_test_list)).T

        y_hat = lr.predict(X_test_exp)
        mse_test = mse(y_test, y_hat)

        # Almacenar el error de test para ese "n"
        best_models[i].append(mse_test)
        print("Polinomio grado:", n, 'Test MSE', best_models[i][3])
    

    best_model = best_models[0]
    for model in best_models[1:]:
        if model[3] < best_model[0]:
            best_model = model

    print('El modelo de mejor polinomio por Val MSE es n=', best_model[2])

    best_model = best_models[0]
    for model in best_models[1:]:
        if model[3] < best_model[3]:
            best_model = model

    print('El modelo de mejor polinomio por Test MSE es n=', best_model[2])

    # Finalmente nos quedamos con el mejor modelo validado por el error de test
    # Aunque se puede obserar que el error de validacion no varia mucho del test
    # Esto quiere decir que no hay overfitting.

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
    plt.scatter(X_test, y_test, label='dataset')
    plt.scatter(X_test, y_hat, label=f'poly n={n}')
    plt.legend()
    if show_plots == True:
        plt.show()

    return best_model


if __name__ == "__main__":
    script_path = os.path.dirname(os.path.realpath(__file__))
    dataset_path = os.path.join(script_path, 'clase_8_dataset.csv')
    dt = Data(dataset_path)

    # Visualizar el dataset entregado
    plt.scatter(dt.dataset[:,0], dt.dataset[:,1], label='dataset')
    plt.legend()
    if show_plots == True:
        plt.show()

    # Partir el dataset en train 80% test 20%
    X_train, X_test, y_train, y_test  = dt.split(0.8)

    # Obtenemos el mejor modelo utilizando la formula cerrada de regresion
    # para los 4 polinimios (start n=1, end n=4)
    best_model = polinomic_regresion(X_train, X_test, y_train, y_test, start_n=1, end_n=4)

    # El mejor modelo resuelto por formula cerrada
    lr = best_model[1]

    # Con el mejor "n=1" crear nuevamente el polinimio pero entrearlo
    # con mini-batch
    n = 1
    X_train_list = [np.power(X_train, i) for i in range(0,n+1)]
    X_train_exp = np.vstack((X_train_list)).T

    X_test_list = [np.power(X_test, i) for i in range(0,n+1)]
    X_test_exp = np.vstack((X_test_list)).T

    model = model.LinearRegresionGradientDescent(model.mini_batch_gradient_descent)
    # NOTA: Por alguna razón el mini-batch diverge para polinomios n >= 3 :(
    model.fit(X_train_exp, y_train, 0.000001, 10)
    error_train_list = model.error_train_list
    error_val_list = model.error_val_list

    plt.plot(error_train_list, label='train error')
    plt.plot(error_val_list, label='val error')
    plt.legend()
    if show_plots == True:
        plt.show()

    y_hat = model.predict(X_test_exp)
    mse_test = mse(y_test, y_hat)

    # El mejor modelo resuelto por formula cerrada
    n_best = best_model[2]
    X_test_list = [np.power(X_test, i) for i in range(0,n_best+1)]
    X_test_exp = np.vstack((X_test_list)).T

    y_hat_lr = lr.predict(X_test_exp)
    mse_test = mse(y_test, y_hat_lr)

    plt.scatter(X_test, y_test, label='dataset')
    plt.scatter(X_test, y_hat_lr, label=f'poly n={n}')
    plt.scatter(X_test, y_hat, label='mini_bath')
    plt.legend()
    if show_plots == True:
        plt.show()