'''
1. Simular una función sin(x) con ruido
2. Hacer el gráfico de los datos
3. Hacer fit con y = w1 X + w2
4. Hacer fit con y = w1 X^2 + w2 X + w3
5. Hacer fir para diferentes polinomios hasta 10
6. Obtener mediante cross-validation para cada polinomio el error de validación (k-folds)
7. Seleccionar el modelo con complejidad correcta para el dataset (usando el modelo que minimiza el validation error obtenido en 6)
8. Obtener el ECM sobre el dataset de test.
9. Regularizar el modelo para mejorar la generalización del modelo (probar agregando mas ruido al sin(x))
'''

import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold


class Data(object):

    def __init__(self, path):
        self.dataset = self._build_dataset(path)
        
    def _build_dataset(self, path):
        # usar numpy esctructurado
        data = np.genfromtxt(path, delimiter=',')
        data = data[1:,1:]
        return data
        # np.fromiter

    def shape(self):
        return self.dataset.shape

    def add(self, offset):
        self.dataset += offset

    def split(self, percentage):
        # retornar train y test segun el %
        X = self.dataset[:,0]
        y = self.dataset[:,1]
        #train_test_split(X, y, test_size=(1-percentage))
        X_train, X_test, y_train, y_test  = train_test_split(X, y, test_size=(1-percentage))
        return X_train, X_test, y_train, y_test


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


if __name__ == '__main__':
    # Simular señal senoidal con desvio 0.4
    sample_size = 1000
    X = np.linspace(0, 4*np.pi, sample_size)
    y = np.sin(X) + np.random.normal(loc=0, scale=0.40, size=sample_size)

    X_train, X_test, y_train, y_test  = train_test_split(X, y, test_size=(0.2))

    # Graficar la señal
    # plt.scatter(X, y)
    # plt.show()

    mse = MSE()
    best_models = []

    for n in range(11):
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
            lr = LinearRegresion()
            lr.fit(Xf_train, yf_train)
        
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
