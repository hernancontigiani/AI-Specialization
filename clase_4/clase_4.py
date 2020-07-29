'''
1. Escribir una clase dataset que levante los datos en un numpy array estructurado
2. La clase tiene que tener un metodo para separar los datos en 80% train y 20% test
3. Hacer una clase que implemente regresion lineal
4. Regresion lineal con b
5. Prediccion
6. Graficos
7. ECM --> MCE+
'''

import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

'''
structure = [('word', np.dtype('U' + str(cls.WORD_MAX_SIZE))),
             ('embedding', np.float32, (cls.N_FEATURES,))]
structure = np.dtype(structure)
# load numpy array from disk using a generator
with open(cls.WORD_TO_VEC_MODEL_TXT_PATH, encoding="utf8") as words_embeddings_txt:
    embeddings_gen = ((line.split()[0], line.split()[1:])
                      for line in words_embeddings_txt)
    embeddings = np.fromiter(embeddings_gen, structure)
'''

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

    dataset = Data('income.data.csv')

    offset = np.zeros(dataset.shape())
    offset[:, 1] = 3*np.ones(dataset.shape()[0])
    dataset.add(offset)

    X_train, X_test, y_train, y_test  = dataset.split(0.8)
   
    lr = LinearRegresion()

    val_mse = []
    test_mse = []
    
    for n in range(0,11):
        # Armamos el polinomio de grado n
        X_train_list = [np.power(X_train, i) for i in range(0,n+1)]
        X_test_list = [np.power(X_test, i) for i in range(0,n+1)]
    
        X_train_exp = np.vstack((X_train_list)).T
        X_test_exp = np.vstack((X_test_list)).T

        lr.fit(X_train_exp, y_train)

        mse = MSE()
        
        lr_y_hat_val = lr.predict(X_train_exp)
        lr_y_hat_test = lr.predict(X_test_exp)
        
        lr_mse_val = mse(y_train, lr_y_hat_val)
        lr_mse_test = mse(y_test, lr_y_hat_test)

        val_mse.append(lr_mse_val)
        test_mse.append(lr_mse_test)


    plt.plot(val_mse, color='b', label='val_mse')
    plt.plot(test_mse, color='g', label='test_mse')
    plt.legend()
    plt.show()

    